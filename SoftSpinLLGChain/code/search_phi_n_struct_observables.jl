#!/usr/bin/env julia

include(joinpath(@__DIR__, "search_phi_n_observables.jl"))

using LinearAlgebra
using Printf
using Statistics

Base.@kwdef struct PhiNStructSearchConfig
    dm_config::String
    right_family::String = "core"
    suffix::String = "phin_struct_core_v1"
    pairs_per_lag::Int = 18000
    max_channels::Int = 48
    signal_fraction_min::Float64 = 0.020
    split_corr_min::Float64 = 0.65
    split_rel_max::Float64 = 1.10
    ridge::Float64 = 1.0e-6
end

struct PhiNStructRow
    left_observable::String
    left_index::Int
    right_observable::String
    right_index::Int
    signal_rms::Float64
    signal_fraction::Float64
    split_corr::Float64
    split_rel::Float64
    struct_rms::NTuple{3, Float64}
    selected_rank::Int
    greedy_gain::Float64
end

function parse_struct_args(args)
    dm_config = length(args) >= 1 ? args[1] :
        normpath(joinpath(@__DIR__, "..", "configs",
            "fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.toml"))
    right_family = length(args) >= 2 ? args[2] : "core"
    suffix = length(args) >= 3 ? args[3] : "phin_struct_core_v1"
    pairs = length(args) >= 4 ? parse(Int, args[4]) : 18000
    max_channels = length(args) >= 5 ? parse(Int, args[5]) : 48
    return PhiNStructSearchConfig(; dm_config, right_family, suffix,
        pairs_per_lag=pairs, max_channels)
end

function equivariant_basis_right_actions(x0::Array{Float32, 3},
        rraw::Array{Float32, 3}, right_grad::Array{Float32, 4})
    N, R, _, B = size(right_grad)
    outs = [zeros(Float32, N, R, B) for _ in 1:3]
    @inbounds for b in 1:B, i in 1:N
        x = Float64(x0[i, 1, b])
        y = Float64(x0[i, 2, b])
        z = Float64(x0[i, 3, b])
        r1 = Float64(rraw[i, 1, b])
        r2 = Float64(rraw[i, 2, b])
        r3 = Float64(rraw[i, 3, b])
        rr = max(x*x + y*y + z*z, 1e-10)
        dotxr = x*r1 + y*r2 + z*r3
        para = (x*dotxr/rr, y*dotxr/rr, z*dotxr/rr)
        perp = (r1 - para[1], r2 - para[2], r3 - para[3])
        skew = (z*r2 - y*r3, -z*r1 + x*r3, y*r1 - x*r2)
        for a in 1:R
            g1 = Float64(right_grad[i, a, 1, b])
            g2 = Float64(right_grad[i, a, 2, b])
            g3 = Float64(right_grad[i, a, 3, b])
            outs[1][i, a, b] = Float32(perp[1]*g1 + perp[2]*g2 + perp[3]*g3)
            outs[2][i, a, b] = Float32(para[1]*g1 + para[2]*g2 + para[3]*g3)
            outs[3][i, a, b] = Float32(skew[1]*g1 + skew[2]*g2 + skew[3]*g3)
        end
    end
    return outs
end

function evaluate_struct_library(cfg::PhiNStructSearchConfig)
    dm_cfg = load_dm_config(cfg.dm_config)
    base = dirname(cfg.dm_config)
    device = detect_spin_device(dm_cfg.device, dm_cfg.required_gpu_name)
    activate_and_describe_device!(device, dm_cfg.device, dm_cfg.required_gpu_name)

    data_h5 = resolve_path(base, dm_cfg.input_hdf5)
    score_path = resolve_path(base, dm_cfg.score_bson)
    cond_path = resolve_path(base, dm_cfg.cond_score_bson)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, dm_cfg.burnin_fraction,
        dm_cfg.tau_max_decorrelation_multiples, dm_cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    cond_kind = configured_cond_score_kind(cfg.dm_config)
    cond_model, cond_params = load_transition_source(cond_kind, cfg.dm_config,
        cond_path, base, device)
    @printf("Structured phi_n search transition source kind: %s\n", String(cond_kind))

    retained = load_retained_channels(resolve_path(base, dm_cfg.retained_channels_toml))
    left_names = unique_observable_names(retained)
    right_names = right_candidate_names(cfg.right_family)
    left_lib = NonlinearLibrary(left_names)
    right_lib = RightObservableLibrary(right_names)
    nleft = length(left_names)
    nright = length(right_names)
    active_lag_indices = configured_lag_indices(cfg.dm_config,
        min(dm_cfg.max_lags, length(sampler.lag_steps)))
    lags = sampler.lag_steps[1:min(dm_cfg.max_lags, length(sampler.lag_steps))]
    active_lags = lags[active_lag_indices]
    left_means = estimate_nonlinear_means(sampler, p, left_lib,
        min(dm_cfg.target_mean_samples, 160000), MersenneTwister(dm_cfg.seed + 710))
    _Phi_true, phi_block = oracle_true_mean_mobility(sampler, p,
        min(dm_cfg.target_mean_samples, 160000), dm_cfg.seed + 711)

    N = sampler.N
    L = length(active_lags)
    signal = zeros(Float64, L, nleft, nright, N)
    split1 = similar(signal)
    split2 = similar(signal)
    parts = zeros(Float64, L, nleft, nright, 3, N)
    rng = MersenneTwister(dm_cfg.seed + 712)
    half = cfg.pairs_per_lag ÷ 2
    require_condition(half >= 1000, "pairs_per_lag too small for split diagnostics.")

    for (li, lag) in enumerate(active_lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, cfg.pairs_per_lag, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm,
            stats, cond_params, device; batch_size=min(cond_params.batch_size, cfg.pairs_per_lag),
            score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        left = nonlinear_observables(xt, p, left_lib)
        center_observables!(left, left_means)
        _, right_grad, _ = right_observable_value_grad_hess(x0, right_lib)
        full_right, _ = delta_entry_right_actions(x0, rraw, right_grad, p, phi_block)
        basis_right = equivariant_basis_right_actions(x0, rraw, right_grad)
        signal[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            1:cfg.pairs_per_lag)
        split1[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            1:half)
        split2[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            (half + 1):cfg.pairs_per_lag)
        for e in 1:3
            parts[li, :, :, e, :] .= rightobs_profiles(left, basis_right[e], N,
                nleft, nright, 1:cfg.pairs_per_lag)
        end
        @printf("Structured phi_n lag %.5g (%d/%d), left=%d right=%d pairs=%d\n",
            lag * sampler.save_dt, li, L, nleft, nright, cfg.pairs_per_lag)
        GC.gc()
    end

    return (; dm_cfg, sampler, cond_kind, left_names, right_names,
        active_lag_indices, active_lags, taus=active_lags .* sampler.save_dt,
        signal, split1, split2, parts, phi_block, device_request=dm_cfg.device,
        required_gpu_name=dm_cfg.required_gpu_name)
end

function struct_rows(eval, cfg::PhiNStructSearchConfig)
    max_signal = maximum([sqrt(mean(abs2, phin_channel_vector(eval.signal, a, b)))
        for a in eachindex(eval.left_names), b in eachindex(eval.right_names)])
    rows = PhiNStructRow[]
    matrices = Matrix{Float64}[]
    for a in eachindex(eval.left_names), b in eachindex(eval.right_names)
        v = phin_channel_vector(eval.signal, a, b)
        v1 = phin_channel_vector(eval.split1, a, b)
        v2 = phin_channel_vector(eval.split2, a, b)
        sig = sqrt(mean(abs2, v))
        sfrac = sig / max(max_signal, eps(Float64))
        scorr = safe_corr(v1, v2)
        srel = sqrt(mean((v1 .- v2) .^ 2)) / max(sqrt(mean(abs2, v)), eps(Float64))
        erms = ntuple(e -> sqrt(mean(abs2, @view eval.parts[:, a, b, e, :])), 3)
        G = zeros(Float64, size(eval.parts, 1) * size(eval.parts, 5), 3)
        for e in 1:3
            G[:, e] .= vec(@view eval.parts[:, a, b, e, :])
        end
        push!(rows, PhiNStructRow(eval.left_names[a], a, eval.right_names[b], b,
            sig, sfrac, scorr, srel, erms, 0, 0.0))
        push!(matrices, G)
    end
    return rows, matrices
end

function valid_candidate(row::PhiNStructRow, cfg::PhiNStructSearchConfig)
    return row.signal_fraction >= cfg.signal_fraction_min &&
        row.split_corr >= cfg.split_corr_min &&
        row.split_rel <= cfg.split_rel_max &&
        maximum(row.struct_rms) > 0
end

function greedy_select_struct(rows::Vector{PhiNStructRow}, matrices, cfg::PhiNStructSearchConfig)
    valid = findall(row -> valid_candidate(row, cfg), rows)
    require_condition(!isempty(valid), "No structured phi_n channel passed stability filters.")
    col_norm = zeros(Float64, 3)
    for idx in valid
        col_norm .+= vec(sum(abs2, matrices[idx]; dims=1))
    end
    col_scale = sqrt.(max.(col_norm, eps(Float64)))
    scaled = [mat ./ reshape(col_scale, 1, 3) for mat in matrices]
    gram = cfg.ridge .* Matrix{Float64}(I, 3, 3)
    selected = Int[]
    selected_pairs = Set{Tuple{String, String}}()
    current_logdet = logdet_psd(gram)
    while length(selected) < cfg.max_channels
        best_idx = 0
        best_score = -Inf
        best_gain = -Inf
        for idx in valid
            idx in selected && continue
            key = (rows[idx].left_observable, rows[idx].right_observable)
            key in selected_pairs && continue
            G = scaled[idx]
            cand_gram = gram .+ transpose(G) * G
            gain = logdet_psd(cand_gram) - current_logdet
            score = gain + 0.03 * log(max(rows[idx].signal_fraction, 1e-12)) +
                0.01 * rows[idx].split_corr
            if score > best_score
                best_idx = idx
                best_score = score
                best_gain = gain
            end
        end
        best_idx == 0 && break
        push!(selected, best_idx)
        push!(selected_pairs, (rows[best_idx].left_observable, rows[best_idx].right_observable))
        gram .+= transpose(scaled[best_idx]) * scaled[best_idx]
        current_logdet = logdet_psd(gram)
        old = rows[best_idx]
        rows[best_idx] = PhiNStructRow(old.left_observable, old.left_index,
            old.right_observable, old.right_index, old.signal_rms,
            old.signal_fraction, old.split_corr, old.split_rel, old.struct_rms,
            length(selected), best_gain)
    end
    return selected, gram, col_scale
end

function write_struct_toml(path, rows, selected, cfg, eval)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "# Generalized right-observable channels selected for the equivariant r2 M parameterization.")
        println(io, "# Generated by code/search_phi_n_struct_observables.jl.")
        println(io, "# IMPORTANT: selected with true M/Phi diagnostics; oracle-only library.")
        println(io, "source_dm_config = \"$(relpath(cfg.dm_config, dirname(path)))\"")
        println(io, "selection = \"greedy_logdet_equivariant_r2_basis\"")
        println(io, "target_kind = \"oracle_trueM_rightobs_struct_identifiability\"")
        println(io, "right_family = \"$(cfg.right_family)\"")
        println(io, @sprintf("pairs_per_lag = %d", cfg.pairs_per_lag))
        println(io, "active_lag_indices = \"$(first(eval.active_lag_indices)):$(last(eval.active_lag_indices))\"")
        println(io)
        for idx in selected
            row = rows[idx]
            println(io, "[[channels]]")
            println(io, "observable = \"$(row.left_observable)\"")
            println(io, "right_observable = \"$(row.right_observable)\"")
            println(io, "observable_index = $(row.left_index)")
            println(io, "right_observable_index = $(row.right_index)")
            println(io, @sprintf("data_rms = %.8e", row.signal_rms))
            println(io, @sprintf("signal_fraction = %.8e", row.signal_fraction))
            println(io, @sprintf("split_corr = %.8f", row.split_corr))
            println(io, @sprintf("split_rel = %.8e", row.split_rel))
            println(io, @sprintf("struct_rms = [%.8e, %.8e, %.8e]",
                row.struct_rms[1], row.struct_rms[2], row.struct_rms[3]))
            println(io, "selected_rank = $(row.selected_rank)")
            println(io, @sprintf("greedy_logdet_gain = %.8e", row.greedy_gain))
            println(io)
        end
    end
end

function write_struct_metrics(path, rows, selected, gram, cfg, eval)
    ensure_parent_dir(path)
    vals = eigvals(Symmetric(gram))
    open(path, "w") do io
        println(io, "SoftSpinLLGChain structured oracle phi_n observable search")
        println(io, "Audit: true M and Phi=<M_true> were used only in this oracle selector.")
        println(io, "dm_config = $(cfg.dm_config)")
        println(io, "right_family = $(cfg.right_family)")
        println(io, "transition_source_kind = $(eval.cond_kind)")
        println(io, @sprintf("candidate_channels = %d", length(rows)))
        println(io, @sprintf("valid_channels = %d", count(row -> valid_candidate(row, cfg), rows)))
        println(io, @sprintf("selected_channels = %d", length(selected)))
        println(io, @sprintf("selected Gram eig min = %.8e", minimum(vals)))
        println(io, @sprintf("selected Gram eig max = %.8e", maximum(vals)))
        println(io, @sprintf("selected Gram condition = %.8e", maximum(vals) / max(minimum(vals), eps(Float64))))
        println(io)
        for idx in selected
            row = rows[idx]
            println(io, @sprintf("%3d %-24s | %-14s signal %.8e frac %.5f split_corr %.5f split_rel %.5f gain %.5e struct [%.3e %.3e %.3e]",
                row.selected_rank, row.left_observable, row.right_observable,
                row.signal_rms, row.signal_fraction, row.split_corr, row.split_rel,
                row.greedy_gain, row.struct_rms[1], row.struct_rms[2], row.struct_rms[3]))
        end
    end
end

function render_struct_summary(path, rows, selected, gram)
    ensure_parent_dir(path)
    selected_set = Set(selected)
    vals = eigvals(Symmetric(gram))
    fig = Figure(; size=(2400, 1500))
    Label(fig[0, 1:2], "Structured oracle search for right observables";
        fontsize=28, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Stability vs oracle residual signal",
        xlabel="signal fraction", ylabel="split correlation")
    for (idx, row) in enumerate(rows)
        scatter!(ax1, [row.signal_fraction], [row.split_corr];
            color=idx in selected_set ? STYLE_PRIMARY : (:gray55, 0.35),
            markersize=idx in selected_set ? 12 : 7)
    end
    ax2 = Axis(fig[1, 2]; title="Equivariant-basis Gram spectrum",
        xlabel="basis direction", ylabel="eigenvalue", yscale=log10)
    lines!(ax2, 1:length(vals), sort(vals; rev=true); color=STYLE_HIGHLIGHT, linewidth=3)
    chosen = sort(selected; by=i -> rows[i].selected_rank)
    ax3 = Axis(fig[2, 1:2]; title="Selected channels", xlabel="rank", ylabel="scale")
    lines!(ax3, 1:length(chosen), [rows[i].signal_rms for i in chosen];
        color=STYLE_PRIMARY, linewidth=2, label="signal RMS")
    lines!(ax3, 1:length(chosen), [rows[i].greedy_gain for i in chosen];
        color=STYLE_HIGHLIGHT, linewidth=2, label="logdet gain")
    axislegend(ax3; position=:rt)
    save_figure_checked(path, fig)
end

function save_struct_artifact(path, rows, selected, gram, col_scale, cfg, eval)
    ensure_parent_dir(path)
    BSON.bson(path, Dict(:rows => rows, :selected_indices => selected,
        :selected_rows => rows[selected], :gram => gram, :column_scale => col_scale,
        :left_names => eval.left_names, :right_names => eval.right_names,
        :active_lags => eval.active_lags, :taus => eval.taus,
        :signal => eval.signal, :split1 => eval.split1, :split2 => eval.split2,
        :parts => eval.parts, :phi_block => Matrix{Float32}(eval.phi_block),
        :config => cfg,
        :audit => "Oracle structured phi_n selector: true M and Phi=<M_true> were used only to select diagnostic observables."))
end

function main()
    cfg = parse_struct_args(ARGS)
    eval = evaluate_struct_library(cfg)
    rows, matrices = struct_rows(eval, cfg)
    selected, gram, col_scale = greedy_select_struct(rows, matrices, cfg)
    root = normpath(joinpath(@__DIR__, ".."))
    toml_path = joinpath(root, "configs", "right_observable_retained_channels_$(cfg.suffix).toml")
    metrics_path = joinpath(root, "logs", "phin_struct_observable_search_$(cfg.suffix)_metrics.txt")
    figure_path = joinpath(root, "figures", "phin_struct_observable_search_$(cfg.suffix).png")
    artifact_path = joinpath(root, "models", "phin_struct_observable_search_$(cfg.suffix).bson")
    write_struct_toml(toml_path, rows, selected, cfg, eval)
    write_struct_metrics(metrics_path, rows, selected, gram, cfg, eval)
    render_struct_summary(figure_path, rows, selected, gram)
    save_struct_artifact(artifact_path, rows, selected, gram, col_scale, cfg, eval)
    @printf("Selected %d structured phi_n channels.\n", length(selected))
    @printf("Saved retained channels to %s\n", toml_path)
    @printf("Saved metrics to %s\n", metrics_path)
    @printf("Saved figure to %s\n", figure_path)
    @printf("Saved artifact to %s\n", artifact_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
