#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM_rightobs.jl"))

using LinearAlgebra
using Printf
using Statistics

Base.@kwdef struct PhiNSearchConfig
    dm_config::String
    left_source::String = "retained"
    right_family::String = "core"
    suffix::String = "phin_oracle_ident_v1"
    pairs_per_lag::Int = 18000
    max_channels::Int = 84
    signal_fraction_min::Float64 = 0.020
    split_corr_min::Float64 = 0.65
    split_rel_max::Float64 = 1.10
    ridge::Float64 = 1.0e-6
end

struct PhiNIdentRow
    left_observable::String
    left_index::Int
    right_observable::String
    right_index::Int
    signal_rms::Float64
    signal_fraction::Float64
    split_corr::Float64
    split_rel::Float64
    entry_rms::NTuple{9, Float64}
    selected_rank::Int
    greedy_gain::Float64
end

function parse_phin_search_args(args)
    dm_config = length(args) >= 1 ? args[1] :
        normpath(joinpath(@__DIR__, "..", "configs",
            "fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.toml"))
    right_family = length(args) >= 2 ? args[2] : "core"
    suffix = length(args) >= 3 ? args[3] : "phin_oracle_ident_v1"
    pairs = length(args) >= 4 ? parse(Int, args[4]) : 18000
    max_channels = length(args) >= 5 ? parse(Int, args[5]) : 84
    return PhiNSearchConfig(; dm_config, right_family, suffix,
        pairs_per_lag=pairs, max_channels)
end

function left_names_from_config(dm_cfg::DMConfig, base::AbstractString, source::AbstractString)
    if source == "retained"
        channels = load_retained_channels(resolve_path(base, dm_cfg.retained_channels_toml))
        return unique_observable_names(channels)
    elseif source in ("all", "baseline", "poly_high", "neighbor_high")
        return nonlinear_candidate_names(source)
    end
    error("Unsupported left_source=$(source).")
end

function right_profile_from_matrix(mat::AbstractMatrix{<:Real}, N::Int, nleft::Int,
        nright::Int)
    profiles = zeros(Float64, nleft, nright, N)
    counts = zeros(Int, N)
    @inbounds for i in 1:N, j in 1:N
        offset = mod(j - i, N) + 1
        counts[offset] += 1
        for a in 1:nleft, b in 1:nright
            row = (i - 1) * nleft + a
            col = (j - 1) * nright + b
            profiles[a, b, offset] += Float64(mat[row, col])
        end
    end
    @inbounds for offset in 1:N
        profiles[:, :, offset] ./= counts[offset]
    end
    return profiles
end

function rightobs_profiles(left::Array{Float32, 3}, right::Array{Float32, 3},
        N::Int, nleft::Int, nright::Int, split_range)
    idx = collect(split_range)
    B = length(idx)
    left_slice = copy(@view left[:, :, idx])
    right_slice = copy(@view right[:, :, idx])
    left_flat = reshape(left_slice, N * nleft, B)
    right_flat = reshape(right_slice, N * nright, B)
    mat = -Matrix{Float64}(left_flat) * transpose(Matrix{Float64}(right_flat)) / B
    return right_profile_from_matrix(mat, N, nleft, nright)
end

function delta_entry_right_actions(x0::Array{Float32, 3}, rraw::Array{Float32, 3},
        right_grad::Array{Float32, 4}, p::SpinParams, phi_block::AbstractMatrix{<:Real})
    N, R, _, B = size(right_grad)
    parts = [zeros(Float32, N, R, B) for _ in 1:9]
    full = zeros(Float32, N, R, B)
    @inbounds for b in 1:B, i in 1:N
        block = true_mobility_site_block(x0[i, 1, b], x0[i, 2, b], x0[i, 3, b], p)
        delta = block .- phi_block
        for u in 1:3, c in 1:3
            e = (c - 1) * 3 + u
            coeff = Float32(delta[u, c] * Float64(rraw[i, u, b]))
            for a in 1:R
                val = coeff * right_grad[i, a, c, b]
                parts[e][i, a, b] = val
                full[i, a, b] += val
            end
        end
    end
    return full, parts
end

function evaluate_phin_library(cfg::PhiNSearchConfig)
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
    @printf("Phi_n search transition source kind: %s\n", String(cond_kind))

    left_names = left_names_from_config(dm_cfg, base, cfg.left_source)
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
        min(dm_cfg.target_mean_samples, 160000), MersenneTwister(dm_cfg.seed + 610))
    _Phi_true, phi_block = oracle_true_mean_mobility(sampler, p,
        min(dm_cfg.target_mean_samples, 160000), dm_cfg.seed + 611)

    N = sampler.N
    L = length(active_lags)
    signal = zeros(Float64, L, nleft, nright, N)
    split1 = similar(signal)
    split2 = similar(signal)
    parts = zeros(Float64, L, nleft, nright, 9, N)
    rng = MersenneTwister(dm_cfg.seed + 612)
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
        full_right, entry_right = delta_entry_right_actions(x0, rraw, right_grad, p,
            phi_block)
        signal[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            1:cfg.pairs_per_lag)
        split1[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            1:half)
        split2[li, :, :, :] .= rightobs_profiles(left, full_right, N, nleft, nright,
            (half + 1):cfg.pairs_per_lag)
        for e in 1:9
            parts[li, :, :, e, :] .= rightobs_profiles(left, entry_right[e], N,
                nleft, nright, 1:cfg.pairs_per_lag)
        end
        @printf("Phi_n oracle-identifiable lag %.5g (%d/%d), left=%d right=%d pairs=%d\n",
            lag * sampler.save_dt, li, L, nleft, nright, cfg.pairs_per_lag)
        GC.gc()
    end

    return (; dm_cfg, sampler, cond_kind, left_names, right_names,
        active_lag_indices, active_lags, taus=active_lags .* sampler.save_dt,
        signal, split1, split2, parts, phi_block, device_request=dm_cfg.device,
        required_gpu_name=dm_cfg.required_gpu_name)
end

phin_channel_vector(A, a::Int, b::Int) = vec(@view A[:, a, b, :])

function phin_channel_part_matrix(parts, a::Int, b::Int)
    L = size(parts, 1)
    N = size(parts, 5)
    G = zeros(Float64, L * N, 9)
    for e in 1:9
        G[:, e] .= vec(@view parts[:, a, b, e, :])
    end
    return G
end

function safe_corr(x::AbstractVector, y::AbstractVector)
    nx = norm(x)
    ny = norm(y)
    return dot(x, y) / max(nx * ny, eps(Float64))
end

function logdet_psd(A::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(A)); check=false)
    if !issuccess(F)
        vals = eigvals(Symmetric(Matrix(A)))
        return sum(log(max(v, eps(Float64))) for v in vals)
    end
    return 2sum(log, diag(F.U))
end

function phin_rows(eval, cfg::PhiNSearchConfig)
    max_signal = maximum([sqrt(mean(abs2, phin_channel_vector(eval.signal, a, b)))
        for a in eachindex(eval.left_names), b in eachindex(eval.right_names)])
    rows = PhiNIdentRow[]
    matrices = Matrix{Float64}[]
    for a in eachindex(eval.left_names), b in eachindex(eval.right_names)
        v = phin_channel_vector(eval.signal, a, b)
        v1 = phin_channel_vector(eval.split1, a, b)
        v2 = phin_channel_vector(eval.split2, a, b)
        sig = sqrt(mean(abs2, v))
        sfrac = sig / max(max_signal, eps(Float64))
        scorr = safe_corr(v1, v2)
        srel = sqrt(mean((v1 .- v2) .^ 2)) / max(sqrt(mean(abs2, v)), eps(Float64))
        erms = ntuple(e -> sqrt(mean(abs2, @view eval.parts[:, a, b, e, :])), 9)
        push!(rows, PhiNIdentRow(eval.left_names[a], a, eval.right_names[b], b,
            sig, sfrac, scorr, srel, erms, 0, 0.0))
        push!(matrices, phin_channel_part_matrix(eval.parts, a, b))
    end
    return rows, matrices
end

function valid_candidate(row::PhiNIdentRow, cfg::PhiNSearchConfig)
    return row.signal_fraction >= cfg.signal_fraction_min &&
        row.split_corr >= cfg.split_corr_min &&
        row.split_rel <= cfg.split_rel_max &&
        maximum(row.entry_rms) > 0
end

function greedy_select_phin(rows::Vector{PhiNIdentRow},
        matrices::Vector{Matrix{Float64}}, cfg::PhiNSearchConfig)
    valid = findall(row -> valid_candidate(row, cfg), rows)
    require_condition(!isempty(valid), "No phi_n observable channel passed stability filters.")
    col_norm = zeros(Float64, 9)
    for idx in valid
        col_norm .+= vec(sum(abs2, matrices[idx]; dims=1))
    end
    col_scale = sqrt.(max.(col_norm, eps(Float64)))
    scaled = [mat ./ reshape(col_scale, 1, 9) for mat in matrices]
    gram = cfg.ridge .* Matrix{Float64}(I, 9, 9)
    selected = Int[]
    selected_pairs = Set{Tuple{String, String}}()
    right_counts = Dict(name => 0 for name in unique(row.right_observable for row in rows))
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
            balance_penalty = 0.010 * right_counts[rows[idx].right_observable]
            score = gain + 0.02 * log(max(rows[idx].signal_fraction, 1e-12)) +
                0.01 * rows[idx].split_corr - balance_penalty
            if score > best_score
                best_idx = idx
                best_score = score
                best_gain = gain
            end
        end
        best_idx == 0 && break
        push!(selected, best_idx)
        push!(selected_pairs, (rows[best_idx].left_observable, rows[best_idx].right_observable))
        right_counts[rows[best_idx].right_observable] += 1
        G = scaled[best_idx]
        gram .+= transpose(G) * G
        current_logdet = logdet_psd(gram)
        old = rows[best_idx]
        rows[best_idx] = PhiNIdentRow(old.left_observable, old.left_index,
            old.right_observable, old.right_index, old.signal_rms,
            old.signal_fraction, old.split_corr, old.split_rel, old.entry_rms,
            length(selected), best_gain)
    end
    return selected, gram, col_scale
end

function write_phin_toml(path, rows, selected, cfg, eval)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "# Generalized right-observable channels for mobility learning.")
        println(io, "# Generated by code/search_phi_n_observables.jl.")
        println(io, "# IMPORTANT: selected with true M and Phi=<M_true>; use only as an oracle diagnostic library.")
        println(io, "source_dm_config = \"$(relpath(cfg.dm_config, dirname(path)))\"")
        println(io, "selection = \"greedy_logdet_local_mobility_entry_decomposition\"")
        println(io, "target_kind = \"oracle_trueM_rightobs_identifiability\"")
        println(io, "left_source = \"$(cfg.left_source)\"")
        println(io, "right_family = \"$(cfg.right_family)\"")
        println(io, @sprintf("pairs_per_lag = %d", cfg.pairs_per_lag))
        println(io, "active_lag_indices = \"$(first(eval.active_lag_indices)):$(last(eval.active_lag_indices))\"")
        println(io, @sprintf("signal_fraction_min = %.8f", cfg.signal_fraction_min))
        println(io, @sprintf("split_corr_min = %.8f", cfg.split_corr_min))
        println(io, @sprintf("split_rel_max = %.8f", cfg.split_rel_max))
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
            println(io, @sprintf("entry_rms = [%s]",
                join([@sprintf("%.8e", row.entry_rms[e]) for e in 1:9], ", ")))
            println(io, "selected_rank = $(row.selected_rank)")
            println(io, @sprintf("greedy_logdet_gain = %.8e", row.greedy_gain))
            println(io, "source = \"oracle_trueM_right_observable_entry_decomposition\"")
            println(io)
        end
    end
end

function write_phin_metrics(path, rows, selected, gram, cfg, eval)
    ensure_parent_dir(path)
    vals = eigvals(Symmetric(gram))
    cond = maximum(vals) / max(minimum(vals), eps(Float64))
    open(path, "w") do io
        println(io, "SoftSpinLLGChain oracle phi_n observable search")
        println(io, "Audit: true M and Phi=<M_true> were intentionally used to select right observables. This is not data-only.")
        println(io, "dm_config = $(cfg.dm_config)")
        println(io, "left_source = $(cfg.left_source)")
        println(io, "right_family = $(cfg.right_family)")
        println(io, "transition_source_kind = $(eval.cond_kind)")
        println(io, "device_request = $(eval.device_request)")
        println(io, "required_gpu_name = $(eval.required_gpu_name)")
        println(io, @sprintf("left_observables = %d", length(eval.left_names)))
        println(io, @sprintf("right_observables = %d", length(eval.right_names)))
        println(io, @sprintf("candidate_channels = %d", length(rows)))
        println(io, @sprintf("valid_channels = %d", count(row -> valid_candidate(row, cfg), rows)))
        println(io, @sprintf("selected_channels = %d", length(selected)))
        println(io, @sprintf("pairs_per_lag = %d", cfg.pairs_per_lag))
        println(io, "active_lag_indices = $(first(eval.active_lag_indices)):$(last(eval.active_lag_indices))")
        println(io, @sprintf("selected Gram eig min = %.8e", minimum(vals)))
        println(io, @sprintf("selected Gram eig max = %.8e", maximum(vals)))
        println(io, @sprintf("selected Gram condition = %.8e", cond))
        println(io)
        println(io, "Selected channels:")
        for idx in selected
            row = rows[idx]
            println(io, @sprintf("%3d %-24s | %-14s signal %.8e frac %.5f split_corr %.5f split_rel %.5f gain %.5e",
                row.selected_rank, row.left_observable, row.right_observable,
                row.signal_rms, row.signal_fraction, row.split_corr, row.split_rel,
                row.greedy_gain))
        end
        println(io)
        println(io, "Top candidates by signal:")
        order = sortperm(collect(eachindex(rows)); by=i -> -rows[i].signal_fraction)
        for idx in order[1:min(120, length(order))]
            row = rows[idx]
            status = idx in selected ? "SELECT" :
                valid_candidate(row, cfg) ? "valid" : "reject"
            println(io, @sprintf("%-6s %-24s | %-14s signal %.8e frac %.5f split_corr %.5f split_rel %.5f",
                status, row.left_observable, row.right_observable, row.signal_rms,
                row.signal_fraction, row.split_corr, row.split_rel))
        end
    end
end

function render_phin_summary(path, rows, selected, gram)
    ensure_parent_dir(path)
    selected_set = Set(selected)
    vals = eigvals(Symmetric(gram))
    fig = Figure(; size=(2800, 1800))
    Label(fig[0, 1:2], "Oracle search for generalized right observables";
        fontsize=30, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Split stability vs oracle A signal",
        xlabel="signal fraction", ylabel="split correlation")
    for (idx, row) in enumerate(rows)
        scatter!(ax1, [row.signal_fraction], [row.split_corr];
            color=idx in selected_set ? STYLE_PRIMARY : (:gray55, 0.35),
            marker=idx in selected_set ? :circle : :xcross,
            markersize=idx in selected_set ? 13 : 8)
    end
    vlines!(ax1, [0.020]; color=:black, linestyle=:dash)
    hlines!(ax1, [0.65]; color=:black, linestyle=:dash)
    ax2 = Axis(fig[1, 2]; title="Selected local-entry Gram spectrum",
        xlabel="entry direction", ylabel="eigenvalue", yscale=log10)
    lines!(ax2, 1:length(vals), sort(vals; rev=true);
        color=STYLE_HIGHLIGHT, linewidth=3)
    chosen = sort(selected; by=i -> rows[i].selected_rank)
    topn = min(length(chosen), 84)
    ax3 = Axis(fig[2, 1:2]; title="Greedy-selected channels",
        xlabel="selection rank", ylabel="oracle A scale")
    xs = collect(1:topn)
    lines!(ax3, xs, [rows[chosen[i]].signal_rms for i in 1:topn];
        color=STYLE_PRIMARY, linewidth=2, label="signal")
    lines!(ax3, xs, [rows[chosen[i]].greedy_gain for i in 1:topn];
        color=STYLE_HIGHLIGHT, linewidth=2, label="logdet gain")
    axislegend(ax3; position=:rt)
    right_names = unique(row.right_observable for row in rows)
    counts = [count(idx -> rows[idx].right_observable == name, selected) for name in right_names]
    order = sortperm(counts; rev=true)
    ax4 = Axis(fig[3, 1:2]; title="Selected right-observable counts",
        xlabel="right observable", ylabel="count")
    show_order = order[1:min(18, length(order))]
    barplot!(ax4, 1:length(show_order), counts[show_order]; color=STYLE_ACCENT)
    ax4.xticks = (1:length(show_order), right_names[show_order])
    ax4.xticklabelrotation = pi / 5
    save_figure_checked(path, fig)
end

function save_phin_artifact(path, rows, selected, gram, col_scale, cfg, eval)
    ensure_parent_dir(path)
    BSON.bson(path, Dict(:rows => rows, :selected_indices => selected,
        :selected_rows => rows[selected], :gram => gram, :column_scale => col_scale,
        :left_names => eval.left_names, :right_names => eval.right_names,
        :active_lags => eval.active_lags, :taus => eval.taus,
        :signal => eval.signal, :split1 => eval.split1, :split2 => eval.split2,
        :parts => eval.parts, :phi_block => Matrix{Float32}(eval.phi_block),
        :config => cfg,
        :audit => "Oracle phi_n identifiability diagnostic: true M and Phi=<M_true> were intentionally used to decompose A targets and select right observables."))
end

function main()
    cfg = parse_phin_search_args(ARGS)
    eval = evaluate_phin_library(cfg)
    rows, matrices = phin_rows(eval, cfg)
    selected, gram, col_scale = greedy_select_phin(rows, matrices, cfg)
    root = normpath(joinpath(@__DIR__, ".."))
    toml_path = joinpath(root, "configs", "right_observable_retained_channels_$(cfg.suffix).toml")
    metrics_path = joinpath(root, "logs", "phin_observable_search_$(cfg.suffix)_metrics.txt")
    figure_path = joinpath(root, "figures", "phin_observable_search_$(cfg.suffix).png")
    artifact_path = joinpath(root, "models", "phin_observable_search_$(cfg.suffix).bson")
    write_phin_toml(toml_path, rows, selected, cfg, eval)
    write_phin_metrics(metrics_path, rows, selected, gram, cfg, eval)
    render_phin_summary(figure_path, rows, selected, gram)
    save_phin_artifact(artifact_path, rows, selected, gram, col_scale, cfg, eval)
    @printf("Selected %d phi_n observable channels.\n", length(selected))
    @printf("Saved retained channels to %s\n", toml_path)
    @printf("Saved metrics to %s\n", metrics_path)
    @printf("Saved figure to %s\n", figure_path)
    @printf("Saved artifact to %s\n", artifact_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
