#!/usr/bin/env julia

include(joinpath(@__DIR__, "prepare_oracle_trueM_dM_targets.jl"))

using LinearAlgebra
using Printf
using Statistics

const ORACLE_ID_TARGETS = ("mx", "my", "mz")

Base.@kwdef struct OracleIdentifiabilityConfig
    dm_config::String
    family::String = "all"
    suffix::String = "oracle_ident_v1"
    pairs_per_lag::Int = 30000
    max_channels::Int = 60
    max_per_target::Int = 22
    signal_fraction_min::Float64 = 0.035
    split_corr_min::Float64 = 0.80
    split_rel_max::Float64 = 0.90
    ridge::Float64 = 1.0e-6
end

struct OracleIdentRow
    observable::String
    observable_index::Int
    target::String
    target_index::Int
    signal_rms::Float64
    signal_fraction::Float64
    split_corr::Float64
    split_rel::Float64
    entry_rms::NTuple{3, Float64}
    selected_rank::Int
    greedy_gain::Float64
end

function parse_oracle_ident_args(args)
    dm_config = length(args) >= 1 ? args[1] :
        normpath(joinpath(@__DIR__, "..", "configs",
            "fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml"))
    family = length(args) >= 2 ? args[2] : "all"
    suffix = length(args) >= 3 ? args[3] : "oracle_ident_v1"
    pairs = length(args) >= 4 ? parse(Int, args[4]) : 30000
    max_channels = length(args) >= 5 ? parse(Int, args[5]) : 60
    return OracleIdentifiabilityConfig(; dm_config, family, suffix,
        pairs_per_lag=pairs, max_channels)
end

function centered_split_profiles(obs::Array{Float32, 3}, action::Array{Float32, 3},
        N::Int, nobs::Int, split_range)
    idx = collect(split_range)
    B = length(idx)
    obs_slice = copy(@view obs[:, :, idx])
    action_slice = copy(@view action[:, :, idx])
    obs_flat = reshape(obs_slice, N * nobs, B)
    action_flat = flatten_batch(action_slice)
    mat = -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(action_flat)) / B
    return profile_from_operator_matrix(mat, N, nobs)
end

function profile_from_operator_matrix(mat::AbstractMatrix{<:Real}, N::Int, nobs::Int)
    profiles = zeros(Float64, nobs, 3, N)
    counts = zeros(Int, N)
    @inbounds for i in 1:N, j in 1:N
        offset = mod(j - i, N) + 1
        counts[offset] += 1
        for a in 1:nobs, c in 1:3
            row = (i - 1) * nobs + a
            col = (j - 1) * 3 + c
            profiles[a, c, offset] += Float64(mat[row, col])
        end
    end
    @inbounds for offset in 1:N
        profiles[:, :, offset] ./= counts[offset]
    end
    return profiles
end

function delta_entry_actions(x0::Array{Float32, 3}, rraw::Array{Float32, 3},
        p::SpinParams, phi_block::AbstractMatrix{<:Real})
    N, _, B = size(x0)
    parts = [zeros(Float32, N, 3, B) for _ in 1:3]
    full = zeros(Float32, N, 3, B)
    @inbounds for b in 1:B, i in 1:N
        block = true_mobility_site_block(x0[i, 1, b], x0[i, 2, b], x0[i, 3, b], p)
        delta = block .- phi_block
        for u in 1:3
            ru = Float64(rraw[i, u, b])
            for c in 1:3
                val = Float32(delta[u, c] * ru)
                parts[u][i, c, b] = val
                full[i, c, b] += val
            end
        end
    end
    return full, parts
end

function evaluate_oracle_library(cfg::OracleIdentifiabilityConfig)
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
    @printf("Oracle identifiability transition source kind: %s\n", String(cond_kind))

    names = nonlinear_candidate_names(cfg.family)
    lib = NonlinearLibrary(names)
    nobs = length(names)
    active_lag_indices = configured_lag_indices(cfg.dm_config,
        min(dm_cfg.max_lags, length(sampler.lag_steps)))
    lags = sampler.lag_steps[1:min(dm_cfg.max_lags, length(sampler.lag_steps))]
    active_lags = lags[active_lag_indices]
    means = estimate_nonlinear_means(sampler, p, lib,
        min(dm_cfg.target_mean_samples, 160000), MersenneTwister(dm_cfg.seed + 501))
    _Phi_true, phi_block = oracle_true_mean_mobility(sampler, p,
        min(dm_cfg.target_mean_samples, 160000), dm_cfg.seed + 502)

    N = sampler.N
    L = length(active_lags)
    signal = zeros(Float64, L, nobs, 3, N)
    split1 = similar(signal)
    split2 = similar(signal)
    parts = zeros(Float64, L, nobs, 3, 3, N)
    rng = MersenneTwister(dm_cfg.seed + 503)
    half = cfg.pairs_per_lag ÷ 2
    require_condition(half >= 1000, "pairs_per_lag too small for split diagnostics.")

    for (li, lag) in enumerate(active_lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, cfg.pairs_per_lag, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm,
            stats, cond_params, device; batch_size=min(cond_params.batch_size, cfg.pairs_per_lag),
            score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        full_action, entry_actions = delta_entry_actions(x0, rraw, p, phi_block)

        signal[li, :, :, :] .= centered_split_profiles(obs, full_action, N, nobs,
            1:cfg.pairs_per_lag)
        split1[li, :, :, :] .= centered_split_profiles(obs, full_action, N, nobs,
            1:half)
        split2[li, :, :, :] .= centered_split_profiles(obs, full_action, N, nobs,
            (half + 1):cfg.pairs_per_lag)
        for u in 1:3
            parts[li, :, :, u, :] .= centered_split_profiles(obs, entry_actions[u],
                N, nobs, 1:cfg.pairs_per_lag)
        end
        @printf("Oracle identifiable observable lag %.5g (%d/%d), %d observables, pairs=%d\n",
            lag * sampler.save_dt, li, L, nobs, cfg.pairs_per_lag)
        GC.gc()
    end

    return (; dm_cfg, sampler, cond_kind, names, active_lag_indices, active_lags,
        taus=active_lags .* sampler.save_dt, signal, split1, split2, parts,
        phi_block, device_request=dm_cfg.device, required_gpu_name=dm_cfg.required_gpu_name)
end

function channel_vector(A, a::Int, c::Int)
    return vec(@view A[:, a, c, :])
end

function channel_part_matrix(parts, a::Int, c::Int)
    L = size(parts, 1)
    N = size(parts, 5)
    G = zeros(Float64, L * N, 9)
    for u in 1:3
        col = (c - 1) * 3 + u
        G[:, col] .= vec(@view parts[:, a, c, u, :])
    end
    return G
end

function safe_corr(x::AbstractVector, y::AbstractVector)
    nx = norm(x)
    ny = norm(y)
    return dot(x, y) / max(nx * ny, eps(Float64))
end

function channel_rows(eval, cfg::OracleIdentifiabilityConfig)
    max_signal = maximum([sqrt(mean(abs2, channel_vector(eval.signal, a, c)))
        for a in eachindex(eval.names), c in 1:3])
    rows = OracleIdentRow[]
    matrices = Matrix{Float64}[]
    for a in eachindex(eval.names), c in 1:3
        v = channel_vector(eval.signal, a, c)
        v1 = channel_vector(eval.split1, a, c)
        v2 = channel_vector(eval.split2, a, c)
        sig = sqrt(mean(abs2, v))
        sfrac = sig / max(max_signal, eps(Float64))
        scorr = safe_corr(v1, v2)
        srel = sqrt(mean((v1 .- v2) .^ 2)) / max(sqrt(mean(abs2, v)), eps(Float64))
        erms = ntuple(u -> sqrt(mean(abs2, @view eval.parts[:, a, c, u, :])), 3)
        push!(rows, OracleIdentRow(eval.names[a], a, ORACLE_ID_TARGETS[c], c,
            sig, sfrac, scorr, srel, erms, 0, 0.0))
        push!(matrices, channel_part_matrix(eval.parts, a, c))
    end
    return rows, matrices
end

function valid_candidate(row::OracleIdentRow, cfg::OracleIdentifiabilityConfig)
    return row.signal_fraction >= cfg.signal_fraction_min &&
        row.split_corr >= cfg.split_corr_min &&
        row.split_rel <= cfg.split_rel_max &&
        maximum(row.entry_rms) > 0
end

function logdet_psd(A::AbstractMatrix{<:Real})
    F = cholesky(Symmetric(Matrix(A)); check=false)
    if !issuccess(F)
        vals = eigvals(Symmetric(Matrix(A)))
        return sum(log(max(v, eps(Float64))) for v in vals)
    end
    return 2sum(log, diag(F.U))
end

function greedy_select(rows::Vector{OracleIdentRow}, matrices::Vector{Matrix{Float64}},
        cfg::OracleIdentifiabilityConfig)
    valid = findall(row -> valid_candidate(row, cfg), rows)
    require_condition(!isempty(valid), "No oracle-identifiable observable channels passed stability filters.")
    col_norm = zeros(Float64, 9)
    for idx in valid
        col_norm .+= vec(sum(abs2, matrices[idx]; dims=1))
    end
    col_scale = sqrt.(max.(col_norm, eps(Float64)))
    scaled = [mat ./ reshape(col_scale, 1, 9) for mat in matrices]
    gram = cfg.ridge .* Matrix{Float64}(I, 9, 9)
    selected = Int[]
    target_counts = zeros(Int, 3)
    selected_names = Set{Tuple{String, Int}}()
    current_logdet = logdet_psd(gram)

    while length(selected) < cfg.max_channels
        best_idx = 0
        best_score = -Inf
        best_gain = -Inf
        for idx in valid
            idx in selected && continue
            key = (rows[idx].observable, rows[idx].target_index)
            key in selected_names && continue
            rows[idx].target_index <= 3 || continue
            target_counts[rows[idx].target_index] >= cfg.max_per_target && continue
            G = scaled[idx]
            cand_gram = gram .+ transpose(G) * G
            gain = logdet_psd(cand_gram) - current_logdet
            balance_penalty = 0.015 * target_counts[rows[idx].target_index]
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
        push!(selected_names, (rows[best_idx].observable, rows[best_idx].target_index))
        target_counts[rows[best_idx].target_index] += 1
        G = scaled[best_idx]
        gram .+= transpose(G) * G
        current_logdet = logdet_psd(gram)
        old = rows[best_idx]
        rows[best_idx] = OracleIdentRow(old.observable, old.observable_index, old.target,
            old.target_index, old.signal_rms, old.signal_fraction, old.split_corr,
            old.split_rel, old.entry_rms, length(selected), best_gain)
    end
    return selected, gram, col_scale
end

function write_oracle_ident_toml(path, rows, selected, cfg, eval)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "# Oracle mobility-identifiability observable channels.")
        println(io, "# Generated by code/search_oracle_identifiable_observables.jl.")
        println(io, "# IMPORTANT: this library was selected using true M and is not data-only.")
        println(io, "# data_rms below is the oracle A-target RMS used as a loss scale.")
        println(io, "source_dm_config = \"$(relpath(cfg.dm_config, dirname(path)))\"")
        println(io, "family = \"$(cfg.family)\"")
        println(io, "selection = \"greedy_logdet_entry_decomposition\"")
        println(io, "target_kind = \"oracle_trueM_identifiability\"")
        println(io, @sprintf("pairs_per_lag = %d", cfg.pairs_per_lag))
        println(io, "active_lag_indices = \"$(first(eval.active_lag_indices)):$(last(eval.active_lag_indices))\"")
        println(io, @sprintf("signal_fraction_min = %.8f", cfg.signal_fraction_min))
        println(io, @sprintf("split_corr_min = %.8f", cfg.split_corr_min))
        println(io, @sprintf("split_rel_max = %.8f", cfg.split_rel_max))
        println(io, "translation_offsets = \"all\"")
        println(io)
        for idx in selected
            row = rows[idx]
            println(io, "[[channels]]")
            println(io, "observable = \"$(row.observable)\"")
            println(io, "target_component = \"$(row.target)\"")
            println(io, "observable_index = $(row.observable_index)")
            println(io, "target_component_index = $(row.target_index)")
            println(io, @sprintf("data_rms = %.8e", row.signal_rms))
            println(io, @sprintf("signal_fraction = %.8e", row.signal_fraction))
            println(io, @sprintf("split_corr = %.8f", row.split_corr))
            println(io, @sprintf("split_rel = %.8e", row.split_rel))
            println(io, @sprintf("entry_rms = [%.8e, %.8e, %.8e]",
                row.entry_rms[1], row.entry_rms[2], row.entry_rms[3]))
            println(io, "selected_rank = $(row.selected_rank)")
            println(io, @sprintf("greedy_logdet_gain = %.8e", row.greedy_gain))
            println(io, "source = \"oracle_trueM_entry_decomposition\"")
            println(io)
        end
    end
end

function write_oracle_ident_metrics(path, rows, selected, gram, cfg, eval)
    ensure_parent_dir(path)
    vals = eigvals(Symmetric(gram))
    cond = maximum(vals) / max(minimum(vals), eps(Float64))
    target_counts = Dict(t => count(idx -> rows[idx].target == t, selected)
        for t in ORACLE_ID_TARGETS)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain oracle mobility-identifiability observable search")
        println(io, "Audit: true M and Phi=<M_true> were intentionally used to decompose the learned-transition-score residual target. This is an oracle diagnostic library, not data-only.")
        println(io, "dm_config = $(cfg.dm_config)")
        println(io, "family = $(cfg.family)")
        println(io, "transition_source_kind = $(eval.cond_kind)")
        println(io, "device_request = $(eval.device_request)")
        println(io, "required_gpu_name = $(eval.required_gpu_name)")
        println(io, @sprintf("candidate_channels = %d", length(rows)))
        println(io, @sprintf("valid_channels = %d", count(row -> valid_candidate(row, cfg), rows)))
        println(io, @sprintf("selected_channels = %d", length(selected)))
        println(io, @sprintf("pairs_per_lag = %d", cfg.pairs_per_lag))
        println(io, "active_lag_indices = $(first(eval.active_lag_indices)):$(last(eval.active_lag_indices))")
        println(io, @sprintf("selected Gram eig min = %.8e", minimum(vals)))
        println(io, @sprintf("selected Gram eig max = %.8e", maximum(vals)))
        println(io, @sprintf("selected Gram condition = %.8e", cond))
        for t in ORACLE_ID_TARGETS
            println(io, "$(t) selected channels = $(target_counts[t])")
        end
        println(io)
        println(io, "Selected channels:")
        for idx in selected
            row = rows[idx]
            println(io, @sprintf("%3d %-24s -> %-2s signal %.8e frac %.5f split_corr %.5f split_rel %.5f entry_rms [%.3e %.3e %.3e] gain %.5e",
                row.selected_rank, row.observable, row.target, row.signal_rms,
                row.signal_fraction, row.split_corr, row.split_rel,
                row.entry_rms[1], row.entry_rms[2], row.entry_rms[3], row.greedy_gain))
        end
        println(io)
        println(io, "Top rejected/valid candidates by signal:")
        order = sortperm(collect(eachindex(rows)); by=i -> -rows[i].signal_fraction)
        for idx in order[1:min(80, length(order))]
            row = rows[idx]
            status = idx in selected ? "SELECT" :
                valid_candidate(row, cfg) ? "valid" : "reject"
            println(io, @sprintf("%-6s %-24s -> %-2s signal %.8e frac %.5f split_corr %.5f split_rel %.5f",
                status, row.observable, row.target, row.signal_rms,
                row.signal_fraction, row.split_corr, row.split_rel))
        end
    end
end

function render_oracle_ident_summary(path, rows, selected, gram)
    ensure_parent_dir(path)
    selected_set = Set(selected)
    vals = eigvals(Symmetric(gram))
    fig = Figure(; size=(2800, 1800))
    Label(fig[0, 1:2], "Oracle observable search for mobility identifiability";
        fontsize=30, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Split stability vs oracle A signal",
        xlabel="signal fraction", ylabel="split correlation")
    for (idx, row) in enumerate(rows)
        scatter!(ax1, [row.signal_fraction], [row.split_corr];
            color=idx in selected_set ? STYLE_PRIMARY : (:gray55, 0.35),
            marker=idx in selected_set ? :circle : :xcross,
            markersize=idx in selected_set ? 13 : 8)
    end
    vlines!(ax1, [0.035]; color=:black, linestyle=:dash)
    hlines!(ax1, [0.80]; color=:black, linestyle=:dash)
    ax2 = Axis(fig[1, 2]; title="Selected entry-decomposition Gram spectrum",
        xlabel="singular direction", ylabel="eigenvalue", yscale=log10)
    lines!(ax2, 1:length(vals), sort(vals; rev=true);
        color=STYLE_HIGHLIGHT, linewidth=3)

    chosen = sort(selected; by=i -> rows[i].selected_rank)
    topn = min(length(chosen), 60)
    ax3 = Axis(fig[2, 1:2]; title="Greedy-selected channels",
        xlabel="selection rank", ylabel="oracle A RMS")
    xs = collect(1:topn)
    lines!(ax3, xs, [rows[chosen[i]].signal_rms for i in 1:topn];
        color=STYLE_PRIMARY, linewidth=2, label="signal")
    lines!(ax3, xs, [rows[chosen[i]].greedy_gain for i in 1:topn];
        color=STYLE_HIGHLIGHT, linewidth=2, label="logdet gain")
    axislegend(ax3; position=:rt)

    ax4 = Axis(fig[3, 1:2]; title="Selected target-component counts",
        xlabel="target", ylabel="count")
    counts = [count(idx -> rows[idx].target == t, selected) for t in ORACLE_ID_TARGETS]
    barplot!(ax4, 1:3, counts; color=STYLE_ACCENT)
    ax4.xticks = (1:3, collect(ORACLE_ID_TARGETS))
    save_figure_checked(path, fig)
end

function save_artifact(path, rows, selected, gram, col_scale, cfg, eval)
    ensure_parent_dir(path)
    BSON.bson(path, Dict(:rows => rows, :selected_indices => selected,
        :selected_rows => rows[selected], :gram => gram, :column_scale => col_scale,
        :names => eval.names, :active_lags => eval.active_lags, :taus => eval.taus,
        :signal => eval.signal, :split1 => eval.split1, :split2 => eval.split2,
        :parts => eval.parts, :phi_block => Matrix{Float32}(eval.phi_block),
        :config => cfg,
        :audit => "Oracle identifiability diagnostic: true M and Phi=<M_true> were intentionally used to decompose A targets and select observables."))
end

function main()
    cfg = parse_oracle_ident_args(ARGS)
    eval = evaluate_oracle_library(cfg)
    rows, matrices = channel_rows(eval, cfg)
    selected, gram, col_scale = greedy_select(rows, matrices, cfg)
    root = normpath(joinpath(@__DIR__, ".."))
    toml_path = joinpath(root, "configs", "nonlinear_observable_retained_channels_$(cfg.suffix).toml")
    metrics_path = joinpath(root, "logs", "oracle_ident_observable_search_$(cfg.suffix)_metrics.txt")
    figure_path = joinpath(root, "figures", "oracle_ident_observable_search_$(cfg.suffix).png")
    artifact_path = joinpath(root, "models", "oracle_ident_observable_search_$(cfg.suffix).bson")
    write_oracle_ident_toml(toml_path, rows, selected, cfg, eval)
    write_oracle_ident_metrics(metrics_path, rows, selected, gram, cfg, eval)
    render_oracle_ident_summary(figure_path, rows, selected, gram)
    save_artifact(artifact_path, rows, selected, gram, col_scale, cfg, eval)
    @printf("Selected %d oracle-identifiable observable channels.\n", length(selected))
    @printf("Saved retained channels to %s\n", toml_path)
    @printf("Saved metrics to %s\n", metrics_path)
    @printf("Saved figure to %s\n", figure_path)
    @printf("Saved artifact to %s\n", artifact_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
