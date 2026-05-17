#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using Printf

const DEFAULT_ORACLE_DM_CONFIG = normpath(joinpath(@__DIR__, "..", "configs",
    "fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml"))

function oracle_full_onsite_matrix_from_block(block::AbstractMatrix{<:Real}, N::Int)
    A = zeros(Float64, 3N, 3N)
    @inbounds for i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        A[rows, rows] .= block
    end
    return A
end

function true_mobility_site_block(x1::Real, x2::Real, x3::Real, p::SpinParams)
    xx = Float64[x1, x2, x3]
    r2 = dot(xx, xx)
    A = p.eps .* Matrix{Float64}(I, 3, 3) .+
        p.alpha_perp .* (r2 .* Matrix{Float64}(I, 3, 3) .- xx * transpose(xx)) .+
        p.alpha_parallel .* (xx * transpose(xx))
    R = -p.gamma * p.theta .* [0.0 -xx[3] xx[2]; xx[3] 0.0 -xx[1]; -xx[2] xx[1] 0.0]
    return p.theta .* A .+ R
end

function oracle_true_mean_mobility(sampler::CondPairSampler, p::SpinParams,
        nsamples::Int, seed::Int)
    raw = sample_raw_states_cond(sampler, nsamples, MersenneTwister(seed))
    block = zeros(Float64, 3, 3)
    N, _, B = size(raw)
    @inbounds for b in 1:B, i in 1:N
        block .+= true_mobility_site_block(raw[i, 1, b], raw[i, 2, b], raw[i, 3, b], p)
    end
    block ./= N * B
    return oracle_full_onsite_matrix_from_block(block, N), block
end

function oracle_true_action_batch(x0::Array{Float32, 3}, rraw::Array{Float32, 3},
        p::SpinParams)
    N, _, B = size(x0)
    out = Array{Float32}(undef, N, 3, B)
    @inbounds for b in 1:B, i in 1:N
        block = true_mobility_site_block(x0[i, 1, b], x0[i, 2, b], x0[i, 3, b], p)
        r = Float64[rraw[i, 1, b], rraw[i, 2, b], rraw[i, 3, b]]
        a = transpose(block) * r
        out[i, 1, b] = Float32(a[1])
        out[i, 2, b] = Float32(a[2])
        out[i, 3, b] = Float32(a[3])
    end
    return out
end

function prepare_oracle_trueM_targets(cfg_path::AbstractString)
    base = dirname(cfg_path)
    cfg = load_dm_config(cfg_path)
    target_kind = configured_target_kind(cfg_path)
    require_condition(target_kind == :oracle_truem,
        "Expected [targets] target_kind = \"oracle_trueM\" in $(cfg_path).")
    device = detect_spin_device(cfg.device, cfg.required_gpu_name)
    activate_and_describe_device!(device, cfg.device, cfg.required_gpu_name)

    data_h5 = resolve_path(base, cfg.input_hdf5)
    score_path = resolve_path(base, cfg.score_bson)
    cond_path = resolve_path(base, cfg.cond_score_bson)
    retained_path = resolve_path(base, cfg.retained_channels_toml)
    out_path = resolve_path(base, cfg.target_artifact_bson)

    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    cond_kind = configured_cond_score_kind(cfg_path)
    cond_model, cond_cfg = load_transition_source(cond_kind, cfg_path, cond_path, base, device)
    @printf("Oracle target transition source kind: %s\n", String(cond_kind))

    channels = load_retained_channels(retained_path)
    names = unique_observable_names(channels)
    lib = NonlinearLibrary(names)
    obs_index = Dict(name => i for (i, name) in enumerate(names))
    selected_inds, selected_channel_ids = selected_linear_indices(channels, obs_index, sampler.N)
    means = estimate_nonlinear_means(sampler, p, lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 11))
    Phi_true, Phi_block = oracle_true_mean_mobility(sampler, p, cfg.target_mean_samples,
        cfg.seed + 40)

    lags = sampler.lag_steps[1:min(cfg.max_lags, length(sampler.lag_steps))]
    nobs = length(names)
    Ctrue = Array{Float64}(undef, length(lags), sampler.N, nobs, sampler.D)
    Cphi = similar(Ctrue)
    rng = MersenneTwister(cfg.seed + 120)
    for (li, lag) in enumerate(lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, cfg.target_pairs_per_lag, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats,
            cond_cfg, device; batch_size=min(cond_cfg.batch_size, cfg.target_pairs_per_lag),
            score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * nobs, cfg.target_pairs_per_lag)
        true_action = oracle_true_action_batch(x0, rraw, p)
        true_flat = flatten_batch(true_action)
        phi_action = transpose(Phi_true) * Matrix{Float64}(flatten_batch(rraw))
        Ctrue[li, :, :, :] .= reshape(
            -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(true_flat)) /
                cfg.target_pairs_per_lag,
            sampler.N, nobs, sampler.D)
        Cphi[li, :, :, :] .= reshape(
            -Matrix{Float64}(obs_flat) * transpose(phi_action) / cfg.target_pairs_per_lag,
            sampler.N, nobs, sampler.D)
        @printf("Prepared oracle true-M target lag %.5g (%d/%d), pairs=%d\n",
            lag * sampler.save_dt, li, length(lags), cfg.target_pairs_per_lag)
        GC.gc()
    end

    A_target = Ctrue .- Cphi
    target_vec = Array{Float32}(undef, length(lags), length(selected_inds))
    scale_vec = Array{Float32}(undef, length(selected_inds))
    for li in eachindex(lags)
        target_vec[li, :] .= Float32.(vec(A_target[li, :, :, :])[selected_inds])
    end
    for (k, cid) in enumerate(selected_channel_ids)
        scale_vec[k] = Float32(max(channels[cid].data_rms, 1f-6))
    end

    ensure_parent_dir(out_path)
    BSON.bson(out_path, Dict(:target_kind => :oracle_trueM,
        :names => names, :channels => channels,
        :observable_means => means, :lags => lags, :taus => lags .* sampler.save_dt,
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Cdot_data => Ctrue, :Cdot_trueM_cond => Ctrue, :Cdot_phi => Cphi,
        :Cdot_phi_trueMmean_cond => Cphi, :A_target => A_target,
        :Phi => Matrix{Float32}(Phi_true), :Phi_block => Matrix{Float32}(Phi_block),
        :save_dt => sampler.save_dt, :N => sampler.N, :D => sampler.D,
        :conditional_source_kind => cond_kind,
        :conditional_source_bson => cfg.cond_score_bson,
        :audit => "Oracle diagnostic target: Cdot and A were generated with true M and Phi=<M_true> using the learned transition score. This artifact is intentionally not data-only."))

    target_metrics = resolve_path(base, joinpath("..", "logs",
        replace(basename(out_path), ".bson" => "_metrics.txt")))
    ensure_parent_dir(target_metrics)
    open(target_metrics, "w") do io
        println(io, "SoftSpinLLGChain oracle true-M dM target metrics")
        println(io, "config = $(basename(cfg_path))")
        println(io, "target_kind = oracle_trueM")
        println(io, "conditional_source_kind = $(cond_kind)")
        println(io, "conditional_source_bson = $(cfg.cond_score_bson)")
        println(io, @sprintf("lags = %d", length(lags)))
        println(io, @sprintf("target_pairs_per_lag = %d", cfg.target_pairs_per_lag))
        println(io, @sprintf("target_mean_samples = %d", cfg.target_mean_samples))
        println(io, @sprintf("selected_channels = %d", length(selected_inds)))
        println(io, @sprintf("A_target rms = %.8e", sqrt(mean(abs2, A_target))))
        println(io, @sprintf("Cdot trueM rms = %.8e", sqrt(mean(abs2, Ctrue))))
        println(io, @sprintf("Cdot Phi rms = %.8e", sqrt(mean(abs2, Cphi))))
        println(io, @sprintf("Phi_true block = [%.8e %.8e %.8e; %.8e %.8e %.8e; %.8e %.8e %.8e]",
            Phi_block[1, 1], Phi_block[1, 2], Phi_block[1, 3],
            Phi_block[2, 1], Phi_block[2, 2], Phi_block[2, 3],
            Phi_block[3, 1], Phi_block[3, 2], Phi_block[3, 3]))
        println(io, "Audit: true M and <M_true> intentionally entered this oracle target and later mobility training.")
    end
    @printf("Saved oracle true-M target artifact to %s\n", out_path)
    @printf("Saved oracle target metrics to %s\n", target_metrics)
    return out_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    prepare_oracle_trueM_targets(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_ORACLE_DM_CONFIG)
end
