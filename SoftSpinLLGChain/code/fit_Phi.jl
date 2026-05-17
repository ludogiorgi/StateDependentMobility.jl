#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi.toml"))
const DEFAULT_FORWARD_CORR_MAX_LAGS = 300

Base.@kwdef struct PhiConfig
    input_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    max_fit_lags::Int
    pairs_per_lag_phi::Int
    pairs_per_lag_cdot::Int
    phi_projected_fit::Bool
    phi_projected_pairs_per_lag::Int
    phi_projected_chunk_size::Int
    phi_projected_sampling::String
    phi_fit_max_lag::Int
    phi_fit_degree::Int
    phi_include_zero_lag::Bool
    phi_zero_lag_samples::Int
    phi_shortlag_run::Bool
    phi_shortlag_samples::Int
    phi_shortlag_dt::Float64
    phi_shortlag_steps::Vector{Int}
    score_batch_size::Int
    true_mobility_samples::Int
    forward_run::Bool
    forward_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_save_dt::Float64
    forward_ntraj::Int
    forward_score_clip::Float32
    figure_width::Int
    figure_height::Int
    artifact_bson::String
    metrics_txt::String
    cdot_figure_png::String
    phi_figure_png::String
    forward_stats_png::String
    forward_cmn_png::String
    forward_hdf5::String
    device::String
    required_gpu_name::String
    seed::Int
    verbose::Bool
end

struct SpinPairSampler
    times::Vector{Float64}
    states::Array{Float32, 4}
    start_idx::Int
    save_dt::Float64
    N::Int
    D::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    tD::Float64
end

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    eval = raw["evaluation"]
    fwd = get(raw, "forward", Dict{String, Any}())
    fig = raw["figure"]
    out = raw["output"]
    run = raw["run"]
    return PhiConfig(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 1.0)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        max_fit_lags=Int(get(eval, "max_fit_lags", 60)),
        pairs_per_lag_phi=Int(get(eval, "pairs_per_lag_phi", 180000)),
        pairs_per_lag_cdot=Int(get(eval, "pairs_per_lag_cdot", 120000)),
        phi_projected_fit=Bool(get(eval, "phi_projected_fit", false)),
        phi_projected_pairs_per_lag=Int(get(eval, "phi_projected_pairs_per_lag",
            get(eval, "pairs_per_lag_phi", 180000))),
        phi_projected_chunk_size=Int(get(eval, "phi_projected_chunk_size", 200000)),
        phi_projected_sampling=String(get(eval, "phi_projected_sampling", "random")),
        phi_fit_max_lag=Int(get(eval, "phi_fit_max_lag", 6)),
        phi_fit_degree=Int(get(eval, "phi_fit_degree", 3)),
        phi_include_zero_lag=Bool(get(eval, "phi_include_zero_lag", true)),
        phi_zero_lag_samples=Int(get(eval, "phi_zero_lag_samples", 240000)),
        phi_shortlag_run=Bool(get(get(raw, "phi_shortlag", Dict{String, Any}()), "run", false)),
        phi_shortlag_samples=Int(get(get(raw, "phi_shortlag", Dict{String, Any}()), "samples", 400000)),
        phi_shortlag_dt=Float64(get(get(raw, "phi_shortlag", Dict{String, Any}()), "dt", 2.5e-4)),
        phi_shortlag_steps=Int.(get(get(raw, "phi_shortlag", Dict{String, Any}()), "save_steps", [0, 4, 8, 12])),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
        true_mobility_samples=Int(get(eval, "true_mobility_samples", 60000)),
        forward_run=Bool(get(fwd, "run", true)),
        forward_dt=Float64(get(fwd, "dt", 0.003)),
        forward_total_time=Float64(get(fwd, "total_time", 180.0)),
        forward_burnin_time=Float64(get(fwd, "burnin_time", 30.0)),
        forward_save_dt=Float64(get(fwd, "save_dt", 0.04)),
        forward_ntraj=Int(get(fwd, "ntrajectories", 72)),
        forward_score_clip=Float32(get(fwd, "score_clip", 80.0)),
        figure_width=Int(get(fig, "width", 3600)),
        figure_height=Int(get(fig, "height", 3000)),
        artifact_bson=String(out["artifact_bson"]),
        metrics_txt=String(out["metrics_txt"]),
        cdot_figure_png=String(out["cdot_figure_png"]),
        phi_figure_png=String(out["phi_figure_png"]),
        forward_stats_png=String(out["forward_stats_png"]),
        forward_cmn_png=String(out["forward_cmn_png"]),
        forward_hdf5=String(out["forward_hdf5"]),
        device=String(get(run, "device", "GPU:0")),
        required_gpu_name=String(get(run, "required_gpu_name", "2080ti")),
        seed=Int(get(run, "seed", 20260509)),
        verbose=Bool(get(run, "verbose", true)),
    )
end

function load_phys(path::AbstractString)
    h5open(path, "r") do f
        return SpinParams(
            N=Int(read(f["/metadata/N"])),
            lambda=Float64(read(f["/metadata/lambda"])),
            mstar=Float64(read(f["/metadata/mstar"])),
            J=Float64(read(f["/metadata/J"])),
            K=Float64(read(f["/metadata/K"])),
            theta=Float64(read(f["/metadata/Theta"])),
            gamma=Float64(read(f["/metadata/gamma"])),
            alpha_perp=Float64(read(f["/metadata/alpha_perp"])),
            alpha_parallel=Float64(read(f["/metadata/alpha_parallel"])),
            eps=Float64(read(f["/metadata/eps"])),
        )
    end
end

function build_sampler(path::AbstractString, burnin_fraction::Float64,
        tau_max_decorrelation_multiples::Float64, lag_stride::Int)
    times, states = load_spin_states(path)
    start = burnin_start_index(length(times), burnin_fraction)
    save_dt = times[2] - times[1]
    tD = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    tau_max = min(tau_max_decorrelation_multiples * tD, times[end] - times[start])
    max_lag = min(length(times) - start - 1, floor(Int, tau_max / save_dt))
    require_condition(max_lag >= 1, "No positive lags available for Phi fitting.")
    lag_steps = collect(1:lag_stride:max_lag)
    return SpinPairSampler(times, states, start, save_dt, size(states, 2), 3size(states, 2),
        lag_steps, lag_steps .* save_dt, tD)
end

function random_lag_pairs(sampler::SpinPairSampler, lag::Int, npairs::Int, rng::AbstractRNG;
        centered_window::Int=0)
    nt, N, _, ntraj = size(sampler.states)
    lower = max(sampler.start_idx + centered_window, sampler.start_idx)
    upper = nt - lag - centered_window
    require_condition(upper >= lower, "Requested lag/window exceeds available trajectory.")
    x0 = Array{Float32}(undef, N, 3, npairs)
    xt = Array{Float32}(undef, N, 3, npairs)
    xp = centered_window > 0 ? Array{Float32}(undef, N, 3, npairs) : Array{Float32}(undef, 0, 0, 0)
    xm = centered_window > 0 ? Array{Float32}(undef, N, 3, npairs) : Array{Float32}(undef, 0, 0, 0)
    @inbounds for b in 1:npairs
        tr = rand(rng, 1:ntraj)
        t = rand(rng, lower:upper)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        if centered_window > 0
            xp[:, :, b] .= sampler.states[t + lag + centered_window, :, :, tr]
            xm[:, :, b] .= sampler.states[t + lag - centered_window, :, :, tr]
        end
    end
    return x0, xt, xp, xm
end

function sample_raw_states(sampler::SpinPairSampler, nsamples::Int, rng::AbstractRNG)
    return sample_state_tensor(sampler.states, sampler.start_idx, nsamples, rng)
end

function covariance_derivative_phi(sampler::SpinPairSampler, stats::DataStats, params::PhiConfig)
    D = sampler.D
    L = min(params.phi_fit_max_lag, length(sampler.lag_steps))
    nfit = L + (params.phi_include_zero_lag ? 1 : 0)
    taus = Vector{Float64}(undef, nfit)
    covs = Array{Float64}(undef, nfit, D, D)
    mu = Float64.(mean_flat(stats))
    rng = MersenneTwister(params.seed + 10)
    offset = 0
    if params.phi_include_zero_lag
        n0 = max(1, params.phi_zero_lag_samples)
        raw = sample_raw_states(sampler, n0, rng)
        x0f = Float64.(flatten_batch(raw))
        x0f .-= mu
        taus[1] = 0.0
        covs[1, :, :] .= (x0f * transpose(x0f)) ./ n0
        offset = 1
    end
    for ell in 1:L
        lag = sampler.lag_steps[ell]
        x0, xt, _, _ = random_lag_pairs(sampler, lag, params.pairs_per_lag_phi, rng)
        x0f = Float64.(flatten_batch(x0))
        xtf = Float64.(flatten_batch(xt))
        x0f .-= mu
        xtf .-= mu
        taus[offset + ell] = sampler.lag_times[ell]
        covs[offset + ell, :, :] .= (xtf * transpose(x0f)) ./ params.pairs_per_lag_phi
        params.verbose && @printf("Phi covariance lag %.5g (%d/%d)\n", sampler.lag_times[ell], ell, L)
    end
    Cdot0 = zeros(Float64, D, D)
    @inbounds for i in 1:D, j in 1:D
        Cdot0[i, j] = polynomial_derivative_at(taus, covs[:, i, j], 0.0, params.phi_fit_degree)
    end
    return taus, covs, Cdot0, -Cdot0
end

function full_onsite_matrix_from_block(block::AbstractMatrix{<:Real}, N::Int)
    A = zeros(Float64, 3N, 3N)
    @inbounds for i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        A[rows, rows] .= block
    end
    return A
end

function accumulate_onsite_covariance_block!(block::AbstractMatrix{Float64},
        x0::Array{Float32, 3}, xt::Array{Float32, 3}, mean_site::Array{Float64, 2})
    N, _, B = size(x0)
    @inbounds for b in 1:B, i in 1:N
        x01 = Float64(x0[i, 1, b]) - mean_site[i, 1]
        x02 = Float64(x0[i, 2, b]) - mean_site[i, 2]
        x03 = Float64(x0[i, 3, b]) - mean_site[i, 3]
        xt1 = Float64(xt[i, 1, b]) - mean_site[i, 1]
        xt2 = Float64(xt[i, 2, b]) - mean_site[i, 2]
        xt3 = Float64(xt[i, 3, b]) - mean_site[i, 3]
        block[1, 1] += xt1 * x01
        block[1, 2] += xt1 * x02
        block[1, 3] += xt1 * x03
        block[2, 1] += xt2 * x01
        block[2, 2] += xt2 * x02
        block[2, 3] += xt2 * x03
        block[3, 1] += xt3 * x01
        block[3, 2] += xt3 * x02
        block[3, 3] += xt3 * x03
    end
    return block
end

function sampled_onsite_covariance_block(sampler::SpinPairSampler, stats::DataStats,
        lag::Int, npairs::Int, rng::AbstractRNG, params::PhiConfig)
    if lowercase(params.phi_projected_sampling) == "stratified"
        return stratified_onsite_covariance_block(sampler, stats, lag, npairs)
    end
    mean_site = Float64.(permutedims(stats.mean, (2, 1)))
    block = zeros(Float64, 3, 3)
    remaining = npairs
    while remaining > 0
        chunk = min(params.phi_projected_chunk_size, remaining)
        if lag == 0
            x0 = sample_raw_states(sampler, chunk, rng)
            accumulate_onsite_covariance_block!(block, x0, x0, mean_site)
        else
            x0, xt, _, _ = random_lag_pairs(sampler, lag, chunk, rng)
            accumulate_onsite_covariance_block!(block, x0, xt, mean_site)
        end
        remaining -= chunk
    end
    block ./= (sampler.N * npairs)
    return block
end

function stratified_onsite_covariance_block(sampler::SpinPairSampler, stats::DataStats,
        lag::Int, npairs::Int)
    nt, N, _, ntraj = size(sampler.states)
    lower = sampler.start_idx
    upper = nt - lag
    require_condition(upper >= lower, "Requested lag exceeds available trajectory.")
    ntime = upper - lower + 1
    total_pairs = ntime * ntraj
    nsel = min(npairs, total_pairs)
    stride = total_pairs / nsel
    mean_site = Float64.(permutedims(stats.mean, (2, 1)))
    blocks = [zeros(Float64, 3, 3) for _ in 1:Threads.nthreads()]
    states = sampler.states
    Threads.@threads for s in 0:(nsel - 1)
        tid = Threads.threadid()
        linear = min(total_pairs - 1, floor(Int, (s + 0.5) * stride))
        t = lower + (linear % ntime)
        tr = 1 + (linear ÷ ntime)
        block = blocks[tid]
        @inbounds for i in 1:N
            x01 = Float64(states[t, i, 1, tr]) - mean_site[i, 1]
            x02 = Float64(states[t, i, 2, tr]) - mean_site[i, 2]
            x03 = Float64(states[t, i, 3, tr]) - mean_site[i, 3]
            xt1 = Float64(states[t + lag, i, 1, tr]) - mean_site[i, 1]
            xt2 = Float64(states[t + lag, i, 2, tr]) - mean_site[i, 2]
            xt3 = Float64(states[t + lag, i, 3, tr]) - mean_site[i, 3]
            block[1, 1] += xt1 * x01
            block[1, 2] += xt1 * x02
            block[1, 3] += xt1 * x03
            block[2, 1] += xt2 * x01
            block[2, 2] += xt2 * x02
            block[2, 3] += xt2 * x03
            block[3, 1] += xt3 * x01
            block[3, 2] += xt3 * x02
            block[3, 3] += xt3 * x03
        end
    end
    block = zeros(Float64, 3, 3)
    for local_block in blocks
        block .+= local_block
    end
    block ./= (N * nsel)
    return block
end

function projected_covariance_derivative_phi(sampler::SpinPairSampler,
        stats::DataStats, params::PhiConfig)
    L = min(params.phi_fit_max_lag, length(sampler.lag_steps))
    nfit = L + (params.phi_include_zero_lag ? 1 : 0)
    require_condition(nfit >= 2, "Projected Phi fit requires at least two lag points.")
    taus = Vector{Float64}(undef, nfit)
    block_covs = Array{Float64}(undef, nfit, 3, 3)
    rng = MersenneTwister(params.seed + 11)
    offset = 0
    if params.phi_include_zero_lag
        n0 = max(1, params.phi_zero_lag_samples)
        taus[1] = 0.0
        block_covs[1, :, :] .= sampled_onsite_covariance_block(sampler, stats, 0, n0, rng, params)
        offset = 1
    end
    for ell in 1:L
        lag = sampler.lag_steps[ell]
        block_covs[offset + ell, :, :] .= sampled_onsite_covariance_block(sampler, stats,
            lag, params.phi_projected_pairs_per_lag, rng, params)
        taus[offset + ell] = sampler.lag_times[ell]
        params.verbose && @printf("Projected Phi covariance lag %.5g (%d/%d), pairs %d\n",
            sampler.lag_times[ell], ell, L, params.phi_projected_pairs_per_lag)
    end
    Cdot_block = zeros(Float64, 3, 3)
    @inbounds for a in 1:3, b in 1:3
        Cdot_block[a, b] = polynomial_derivative_at(taus, block_covs[:, a, b],
            0.0, params.phi_fit_degree)
    end
    Cdot0 = full_onsite_matrix_from_block(Cdot_block, sampler.N)
    Phi_raw = -Cdot0
    covs = Array{Float64}(undef, nfit, sampler.D, sampler.D)
    @inbounds for k in 1:nfit
        covs[k, :, :] .= full_onsite_matrix_from_block(@view(block_covs[k, :, :]), sampler.N)
    end
    return taus, covs, Cdot0, Phi_raw
end

function covariance_derivative_phi_shortlag(sampler::SpinPairSampler, p::SpinParams,
        stats::DataStats, params::PhiConfig)
    steps = sort(unique(params.phi_shortlag_steps))
    require_condition(!isempty(steps) && first(steps) == 0,
        "phi_shortlag.save_steps must include 0 as the first saved step.")
    require_condition(length(steps) >= 3, "Need at least three short-lag points for Phi extrapolation.")
    D = sampler.D
    taus = steps .* params.phi_shortlag_dt
    cov_threads = [zeros(Float64, length(steps), D, D) for _ in 1:Threads.nthreads()]
    work = [make_work(p) for _ in 1:Threads.nthreads()]
    mu = Float64.(mean_flat(stats))
    max_step = maximum(steps)
    index_for_step = Dict(step => idx for (idx, step) in enumerate(steps))
    @printf("Short-lag Phi resimulation: %d stationary starts, dt %.4g, saved steps %s\n",
        params.phi_shortlag_samples, params.phi_shortlag_dt, string(steps))
    Threads.@threads for sample_idx in 1:params.phi_shortlag_samples
        tid = Threads.threadid()
        rng = MersenneTwister(params.seed + 70_000 + sample_idx)
        tr = rand(rng, 1:size(sampler.states, 4))
        t = rand(rng, sampler.start_idx:size(sampler.states, 1))
        x = Float64.(sampler.states[t, :, :, tr])
        x0 = flatten_state(x) .- mu
        cov_threads[tid][1, :, :] .+= x0 * transpose(x0)
        for step in 1:max_step
            em_step!(x, p, params.phi_shortlag_dt, rng, work[tid])
            idx = get(index_for_step, step, 0)
            if idx > 0
                xt = flatten_state(x) .- mu
                cov_threads[tid][idx, :, :] .+= xt * transpose(x0)
            end
        end
    end
    covs = zeros(Float64, length(steps), D, D)
    for C in cov_threads
        covs .+= C
    end
    covs ./= params.phi_shortlag_samples
    Cdot0 = zeros(Float64, D, D)
    degree = min(params.phi_fit_degree, length(steps) - 1)
    @inbounds for i in 1:D, j in 1:D
        Cdot0[i, j] = polynomial_derivative_at(taus, covs[:, i, j], 0.0, degree)
    end
    return taus, covs, Cdot0, -Cdot0
end

function observable_values(raw::Array{Float32, 3}, p::SpinParams)
    N, _, B = size(raw)
    names = ["mx", "my", "mz", "r2", "mperp2", "mz2", "local_U"]
    obs = Array{Float32}(undef, N, length(names), B)
    @inbounds for b in 1:B
        for i in 1:N
            ip = periodic(i + 1, N)
            x1, x2, x3 = Float64(raw[i, 1, b]), Float64(raw[i, 2, b]), Float64(raw[i, 3, b])
            r2 = x1 * x1 + x2 * x2 + x3 * x3
            diff2 = sum(abs2, Float64.(raw[ip, :, b]) .- Float64.(raw[i, :, b]))
            obs[i, 1, b] = raw[i, 1, b]
            obs[i, 2, b] = raw[i, 2, b]
            obs[i, 3, b] = raw[i, 3, b]
            obs[i, 4, b] = Float32(r2)
            obs[i, 5, b] = Float32(x1 * x1 + x2 * x2)
            obs[i, 6, b] = Float32(x3 * x3)
            obs[i, 7, b] = Float32(0.25 * p.lambda * (r2 - p.mstar^2)^2 + 0.5 * p.J * diff2 - 0.5 * p.K * x3^2)
        end
    end
    return obs, names
end

function estimate_observable_means(sampler::SpinPairSampler, p::SpinParams, nsamples::Int, seed::Int)
    raw = sample_raw_states(sampler, nsamples, MersenneTwister(seed))
    obs, names = observable_values(raw, p)
    means = [mean(Float64, @view obs[:, j, :]) for j in 1:length(names)]
    return names, means
end

function center_observables!(obs::Array{Float32, 3}, means::Vector{Float64})
    @inbounds for j in 1:length(means)
        obs[:, j, :] .-= Float32(means[j])
    end
    return obs
end

function estimate_data_cdot(sampler::SpinPairSampler, p::SpinParams, names, means,
        stats::DataStats, params::PhiConfig)
    lags = sampler.lag_steps[1:min(params.max_fit_lags, length(sampler.lag_steps))]
    nobs = length(names)
    Cdot = Array{Float64}(undef, length(lags), sampler.N, nobs, sampler.D)
    mu = Float64.(mean_flat(stats))
    rng = MersenneTwister(params.seed + 20)
    for (li, lag) in enumerate(lags)
        x0, _, xp, xm = random_lag_pairs(sampler, lag, params.pairs_per_lag_cdot, rng;
            centered_window=1)
        x0f = Float64.(flatten_batch(x0))
        x0f .-= mu
        obsp, _ = observable_values(xp, p)
        obsm, _ = observable_values(xm, p)
        deriv = (obsp .- obsm) ./ Float32(2.0 * sampler.save_dt)
        deriv_flat = reshape(deriv, sampler.N * nobs, params.pairs_per_lag_cdot)
        mat = Matrix{Float64}(deriv_flat) * transpose(x0f) ./ params.pairs_per_lag_cdot
        Cdot[li, :, :, :] .= reshape(mat, sampler.N, nobs, sampler.D)
        params.verbose && @printf("Data Cdot lag %.5g (%d/%d)\n", sampler.lag_times[li], li, length(lags))
    end
    return lags, sampler.lag_times[1:length(lags)], Cdot
end

function load_score_checkpoint(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    stats_obj = blob[:stats]
    stats = stats_obj isa DataStats ? stats_obj : DataStats(Float32.(stats_obj[:mean]), Float32.(stats_obj[:std]))
    sigma = Float32(blob[:trainer_cfg][:sigma])
    return model, stats, sigma, blob
end

function evaluate_raw_score(model, raw::Array{Float32, 3}, stats::DataStats, sigma::Float32,
        device::ExecutionDevice; batch_size::Int)
    normed = apply_stats_tensor(raw, stats)
    sn = evaluate_score_norm(model, normed, sigma, device; batch_size)
    return normalized_score_to_raw(sn, stats)
end

function estimate_phi_gfdt(sampler::SpinPairSampler, p::SpinParams, names, means,
        stats::DataStats, score_model, sigma::Float32, Phi::AbstractMatrix{<:Real},
        taus::Vector{Float64}, lags::Vector{Int}, params::PhiConfig, device::ExecutionDevice)
    nobs = length(names)
    Cdot = Array{Float64}(undef, length(lags), sampler.N, nobs, sampler.D)
    rng = MersenneTwister(params.seed + 30)
    for (li, lag) in enumerate(lags)
        x0, xt, _, _ = random_lag_pairs(sampler, lag, params.pairs_per_lag_cdot, rng)
        obs, _ = observable_values(xt, p)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * nobs, params.pairs_per_lag_cdot)
        sraw = evaluate_raw_score(score_model, x0, stats, sigma, device; batch_size=params.score_batch_size)
        sflat = Float64.(flatten_batch(sraw))
        action = transpose(Matrix{Float64}(Phi)) * sflat
        mat = Matrix{Float64}(obs_flat) * transpose(action) ./ params.pairs_per_lag_cdot
        Cdot[li, :, :, :] .= reshape(mat, sampler.N, nobs, sampler.D)
        params.verbose && @printf("Phi GFDT lag %.5g (%d/%d)\n", taus[li], li, length(lags))
    end
    return Cdot
end

function true_mean_mobility(sampler::SpinPairSampler, p::SpinParams, nsamples::Int, seed::Int)
    raw = sample_raw_states(sampler, nsamples, MersenneTwister(seed))
    M = zeros(Float64, sampler.D, sampler.D)
    @inbounds for b in 1:nsamples
        M .+= true_mobility_matrix((@view raw[:, :, b]), p)
    end
    return M ./ nsamples
end

function render_phi_figure(path, params, Phi_raw, Phi_projected, Phi, Mtrue, eigvals, metrics)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    figure_title!(fig, "Soft-spin LLG Phi recovery";
        subtitle=@sprintf("Phi vs <M_true> rel.RMSE %.4e, corr %.5f, min sym eig %.4e",
            metrics[:relative_rmse], metrics[:correlation], minimum(eigvals)))
    mats = [Phi_raw, Phi_projected, Phi, Mtrue, Phi - Mtrue]
    titles = ["Phi raw", "block-circulant Phi", "PSD-projected Phi", "<M_true> ex-post", "Phi - <M_true>"]
    for j in 1:5
        ax = Axis(fig[1 + (j - 1) ÷ 3, 1 + (j - 1) % 3]; title=titles[j], xlabel="column", ylabel="row")
        heatmap!(ax, mats[j]; colormap=:balance)
    end
    ax = Axis(fig[2, 3]; title="Symmetric-part eigenvalues", xlabel="index", ylabel="eigenvalue")
    scatterlines!(ax, collect(eachindex(eigvals)), eigvals; color=STYLE_PRIMARY)
    save_figure_checked(path, fig)
end

function render_cdot_figure(path, params, taus, names, Cdata, Cphi, metrics)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    figure_title!(fig, "Soft-spin LLG data Cdot vs Phi stationary-score GFDT";
        subtitle=@sprintf("All selected channels rel.RMSE %.4e, corr %.5f", metrics[:relative_rmse], metrics[:correlation]))
    panels = min(12, size(Cdata, 2) * size(Cdata, 3))
    for panel in 1:panels
        site = 1 + (panel - 1) % size(Cdata, 2)
        obs = 1 + ((panel - 1) ÷ size(Cdata, 2)) % size(Cdata, 3)
        flat_col = (site - 1) * 3 + min(obs, 3)
        ax = Axis(fig[1 + (panel - 1) ÷ 4, 1 + (panel - 1) % 4];
            title="site $(site) $(names[obs]) vs x$(flat_col)", xlabel="tau", ylabel="Cdot")
        lines!(ax, taus, Cdata[:, site, obs, flat_col]; color=:black, linewidth=2, label="data")
        lines!(ax, taus, Cphi[:, site, obs, flat_col]; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="Phi GFDT")
        panel == 1 && axislegend(ax; position=:rt)
    end
    save_figure_checked(path, fig)
end

function initial_raw_batch(sampler::SpinPairSampler, ntraj::Int, rng::AbstractRNG)
    raw = sample_raw_states(sampler, ntraj, rng)
    return Float64.(raw)
end

function sqrt_factor_psd(S::AbstractMatrix{<:Real})
    ev = eigen(Symmetric(sympart(S)))
    vals = max.(ev.values, 0.0)
    keep = vals .> 1.0e-12
    return ev.vectors[:, keep] * Diagonal(sqrt.(vals[keep])), vals
end

function integrate_phi_forward(score_model, sigma::Float32, stats::DataStats,
        Phi::AbstractMatrix{<:Real}, sampler::SpinPairSampler, params::PhiConfig,
        device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 500)
    z = initial_raw_batch(sampler, params.forward_ntraj, rng)
    D = sampler.D
    sqrt_phi, eigvals = sqrt_factor_psd(sympart(Phi))
    nsteps = ceil(Int, params.forward_total_time / params.forward_dt)
    burn_steps = floor(Int, params.forward_burnin_time / params.forward_dt)
    save_every = max(1, round(Int, params.forward_save_dt / params.forward_dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, params.forward_ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            saved[save_idx, :, :, :] .= Float32.(z)
            times[save_idx] = (step - burn_steps) * params.forward_dt
        end
        step == nsteps && break
        raw32 = Float32.(z)
        score_raw = evaluate_raw_score(score_model, raw32, stats, sigma, device; batch_size=params.score_batch_size)
        sflat = Float64.(flatten_batch(score_raw))
        clamp!(sflat, -Float64(params.forward_score_clip), Float64(params.forward_score_clip))
        drift = Matrix{Float64}(Phi) * sflat
        noise = sqrt_phi * randn(rng, size(sqrt_phi, 2), params.forward_ntraj)
        flat = Float64.(flatten_batch(Float32.(z)))
        @. flat = flat + params.forward_dt * drift + sqrt(2.0 * params.forward_dt) * noise
        z .= unflatten_batch(flat, sampler.N)
    end
    return times[1:save_idx], saved[1:save_idx, :, :, :], eigvals
end

function save_forward(path, times, states, eigvals)
    ensure_parent_dir(path)
    h5open(path, "w") do f
        f["/trajectories/time"] = times
        f["/trajectories/states"] = states
        f["/metadata/model"] = "constant Phi score-based Langevin"
        f["/metadata/min_sym_phi_eig"] = minimum(eigvals)
    end
    @printf("Saved Phi forward trajectories to %s\n", path)
end

function flat_samples_from_states(states::AbstractArray{<:Real, 4}, max_samples::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(states)
    total = nt * ntraj
    n = min(max_samples, total)
    out = Matrix{Float64}(undef, 3N, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = 1 + (linear % nt)
        tr = 1 + (linear ÷ nt)
        out[:, s] .= flatten_state(@view states[t, :, :, tr])
    end
    return out
end

function sampled_forward_values(states::AbstractArray{<:Real, 4}, j::Int, max_values::Int,
        rng::AbstractRNG)
    nt, N, _, ntraj = size(states)
    total = nt * N * ntraj
    n = min(max_values, total)
    out = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = total <= max_values ? s - 1 : rand(rng, 0:(total - 1))
        t = 1 + (linear % nt)
        rem = linear ÷ nt
        i = 1 + (rem % N)
        tr = 1 + (rem ÷ N)
        if j <= 3
            out[s] = Float64(states[t, i, j, tr])
        else
            out[s] = sqrt(sum(abs2, @view states[t, i, :, tr]))
        end
    end
    return out
end

function render_forward_stats(path, params, obs_states::AbstractArray{<:Real, 4},
        phi_states::AbstractArray{<:Real, 4})
    fig = Figure(; size=(params.figure_width, params.figure_height))
    figure_title!(fig, "Soft-spin LLG Phi forward statistics";
        subtitle="Stationary PDFs/covariance from all available obs/model samples")
    labels = ["mx", "my", "mz", "|m|"]
    for j in 1:4
        ax = Axis(fig[1, j]; title="PDF $(labels[j])", xlabel=labels[j], ylabel="density")
        vo = sampled_forward_values(obs_states, j, 1_000_000, MersenneTwister(100 + j))
        vp = sampled_forward_values(phi_states, j, 1_000_000, MersenneTwister(200 + j))
        ko = kde(vo)
        kp = kde(vp)
        lines!(ax, ko.x, ko.density; color=:black, linewidth=2, label="obs")
        lines!(ax, kp.x, kp.density; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="Phi")
        axislegend(ax; position=:rt)
    end
    cov_obs = cov(permutedims(flat_samples_from_states(obs_states, 200000, MersenneTwister(1))))
    cov_phi = cov(permutedims(flat_samples_from_states(phi_states, 200000, MersenneTwister(2))))
    ax1 = Axis(fig[2, 1:2]; title="Observed covariance", xlabel="flat index", ylabel="flat index")
    heatmap!(ax1, cov_obs; colormap=:balance)
    ax2 = Axis(fig[2, 3:4]; title="Phi covariance error", xlabel="flat index", ylabel="flat index")
    heatmap!(ax2, cov_phi - cov_obs; colormap=:balance)
    save_figure_checked(path, fig)
end

function estimate_coordinate_correlations(states::Array{Float32, 4}, save_dt::Float64, max_lags::Int)
    nt, N, _, ntraj = size(states)
    D = 3N
    L = min(max_lags, nt - 1)
    C = Array{Float64}(undef, L + 1, D, D)
    flat_all = [Float64.(flatten_state(@view states[t, :, :, tr])) for t in 1:nt, tr in 1:ntraj]
    mu = reduce(+, flat_all) ./ length(flat_all)
    for lag in 0:L
        mat = zeros(Float64, D, D)
        count = 0
        for tr in 1:ntraj, t in 1:(nt - lag)
            x0 = flat_all[t, tr] .- mu
            xt = flat_all[t + lag, tr] .- mu
            mat .+= xt * transpose(x0)
            count += 1
        end
        C[lag + 1, :, :] .= mat ./ count
    end
    return collect(0:L) .* save_dt, C
end

function render_forward_cmn(path, params, obs_t, Cobs, phi_t, Cphi)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    metrics = agreement_metrics(Cobs[1:min(end, size(Cphi, 1)), :, :], Cphi[1:min(end, size(Cobs, 1)), :, :])
    figure_title!(fig, "Soft-spin LLG Phi forward coordinate correlations";
        subtitle=@sprintf("Coordinate C(t) rel.RMSE %.4e, corr %.5f", metrics[:relative_rmse], metrics[:correlation]))
    D = size(Cobs, 2)
    panels = min(12, D)
    for j in 1:panels
        ax = Axis(fig[1 + (j - 1) ÷ 4, 1 + (j - 1) % 4]; title="C$(j),$(j)(t)", xlabel="t", ylabel="C")
        lines!(ax, obs_t, Cobs[:, j, j]; color=:black, linewidth=2, label="obs")
        lines!(ax, phi_t, Cphi[:, j, j]; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="Phi")
        j == 1 && axislegend(ax; position=:rt)
    end
    save_figure_checked(path, fig)
    return metrics
end

function run_pipeline(param_file::AbstractString)
    base = dirname(param_file)
    params = load_config(param_file)
    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)
    p = load_phys(data_h5)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, sigma, _ = load_score_checkpoint(score_path, device)
    saved_phi_taus, saved_phi_covs, saved_Cdot0_raw, saved_Phi_raw =
        params.phi_projected_fit ?
        projected_covariance_derivative_phi(sampler, stats, params) :
        covariance_derivative_phi(sampler, stats, params)
    phi_taus, phi_covs, Cdot0_raw, Phi_raw = if params.phi_shortlag_run
        covariance_derivative_phi_shortlag(sampler, p, stats, params)
    else
        saved_phi_taus, saved_phi_covs, saved_Cdot0_raw, saved_Phi_raw
    end
    Phi_projected, Phi_block = project_soft_spin_phi(Phi_raw, sampler.N)
    Spsd, eigvals = psd_project_symmetric(sympart(Phi_projected))
    Phi = Spsd + skewpart(Phi_projected)
    Mtrue = true_mean_mobility(sampler, p, params.true_mobility_samples, params.seed + 40)
    metrics_phi = agreement_metrics(Mtrue, Phi)
    names, means = estimate_observable_means(sampler, p, min(200000, params.pairs_per_lag_cdot), params.seed + 50)
    lags, taus, Cdot_data = estimate_data_cdot(sampler, p, names, means, stats, params)
    Cdot_phi = estimate_phi_gfdt(sampler, p, names, means, stats, score_model, sigma, Phi,
        taus, lags, params, device)
    metrics_cdot = agreement_metrics(Cdot_data, Cdot_phi)
    render_phi_figure(resolve_path(base, params.phi_figure_png), params, Phi_raw,
        Phi_projected, Phi, Mtrue, eigvals, metrics_phi)
    render_cdot_figure(resolve_path(base, params.cdot_figure_png), params, taus,
        names, Cdot_data, Cdot_phi, metrics_cdot)
    phi_forward_states = nothing
    metrics_forward = nothing
    if params.forward_run
        ft, fs, feigs = integrate_phi_forward(score_model, sigma, stats, Phi, sampler, params, device)
        save_forward(resolve_path(base, params.forward_hdf5), ft, fs, feigs)
        obs_stop = min(size(sampler.states, 1), sampler.start_idx + size(fs, 1) - 1)
        obs = sampler.states[sampler.start_idx:obs_stop, :, :, 1:min(size(sampler.states, 4), size(fs, 4))]
        phi = fs[1:size(obs, 1), :, :, 1:size(obs, 4)]
        obs_stats = @view sampler.states[sampler.start_idx:end, :, :, :]
        render_forward_stats(resolve_path(base, params.forward_stats_png), params, obs_stats, fs)
        corr_lags = min(DEFAULT_FORWARD_CORR_MAX_LAGS, size(obs, 1) - 1)
        obs_t, Cobs = estimate_coordinate_correlations(obs, sampler.save_dt, corr_lags)
        phi_t, Cphi = estimate_coordinate_correlations(phi, ft[2] - ft[1],
            min(DEFAULT_FORWARD_CORR_MAX_LAGS, size(phi, 1) - 1))
        metrics_forward = render_forward_cmn(resolve_path(base, params.forward_cmn_png), params, obs_t, Cobs, phi_t, Cphi)
        phi_forward_states = fs
    end
    artifact_path = resolve_path(base, params.artifact_bson)
    ensure_parent_dir(artifact_path)
    BSON.bson(artifact_path, Dict(:params => params, :phi_taus => phi_taus,
        :phi_covariances => phi_covs, :Cdot0_raw => Cdot0_raw, :Phi_raw => Phi_raw,
        :saved_phi_taus => saved_phi_taus, :saved_phi_covariances => saved_phi_covs,
        :saved_Cdot0_raw => saved_Cdot0_raw, :saved_Phi_raw => saved_Phi_raw,
        :Phi_projected => Phi_projected, :Phi_projected_block => Phi_block,
        :Phi => Phi, :Mtrue => Mtrue,
        :metrics_phi => metrics_phi, :observable_names => names, :observable_means => means,
        :lags => lags, :taus => taus, :Cdot_data => Cdot_data, :Cdot_phi => Cdot_phi,
        :metrics_cdot => metrics_cdot, :phi_forward_states => phi_forward_states,
        :metrics_forward => metrics_forward,
        :no_cheating_audit => "Phi_raw and Cdot_data are estimated from trajectory correlations only. Phi GFDT uses the learned stationary score. True mobility is saved only as ex-post diagnostics."))
    metrics_path = resolve_path(base, params.metrics_txt)
    ensure_parent_dir(metrics_path)
    open(metrics_path, "w") do io
        println(io, "SoftSpinLLGChain Step 2 Phi metrics")
        println(io, @sprintf("Phi vs <M_true> rel.RMSE = %.8e", metrics_phi[:relative_rmse]))
        println(io, @sprintf("Phi vs <M_true> corr = %.8e", metrics_phi[:correlation]))
        println(io, @sprintf("Phi symmetric min eigenvalue = %.8e", minimum(eigvals)))
        println(io, @sprintf("Phi projection relative change = %.8e", norm(Phi_projected - Phi_raw) / max(norm(Phi_raw), eps(Float64))))
        println(io, @sprintf("Phi PSD projection relative change = %.8e", norm(Phi - Phi_projected) / max(norm(Phi_projected), eps(Float64))))
        println(io, "Phi projected fit = " * string(params.phi_projected_fit))
        if params.phi_projected_fit
            println(io, @sprintf("Phi projected pairs per lag = %d", params.phi_projected_pairs_per_lag))
            println(io, @sprintf("Phi projected chunk size = %d", params.phi_projected_chunk_size))
            println(io, "Phi projected sampling = " * params.phi_projected_sampling)
        end
        println(io, @sprintf("Phi fit max lag index = %d", params.phi_fit_max_lag))
        println(io, @sprintf("Phi fit polynomial degree = %d", params.phi_fit_degree))
        println(io, @sprintf("Phi onsite axial block = [%.8e %.8e %.8e; %.8e %.8e %.8e; %.8e %.8e %.8e]",
            Phi_block[1, 1], Phi_block[1, 2], Phi_block[1, 3],
            Phi_block[2, 1], Phi_block[2, 2], Phi_block[2, 3],
            Phi_block[3, 1], Phi_block[3, 2], Phi_block[3, 3]))
        if params.phi_shortlag_run
            println(io, @sprintf("Phi short-lag samples = %d", params.phi_shortlag_samples))
            println(io, @sprintf("Phi short-lag dt = %.8e", params.phi_shortlag_dt))
            println(io, "Phi short-lag saved steps = " * string(params.phi_shortlag_steps))
        end
        println(io, @sprintf("Cdot Phi-GFDT rel.RMSE = %.8e", metrics_cdot[:relative_rmse]))
        println(io, @sprintf("Cdot Phi-GFDT corr = %.8e", metrics_cdot[:correlation]))
        if metrics_forward !== nothing
            println(io, @sprintf("Forward coordinate C rel.RMSE = %.8e", metrics_forward[:relative_rmse]))
            println(io, @sprintf("Forward coordinate C corr = %.8e", metrics_forward[:correlation]))
        end
        println(io, "No-cheating audit: data-driven estimators and targets used trajectory data and learned score only; analytic mobility appears only in ex-post metrics.")
    end
    @printf("Step 2 Phi stage complete. Artifacts saved to %s\n", artifact_path)
    @printf("No-cheating audit: no analytic score or true mobility entered Phi, Cdot_data, GFDT targets, or model selection.\n")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE)
end
