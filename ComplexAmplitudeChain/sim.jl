#!/usr/bin/env julia

# Simulates the stochastic complex-amplitude chain with state-dependent
# dissipative and reactive mobility from paper.tex.
#
# The script deliberately stops at data generation:
#   1. run a pilot integration and estimate the decorrelation time t_D;
#   2. match the L96 uncorrelated-sample budget, T = N_L96 * t_D,
#      up to the integer dt grid;
#   3. save at the requested number of snapshots per decorrelation time;
#   4. save trajectories and diagnostic figures.

import Pkg

function ensure_packages(packages::Vector{String})
    missing = String[]
    for pkg in packages
        if Base.find_package(pkg) === nothing
            push!(missing, pkg)
        end
    end
    if !isempty(missing)
        @info "Installing missing Julia packages" missing
        Pkg.add(missing)
    end
    return nothing
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ensure_packages(["FFTW", "HDF5", "KernelDensity", "GLMakie"])

using FFTW
using HDF5
using KernelDensity
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

const STYLE_FILE_CANDIDATES = (
    joinpath(@__DIR__, "src", "figure_style.jl"),
    normpath(joinpath(@__DIR__, "..", "2D", "src", "figure_style.jl")),
)
for style_file in STYLE_FILE_CANDIDATES
    if isfile(style_file)
        include(style_file)
        break
    end
end

using GLMakie

if !isdefined(@__MODULE__, :STYLE_PRIMARY)
    const STYLE_PRIMARY = :dodgerblue4
    const STYLE_SECONDARY = :darkorange2
    const STYLE_ACCENT = :seagreen4
    const STYLE_HIGHLIGHT = :firebrick3
    const STYLE_VIOLET = :mediumpurple4
    const STYLE_ZERO = :gray40
    const STYLE_SEQUENTIAL_BLUE = :viridis
    const STYLE_DIVERGING = :balance
end
if !isdefined(@__MODULE__, :figure_title!)
    function figure_title!(fig::Figure, title::AbstractString; subtitle::AbstractString="")
        Label(fig[0, :], isempty(subtitle) ? title : string(title, "\n", subtitle);
              fontsize=24, font=:bold, tellwidth=false)
        return nothing
    end
end
if !isdefined(@__MODULE__, :save_figure)
    save_figure(path::AbstractString, fig::Figure) = (save(path, fig); nothing)
end

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "sim.toml")

struct SimParams
    K::Int
    alpha::Float64
    beta::Float64
    kappa::Float64
    d0::Float64
    d1::Float64
    omega0::Float64
    omega1::Float64
    t0::Float64
    dt::Float64
    ntrajectories::Int
    requested_threads::Int
    seed::Int
    production_snapshots_per_decorrelation::Int
    q0::Float64
    p0::Float64
    perturb_site::Int
    perturb_q::Float64
    perturb_p::Float64
    reference_hdf5::String
    fallback_t_total::Float64
    fallback_t_decorrelation::Float64
    fallback_save_dt::Float64
    fallback_saved_intervals::Int
    pilot_t1::Float64
    pilot_save_dt::Float64
    pilot_burnin_fraction::Float64
    pilot_correlation_stride::Int
    pilot_max_decorrelation_time::Float64
    decorrelation_threshold::Float64
    burnin_fraction::Float64
    histogram_bins::Int
    correlation_stride::Int
    max_correlation_lags::Int
    cross_offsets::Vector{Int}
    bivariate_offsets::Vector{Int}
    max_pdf_samples::Int
    figure_width::Int
    figure_height::Int
    dynamics_figure_width::Int
    dynamics_figure_height::Int
    dynamics_window_decorrelation_times::Float64
    dynamics_trajectory::Int
    dynamics_max_frames::Int
    dynamics_trace_sites::Vector{Int}
    output_hdf5::String
    output_png::String
    output_dynamics_png::String
end

struct L96Reference
    t_total::Float64
    t_decorrelation::Float64
    save_dt::Float64
    saved_intervals::Int
    uncorrelated_count::Float64
    snapshots_per_decorrelation::Float64
    source::String
end

struct ProductionSchedule
    t1::Float64
    dt::Float64
    save_dt::Float64
    save_every::Int
    nsteps::Int
    nsaved::Int
    uncorrelated_count::Float64
    snapshots_per_decorrelation::Float64
end

struct MarginalPdf
    centers::Vector{Float64}
    density::Vector{Float64}
end

struct PairPdf
    label::String
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct CorrelationResult
    lags::Vector{Float64}
    acf_q::Vector{Float64}
    acf_p::Vector{Float64}
    cross_qp::Vector{Float64}
    cross_pq::Vector{Float64}
    cross_offsets::Vector{Int}
    spatial_q::Matrix{Float64}
    t_decorrelation::Float64
    mean_q::Float64
    mean_p::Float64
    var_q::Float64
    var_p::Float64
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? path : normpath(joinpath(base_dir, path))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    sim = raw["simulation"]
    ic = raw["initial_condition"]
    ref = raw["reference_l96"]
    cal = raw["calibration"]
    stats = raw["statistics"]
    fig = raw["figure"]
    out = raw["output"]

    params = SimParams(
        Int(sim["K"]),
        Float64(sim["alpha"]),
        Float64(sim["beta"]),
        Float64(sim["kappa"]),
        Float64(sim["d0"]),
        Float64(sim["d1"]),
        Float64(sim["omega0"]),
        Float64(sim["omega1"]),
        Float64(sim["t0"]),
        Float64(sim["dt"]),
        Int(sim["ntrajectories"]),
        Int(sim["requested_threads"]),
        Int(sim["seed"]),
        Int(get(sim, "production_snapshots_per_decorrelation", 100)),
        Float64(ic["q0"]),
        Float64(ic["p0"]),
        Int(ic["perturb_site"]),
        Float64(ic["perturb_q"]),
        Float64(ic["perturb_p"]),
        String(ref["hdf5_file"]),
        Float64(ref["fallback_t_total"]),
        Float64(ref["fallback_t_decorrelation"]),
        Float64(ref["fallback_save_dt"]),
        Int(ref["fallback_saved_intervals"]),
        Float64(cal["pilot_t1"]),
        Float64(cal["pilot_save_dt"]),
        Float64(cal["pilot_burnin_fraction"]),
        Int(cal["pilot_correlation_stride"]),
        Float64(cal["pilot_max_decorrelation_time"]),
        Float64(cal["decorrelation_threshold"]),
        Float64(stats["burnin_fraction"]),
        Int(stats["histogram_bins"]),
        Int(stats["correlation_stride"]),
        Int(stats["max_correlation_lags"]),
        Int.(stats["cross_offsets"]),
        Int.(stats["bivariate_offsets"]),
        Int(stats["max_pdf_samples"]),
        Int(fig["width"]),
        Int(fig["height"]),
        Int(get(fig, "dynamics_width", fig["width"])),
        Int(get(fig, "dynamics_height", fig["height"])),
        Float64(get(fig, "dynamics_window_decorrelation_times", 20.0)),
        Int(get(fig, "dynamics_trajectory", 1)),
        Int(get(fig, "dynamics_max_frames", 900)),
        Int.(get(fig, "dynamics_trace_sites", [1, max(1, Int(sim["K"]) ÷ 4), max(1, Int(sim["K"]) ÷ 2), Int(sim["K"])])),
        String(out["hdf5_file"]),
        String(out["figure_png"]),
        String(get(out, "dynamics_figure_png", "outputs/complex_amplitude_chain_dynamics.png")),
    )

    require_condition(params.K >= 4, "K must be at least 4.")
    require_condition(params.alpha > 0.0, "alpha must be positive.")
    require_condition(params.beta > 0.0, "beta must be positive.")
    require_condition(params.kappa >= 0.0, "kappa must be nonnegative.")
    require_condition(params.d0 > 0.0, "d0 must be positive.")
    require_condition(params.d1 >= 0.0, "d1 must be nonnegative.")
    require_condition(params.omega0 >= 0.0, "omega0 must be nonnegative.")
    require_condition(params.omega1 >= 0.0, "omega1 must be nonnegative.")
    require_condition(params.dt > 0.0, "dt must be positive.")
    require_condition(params.ntrajectories >= 1, "ntrajectories must be positive.")
    require_condition(params.requested_threads >= 0, "requested_threads must be nonnegative.")
    require_condition(params.production_snapshots_per_decorrelation >= 1,
        "production_snapshots_per_decorrelation must be positive.")
    require_condition(1 <= params.perturb_site <= params.K, "perturb_site must lie in 1:K.")
    require_condition(params.fallback_t_total > 0.0, "fallback_t_total must be positive.")
    require_condition(params.fallback_t_decorrelation > 0.0, "fallback_t_decorrelation must be positive.")
    require_condition(params.fallback_save_dt > 0.0, "fallback_save_dt must be positive.")
    require_condition(params.fallback_saved_intervals >= 1, "fallback_saved_intervals must be positive.")
    require_condition(params.pilot_t1 > params.t0, "pilot_t1 must exceed t0.")
    require_condition(params.pilot_save_dt >= params.dt, "pilot_save_dt must be at least dt.")
    require_condition(isapprox(params.pilot_save_dt / params.dt, round(params.pilot_save_dt / params.dt); atol=1e-12),
        "pilot_save_dt must be an integer multiple of dt.")
    require_condition(0.0 <= params.pilot_burnin_fraction < 1.0, "pilot_burnin_fraction must be in [0, 1).")
    require_condition(params.pilot_correlation_stride >= 1, "pilot_correlation_stride must be positive.")
    require_condition(params.pilot_max_decorrelation_time > 0.0, "pilot_max_decorrelation_time must be positive.")
    require_condition(0.0 < params.decorrelation_threshold < 1.0, "decorrelation_threshold must lie in (0, 1).")
    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.histogram_bins >= 32, "histogram_bins must be at least 32.")
    require_condition(params.correlation_stride >= 1, "correlation_stride must be positive.")
    require_condition(params.max_correlation_lags >= 8, "max_correlation_lags must be at least 8.")
    require_condition(!isempty(params.cross_offsets), "cross_offsets must not be empty.")
    require_condition(!isempty(params.bivariate_offsets), "bivariate_offsets must not be empty.")
    require_condition(all(1 .<= params.cross_offsets) && all(params.cross_offsets .<= params.K - 1),
        "cross_offsets must lie in 1:(K-1).")
    require_condition(all(1 .<= params.bivariate_offsets) && all(params.bivariate_offsets .<= params.K - 1),
        "bivariate_offsets must lie in 1:(K-1).")
    require_condition(params.max_pdf_samples >= 10_000, "max_pdf_samples should be at least 10000.")
    require_condition(params.figure_width >= 1600 && params.figure_height >= 1200, "Figure dimensions are too small.")
    require_condition(params.dynamics_figure_width >= 1600 && params.dynamics_figure_height >= 1000,
        "Dynamics figure dimensions are too small.")
    require_condition(params.dynamics_window_decorrelation_times > 0.0,
        "dynamics_window_decorrelation_times must be positive.")
    require_condition(1 <= params.dynamics_trajectory <= params.ntrajectories,
        "dynamics_trajectory must lie in 1:ntrajectories.")
    require_condition(params.dynamics_max_frames >= 16, "dynamics_max_frames must be at least 16.")
    require_condition(!isempty(params.dynamics_trace_sites), "dynamics_trace_sites must not be empty.")
    require_condition(all(1 .<= params.dynamics_trace_sites) && all(params.dynamics_trace_sites .<= params.K),
        "dynamics_trace_sites must lie in 1:K.")
    return params
end

function ensure_thread_count(params::SimParams)
    if params.requested_threads > 0
        actual = Threads.nthreads()
        require_condition(actual == params.requested_threads,
            @sprintf("Expected %d Julia threads, found %d. Run with `julia --threads %d ...` or set requested_threads = 0.",
                     params.requested_threads, actual, params.requested_threads))
    end
    return nothing
end

periodic_index(i::Int, K::Int) = mod1(i, K)

function load_l96_reference(params::SimParams, base_dir::AbstractString)
    ref_path = resolve_path(base_dir, params.reference_hdf5)
    if isfile(ref_path)
        return h5open(ref_path, "r") do file
            times = read(file["/trajectories/time"])
            t_total = Float64(times[end] - times[1])
            t_dec = Float64(read(file["/statistics/correlations/t_decorrelation"]))
            save_dt = Float64(read(file["/metadata/save_dt"]))
            saved_intervals = length(times) - 1
            return L96Reference(t_total, t_dec, save_dt, saved_intervals,
                t_total / t_dec, t_dec / save_dt, ref_path)
        end
    end

    t_total = params.fallback_t_total
    t_dec = params.fallback_t_decorrelation
    save_dt = params.fallback_save_dt
    return L96Reference(t_total, t_dec, save_dt, params.fallback_saved_intervals,
        t_total / t_dec, t_dec / save_dt, "fallback values from sim.toml")
end

function initial_state(params::SimParams)
    z = Array{Float64}(undef, params.K, 2)
    z[:, 1] .= params.q0
    z[:, 2] .= params.p0
    z[params.perturb_site, 1] += params.perturb_q
    z[params.perturb_site, 2] += params.perturb_p
    return z
end

function potential_value(z::Array{Float64, 2}, params::SimParams)
    K = params.K
    total = 0.0
    @inbounds for i in 1:K
        ip1 = periodic_index(i + 1, K)
        q = z[i, 1]
        p = z[i, 2]
        r2 = q * q + p * p
        dq = z[ip1, 1] - q
        dp = z[ip1, 2] - p
        total += 0.5 * params.alpha * r2 + 0.25 * params.beta * r2 * r2 +
                 0.5 * params.kappa * (dq * dq + dp * dp)
    end
    return total
end

function drift!(b::Array{Float64, 2}, z::Array{Float64, 2}, params::SimParams)
    K = params.K
    @inbounds for i in 1:K
        im1 = periodic_index(i - 1, K)
        ip1 = periodic_index(i + 1, K)
        q = z[i, 1]
        p = z[i, 2]
        r2 = q * q + p * p
        d = params.d0 + params.d1 * r2
        omega = params.omega0 + params.omega1 * r2

        gq = params.alpha * q + params.beta * r2 * q +
             params.kappa * (2.0 * q - z[im1, 1] - z[ip1, 1])
        gp = params.alpha * p + params.beta * r2 * p +
             params.kappa * (2.0 * p - z[im1, 2] - z[ip1, 2])

        b[i, 1] = -d * gq + omega * gp + 2.0 * params.d1 * q - 2.0 * params.omega1 * p
        b[i, 2] = -d * gp - omega * gq + 2.0 * params.d1 * p + 2.0 * params.omega1 * q
    end
    return nothing
end

function integrate_chain_ensemble(params::SimParams, t1::Float64, save_dt::Float64; label::AbstractString, dt::Float64=params.dt)
    ensure_thread_count(params)

    require_condition(dt > 0.0, "Integration dt must be positive.")
    nsteps_float = (t1 - params.t0) / dt
    require_condition(isapprox(nsteps_float, round(nsteps_float); atol=1e-8), "t1 - t0 must be an integer multiple of dt.")
    nsteps = round(Int, nsteps_float)
    save_every = round(Int, save_dt / dt)
    require_condition(save_every >= 1, "save_every must be positive.")
    require_condition(isapprox(save_every * dt, save_dt; rtol=1e-10, atol=1e-10),
        "save_dt must be an integer multiple of the integration dt.")
    require_condition(nsteps % save_every == 0, "nsteps must be divisible by save_every.")

    nsaved = nsteps ÷ save_every + 1
    times = collect(range(params.t0, step=save_dt, length=nsaved))
    states = Array{Float64}(undef, nsaved, params.K, 2, params.ntrajectories)
    z0 = initial_state(params)

    @printf("Integrating %s complex-amplitude chain: K=%d, trajectories=%d, dt=%.5g, save_dt=%.5g, T=%.4f, saved=%d, threads=%d\n",
        label, params.K, params.ntrajectories, dt, save_dt, t1 - params.t0, nsaved, Threads.nthreads())

    Threads.@threads for traj_idx in 1:params.ntrajectories
        rng = MersenneTwister(params.seed + 100_000 * (label == "pilot" ? 0 : 1) + traj_idx)
        z = copy(z0)
        b = similar(z)
        states[1, :, :, traj_idx] .= z
        save_idx = 2

        @inbounds for step in 1:nsteps
            drift!(b, z, params)
            for i in 1:params.K
                q = z[i, 1]
                p = z[i, 2]
                r2 = q * q + p * p
                d = params.d0 + params.d1 * r2
                noise_scale = sqrt(2.0 * d * dt)
                z[i, 1] = q + dt * b[i, 1] + noise_scale * randn(rng)
                z[i, 2] = p + dt * b[i, 2] + noise_scale * randn(rng)
            end

            if step % save_every == 0
                if !all(isfinite, z)
                    error(@sprintf("Non-finite state in %s trajectory %d at step %d. Reduce dt.", label, traj_idx, step))
                end
                states[save_idx, :, :, traj_idx] .= z
                save_idx += 1
            end
        end
    end

    return times, states
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function nearest_schedule(params::SimParams, t_dec::Float64, ref::L96Reference)
    dt = params.dt
    target_save_dt = t_dec / params.production_snapshots_per_decorrelation
    save_every = max(1, round(Int, target_save_dt / dt))
    save_dt = save_every * dt
    nsteps_raw = max(1, round(Int, ref.uncorrelated_count * t_dec / dt))
    nsteps = max(save_every, round(Int, nsteps_raw / save_every) * save_every)
    saved_intervals = nsteps
    t1 = params.t0 + nsteps * dt
    nsaved = nsteps ÷ save_every + 1
    return ProductionSchedule(t1, dt, save_dt, save_every, nsteps, nsaved,
        (t1 - params.t0) / t_dec, t_dec / save_dt)
end

function channel_mean_var(states::Array{Float64, 4}, start_idx::Int, channel::Int)
    nt, K, _, ntraj = size(states)
    total = 0.0
    count = 0
    @inbounds for traj in 1:ntraj, i in 1:K, t in start_idx:nt
        total += states[t, i, channel, traj]
        count += 1
    end
    mu = total / count
    ss = 0.0
    @inbounds for traj in 1:ntraj, i in 1:K, t in start_idx:nt
        delta = states[t, i, channel, traj] - mu
        ss += delta * delta
    end
    return mu, ss / count
end

function estimate_decorrelation_time(lags::Vector{Float64}, acf_q::Vector{Float64}, acf_p::Vector{Float64}, threshold::Float64)
    n = length(lags)
    envelope = [max(abs(acf_q[i]), abs(acf_p[i])) for i in 1:n]
    running_max = similar(envelope)
    running_max[end] = envelope[end]
    @inbounds for i in (n - 1):-1:1
        running_max[i] = max(envelope[i], running_max[i + 1])
    end
    for i in 2:n
        if running_max[i] <= threshold
            return lags[i]
        end
    end
    return lags[end]
end

function fft_corr_future_present(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int)
    n = length(x)
    nfft = 1 << ceil(Int, log2(2n - 1))
    xpad = zeros(Float64, nfft)
    ypad = zeros(Float64, nfft)
    xpad[1:n] .= x
    ypad[1:n] .= y
    c = real.(ifft(fft(xpad) .* conj.(fft(ypad))))
    return c[1:(max_lag + 1)]
end

function downsample_states(states::Array{Float64, 4}, start_idx::Int, stride::Int)
    indices = collect(start_idx:stride:size(states, 1))
    return Array(@view states[indices, :, :, :])
end

function compute_chain_correlations(data::Array{Float64, 4}, dt_corr::Float64, threshold::Float64,
                                    cross_offsets::Vector{Int}, max_lags::Int)
    nt, K, _, ntraj = size(data)
    max_lag = min(max_lags, nt - 1)
    require_condition(max_lag >= 1, "Correlation window is empty.")
    lags = collect(0:max_lag) .* dt_corr
    mean_q, var_q = channel_mean_var(data, 1, 1)
    mean_p, var_p = channel_mean_var(data, 1, 2)
    require_condition(var_q > 0.0 && var_p > 0.0, "Cannot normalize correlations with zero variance.")

    acf_q_sum = zeros(Float64, max_lag + 1)
    acf_p_sum = zeros(Float64, max_lag + 1)
    qp_sum = zeros(Float64, max_lag + 1)
    pq_sum = zeros(Float64, max_lag + 1)
    spatial_sum = zeros(Float64, max_lag + 1, length(cross_offsets))

    scratch_q = Vector{Float64}(undef, nt)
    scratch_p = Vector{Float64}(undef, nt)
    scratch_pair = Vector{Float64}(undef, nt)

    @inbounds for traj in 1:ntraj
        for i in 1:K
            for t in 1:nt
                scratch_q[t] = data[t, i, 1, traj] - mean_q
                scratch_p[t] = data[t, i, 2, traj] - mean_p
            end
            acf_q_sum .+= fft_corr_future_present(scratch_q, scratch_q, max_lag)
            acf_p_sum .+= fft_corr_future_present(scratch_p, scratch_p, max_lag)
            qp_sum .+= fft_corr_future_present(scratch_q, scratch_p, max_lag)
            pq_sum .+= fft_corr_future_present(scratch_p, scratch_q, max_lag)

            for (offset_idx, offset) in enumerate(cross_offsets)
                j = periodic_index(i + offset, K)
                for t in 1:nt
                    scratch_pair[t] = data[t, j, 1, traj] - mean_q
                end
                spatial_sum[:, offset_idx] .+= fft_corr_future_present(scratch_pair, scratch_q, max_lag)
            end
        end
    end

    nseries = K * ntraj
    denom_counts = Float64[nseries * (nt - lag) for lag in 0:max_lag]
    acf_q = acf_q_sum ./ (denom_counts .* var_q)
    acf_p = acf_p_sum ./ (denom_counts .* var_p)
    denom_qp = denom_counts .* sqrt(var_q * var_p)
    cross_qp = qp_sum ./ denom_qp
    cross_pq = pq_sum ./ denom_qp
    spatial_q = similar(spatial_sum)
    for offset_idx in eachindex(cross_offsets)
        spatial_q[:, offset_idx] .= spatial_sum[:, offset_idx] ./ (denom_counts .* var_q)
    end

    t_dec = estimate_decorrelation_time(lags, acf_q, acf_p, threshold)
    return CorrelationResult(lags, acf_q, acf_p, cross_qp, cross_pq, copy(cross_offsets),
        spatial_q, t_dec, mean_q, mean_p, var_q, var_p)
end

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function draw_scalar_samples(states::Array{Float64, 4}, start_idx::Int, max_samples::Int,
                             rng::AbstractRNG, selector::Function)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        site = (tmp % K) + 1
        traj = (tmp ÷ K) + 1
        t = start_idx + time_local
        values[sample_idx] = selector(states, t, site, traj)
    end
    return values
end

function compute_marginal_pdfs(states::Array{Float64, 4}, start_idx::Int, bins::Int, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed + 10_011)
    qs = draw_scalar_samples(states, start_idx, max_samples, rng, (s, t, i, tr) -> s[t, i, 1, tr])
    ps = draw_scalar_samples(states, start_idx, max_samples, rng, (s, t, i, tr) -> s[t, i, 2, tr])
    amps = sqrt.(qs .* qs .+ ps .* ps)

    qk = kde(qs; npoints=bins, boundary=kde_range(qs))
    pk = kde(ps; npoints=bins, boundary=kde_range(ps))
    rk = kde(amps; npoints=bins, boundary=(0.0, kde_range(amps)[2]))
    return MarginalPdf(collect(qk.x), collect(qk.density)),
           MarginalPdf(collect(pk.x), collect(pk.density)),
           MarginalPdf(collect(rk.x), collect(rk.density))
end

function draw_pair_samples(states::Array{Float64, 4}, start_idx::Int, max_samples::Int,
                           rng::AbstractRNG, selector::Function)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    x = Vector{Float64}(undef, nsamples)
    y = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        site = (tmp % K) + 1
        traj = (tmp ÷ K) + 1
        t = start_idx + time_local
        x[sample_idx], y[sample_idx] = selector(states, t, site, traj)
    end
    return x, y
end

function compute_pair_pdfs(states::Array{Float64, 4}, start_idx::Int, offsets::Vector{Int},
                           bins::Int, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed + 21_019)
    pair_pdfs = PairPdf[]

    q, p = draw_pair_samples(states, start_idx, max_samples, rng,
        (s, t, i, tr) -> (s[t, i, 1, tr], s[t, i, 2, tr]))
    kde_qp = kde((q, p); npoints=(bins, bins), boundary=(kde_range(q), kde_range(p)))
    push!(pair_pdfs, PairPdf("(q_i, p_i)", collect(kde_qp.x), collect(kde_qp.y), Array(kde_qp.density)))

    for offset in offsets
        x, y = draw_pair_samples(states, start_idx, max_samples, rng,
            (s, t, i, tr) -> (s[t, i, 1, tr], s[t, periodic_index(i + offset, size(s, 2)), 1, tr]))
        kde_pair = kde((x, y); npoints=(bins, bins), boundary=(kde_range(x), kde_range(y)))
        push!(pair_pdfs, PairPdf(@sprintf("(q_i, q_{i+%d})", offset),
            collect(kde_pair.x), collect(kde_pair.y), Array(kde_pair.density)))
    end

    return pair_pdfs
end

function compute_spatial_covariance(states::Array{Float64, 4}, start_idx::Int)
    nt, K, _, ntraj = size(states)
    D = 2K
    mean_flat = zeros(Float64, D)
    count = (nt - start_idx + 1) * ntraj
    @inbounds for traj in 1:ntraj, t in start_idx:nt, i in 1:K
        mean_flat[2i - 1] += states[t, i, 1, traj]
        mean_flat[2i] += states[t, i, 2, traj]
    end
    mean_flat ./= count

    cov = zeros(Float64, D, D)
    x = Vector{Float64}(undef, D)
    @inbounds for traj in 1:ntraj, t in start_idx:nt
        for i in 1:K
            x[2i - 1] = states[t, i, 1, traj] - mean_flat[2i - 1]
            x[2i] = states[t, i, 2, traj] - mean_flat[2i]
        end
        BLAS.syr!('U', 1.0, x, cov)
    end
    cov ./= count
    cov = Symmetric(cov, :U) |> Matrix
    return cov
end

function mobility_summary(states::Array{Float64, 4}, start_idx::Int, params::SimParams)
    nt, K, _, ntraj = size(states)
    d_total = 0.0
    w_total = 0.0
    r2_total = 0.0
    count = 0
    @inbounds for traj in 1:ntraj, t in start_idx:nt, i in 1:K
        q = states[t, i, 1, traj]
        p = states[t, i, 2, traj]
        r2 = q * q + p * p
        d_total += params.d0 + params.d1 * r2
        w_total += params.omega0 + params.omega1 * r2
        r2_total += r2
        count += 1
    end
    return d_total / count, w_total / count, r2_total / count
end

function save_hdf5(path::AbstractString, params::SimParams, ref::L96Reference, schedule::ProductionSchedule,
                   pilot_corr::CorrelationResult, times::Vector{Float64}, states::Array{Float64, 4},
                   q_pdf::MarginalPdf, p_pdf::MarginalPdf, r_pdf::MarginalPdf,
                   pair_pdfs::Vector{PairPdf}, corr::CorrelationResult, cov::Matrix{Float64},
                   mean_d::Float64, mean_omega::Float64, mean_r2::Float64)
    nt, K, _, ntraj = size(states)
    states_flat = Array{Float64}(undef, nt, 2K, ntraj)
    @inbounds for traj in 1:ntraj, t in 1:nt, i in 1:K
        states_flat[t, 2i - 1, traj] = states[t, i, 1, traj]
        states_flat[t, 2i, traj] = states[t, i, 2, traj]
    end

    h5open(path, "w") do file
        file["/trajectories/time"] = times
        file["/trajectories/states"] = states
        file["/trajectories/states_flat"] = states_flat
        file["/trajectories/channel_names"] = ["q", "p"]

        file["/statistics/pdf/q_centers"] = q_pdf.centers
        file["/statistics/pdf/q_density"] = q_pdf.density
        file["/statistics/pdf/p_centers"] = p_pdf.centers
        file["/statistics/pdf/p_density"] = p_pdf.density
        file["/statistics/pdf/amplitude_centers"] = r_pdf.centers
        file["/statistics/pdf/amplitude_density"] = r_pdf.density
        file["/statistics/pdf/bivariate_labels"] = [pdf.label for pdf in pair_pdfs]
        for (idx, pdf) in enumerate(pair_pdfs)
            base = @sprintf("/statistics/pdf/bivariate/pair_%02d", idx)
            file[string(base, "/label")] = pdf.label
            file[string(base, "/x_grid")] = pdf.x_grid
            file[string(base, "/y_grid")] = pdf.y_grid
            file[string(base, "/density")] = pdf.density
        end

        file["/statistics/correlations/lags"] = corr.lags
        file["/statistics/correlations/acf_q"] = corr.acf_q
        file["/statistics/correlations/acf_p"] = corr.acf_p
        file["/statistics/correlations/cross_qp"] = corr.cross_qp
        file["/statistics/correlations/cross_pq"] = corr.cross_pq
        file["/statistics/correlations/cross_offsets"] = corr.cross_offsets
        file["/statistics/correlations/spatial_q"] = corr.spatial_q
        file["/statistics/correlations/t_decorrelation"] = pilot_corr.t_decorrelation
        file["/statistics/correlations/t_decorrelation_final_check"] = corr.t_decorrelation
        file["/statistics/correlations/global_mean_q"] = corr.mean_q
        file["/statistics/correlations/global_mean_p"] = corr.mean_p
        file["/statistics/correlations/global_variance_q"] = corr.var_q
        file["/statistics/correlations/global_variance_p"] = corr.var_p
        file["/statistics/covariance_flat"] = cov

        file["/mobility/mean_d"] = mean_d
        file["/mobility/mean_omega"] = mean_omega
        file["/mobility/mean_r2"] = mean_r2

        file["/reference_l96/source"] = ref.source
        file["/reference_l96/t_total"] = ref.t_total
        file["/reference_l96/t_decorrelation"] = ref.t_decorrelation
        file["/reference_l96/save_dt"] = ref.save_dt
        file["/reference_l96/saved_intervals"] = ref.saved_intervals
        file["/reference_l96/uncorrelated_count"] = ref.uncorrelated_count
        file["/reference_l96/snapshots_per_decorrelation"] = ref.snapshots_per_decorrelation

        file["/metadata/model_name"] = "stochastic_complex_amplitude_chain_state_dependent_mobility"
        file["/metadata/K"] = params.K
        file["/metadata/state_dimension"] = 2params.K
        file["/metadata/alpha"] = params.alpha
        file["/metadata/beta"] = params.beta
        file["/metadata/kappa"] = params.kappa
        file["/metadata/d0"] = params.d0
        file["/metadata/d1"] = params.d1
        file["/metadata/omega0"] = params.omega0
        file["/metadata/omega1"] = params.omega1
        file["/metadata/pilot_dt"] = params.dt
        file["/metadata/dt"] = schedule.dt
        file["/metadata/save_dt"] = schedule.save_dt
        file["/metadata/t0"] = params.t0
        file["/metadata/t1"] = schedule.t1
        file["/metadata/nsteps"] = schedule.nsteps
        file["/metadata/save_every"] = schedule.save_every
        file["/metadata/nsaved"] = schedule.nsaved
        file["/metadata/ntrajectories"] = params.ntrajectories
        file["/metadata/burnin_fraction"] = params.burnin_fraction
        file["/metadata/production_uncorrelated_count"] = schedule.uncorrelated_count
        file["/metadata/production_snapshots_per_decorrelation"] = schedule.snapshots_per_decorrelation
        file["/metadata/sde_noise_prefactor"] = "sqrt(2*d_i(z)) per q/p site component"
    end
    return nothing
end

function truncate_series(xs::Vector{Float64}, ys::Vector{Float64}, xmax::Float64)
    last_idx = clamp(searchsortedlast(xs, xmax), 2, length(xs))
    return xs[1:last_idx], ys[1:last_idx]
end

function heatmap_panel!(parent, x, y, z; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString,
                        colorbar_label::AbstractString="", colormap=STYLE_SEQUENTIAL_BLUE)
    layout = GridLayout(parent)
    ax = Axis(layout[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)
    hm = heatmap!(ax, x, y, z; colormap=colormap)
    Colorbar(layout[1, 2], hm; label=colorbar_label)
    return ax
end

function render_summary_figure(path::AbstractString, params::SimParams, ref::L96Reference,
                               schedule::ProductionSchedule, q_pdf::MarginalPdf, p_pdf::MarginalPdf,
                               r_pdf::MarginalPdf, pair_pdfs::Vector{PairPdf},
                               corr::CorrelationResult, cov::Matrix{Float64},
                               mean_d::Float64, mean_omega::Float64, mean_r2::Float64,
                               sample_state::Matrix{Float64})
    fig = Figure(; size=(params.figure_width, params.figure_height))
    subtitle = @sprintf("K=%d dim=%d  dt=%.3g  save_dt=%.4f  T=%.2f  t_D=%.3f  saves/t_D=%.1f",
        params.K, 2params.K, schedule.dt, schedule.save_dt, schedule.t1 - params.t0,
        corr.t_decorrelation, schedule.snapshots_per_decorrelation)
    figure_title!(fig, "Stochastic complex-amplitude chain - observational summary"; subtitle=subtitle)

    ax_pdf = Axis(fig[1, 1]; title="Translation-averaged marginals", xlabel="value", ylabel="density")
    lines!(ax_pdf, q_pdf.centers, q_pdf.density; color=STYLE_PRIMARY, linewidth=3, label="q")
    lines!(ax_pdf, p_pdf.centers, p_pdf.density; color=STYLE_SECONDARY, linewidth=3, label="p")
    axislegend(ax_pdf; position=:rt)

    ax_amp = Axis(fig[1, 2]; title="Amplitude PDF", xlabel="r_i", ylabel="density")
    lines!(ax_amp, r_pdf.centers, r_pdf.density; color=STYLE_ACCENT, linewidth=3)

    corr_t_max = min(corr.lags[end], 10.0)
    lags_q, acf_q = truncate_series(corr.lags, corr.acf_q, corr_t_max)
    _, acf_p = truncate_series(corr.lags, corr.acf_p, corr_t_max)
    ax_acf = Axis(fig[1, 3]; title="Coordinate autocorrelations", xlabel="lag tau", ylabel="C(tau)")
    hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_acf, lags_q, acf_q; color=STYLE_PRIMARY, linewidth=3, label="q")
    lines!(ax_acf, lags_q, acf_p; color=STYLE_SECONDARY, linewidth=3, label="p")
    xlims!(ax_acf, 0.0, corr_t_max)
    axislegend(ax_acf; position=:rt)

    _, qp = truncate_series(corr.lags, corr.cross_qp, corr_t_max)
    _, pq = truncate_series(corr.lags, corr.cross_pq, corr_t_max)
    ax_cross = Axis(fig[2, 1]; title="Local q/p cross-correlations", xlabel="lag tau", ylabel="C(tau)")
    hlines!(ax_cross, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_cross, lags_q, qp; color=STYLE_HIGHLIGHT, linewidth=3, label="C_qp")
    lines!(ax_cross, lags_q, pq; color=STYLE_VIOLET, linewidth=3, label="C_pq")
    xlims!(ax_cross, 0.0, corr_t_max)
    axislegend(ax_cross; position=:rt)

    ax_spatial = Axis(fig[2, 2]; title="Shifted q correlations", xlabel="lag tau", ylabel="C_r(tau)")
    hlines!(ax_spatial, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for (idx, offset) in enumerate(corr.cross_offsets)
        _, vals = truncate_series(corr.lags, corr.spatial_q[:, idx], corr_t_max)
        lines!(ax_spatial, lags_q, vals; linewidth=2, label=@sprintf("r=%d", offset))
    end
    xlims!(ax_spatial, 0.0, corr_t_max)
    axislegend(ax_spatial; position=:rt, nbanks=2)

    max_pair_panels = min(length(pair_pdfs), 3)
    pair_positions = [(2, 3), (3, 1), (3, 2)]
    for idx in 1:max_pair_panels
        row, col = pair_positions[idx]
        pdf = pair_pdfs[idx]
        heatmap_panel!(fig[row, col], pdf.x_grid, pdf.y_grid, pdf.density;
            title="Bivariate PDF " * pdf.label, xlabel="first", ylabel="second",
            colorbar_label="density", colormap=STYLE_SEQUENTIAL_BLUE)
    end

    heatmap_panel!(fig[3, 3], collect(1:size(cov, 1)), collect(1:size(cov, 2)), cov;
        title="Empirical covariance of flattened state", xlabel="component", ylabel="component",
        colorbar_label="cov", colormap=STYLE_DIVERGING)

    ax_sample = heatmap_panel!(fig[4, 1], collect(1:params.K), collect(1:2), sample_state;
        title="Final sample state, trajectory 1", xlabel="site", ylabel="channel",
        colorbar_label="state", colormap=STYLE_DIVERGING)
    ax_sample.yticks = ([1, 2], ["q", "p"])

    ax_meta = Axis(fig[4, 2:3]; title="Sampling budget check")
    hidedecorations!(ax_meta)
    hidespines!(ax_meta)
    lines = [
        @sprintf("L96 source: %s", ref.source),
        @sprintf("L96: T=%.4f, t_D=%.4f, save_dt=%.4f, N=T/t_D=%.4f, saved intervals=%d",
            ref.t_total, ref.t_decorrelation, ref.save_dt, ref.uncorrelated_count, ref.saved_intervals),
        @sprintf("Complex chain: T=%.4f, t_D=%.4f, save_dt=%.4f, N=T/t_D=%.4f, saved intervals=%d",
            schedule.t1 - params.t0, corr.t_decorrelation, schedule.save_dt,
            schedule.uncorrelated_count, schedule.nsaved - 1),
        @sprintf("Snapshots per decorrelation: L96 %.4f, complex chain %.4f",
            ref.snapshots_per_decorrelation, schedule.snapshots_per_decorrelation),
        @sprintf("Mean r^2=%.4f, mean d=%.4f, mean omega=%.4f", mean_r2, mean_d, mean_omega),
    ]
    text!(ax_meta, 0.02, 0.92; text=join(lines, "\n"), align=(:left, :top),
        space=:relative, fontsize=18)

    colsize!(fig.layout, 1, Relative(1 / 3))
    colsize!(fig.layout, 2, Relative(1 / 3))
    colsize!(fig.layout, 3, Relative(1 / 3))
    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
    save_figure(path, fig)
    return nothing
end

function dynamics_window_indices(times::Vector{Float64}, start_idx::Int, t_dec::Float64,
                                window_decorrelation_times::Float64, max_frames::Int)
    t_start = times[start_idx]
    t_stop = min(times[end], t_start + window_decorrelation_times * t_dec)
    stop_idx = searchsortedlast(times, t_stop)
    stop_idx = max(stop_idx, start_idx + 1)
    raw = collect(start_idx:stop_idx)
    if length(raw) <= max_frames
        return raw
    end
    stride = ceil(Int, length(raw) / max_frames)
    return collect(start_idx:stride:stop_idx)
end

function render_dynamics_figure(path::AbstractString, params::SimParams, schedule::ProductionSchedule,
                                times::Vector{Float64}, states::Array{Float64, 4},
                                start_idx::Int, t_dec::Float64)
    indices = dynamics_window_indices(times, start_idx, t_dec,
        params.dynamics_window_decorrelation_times, params.dynamics_max_frames)
    traj = params.dynamics_trajectory
    t_axis = (times[indices] .- times[first(indices)]) ./ t_dec
    sites = collect(1:params.K)

    q = Array(@view states[indices, :, 1, traj])
    p = Array(@view states[indices, :, 2, traj])
    amp = sqrt.(q .* q .+ p .* p)
    qp_limit = maximum(abs, vcat(vec(q), vec(p)))
    qp_clims = (-qp_limit, qp_limit)

    fig = Figure(; size=(params.dynamics_figure_width, params.dynamics_figure_height))
    subtitle = @sprintf("trajectory %d   window %.2f t_D   t_D=%.3f   save_dt=%.4f   frames=%d   sites=%d",
        traj, t_axis[end] - t_axis[1], t_dec, schedule.save_dt, length(indices), params.K)
    figure_title!(fig, "Complex-amplitude chain dynamics"; subtitle=subtitle)

    q_layout = GridLayout(fig[1, 1])
    ax_q = Axis(q_layout[1, 1]; title="q_i(t)", xlabel="time / t_D", ylabel="site i")
    hm_q = heatmap!(ax_q, t_axis, sites, q; colormap=STYLE_DIVERGING, colorrange=qp_clims)
    Colorbar(q_layout[1, 2], hm_q; label="q")

    p_layout = GridLayout(fig[1, 2])
    ax_p = Axis(p_layout[1, 1]; title="p_i(t)", xlabel="time / t_D", ylabel="site i")
    hm_p = heatmap!(ax_p, t_axis, sites, p; colormap=STYLE_DIVERGING, colorrange=qp_clims)
    Colorbar(p_layout[1, 2], hm_p; label="p")

    amp_layout = GridLayout(fig[2, 1])
    ax_amp = Axis(amp_layout[1, 1]; title="amplitude r_i(t)", xlabel="time / t_D", ylabel="site i")
    hm_amp = heatmap!(ax_amp, t_axis, sites, amp; colormap=STYLE_SEQUENTIAL_BLUE)
    Colorbar(amp_layout[1, 2], hm_amp; label="r")

    ax_phase = Axis(fig[2, 2]; title="complex-amplitude traces", xlabel="q_i", ylabel="p_i")
    hlines!(ax_phase, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    vlines!(ax_phase, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for site in params.dynamics_trace_sites
        lines!(ax_phase, q[:, site], p[:, site]; linewidth=1.2, label=@sprintf("i=%d", site))
        scatter!(ax_phase, [q[1, site]], [p[1, site]]; markersize=8)
        scatter!(ax_phase, [q[end, site]], [p[end, site]]; markersize=10, marker=:rect)
    end
    axislegend(ax_phase; position=:rt, nbanks=2)

    Label(fig[3, 1:2],
        "Hovmoller panels show one post-burn-in trajectory as two lattice fields, q_i(t) and p_i(t). " *
        "The bottom-left panel shows the local amplitude r_i(t)=sqrt(q_i(t)^2+p_i(t)^2); " *
        "the bottom-right panel shows selected site trajectories in the complex plane.",
        tellwidth=false, fontsize=18)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))
    save_figure(path, fig)
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    ensure_thread_count(params)
    base_dir = dirname(abspath(param_file))
    output_hdf5 = resolve_path(base_dir, params.output_hdf5)
    output_png = resolve_path(base_dir, params.output_png)
    output_dynamics_png = resolve_path(base_dir, params.output_dynamics_png)
    ensure_parent_dir(output_hdf5)
    ensure_parent_dir(output_png)
    ensure_parent_dir(output_dynamics_png)

    ref = load_l96_reference(params, base_dir)
    @printf("L96 reference from %s\n", ref.source)
    @printf("L96 budget: T=%.6f, tD=%.6f, save_dt=%.6f, N=%.6f, saves/tD=%.6f, saved intervals=%d\n",
        ref.t_total, ref.t_decorrelation, ref.save_dt, ref.uncorrelated_count,
        ref.snapshots_per_decorrelation, ref.saved_intervals)

    pilot_times, pilot_states = integrate_chain_ensemble(params, params.pilot_t1, params.pilot_save_dt; label="pilot")
    pilot_start = burnin_start_index(length(pilot_times), params.pilot_burnin_fraction)
    pilot_data = downsample_states(pilot_states, pilot_start, params.pilot_correlation_stride)
    pilot_dt_corr = params.pilot_save_dt * params.pilot_correlation_stride
    pilot_max_lags = min(size(pilot_data, 1) - 1, floor(Int, params.pilot_max_decorrelation_time / pilot_dt_corr))
    pilot_corr = compute_chain_correlations(pilot_data, pilot_dt_corr, params.decorrelation_threshold,
        params.cross_offsets, pilot_max_lags)
    @printf("Pilot decorrelation estimate: tD = %.6f\n", pilot_corr.t_decorrelation)

    schedule = nearest_schedule(params, pilot_corr.t_decorrelation, ref)
    @printf("Production schedule: T=%.6f, dt=%.8f, save_dt=%.6f, save_every=%d, nsteps=%d, nsaved=%d\n",
        schedule.t1 - params.t0, schedule.dt, schedule.save_dt, schedule.save_every, schedule.nsteps, schedule.nsaved)
    @printf("Production budget check: N=%.6f (target %.6f), saves/tD=%.6f (target %d, L96 %.6f)\n",
        schedule.uncorrelated_count, ref.uncorrelated_count,
        schedule.snapshots_per_decorrelation, params.production_snapshots_per_decorrelation,
        ref.snapshots_per_decorrelation)

    times, states = integrate_chain_ensemble(params, schedule.t1, schedule.save_dt; label="production", dt=schedule.dt)
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    @printf("Stationary diagnostic window starts at saved index %d / %d\n", start_idx, length(times))

    q_pdf, p_pdf, r_pdf = compute_marginal_pdfs(states, start_idx, params.histogram_bins,
        params.max_pdf_samples, params.seed)
    pair_pdfs = compute_pair_pdfs(states, start_idx, params.bivariate_offsets,
        params.histogram_bins, params.max_pdf_samples, params.seed)

    corr_data = downsample_states(states, start_idx, params.correlation_stride)
    dt_corr = schedule.save_dt * params.correlation_stride
    corr = compute_chain_correlations(corr_data, dt_corr, params.decorrelation_threshold,
        params.cross_offsets, params.max_correlation_lags)
    @printf("Final-window decorrelation check from downsampled saved trajectory: tD = %.6f\n", corr.t_decorrelation)

    cov = compute_spatial_covariance(states, start_idx)
    mean_d, mean_omega, mean_r2 = mobility_summary(states, start_idx, params)

    @printf("Writing HDF5 output to %s\n", output_hdf5)
    save_hdf5(output_hdf5, params, ref, schedule, pilot_corr, times, states,
        q_pdf, p_pdf, r_pdf, pair_pdfs, corr, cov, mean_d, mean_omega, mean_r2)

    sample_state = Array(states[end, :, :, 1])
    @printf("Rendering summary figure to %s\n", output_png)
    render_summary_figure(output_png, params, ref, schedule, q_pdf, p_pdf, r_pdf,
        pair_pdfs, corr, cov, mean_d, mean_omega, mean_r2, sample_state)

    @printf("Rendering dynamics figure to %s\n", output_dynamics_png)
    render_dynamics_figure(output_dynamics_png, params, schedule, times, states,
        start_idx, pilot_corr.t_decorrelation)

    @printf("Completed complex-amplitude simulation. Pilot tD = %.6f, production T = %.6f, save_dt = %.6f\n",
        pilot_corr.t_decorrelation, schedule.t1 - params.t0, schedule.save_dt)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
