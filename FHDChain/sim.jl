#!/usr/bin/env julia

# Simulates the strong state-dependence periodic fluctuating-hydrodynamic chain
# described in system.txt. This script deliberately stops at data generation:
#   1. run a pilot integration and estimate the decorrelation time t_D;
#   2. match the previous ComplexAmplitudeChain uncorrelated-sample budget;
#   3. save 100 snapshots per decorrelation time up to the integer dt grid;
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

function display_is_usable(display::AbstractString)
    xdpyinfo = Sys.which("xdpyinfo")
    xdpyinfo === nothing && return true
    try
        run(pipeline(`$xdpyinfo -display $display`, stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

const XVFB_PROCESS = Ref{Any}(nothing)

function start_verified_xvfb!()
    xvfb = Sys.which("Xvfb")
    xvfb === nothing && return nothing
    for display_id in 101:140
        display = ":" * string(display_id)
        lock_path = "/tmp/.X" * string(display_id) * "-lock"
        isfile(lock_path) && continue
        proc = run(pipeline(`$xvfb $display -screen 0 1920x1200x24 -nolisten tcp`,
            stdout=devnull, stderr=devnull); wait=false)
        sleep(1.5)
        if display_is_usable(display)
            XVFB_PROCESS[] = proc
            ENV["DISPLAY"] = display
            ENV["STATEDEP_XVFB_DISPLAY"] = display
            return nothing
        end
    end
    return nothing
end

ENV["STATEDEP_XVFB_DISPLAY"] = get(ENV, "STATEDEP_XVFB_DISPLAY", ":101")
if haskey(ENV, "DISPLAY") && !display_is_usable(ENV["DISPLAY"])
    delete!(ENV, "DISPLAY")
end
if !haskey(ENV, "DISPLAY") || isempty(ENV["DISPLAY"])
    start_verified_xvfb!()
end

using FFTW
using HDF5
using KernelDensity
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

BLAS.set_num_threads(1)

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
    N::Int
    L::Float64
    rho0::Float64
    cs::Float64
    theta::Float64
    eta0::Float64
    zeta::Float64
    t0::Float64
    dt::Float64
    ntrajectories::Int
    requested_threads::Int
    seed::Int
    production_snapshots_per_decorrelation::Int
    max_substep_depth::Int
    density_floor::Float64
    density_log_step_limit::Float64
    velocity_density_floor::Float64
    rho_perturb_amplitude::Float64
    momentum_perturb_amplitude::Float64
    reference_hdf5::String
    fallback_uncorrelated_count::Float64
    fallback_snapshots_per_decorrelation::Float64
    fallback_ntrajectories::Int
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
    trajectories_figure_width::Int
    trajectories_figure_height::Int
    dynamics_window_decorrelation_times::Float64
    dynamics_trajectory::Int
    dynamics_max_frames::Int
    dynamics_trace_sites::Vector{Int}
    output_hdf5::String
    output_png::String
    output_dynamics_png::String
    output_trajectories_png::String
end

struct PreviousReference
    uncorrelated_count::Float64
    snapshots_per_decorrelation::Float64
    ntrajectories::Int
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
    acf_rho::Vector{Float64}
    acf_m::Vector{Float64}
    cross_rhom::Vector{Float64}
    cross_mrho::Vector{Float64}
    cross_offsets::Vector{Int}
    spatial_rho::Matrix{Float64}
    t_decorrelation::Float64
    mean_rho::Float64
    mean_m::Float64
    var_rho::Float64
    var_m::Float64
end

struct ConservationStats
    mass::Matrix{Float64}
    momentum::Matrix{Float64}
    min_rho::Matrix{Float64}
    mass_max_abs_drift::Float64
    momentum_max_abs_drift::Float64
    min_density::Float64
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
    ref = raw["reference_previous"]
    cal = raw["calibration"]
    stats = raw["statistics"]
    fig = raw["figure"]
    out = raw["output"]

    params = SimParams(
        Int(sim["N"]),
        Float64(sim["L"]),
        Float64(sim["rho0"]),
        Float64(sim["sound_speed"]),
        Float64(sim["Theta"]),
        Float64(sim["eta0"]),
        Float64(sim["zeta"]),
        Float64(sim["t0"]),
        Float64(sim["dt"]),
        Int(sim["ntrajectories"]),
        Int(sim["requested_threads"]),
        Int(sim["seed"]),
        Int(get(sim, "production_snapshots_per_decorrelation", 100)),
        Int(get(sim, "max_substep_depth", 12)),
        Float64(get(sim, "density_floor", 1e-8)),
        Float64(get(sim, "density_log_step_limit", 0.25)),
        Float64(get(sim, "velocity_density_floor", 0.0)),
        Float64(ic["rho_perturb_amplitude"]),
        Float64(ic["momentum_perturb_amplitude"]),
        String(ref["hdf5_file"]),
        Float64(ref["fallback_uncorrelated_count"]),
        Float64(ref["fallback_snapshots_per_decorrelation"]),
        Int(ref["fallback_ntrajectories"]),
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
        Int(get(fig, "trajectories_width", fig["width"])),
        Int(get(fig, "trajectories_height", fig["height"])),
        Float64(get(fig, "dynamics_window_decorrelation_times", 20.0)),
        Int(get(fig, "dynamics_trajectory", 1)),
        Int(get(fig, "dynamics_max_frames", 900)),
        Int.(get(fig, "dynamics_trace_sites", [1, max(1, Int(sim["N"]) ÷ 3), max(1, 2Int(sim["N"]) ÷ 3), Int(sim["N"])])),
        String(out["hdf5_file"]),
        String(out["figure_png"]),
        String(get(out, "dynamics_figure_png", "outputs/fhd_chain_dynamics.png")),
        String(get(out, "trajectories_figure_png", "outputs/fhd_chain_trajectories.png")),
    )

    require_condition(params.N >= 4, "N must be at least 4.")
    require_condition(params.L > 0.0, "L must be positive.")
    require_condition(params.rho0 > 0.0, "rho0 must be positive.")
    require_condition(params.cs > 0.0, "sound_speed must be positive.")
    require_condition(params.theta > 0.0, "Theta must be positive.")
    require_condition(params.eta0 > 0.0, "eta0 must be positive.")
    require_condition(params.zeta > 0.0, "zeta must be positive.")
    require_condition(params.dt > 0.0, "dt must be positive.")
    require_condition(params.ntrajectories >= 1, "ntrajectories must be positive.")
    require_condition(params.requested_threads >= 0, "requested_threads must be nonnegative.")
    require_condition(params.production_snapshots_per_decorrelation >= 1,
        "production_snapshots_per_decorrelation must be positive.")
    require_condition(params.max_substep_depth >= 0, "max_substep_depth must be nonnegative.")
    require_condition(params.density_floor >= 0.0, "density_floor must be nonnegative.")
    require_condition(params.density_log_step_limit > 0.0, "density_log_step_limit must be positive.")
    require_condition(params.velocity_density_floor >= 0.0, "velocity_density_floor must be nonnegative.")
    require_condition(params.rho_perturb_amplitude >= 0.0, "rho perturbation amplitude must be nonnegative.")
    require_condition(params.momentum_perturb_amplitude >= 0.0, "momentum perturbation amplitude must be nonnegative.")
    require_condition(params.fallback_uncorrelated_count > 0.0, "fallback_uncorrelated_count must be positive.")
    require_condition(params.fallback_snapshots_per_decorrelation > 0.0,
        "fallback_snapshots_per_decorrelation must be positive.")
    require_condition(params.fallback_ntrajectories >= 1, "fallback_ntrajectories must be positive.")
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
    require_condition(all(1 .<= params.cross_offsets) && all(params.cross_offsets .<= params.N - 1),
        "cross_offsets must lie in 1:(N-1).")
    require_condition(all(1 .<= params.bivariate_offsets) && all(params.bivariate_offsets .<= params.N - 1),
        "bivariate_offsets must lie in 1:(N-1).")
    require_condition(params.max_pdf_samples >= 10_000, "max_pdf_samples should be at least 10000.")
    require_condition(params.figure_width >= 1600 && params.figure_height >= 1200, "Figure dimensions are too small.")
    require_condition(params.dynamics_figure_width >= 1600 && params.dynamics_figure_height >= 1000,
        "Dynamics figure dimensions are too small.")
    require_condition(params.trajectories_figure_width >= 1600 && params.trajectories_figure_height >= 1000,
        "Trajectories figure dimensions are too small.")
    require_condition(params.dynamics_window_decorrelation_times > 0.0,
        "dynamics_window_decorrelation_times must be positive.")
    require_condition(1 <= params.dynamics_trajectory <= params.ntrajectories,
        "dynamics_trajectory must lie in 1:ntrajectories.")
    require_condition(params.dynamics_max_frames >= 16, "dynamics_max_frames must be at least 16.")
    require_condition(!isempty(params.dynamics_trace_sites), "dynamics_trace_sites must not be empty.")
    require_condition(all(1 .<= params.dynamics_trace_sites) && all(params.dynamics_trace_sites .<= params.N),
        "dynamics_trace_sites must lie in 1:N.")
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

periodic_index(i::Int, N::Int) = mod1(i, N)
dx(params::SimParams) = params.L / params.N

function load_previous_reference(params::SimParams, base_dir::AbstractString)
    ref_path = resolve_path(base_dir, params.reference_hdf5)
    if isfile(ref_path)
        return h5open(ref_path, "r") do file
            uncorrelated_count = Float64(read(file["/metadata/production_uncorrelated_count"]))
            snapshots_per_decorrelation = Float64(read(file["/metadata/production_snapshots_per_decorrelation"]))
            ntraj = Int(read(file["/metadata/ntrajectories"]))
            return PreviousReference(uncorrelated_count, snapshots_per_decorrelation, ntraj, ref_path)
        end
    end
    return PreviousReference(params.fallback_uncorrelated_count,
        params.fallback_snapshots_per_decorrelation, params.fallback_ntrajectories,
        "fallback values from sim.toml")
end

function initial_state(params::SimParams, rng::AbstractRNG)
    N = params.N
    z = Array{Float64}(undef, N, 2)
    z[:, 1] .= params.rho0
    z[:, 2] .= 0.0

    if params.rho_perturb_amplitude > 0.0
        vals = [cos(2pi * (i - 1) / N) + 0.5sin(4pi * (i - 1) / N) for i in 1:N]
        vals .-= mean(vals)
        z[:, 1] .+= params.rho_perturb_amplitude .* vals
        z[:, 1] .*= (params.rho0 * N) / sum(z[:, 1])
    end
    if params.momentum_perturb_amplitude > 0.0
        vals = [sin(2pi * (i - 1) / N) + 0.35cos(4pi * (i - 1) / N) for i in 1:N]
        vals .-= mean(vals)
        z[:, 2] .+= params.momentum_perturb_amplitude .* vals
        z[:, 2] .-= mean(z[:, 2])
    end

    # Add tiny trajectory-specific zero-mean perturbations so the ensemble does
    # not begin from identical deterministic fluxes.
    rho_noise = 1e-4 .* randn(rng, N)
    rho_noise .-= mean(rho_noise)
    z[:, 1] .+= rho_noise
    z[:, 1] .*= (params.rho0 * N) / sum(z[:, 1])
    m_noise = 1e-4 .* randn(rng, N)
    m_noise .-= mean(m_noise)
    z[:, 2] .+= m_noise
    z[:, 2] .-= mean(z[:, 2])
    return z
end

function edge_viscosity!(eta::Vector{Float64}, z::Array{Float64, 2}, params::SimParams)
    N = params.N
    @inbounds for i in 1:N
        ip1 = periodic_index(i + 1, N)
        rho_edge = 0.5 * (z[i, 1] + z[ip1, 1])
        if rho_edge <= 0.0
            error(@sprintf("Nonpositive edge density %.6e between sites %d and %d.", rho_edge, i, ip1))
        end
        eta[i] = params.eta0 * (rho_edge / params.rho0)^params.zeta
    end
    return nothing
end

function drift!(b::Array{Float64, 2}, eta::Vector{Float64}, z::Array{Float64, 2}, params::SimParams)
    N = params.N
    h = dx(params)
    edge_viscosity!(eta, z, params)

    @inbounds for i in 1:N
        if z[i, 1] <= 0.0
            error(@sprintf("Nonpositive density %.6e at site %d.", z[i, 1], i))
        end
    end

    @inbounds for i in 1:N
        im1 = periodic_index(i - 1, N)
        ip1 = periodic_index(i + 1, N)

        rho_i = z[i, 1]
        rho_ip1 = z[ip1, 1]
        rho_im1 = z[im1, 1]
        m_i = z[i, 2]
        m_ip1 = z[ip1, 2]
        m_im1 = z[im1, 2]

        u_i = m_i / max(rho_i, params.velocity_density_floor)
        u_ip1 = m_ip1 / max(rho_ip1, params.velocity_density_floor)
        u_im1 = m_im1 / max(rho_im1, params.velocity_density_floor)

        j_right = 0.5 * (m_i + m_ip1)
        j_left = 0.5 * (m_im1 + m_i)
        b[i, 1] = -(j_right - j_left) / h

        pi_right = 0.25 * (m_i + m_ip1) * (u_i + u_ip1) + params.cs^2 * 0.5 * (rho_i + rho_ip1)
        pi_left = 0.25 * (m_im1 + m_i) * (u_im1 + u_i) + params.cs^2 * 0.5 * (rho_im1 + rho_i)
        tau_right = eta[i] * (u_ip1 - u_i) / h
        tau_left = eta[im1] * (u_i - u_im1) / h

        b[i, 2] = -(pi_right - pi_left) / h + (tau_right - tau_left) / h
    end
    return nothing
end

function noise_increment!(dm_noise::Vector{Float64}, eta::Vector{Float64}, rng::AbstractRNG,
                          params::SimParams, dt::Float64)
    N = params.N
    h = dx(params)
    fill!(dm_noise, 0.0)
    @inbounds for edge in 1:N
        ip1 = periodic_index(edge + 1, N)
        amp = sqrt(2.0 * params.theta * eta[edge]) / h * sqrt(dt) * randn(rng)
        dm_noise[edge] += amp
        dm_noise[ip1] -= amp
    end
    return nothing
end

function euler_candidate!(candidate::Array{Float64, 2}, z::Array{Float64, 2},
                          b::Array{Float64, 2}, eta::Vector{Float64}, dm_noise::Vector{Float64},
                          rng::AbstractRNG, params::SimParams, dt::Float64)
    drift!(b, eta, z, params)
    noise_increment!(dm_noise, eta, rng, params, dt)
    rho_sum = 0.0
    @inbounds for i in 1:params.N
        log_increment = clamp(dt * b[i, 1] / z[i, 1],
            -params.density_log_step_limit, params.density_log_step_limit)
        candidate[i, 1] = z[i, 1] * exp(log_increment)
        rho_sum += candidate[i, 1]
        candidate[i, 2] = z[i, 2] + dt * b[i, 2] + dm_noise[i]
    end
    rho_scale = params.rho0 * params.N / rho_sum
    momentum_mean = 0.0
    @inbounds for i in 1:params.N
        candidate[i, 1] *= rho_scale
        momentum_mean += candidate[i, 2]
    end
    momentum_mean /= params.N
    @inbounds for i in 1:params.N
        candidate[i, 2] -= momentum_mean
    end
    return all(isfinite, candidate) && minimum(@view candidate[:, 1]) > params.density_floor
end

function positive_step!(z::Array{Float64, 2}, candidate::Array{Float64, 2},
                        b::Array{Float64, 2}, eta::Vector{Float64}, dm_noise::Vector{Float64},
                        rng::AbstractRNG, params::SimParams, dt::Float64, depth::Int=0)
    if euler_candidate!(candidate, z, b, eta, dm_noise, rng, params, dt)
        copyto!(z, candidate)
        return 0
    end

    if depth >= params.max_substep_depth
        error(@sprintf("Adaptive positivity retry exceeded max_substep_depth=%d at dt=%.6e; min candidate rho=%.6e.",
            params.max_substep_depth, dt, minimum(@view candidate[:, 1])))
    end

    z_half = copy(z)
    candidate_half = similar(z)
    retries = 1
    retries += positive_step!(z_half, candidate_half, b, eta, dm_noise, rng, params, 0.5dt, depth + 1)
    retries += positive_step!(z_half, candidate_half, b, eta, dm_noise, rng, params, 0.5dt, depth + 1)
    copyto!(z, z_half)
    return retries
end

function integrate_fhd_ensemble(params::SimParams, t1::Float64, save_dt::Float64; label::AbstractString, dt::Float64=params.dt)
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
    states = Array{Float64}(undef, nsaved, params.N, 2, params.ntrajectories)

    @printf("Integrating %s FHD chain: N=%d, trajectories=%d, dt=%.5g, save_dt=%.5g, T=%.4f, saved=%d, threads=%d\n",
        label, params.N, params.ntrajectories, dt, save_dt, t1 - params.t0, nsaved, Threads.nthreads())

    Threads.@threads :dynamic for traj_idx in 1:params.ntrajectories
        rng = MersenneTwister(params.seed + 100_000 * (label == "pilot" ? 0 : 1) + traj_idx)
        z = initial_state(params, rng)
        b = similar(z)
        candidate = similar(z)
        eta = Vector{Float64}(undef, params.N)
        dm_noise = Vector{Float64}(undef, params.N)
        substep_retries = 0
        states[1, :, :, traj_idx] .= z
        save_idx = 2

        @inbounds for step in 1:nsteps
            substep_retries += positive_step!(z, candidate, b, eta, dm_noise, rng, params, dt)

            if step % save_every == 0
                if !all(isfinite, z)
                    error(@sprintf("Non-finite state in %s trajectory %d at step %d. Reduce dt.", label, traj_idx, step))
                end
                min_rho = minimum(@view z[:, 1])
                if min_rho <= 0.0
                    error(@sprintf("Nonpositive density in %s trajectory %d at step %d: min rho = %.6e. Reduce dt.",
                        label, traj_idx, step, min_rho))
                end
                states[save_idx, :, :, traj_idx] .= z
                save_idx += 1
            end
        end

        if substep_retries > 0
            @printf("%s trajectory %d used %d adaptive positivity retries\n", label, traj_idx, substep_retries)
        end
    end

    return times, states
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function nearest_schedule(params::SimParams, t_dec::Float64, ref::PreviousReference)
    dt = params.dt
    target_save_dt = t_dec / params.production_snapshots_per_decorrelation
    save_every = max(1, round(Int, target_save_dt / dt))
    save_dt = save_every * dt
    nsteps_raw = max(1, round(Int, ref.uncorrelated_count * t_dec / dt))
    nsteps = max(save_every, round(Int, nsteps_raw / save_every) * save_every)
    t1 = params.t0 + nsteps * dt
    nsaved = nsteps ÷ save_every + 1
    return ProductionSchedule(t1, dt, save_dt, save_every, nsteps, nsaved,
        (t1 - params.t0) / t_dec, t_dec / save_dt)
end

function channel_mean_var_nonzero(states::Array{Float64, 4}, start_idx::Int, channel::Int)
    nt, N, _, ntraj = size(states)
    total = 0.0
    count = 0
    @inbounds for traj in 1:ntraj, t in start_idx:nt
        spatial_mean = mean(@view states[t, :, channel, traj])
        for i in 1:N
            total += states[t, i, channel, traj] - spatial_mean
            count += 1
        end
    end
    mu = total / count
    ss = 0.0
    @inbounds for traj in 1:ntraj, t in start_idx:nt
        spatial_mean = mean(@view states[t, :, channel, traj])
        for i in 1:N
            delta = states[t, i, channel, traj] - spatial_mean - mu
            ss += delta * delta
        end
    end
    return mu, ss / count
end

function estimate_decorrelation_time(lags::Vector{Float64}, acf_rho::Vector{Float64}, acf_m::Vector{Float64}, threshold::Float64)
    n = length(lags)
    envelope = [max(abs(acf_rho[i]), abs(acf_m[i])) for i in 1:n]
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

function compute_fhd_correlations(data::Array{Float64, 4}, dt_corr::Float64, threshold::Float64,
                                  cross_offsets::Vector{Int}, max_lags::Int)
    nt, N, _, ntraj = size(data)
    max_lag = min(max_lags, nt - 1)
    require_condition(max_lag >= 1, "Correlation window is empty.")
    lags = collect(0:max_lag) .* dt_corr
    mean_rho, var_rho = channel_mean_var_nonzero(data, 1, 1)
    mean_m, var_m = channel_mean_var_nonzero(data, 1, 2)
    require_condition(var_rho > 0.0 && var_m > 0.0, "Cannot normalize correlations with zero variance.")

    acf_rho_sum = zeros(Float64, max_lag + 1)
    acf_m_sum = zeros(Float64, max_lag + 1)
    rhom_sum = zeros(Float64, max_lag + 1)
    mrho_sum = zeros(Float64, max_lag + 1)
    spatial_sum = zeros(Float64, max_lag + 1, length(cross_offsets))

    scratch_rho = Vector{Float64}(undef, nt)
    scratch_m = Vector{Float64}(undef, nt)
    scratch_pair = Vector{Float64}(undef, nt)

    rho_means = Matrix{Float64}(undef, nt, ntraj)
    m_means = Matrix{Float64}(undef, nt, ntraj)
    @inbounds for traj in 1:ntraj, t in 1:nt
        rho_means[t, traj] = mean(@view data[t, :, 1, traj])
        m_means[t, traj] = mean(@view data[t, :, 2, traj])
    end

    @inbounds for traj in 1:ntraj
        for i in 1:N
            for t in 1:nt
                scratch_rho[t] = data[t, i, 1, traj] - rho_means[t, traj] - mean_rho
                scratch_m[t] = data[t, i, 2, traj] - m_means[t, traj] - mean_m
            end
            acf_rho_sum .+= fft_corr_future_present(scratch_rho, scratch_rho, max_lag)
            acf_m_sum .+= fft_corr_future_present(scratch_m, scratch_m, max_lag)
            rhom_sum .+= fft_corr_future_present(scratch_rho, scratch_m, max_lag)
            mrho_sum .+= fft_corr_future_present(scratch_m, scratch_rho, max_lag)

            for (offset_idx, offset) in enumerate(cross_offsets)
                j = periodic_index(i + offset, N)
                for t in 1:nt
                    scratch_pair[t] = data[t, j, 1, traj] - rho_means[t, traj] - mean_rho
                end
                spatial_sum[:, offset_idx] .+= fft_corr_future_present(scratch_pair, scratch_rho, max_lag)
            end
        end
    end

    nseries = N * ntraj
    denom_counts = Float64[nseries * (nt - lag) for lag in 0:max_lag]
    acf_rho = acf_rho_sum ./ (denom_counts .* var_rho)
    acf_m = acf_m_sum ./ (denom_counts .* var_m)
    denom_cross = denom_counts .* sqrt(var_rho * var_m)
    cross_rhom = rhom_sum ./ denom_cross
    cross_mrho = mrho_sum ./ denom_cross
    spatial_rho = similar(spatial_sum)
    for offset_idx in eachindex(cross_offsets)
        spatial_rho[:, offset_idx] .= spatial_sum[:, offset_idx] ./ (denom_counts .* var_rho)
    end

    t_dec = estimate_decorrelation_time(lags, acf_rho, acf_m, threshold)
    return CorrelationResult(lags, acf_rho, acf_m, cross_rhom, cross_mrho, copy(cross_offsets),
        spatial_rho, t_dec, mean_rho, mean_m, var_rho, var_m)
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
    nt, N, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * N * ntraj
    nsamples = min(max_samples, total)
    values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        site = (tmp % N) + 1
        traj = (tmp ÷ N) + 1
        t = start_idx + time_local
        values[sample_idx] = selector(states, t, site, traj)
    end
    return values
end

function compute_marginal_pdfs(states::Array{Float64, 4}, start_idx::Int, params::SimParams)
    rng = MersenneTwister(params.seed + 10_011)
    rhos = draw_scalar_samples(states, start_idx, params.max_pdf_samples, rng, (s, t, i, tr) -> s[t, i, 1, tr])
    ms = draw_scalar_samples(states, start_idx, params.max_pdf_samples, rng, (s, t, i, tr) -> s[t, i, 2, tr])
    us = draw_scalar_samples(states, start_idx, params.max_pdf_samples, rng,
        (s, t, i, tr) -> s[t, i, 2, tr] / max(s[t, i, 1, tr], params.velocity_density_floor))
    etas = draw_scalar_samples(states, start_idx, params.max_pdf_samples, rng, (s, t, i, tr) -> begin
        ip1 = periodic_index(i + 1, size(s, 2))
        rho_edge = 0.5 * (s[t, i, 1, tr] + s[t, ip1, 1, tr])
        params.eta0 * (rho_edge / params.rho0)^params.zeta
    end)

    rho_k = kde(rhos; npoints=params.histogram_bins, boundary=kde_range(rhos))
    m_k = kde(ms; npoints=params.histogram_bins, boundary=kde_range(ms))
    u_k = kde(us; npoints=params.histogram_bins, boundary=kde_range(us))
    eta_k = kde(etas; npoints=params.histogram_bins, boundary=kde_range(etas))
    return MarginalPdf(collect(rho_k.x), collect(rho_k.density)),
           MarginalPdf(collect(m_k.x), collect(m_k.density)),
           MarginalPdf(collect(u_k.x), collect(u_k.density)),
           MarginalPdf(collect(eta_k.x), collect(eta_k.density))
end

function draw_pair_samples(states::Array{Float64, 4}, start_idx::Int, max_samples::Int,
                           rng::AbstractRNG, selector::Function)
    nt, N, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * N * ntraj
    nsamples = min(max_samples, total)
    x = Vector{Float64}(undef, nsamples)
    y = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        site = (tmp % N) + 1
        traj = (tmp ÷ N) + 1
        t = start_idx + time_local
        x[sample_idx], y[sample_idx] = selector(states, t, site, traj)
    end
    return x, y
end

function compute_pair_pdfs(states::Array{Float64, 4}, start_idx::Int, offsets::Vector{Int},
                           params::SimParams)
    rng = MersenneTwister(params.seed + 21_019)
    pair_pdfs = PairPdf[]

    drho, m = draw_pair_samples(states, start_idx, params.max_pdf_samples, rng,
        (s, t, i, tr) -> (s[t, i, 1, tr] - params.rho0, s[t, i, 2, tr]))
    kde_drhom = kde((drho, m); npoints=(params.histogram_bins, params.histogram_bins),
        boundary=(kde_range(drho), kde_range(m)))
    push!(pair_pdfs, PairPdf("(rho_i-rho0, m_i)", collect(kde_drhom.x), collect(kde_drhom.y), Array(kde_drhom.density)))

    for offset in offsets
        x, y = draw_pair_samples(states, start_idx, params.max_pdf_samples, rng,
            (s, t, i, tr) -> (s[t, i, 1, tr] - params.rho0,
                              s[t, periodic_index(i + offset, size(s, 2)), 1, tr] - params.rho0))
        kde_pair = kde((x, y); npoints=(params.histogram_bins, params.histogram_bins),
            boundary=(kde_range(x), kde_range(y)))
        push!(pair_pdfs, PairPdf(@sprintf("(rho_i-rho0, rho_{i+%d}-rho0)", offset),
            collect(kde_pair.x), collect(kde_pair.y), Array(kde_pair.density)))
    end

    return pair_pdfs
end

function compute_spatial_covariance(states::Array{Float64, 4}, start_idx::Int)
    nt, N, _, ntraj = size(states)
    D = 2N
    mean_flat = zeros(Float64, D)
    count = (nt - start_idx + 1) * ntraj
    @inbounds for traj in 1:ntraj, t in start_idx:nt
        for i in 1:N
            mean_flat[i] += states[t, i, 1, traj]
            mean_flat[N + i] += states[t, i, 2, traj]
        end
    end
    mean_flat ./= count

    cov = zeros(Float64, D, D)
    x = Vector{Float64}(undef, D)
    @inbounds for traj in 1:ntraj, t in start_idx:nt
        for i in 1:N
            x[i] = states[t, i, 1, traj] - mean_flat[i]
            x[N + i] = states[t, i, 2, traj] - mean_flat[N + i]
        end
        BLAS.syr!('U', 1.0, x, cov)
    end
    cov ./= count
    cov = Symmetric(cov, :U) |> Matrix
    return cov
end

function viscosity_summary(states::Array{Float64, 4}, start_idx::Int, params::SimParams)
    nt, N, _, ntraj = size(states)
    eta_total = 0.0
    eta2_total = 0.0
    rho_min = Inf
    rho_max = -Inf
    count = 0
    @inbounds for traj in 1:ntraj, t in start_idx:nt, i in 1:N
        ip1 = periodic_index(i + 1, N)
        rho_edge = 0.5 * (states[t, i, 1, traj] + states[t, ip1, 1, traj])
        eta = params.eta0 * (rho_edge / params.rho0)^params.zeta
        eta_total += eta
        eta2_total += eta * eta
        rho_min = min(rho_min, states[t, i, 1, traj])
        rho_max = max(rho_max, states[t, i, 1, traj])
        count += 1
    end
    mean_eta = eta_total / count
    std_eta = sqrt(max(eta2_total / count - mean_eta^2, 0.0))
    return mean_eta, std_eta, rho_min, rho_max
end

function compute_conservation_stats(times::Vector{Float64}, states::Array{Float64, 4}, params::SimParams)
    nt, N, _, ntraj = size(states)
    h = dx(params)
    mass = Matrix{Float64}(undef, nt, ntraj)
    momentum = Matrix{Float64}(undef, nt, ntraj)
    min_rho = Matrix{Float64}(undef, nt, ntraj)
    @inbounds for traj in 1:ntraj, t in 1:nt
        mass[t, traj] = h * sum(@view states[t, :, 1, traj])
        momentum[t, traj] = h * sum(@view states[t, :, 2, traj])
        min_rho[t, traj] = minimum(@view states[t, :, 1, traj])
    end
    mass_target = params.rho0 * params.L
    momentum_target = 0.0
    mass_max_abs_drift = maximum(abs.(mass .- mass_target))
    momentum_max_abs_drift = maximum(abs.(momentum .- momentum_target))
    min_density = minimum(min_rho)
    return ConservationStats(mass, momentum, min_rho, mass_max_abs_drift, momentum_max_abs_drift, min_density)
end

function save_hdf5(path::AbstractString, params::SimParams, ref::PreviousReference, schedule::ProductionSchedule,
                   pilot_corr::CorrelationResult, times::Vector{Float64}, states::Array{Float64, 4},
                   rho_pdf::MarginalPdf, m_pdf::MarginalPdf, u_pdf::MarginalPdf, eta_pdf::MarginalPdf,
                   pair_pdfs::Vector{PairPdf}, corr::CorrelationResult, cov::Matrix{Float64},
                   mean_eta::Float64, std_eta::Float64, rho_min::Float64, rho_max::Float64,
                   conservation::ConservationStats)
    nt, N, _, ntraj = size(states)
    states_flat = Array{Float64}(undef, nt, 2N, ntraj)
    @inbounds for traj in 1:ntraj, t in 1:nt
        for i in 1:N
            states_flat[t, i, traj] = states[t, i, 1, traj]
            states_flat[t, N + i, traj] = states[t, i, 2, traj]
        end
    end

    h5open(path, "w") do file
        file["/trajectories/time"] = times
        file["/trajectories/states"] = states
        file["/trajectories/states_flat"] = states_flat
        file["/trajectories/channel_names"] = ["rho", "m"]
        file["/trajectories/flat_order"] = "rho_1,...,rho_N,m_1,...,m_N"

        file["/statistics/pdf/rho_centers"] = rho_pdf.centers
        file["/statistics/pdf/rho_density"] = rho_pdf.density
        file["/statistics/pdf/m_centers"] = m_pdf.centers
        file["/statistics/pdf/m_density"] = m_pdf.density
        file["/statistics/pdf/u_centers"] = u_pdf.centers
        file["/statistics/pdf/u_density"] = u_pdf.density
        file["/statistics/pdf/eta_centers"] = eta_pdf.centers
        file["/statistics/pdf/eta_density"] = eta_pdf.density
        file["/statistics/pdf/bivariate_labels"] = [pdf.label for pdf in pair_pdfs]
        for (idx, pdf) in enumerate(pair_pdfs)
            base = @sprintf("/statistics/pdf/bivariate/pair_%02d", idx)
            file[string(base, "/label")] = pdf.label
            file[string(base, "/x_grid")] = pdf.x_grid
            file[string(base, "/y_grid")] = pdf.y_grid
            file[string(base, "/density")] = pdf.density
        end

        file["/statistics/correlations/lags"] = corr.lags
        file["/statistics/correlations/acf_rho"] = corr.acf_rho
        file["/statistics/correlations/acf_m"] = corr.acf_m
        file["/statistics/correlations/cross_rhom"] = corr.cross_rhom
        file["/statistics/correlations/cross_mrho"] = corr.cross_mrho
        file["/statistics/correlations/cross_offsets"] = corr.cross_offsets
        file["/statistics/correlations/spatial_rho"] = corr.spatial_rho
        file["/statistics/correlations/t_decorrelation"] = pilot_corr.t_decorrelation
        file["/statistics/correlations/t_decorrelation_final_check"] = corr.t_decorrelation
        file["/statistics/correlations/global_mean_rho_nonzero"] = corr.mean_rho
        file["/statistics/correlations/global_mean_m_nonzero"] = corr.mean_m
        file["/statistics/correlations/global_variance_rho_nonzero"] = corr.var_rho
        file["/statistics/correlations/global_variance_m_nonzero"] = corr.var_m
        file["/statistics/covariance_flat"] = cov

        file["/statistics/conservation/mass"] = conservation.mass
        file["/statistics/conservation/momentum"] = conservation.momentum
        file["/statistics/conservation/min_rho"] = conservation.min_rho
        file["/statistics/conservation/mass_max_abs_drift"] = conservation.mass_max_abs_drift
        file["/statistics/conservation/momentum_max_abs_drift"] = conservation.momentum_max_abs_drift
        file["/statistics/conservation/min_density"] = conservation.min_density

        file["/mobility/mean_eta_edge"] = mean_eta
        file["/mobility/std_eta_edge"] = std_eta
        file["/mobility/rho_min_post_burnin"] = rho_min
        file["/mobility/rho_max_post_burnin"] = rho_max

        file["/reference_previous/source"] = ref.source
        file["/reference_previous/uncorrelated_count"] = ref.uncorrelated_count
        file["/reference_previous/snapshots_per_decorrelation"] = ref.snapshots_per_decorrelation
        file["/reference_previous/ntrajectories"] = ref.ntrajectories

        file["/metadata/model_name"] = "periodic_fluctuating_hydrodynamic_chain_strong_state_dependent_mobility"
        file["/metadata/N"] = params.N
        file["/metadata/state_dimension_full"] = 2params.N
        file["/metadata/state_dimension_effective"] = 2params.N - 2
        file["/metadata/L"] = params.L
        file["/metadata/dx"] = dx(params)
        file["/metadata/rho0"] = params.rho0
        file["/metadata/sound_speed"] = params.cs
        file["/metadata/Theta"] = params.theta
        file["/metadata/eta0"] = params.eta0
        file["/metadata/zeta"] = params.zeta
        file["/metadata/regime"] = "strong_state_dependence"
        file["/metadata/pilot_dt"] = params.dt
        file["/metadata/dt"] = schedule.dt
        file["/metadata/save_dt"] = schedule.save_dt
        file["/metadata/t0"] = params.t0
        file["/metadata/t1"] = schedule.t1
        file["/metadata/nsteps"] = schedule.nsteps
        file["/metadata/save_every"] = schedule.save_every
        file["/metadata/nsaved"] = schedule.nsaved
        file["/metadata/ntrajectories"] = params.ntrajectories
        file["/metadata/thread_parallelism"] = "Threads.@threads :dynamic over independent trajectories"
        file["/metadata/blas_threads"] = BLAS.get_num_threads()
        file["/metadata/burnin_fraction"] = params.burnin_fraction
        file["/metadata/production_uncorrelated_count"] = schedule.uncorrelated_count
        file["/metadata/production_snapshots_per_decorrelation"] = schedule.snapshots_per_decorrelation
        file["/metadata/max_substep_depth"] = params.max_substep_depth
        file["/metadata/density_floor"] = params.density_floor
        file["/metadata/density_log_step_limit"] = params.density_log_step_limit
        file["/metadata/velocity_density_floor"] = params.velocity_density_floor
        file["/metadata/sde_noise_prefactor"] = "conservative edge stress sqrt(2*Theta*eta_edge)/dx"
        file["/metadata/density_update"] = "first-order exponential conservative-flux update with exact mass renormalization and log-step limiter"
        file["/metadata/momentum_update"] = "Euler-Maruyama conservative flux/noise update with velocity denominator floor and zero-mode projection"
    end
    return nothing
end

function truncate_series(xs::Vector{Float64}, ys::Vector{Float64}, xmax::Float64)
    last_idx = clamp(searchsortedlast(xs, xmax), 2, length(xs))
    return xs[1:last_idx], ys[1:last_idx]
end

function heatmap_panel!(parent, x, y, z; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString,
                        colorbar_label::AbstractString="", colormap=STYLE_SEQUENTIAL_BLUE, colorrange=nothing)
    layout = GridLayout(parent)
    ax = Axis(layout[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)
    hm = isnothing(colorrange) ? heatmap!(ax, x, y, z; colormap=colormap) :
         heatmap!(ax, x, y, z; colormap=colormap, colorrange=colorrange)
    Colorbar(layout[1, 2], hm; label=colorbar_label)
    return ax
end

function render_summary_figure(path::AbstractString, params::SimParams, ref::PreviousReference,
                               schedule::ProductionSchedule, rho_pdf::MarginalPdf, m_pdf::MarginalPdf,
                               u_pdf::MarginalPdf, eta_pdf::MarginalPdf,
                               pair_pdfs::Vector{PairPdf}, corr::CorrelationResult, cov::Matrix{Float64},
                               mean_eta::Float64, std_eta::Float64, rho_min::Float64, rho_max::Float64,
                               conservation::ConservationStats, sample_state::Matrix{Float64})
    fig = Figure(; size=(params.figure_width, params.figure_height))
    subtitle = @sprintf("N=%d full dim=%d eff dim=%d  dt=%.3g  save_dt=%.4f  T=%.2f  t_D=%.3f  saves/t_D=%.1f",
        params.N, 2params.N, 2params.N - 2, schedule.dt, schedule.save_dt, schedule.t1 - params.t0,
        corr.t_decorrelation, schedule.snapshots_per_decorrelation)
    figure_title!(fig, "Strong fluctuating-hydrodynamic chain - observational summary"; subtitle=subtitle)

    ax_pdf = Axis(fig[1, 1]; title="Cell marginals", xlabel="value", ylabel="density")
    lines!(ax_pdf, rho_pdf.centers, rho_pdf.density; color=STYLE_PRIMARY, linewidth=3, label="rho")
    lines!(ax_pdf, m_pdf.centers, m_pdf.density; color=STYLE_SECONDARY, linewidth=3, label="m")
    axislegend(ax_pdf; position=:rt)

    ax_u = Axis(fig[1, 2]; title="Velocity and edge-viscosity PDFs", xlabel="value", ylabel="density")
    lines!(ax_u, u_pdf.centers, u_pdf.density; color=STYLE_HIGHLIGHT, linewidth=3, label="u=m/rho_eff")
    lines!(ax_u, eta_pdf.centers, eta_pdf.density; color=STYLE_ACCENT, linewidth=3, label="eta_edge")
    axislegend(ax_u; position=:rt)

    corr_t_max = min(corr.lags[end], 10.0)
    lags_rho, acf_rho = truncate_series(corr.lags, corr.acf_rho, corr_t_max)
    _, acf_m = truncate_series(corr.lags, corr.acf_m, corr_t_max)
    ax_acf = Axis(fig[1, 3]; title="Nonzero-mode autocorrelations", xlabel="lag tau", ylabel="C(tau)")
    hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_acf, lags_rho, acf_rho; color=STYLE_PRIMARY, linewidth=3, label="rho")
    lines!(ax_acf, lags_rho, acf_m; color=STYLE_SECONDARY, linewidth=3, label="m")
    xlims!(ax_acf, 0.0, corr_t_max)
    axislegend(ax_acf; position=:rt)

    _, rhom = truncate_series(corr.lags, corr.cross_rhom, corr_t_max)
    _, mrho = truncate_series(corr.lags, corr.cross_mrho, corr_t_max)
    ax_cross = Axis(fig[2, 1]; title="Local rho/m cross-correlations", xlabel="lag tau", ylabel="C(tau)")
    hlines!(ax_cross, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_cross, lags_rho, rhom; color=STYLE_HIGHLIGHT, linewidth=3, label="C_rho,m")
    lines!(ax_cross, lags_rho, mrho; color=STYLE_VIOLET, linewidth=3, label="C_m,rho")
    xlims!(ax_cross, 0.0, corr_t_max)
    axislegend(ax_cross; position=:rt)

    ax_spatial = Axis(fig[2, 2]; title="Shifted density correlations", xlabel="lag tau", ylabel="C_r(tau)")
    hlines!(ax_spatial, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for (idx, offset) in enumerate(corr.cross_offsets)
        _, vals = truncate_series(corr.lags, corr.spatial_rho[:, idx], corr_t_max)
        lines!(ax_spatial, lags_rho, vals; linewidth=2, label=@sprintf("r=%d", offset))
    end
    xlims!(ax_spatial, 0.0, corr_t_max)
    axislegend(ax_spatial; position=:rt, nbanks=2)

    heatmap_panel!(fig[2, 3], collect(1:size(cov, 1)), collect(1:size(cov, 2)), cov;
        title="Empirical covariance of flattened state", xlabel="component", ylabel="component",
        colorbar_label="cov", colormap=STYLE_DIVERGING)

    max_pair_panels = min(length(pair_pdfs), 3)
    pair_positions = [(3, 1), (3, 2), (3, 3)]
    for idx in 1:max_pair_panels
        row, col = pair_positions[idx]
        pdf = pair_pdfs[idx]
        heatmap_panel!(fig[row, col], pdf.x_grid, pdf.y_grid, pdf.density;
            title="Bivariate PDF " * pdf.label, xlabel="first", ylabel="second",
            colorbar_label="density", colormap=STYLE_SEQUENTIAL_BLUE)
    end

    sample_plot = hcat(sample_state[:, 1] .- params.rho0, sample_state[:, 2])
    ax_sample = heatmap_panel!(fig[4, 1], collect(1:params.N), collect(1:2), sample_plot;
        title="Final sample state, trajectory 1", xlabel="site", ylabel="channel",
        colorbar_label="state", colormap=STYLE_DIVERGING)
    ax_sample.yticks = ([1, 2], ["rho-rho0", "m"])

    ax_cons = Axis(fig[4, 2]; title="Conservation diagnostics", xlabel="saved index", ylabel="drift")
    idx = collect(1:size(conservation.mass, 1))
    mass_drift = conservation.mass[:, 1] .- params.rho0 * params.L
    mom_drift = conservation.momentum[:, 1]
    lines!(ax_cons, idx, mass_drift; color=STYLE_PRIMARY, linewidth=2.2, label="mass drift")
    lines!(ax_cons, idx, mom_drift; color=STYLE_SECONDARY, linewidth=2.2, label="momentum")
    axislegend(ax_cons; position=:rt)

    ax_meta = Axis(fig[4, 3]; title="Sampling and positivity check")
    hidedecorations!(ax_meta)
    hidespines!(ax_meta)
    source_label = occursin("ComplexAmplitudeChain", ref.source) ?
        joinpath("ComplexAmplitudeChain", "outputs", basename(ref.source)) : ref.source
    lines_txt = [
        @sprintf("Previous source: %s", source_label),
        @sprintf("Previous: N=T/t_D=%.4f, saves/t_D=%.4f, ntraj=%d",
            ref.uncorrelated_count, ref.snapshots_per_decorrelation, ref.ntrajectories),
        @sprintf("FHD: T=%.3f, t_D=%.3f, save_dt=%.5f",
            schedule.t1 - params.t0, corr.t_decorrelation, schedule.save_dt,
            ),
        @sprintf("FHD: N=T/t_D=%.4f, saved intervals=%d",
            schedule.uncorrelated_count, schedule.nsaved - 1),
        @sprintf("FHD saves/t_D=%.4f; requested=%d",
            schedule.snapshots_per_decorrelation, params.production_snapshots_per_decorrelation),
        @sprintf("mean eta=%.5f, std eta=%.5f", mean_eta, std_eta),
        @sprintf("rho post-burnin range=[%.3e, %.5f]", rho_min, rho_max),
        @sprintf("max |mass drift|=%.3e, max |momentum|=%.3e",
            conservation.mass_max_abs_drift, conservation.momentum_max_abs_drift),
        @sprintf("min rho=%.3e", conservation.min_density),
    ]
    text!(ax_meta, 0.02, 0.95; text=join(lines_txt, "\n"), align=(:left, :top),
        space=:relative, fontsize=14)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))
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

function edge_viscosity_matrix(states::Array{Float64, 4}, indices::Vector{Int}, traj::Int, params::SimParams)
    eta = Matrix{Float64}(undef, length(indices), params.N)
    @inbounds for (row, t) in enumerate(indices), i in 1:params.N
        ip1 = periodic_index(i + 1, params.N)
        rho_edge = 0.5 * (states[t, i, 1, traj] + states[t, ip1, 1, traj])
        eta[row, i] = params.eta0 * (rho_edge / params.rho0)^params.zeta
    end
    return eta
end

function render_dynamics_figure(path::AbstractString, params::SimParams, schedule::ProductionSchedule,
                                times::Vector{Float64}, states::Array{Float64, 4},
                                start_idx::Int, t_dec::Float64)
    indices = dynamics_window_indices(times, start_idx, t_dec,
        params.dynamics_window_decorrelation_times, params.dynamics_max_frames)
    traj = params.dynamics_trajectory
    t_axis = (times[indices] .- times[first(indices)]) ./ t_dec
    sites = collect(1:params.N)

    rho = Array(@view states[indices, :, 1, traj])
    m = Array(@view states[indices, :, 2, traj])
    drho = rho .- params.rho0
    u = m ./ max.(rho, params.velocity_density_floor)
    eta = edge_viscosity_matrix(states, indices, traj, params)
    rho_limit = maximum(abs, drho)
    m_limit = maximum(abs, m)
    u_limit = maximum(abs, u)

    fig = Figure(; size=(params.dynamics_figure_width, params.dynamics_figure_height))
    subtitle = @sprintf("trajectory %d   window %.2f t_D   t_D=%.3f   save_dt=%.5f   frames=%d   sites=%d",
        traj, t_axis[end] - t_axis[1], t_dec, schedule.save_dt, length(indices), params.N)
    figure_title!(fig, "Strong FHD chain dynamics"; subtitle=subtitle)

    heatmap_panel!(fig[1, 1], t_axis, sites, drho;
        title="density fluctuation rho_i(t)-rho0", xlabel="time / t_D", ylabel="site i",
        colorbar_label="rho-rho0", colormap=STYLE_DIVERGING, colorrange=(-rho_limit, rho_limit))
    heatmap_panel!(fig[1, 2], t_axis, sites, m;
        title="momentum m_i(t)", xlabel="time / t_D", ylabel="site i",
        colorbar_label="m", colormap=STYLE_DIVERGING, colorrange=(-m_limit, m_limit))
    heatmap_panel!(fig[2, 1], t_axis, sites, u;
        title="regularized velocity u_i(t)=m_i/rho_eff", xlabel="time / t_D", ylabel="site i",
        colorbar_label="u", colormap=STYLE_DIVERGING, colorrange=(-u_limit, u_limit))
    heatmap_panel!(fig[2, 2], t_axis, sites, eta;
        title="edge viscosity eta_{i+1/2}(t)", xlabel="time / t_D", ylabel="edge i+1/2",
        colorbar_label="eta", colormap=STYLE_SEQUENTIAL_BLUE)

    Label(fig[3, 1:2],
        "Hovmoller panels show one post-burn-in trajectory with the same saved resolution as the dataset. " *
        "The stochastic forcing is conservative, so density and momentum evolve through spatial flux differences.",
        tellwidth=false, fontsize=18)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))
    save_figure(path, fig)
    return nothing
end

function render_trajectories_figure(path::AbstractString, params::SimParams, schedule::ProductionSchedule,
                                    times::Vector{Float64}, states::Array{Float64, 4},
                                    conservation::ConservationStats, start_idx::Int, t_dec::Float64)
    indices = dynamics_window_indices(times, start_idx, t_dec,
        params.dynamics_window_decorrelation_times, params.dynamics_max_frames)
    traj = params.dynamics_trajectory
    t_axis = (times[indices] .- times[first(indices)]) ./ t_dec

    fig = Figure(; size=(params.trajectories_figure_width, params.trajectories_figure_height))
    subtitle = @sprintf("trajectory %d   saved resolution %.5f   t_D=%.3f   %d displayed frames",
        traj, schedule.save_dt, t_dec, length(indices))
    figure_title!(fig, "Strong FHD chain selected trajectories"; subtitle=subtitle)

    ax_rho = Axis(fig[1, 1]; title="Selected density traces", xlabel="time / t_D", ylabel="rho-rho0")
    hlines!(ax_rho, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for site in params.dynamics_trace_sites
        vals = states[indices, site, 1, traj] .- params.rho0
        lines!(ax_rho, t_axis, vals; linewidth=2, label=@sprintf("i=%d", site))
    end
    axislegend(ax_rho; position=:rt, nbanks=2)

    ax_m = Axis(fig[1, 2]; title="Selected momentum traces", xlabel="time / t_D", ylabel="m")
    hlines!(ax_m, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for site in params.dynamics_trace_sites
        vals = states[indices, site, 2, traj]
        lines!(ax_m, t_axis, vals; linewidth=2, label=@sprintf("i=%d", site))
    end
    axislegend(ax_m; position=:rt, nbanks=2)

    ax_phase = Axis(fig[2, 1]; title="Local state-space traces", xlabel="rho_i-rho0", ylabel="m_i")
    hlines!(ax_phase, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    vlines!(ax_phase, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for site in params.dynamics_trace_sites
        x = states[indices, site, 1, traj] .- params.rho0
        y = states[indices, site, 2, traj]
        lines!(ax_phase, x, y; linewidth=1.4, label=@sprintf("i=%d", site))
        scatter!(ax_phase, [x[1]], [y[1]]; markersize=8)
        scatter!(ax_phase, [x[end]], [y[end]]; markersize=10, marker=:rect)
    end
    axislegend(ax_phase; position=:rt, nbanks=2)

    ax_cons = Axis(fig[2, 2]; title="Conserved quantities and positivity", xlabel="time / t_D", ylabel="value")
    mass_drift = conservation.mass[indices, traj] .- params.rho0 * params.L
    momentum = conservation.momentum[indices, traj]
    min_rho = conservation.min_rho[indices, traj]
    lines!(ax_cons, t_axis, mass_drift; color=STYLE_PRIMARY, linewidth=2.4, label="mass drift")
    lines!(ax_cons, t_axis, momentum; color=STYLE_SECONDARY, linewidth=2.4, label="momentum")
    lines!(ax_cons, t_axis, min_rho; color=STYLE_ACCENT, linewidth=2.4, label="min rho")
    axislegend(ax_cons; position=:rt)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
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
    output_trajectories_png = resolve_path(base_dir, params.output_trajectories_png)
    ensure_parent_dir(output_hdf5)
    ensure_parent_dir(output_png)
    ensure_parent_dir(output_dynamics_png)
    ensure_parent_dir(output_trajectories_png)

    ref = load_previous_reference(params, base_dir)
    @printf("Previous-system reference from %s\n", ref.source)
    @printf("Previous budget: N=%.6f, saves/tD=%.6f, ntrajectories=%d\n",
        ref.uncorrelated_count, ref.snapshots_per_decorrelation, ref.ntrajectories)
    if ref.ntrajectories != params.ntrajectories
        @printf("Warning: configured ntrajectories=%d differs from previous reference ntrajectories=%d\n",
            params.ntrajectories, ref.ntrajectories)
    end

    pilot_times, pilot_states = integrate_fhd_ensemble(params, params.pilot_t1, params.pilot_save_dt; label="pilot")
    pilot_start = burnin_start_index(length(pilot_times), params.pilot_burnin_fraction)
    pilot_data = downsample_states(pilot_states, pilot_start, params.pilot_correlation_stride)
    pilot_dt_corr = params.pilot_save_dt * params.pilot_correlation_stride
    pilot_max_lags = min(size(pilot_data, 1) - 1, floor(Int, params.pilot_max_decorrelation_time / pilot_dt_corr))
    pilot_corr = compute_fhd_correlations(pilot_data, pilot_dt_corr, params.decorrelation_threshold,
        params.cross_offsets, pilot_max_lags)
    @printf("Pilot decorrelation estimate: tD = %.6f\n", pilot_corr.t_decorrelation)
    pilot_corr_window = (size(pilot_data, 1) - 1) * pilot_dt_corr
    if isapprox(pilot_corr.t_decorrelation, pilot_corr_window; rtol=1e-12)
        @printf("Warning: pilot tD reached the available pilot correlation window.\n")
    end

    schedule = nearest_schedule(params, pilot_corr.t_decorrelation, ref)
    @printf("Production schedule: T=%.6f, dt=%.8f, save_dt=%.6f, save_every=%d, nsteps=%d, nsaved=%d\n",
        schedule.t1 - params.t0, schedule.dt, schedule.save_dt, schedule.save_every, schedule.nsteps, schedule.nsaved)
    @printf("Production budget check: N=%.6f (target %.6f), saves/tD=%.6f (target %d, previous %.6f)\n",
        schedule.uncorrelated_count, ref.uncorrelated_count,
        schedule.snapshots_per_decorrelation, params.production_snapshots_per_decorrelation,
        ref.snapshots_per_decorrelation)

    times, states = integrate_fhd_ensemble(params, schedule.t1, schedule.save_dt; label="production", dt=schedule.dt)
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    @printf("Stationary diagnostic window starts at saved index %d / %d\n", start_idx, length(times))

    rho_pdf, m_pdf, u_pdf, eta_pdf = compute_marginal_pdfs(states, start_idx, params)
    pair_pdfs = compute_pair_pdfs(states, start_idx, params.bivariate_offsets, params)

    corr_data = downsample_states(states, start_idx, params.correlation_stride)
    dt_corr = schedule.save_dt * params.correlation_stride
    corr = compute_fhd_correlations(corr_data, dt_corr, params.decorrelation_threshold,
        params.cross_offsets, params.max_correlation_lags)
    @printf("Final-window decorrelation check from downsampled saved trajectory: tD = %.6f\n", corr.t_decorrelation)

    cov = compute_spatial_covariance(states, start_idx)
    mean_eta, std_eta, rho_min, rho_max = viscosity_summary(states, start_idx, params)
    conservation = compute_conservation_stats(times, states, params)

    require_condition(conservation.min_density > 0.0, "Saved trajectory contains nonpositive density.")
    @printf("Conservation: max |mass drift|=%.6e, max |momentum|=%.6e, min rho=%.6f\n",
        conservation.mass_max_abs_drift, conservation.momentum_max_abs_drift, conservation.min_density)
    @printf("Viscosity: mean eta=%.6f, std eta=%.6f, rho range post-burnin=[%.6f, %.6f]\n",
        mean_eta, std_eta, rho_min, rho_max)

    @printf("Writing HDF5 output to %s\n", output_hdf5)
    save_hdf5(output_hdf5, params, ref, schedule, pilot_corr, times, states,
        rho_pdf, m_pdf, u_pdf, eta_pdf, pair_pdfs, corr, cov, mean_eta, std_eta,
        rho_min, rho_max, conservation)

    sample_state = Array(states[end, :, :, 1])
    @printf("Rendering summary figure to %s\n", output_png)
    render_summary_figure(output_png, params, ref, schedule, rho_pdf, m_pdf, u_pdf, eta_pdf,
        pair_pdfs, corr, cov, mean_eta, std_eta, rho_min, rho_max, conservation, sample_state)

    @printf("Rendering dynamics figure to %s\n", output_dynamics_png)
    render_dynamics_figure(output_dynamics_png, params, schedule, times, states,
        start_idx, pilot_corr.t_decorrelation)

    @printf("Rendering trajectories figure to %s\n", output_trajectories_png)
    render_trajectories_figure(output_trajectories_png, params, schedule, times, states,
        conservation, start_idx, pilot_corr.t_decorrelation)

    @printf("Completed FHD simulation. Pilot tD = %.6f, production T = %.6f, save_dt = %.6f\n",
        pilot_corr.t_decorrelation, schedule.t1 - params.t0, schedule.save_dt)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
