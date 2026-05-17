#!/usr/bin/env julia

include(joinpath(@__DIR__, "sim.jl"))

using HDF5
using Printf
using Random
using Statistics
using TOML

const DEFAULT_HIGHFREQ_CONFIG = normpath(joinpath(@__DIR__, "..", "configs", "sim_phi_highfreq.toml"))

Base.@kwdef struct HighFreqParams
    sim_config::String
    source_hdf5::String
    output_hdf5::String
    t_total::Float64
    save_dt::Float64
    burnin_fraction_source::Float64
    ntrajectories::Int
    seed::Int
end

function load_highfreq_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    run = raw["run"]
    output = raw["output"]
    return HighFreqParams(
        sim_config=String(data["sim_config"]),
        source_hdf5=String(data["source_hdf5"]),
        output_hdf5=String(output["hdf5_file"]),
        t_total=Float64(run["t_total"]),
        save_dt=Float64(run["save_dt"]),
        burnin_fraction_source=Float64(get(run, "burnin_fraction_source", 0.1)),
        ntrajectories=Int(get(run, "ntrajectories", 16)),
        seed=Int(get(run, "seed", 20260640)),
    )
end

function load_stationary_initials(path::AbstractString, params::SimParams,
        hf::HighFreqParams, rng::AbstractRNG)
    states = h5open(path, "r") do h5
        Array{Float32}(read(h5["/trajectories/states"]))
    end
    nt, K, C, nobs = size(states)
    require_condition(K == params.N && C == 2, "High-frequency source state shape does not match sim params.")
    start_idx = burnin_start_index(nt, hf.burnin_fraction_source)
    z0 = Array{Float64}(undef, params.N, 2, hf.ntrajectories)
    @inbounds for tr in 1:hf.ntrajectories
        t = rand(rng, start_idx:nt)
        src = rand(rng, 1:nobs)
        z0[:, :, tr] .= Float64.(@view states[t, :, :, src])
        z0[:, 1, tr] .-= mean(@view z0[:, 1, tr]) - params.rho0
        z0[:, 2, tr] .-= mean(@view z0[:, 2, tr])
    end
    return z0
end

function integrate_from_initials(params::SimParams, z0::Array{Float64, 3},
        t_total::Float64, save_dt::Float64; label::AbstractString)
    ensure_thread_count(params)
    dt = params.dt
    nsteps_float = t_total / dt
    require_condition(isapprox(nsteps_float, round(nsteps_float); atol=1e-6),
        "t_total must be an integer multiple of dt.")
    nsteps = round(Int, nsteps_float)
    save_every = round(Int, save_dt / dt)
    require_condition(save_every >= 1, "save_dt must be at least dt.")
    require_condition(isapprox(save_every * dt, save_dt; atol=1e-12, rtol=1e-12),
        "save_dt must be an integer multiple of dt.")
    require_condition(nsteps % save_every == 0, "nsteps must be divisible by save_every.")
    nsaved = nsteps ÷ save_every + 1
    times = collect(range(0.0, step=save_dt, length=nsaved))
    states = Array{Float32}(undef, nsaved, params.N, 2, size(z0, 3))
    @printf("Integrating high-frequency Phi data: T=%.4f dt=%.5g save_dt=%.5g saved=%d ntraj=%d\n",
        t_total, dt, save_dt, nsaved, size(z0, 3))
    Threads.@threads :dynamic for tr in 1:size(z0, 3)
        rng = MersenneTwister(params.seed + 700_000 + tr)
        z = copy(@view z0[:, :, tr])
        b = similar(z)
        candidate = similar(z)
        eta = Vector{Float64}(undef, params.N)
        laprho = Vector{Float64}(undef, params.N)
        pi_cap = Vector{Float64}(undef, params.N)
        dm_noise = Vector{Float64}(undef, params.N)
        states[1, :, :, tr] .= z
        save_idx = 2
        stride = max(1, nsteps ÷ 20)
        retries = 0
        @inbounds for step in 1:nsteps
            retries += positive_step!(z, candidate, b, eta, laprho, pi_cap, dm_noise, rng, params, dt)
            if tr == 1 && step % stride == 0
                @printf("%s trajectory 1 %.1f%%\n", label, 100.0 * step / nsteps)
                flush(stdout)
            end
            if step % save_every == 0
                states[save_idx, :, :, tr] .= z
                save_idx += 1
            end
        end
        retries > 0 && @printf("%s trajectory %d used %d retries\n", label, tr, retries)
    end
    return times, states
end

function save_highfreq(path::AbstractString, times, states, params::SimParams, hf::HighFreqParams)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/trajectories/time", times)
        write(h5, "/trajectories/states", states)
        write(h5, "/metadata/N", params.N)
        write(h5, "/metadata/L", params.L)
        write(h5, "/metadata/dx", dx(params))
        write(h5, "/metadata/rho0", params.rho0)
        write(h5, "/metadata/sound_speed", params.cs)
        write(h5, "/metadata/Theta", params.theta)
        write(h5, "/metadata/eta0", params.eta0)
        write(h5, "/metadata/zeta", params.zeta)
        write(h5, "/metadata/kappa", params.kappa)
        write(h5, "/metadata/velocity_density_floor", params.velocity_density_floor)
        write(h5, "/metadata/dt", params.dt)
        write(h5, "/metadata/save_dt", hf.save_dt)
        write(h5, "/metadata/ntrajectories", size(states, 4))
        write(h5, "/metadata/source", "short high-frequency stationary continuation for data-only Phi estimation")
    end
    @printf("Saved high-frequency Phi data to %s\n", path)
    return nothing
end

function run_pipeline(config_path::AbstractString)
    base = dirname(abspath(config_path))
    hf = load_highfreq_params(config_path)
    sim_config = resolve_path(base, hf.sim_config)
    source_h5 = resolve_path(base, hf.source_hdf5)
    output_h5 = resolve_path(base, hf.output_hdf5)
    params = load_params(sim_config)
    rng = MersenneTwister(hf.seed)
    z0 = load_stationary_initials(source_h5, params, hf, rng)
    times, states = integrate_from_initials(params, z0, hf.t_total, hf.save_dt; label="phi_highfreq")
    save_highfreq(output_h5, times, states, params, hf)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    config_path = isempty(ARGS) ? DEFAULT_HIGHFREQ_CONFIG : abspath(ARGS[1])
    run_pipeline(config_path)
end
