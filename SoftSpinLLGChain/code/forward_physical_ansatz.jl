#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM_physical_ansatz.jl"))

using HDF5
using Printf

Base.@kwdef struct PhysicalForwardConfig
    dt::Float64
    total_time::Float64
    burnin_time::Float64
    save_dt::Float64
    ntraj::Int
    score_clip::Float32
    score_batch_size::Int
end

function load_physical_forward_config(path::AbstractString)
    raw = TOML.parsefile(path)
    fwd = raw["forward"]
    eval = raw["evaluation"]
    return PhysicalForwardConfig(
        dt=Float64(get(fwd, "dt", 0.003)),
        total_time=Float64(get(fwd, "total_time", 180.0)),
        burnin_time=Float64(get(fwd, "burnin_time", 30.0)),
        save_dt=Float64(get(fwd, "save_dt", 0.04)),
        ntraj=Int(get(fwd, "ntrajectories", 72)),
        score_clip=Float32(get(fwd, "score_clip", 80.0)),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
    )
end

function initial_raw_batch_cond_physical(sampler::CondPairSampler, ntraj::Int, rng::AbstractRNG)
    raw = sample_raw_states_cond(sampler, ntraj, rng)
    return Float64.(raw)
end

function physical_ansatz_action_noise_divergence(coeff::AbstractVector{<:Real},
        raw::Array{Float32, 3}, score_raw::Array{Float32, 3}, rng::AbstractRNG)
    N, _, B = size(raw)
    drift = Array{Float32}(undef, N, 3, B)
    noise = Array{Float32}(undef, N, 3, B)
    c0, cp, cq, ck = Float32.(coeff)
    @inbounds for b in 1:B, i in 1:N
        x1, x2, x3 = raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]
        s1, s2, s3 = score_raw[i, 1, b], score_raw[i, 2, b], score_raw[i, 3, b]
        rsq = x1 * x1 + x2 * x2 + x3 * x3
        dotxs = x1 * s1 + x2 * s2 + x3 * s3
        cross1 = x2 * s3 - x3 * s2
        cross2 = x3 * s1 - x1 * s3
        cross3v = x1 * s2 - x2 * s1
        div_coeff = -2f0 * cp + 4f0 * cq
        drift[i, 1, b] = c0 * s1 + cp * (rsq * s1 - x1 * dotxs) +
            cq * x1 * dotxs + ck * cross1 + div_coeff * x1
        drift[i, 2, b] = c0 * s2 + cp * (rsq * s2 - x2 * dotxs) +
            cq * x2 * dotxs + ck * cross2 + div_coeff * x2
        drift[i, 3, b] = c0 * s3 + cp * (rsq * s3 - x3 * dotxs) +
            cq * x3 * dotxs + ck * cross3v + div_coeff * x3
        lam_perp = max(c0 + cp * rsq, 1f-8)
        lam_para = max(c0 + cq * rsq, 1f-8)
        z1, z2, z3 = randn(rng, Float32), randn(rng, Float32), randn(rng, Float32)
        if rsq < 1f-12
            scale = sqrt(lam_perp)
            noise[i, 1, b] = scale * z1
            noise[i, 2, b] = scale * z2
            noise[i, 3, b] = scale * z3
        else
            dotxz = x1 * z1 + x2 * z2 + x3 * z3
            inv = 1f0 / rsq
            p1, p2, p3 = x1 * dotxz * inv, x2 * dotxz * inv, x3 * dotxz * inv
            sperp = sqrt(lam_perp)
            spara = sqrt(lam_para)
            noise[i, 1, b] = sperp * (z1 - p1) + spara * p1
            noise[i, 2, b] = sperp * (z2 - p2) + spara * p2
            noise[i, 3, b] = sperp * (z3 - p3) + spara * p3
        end
    end
    return drift, noise
end

function integrate_physical_ansatz_forward(ansatz_cfg_path::AbstractString,
        phi_cfg_path::AbstractString; output_h5::AbstractString="",
        dt_override::Float64=NaN)
    base = dirname(ansatz_cfg_path)
    cfg = load_physical_ansatz_config(ansatz_cfg_path)
    dm = cfg.dm
    fcfg = load_physical_forward_config(phi_cfg_path)
    dt = isfinite(dt_override) ? dt_override : fcfg.dt
    device = detect_spin_device(dm.device, dm.required_gpu_name)
    activate_and_describe_device!(device, dm.device, dm.required_gpu_name)
    data_h5 = resolve_path(base, dm.input_hdf5)
    sampler = build_cond_sampler(data_h5, dm.burnin_fraction,
        dm.tau_max_decorrelation_multiples, dm.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(resolve_path(base, dm.score_bson), device)
    blob = BSON.load(resolve_path(base, cfg.output_bson))
    coeff = Vector{Float64}(blob[:coefficients])
    psd = blob[:psd_diag]
    require_condition(psd.min_perp > -1e-6 && psd.min_parallel > -1e-6,
        "Physical ansatz has negative symmetric eigenvalue proxies; refusing forward integration.")
    rng = MersenneTwister(dm.seed + 1600)
    z = initial_raw_batch_cond_physical(sampler, fcfg.ntraj, rng)
    nsteps = ceil(Int, fcfg.total_time / dt)
    burn_steps = floor(Int, fcfg.burnin_time / dt)
    save_every = max(1, round(Int, fcfg.save_dt / dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, fcfg.ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    progress = ProgressMeter.Progress(nsteps; desc="Forward physical ansatz $(basename(ansatz_cfg_path))")
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            saved[save_idx, :, :, :] .= Float32.(z)
            times[save_idx] = (step - burn_steps) * dt
        end
        step == nsteps && break
        raw32 = Float32.(z)
        score_raw = evaluate_raw_score_local(score_model, raw32, stats, score_sigma,
            device; batch_size=fcfg.score_batch_size)
        clamp!(score_raw, -fcfg.score_clip, fcfg.score_clip)
        drift, noise = physical_ansatz_action_noise_divergence(coeff, raw32, score_raw, rng)
        @. z = z + dt * Float64(drift) + sqrt(2.0 * dt) * Float64(noise)
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    isempty(output_h5) && (output_h5 = joinpath(@__DIR__, "..", "data",
        replace(splitext(basename(ansatz_cfg_path))[1], "fit_dM_" => "forward_dM_") * ".h5"))
    ensure_parent_dir(output_h5)
    h5open(output_h5, "w") do f
        f["/trajectories/time"] = times[1:save_idx]
        f["/trajectories/states"] = saved[1:save_idx, :, :, :]
        f["/metadata/model"] = "physics-informed parametric mobility ansatz with learned stationary score"
        f["/metadata/source_config"] = ansatz_cfg_path
        f["/metadata/source_model"] = resolve_path(base, cfg.output_bson)
        f["/metadata/coefficients"] = coeff
        f["/metadata/divergence"] = "analytic divergence of c0 I + cp(r2 I-mmT) + cq mmT + ck [m]_x"
        f["/metadata/dt"] = dt
    end
    @printf("Saved physical-ansatz forward trajectories to %s\n", output_h5)
    return output_h5
end

function main()
    ansatz_cfg = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(@__DIR__, "..", "configs", "fit_dM_phys_ansatz_clean37_mean10_gpu0.toml")
    phi_cfg = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(@__DIR__, "..", "configs", "fit_Phi_phys_pC_dataonly.toml")
    out = length(ARGS) >= 3 ? ARGS[3] : ""
    dt = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
    integrate_physical_ansatz_forward(ansatz_cfg, phi_cfg; output_h5=out,
        dt_override=dt)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
