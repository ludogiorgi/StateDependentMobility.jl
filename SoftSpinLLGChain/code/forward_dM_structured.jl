#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM_structured.jl"))

using HDF5
using Printf
using Statistics

Base.@kwdef struct StructuredForwardConfig
    dt::Float64
    total_time::Float64
    burnin_time::Float64
    save_dt::Float64
    ntraj::Int
    score_clip::Float32
    score_batch_size::Int
    mobility_scale::Float64
end

function load_structured_forward_config(path::AbstractString)
    raw = TOML.parsefile(path)
    fwd = raw["forward"]
    eval = raw["evaluation"]
    return StructuredForwardConfig(
        dt=Float64(get(fwd, "dt", 0.003)),
        total_time=Float64(get(fwd, "total_time", 180.0)),
        burnin_time=Float64(get(fwd, "burnin_time", 30.0)),
        save_dt=Float64(get(fwd, "save_dt", 0.04)),
        ntraj=Int(get(fwd, "ntrajectories", 72)),
        score_clip=Float32(get(fwd, "score_clip", 80.0)),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
        mobility_scale=Float64(get(fwd, "mobility_scale", 1.0)),
    )
end

function initial_raw_batch_cond_structured(sampler::CondPairSampler, ntraj::Int, rng::AbstractRNG)
    raw = sample_raw_states_cond(sampler, ntraj, rng)
    return Float64.(raw)
end

function structured_entries_host(model::StructuredMobilityNN, raw::Array{Float32, 3},
        stats::DataStats, device::ExecutionDevice)
    e = structured_block_entries(model, move_array(raw, device), stats)
    return (;
        lambda_perp=Array(e.lambda_perp), lambda_para=Array(e.lambda_para),
        beta=Array(e.beta),
        m11=Array(e.m11), m12=Array(e.m12), m13=Array(e.m13),
        m21=Array(e.m21), m22=Array(e.m22), m23=Array(e.m23),
        m31=Array(e.m31), m32=Array(e.m32), m33=Array(e.m33))
end

function structured_divergence_blocks(model::StructuredMobilityNN, raw::Array{Float32, 3},
        stats::DataStats, device::ExecutionDevice; eps_raw::Float32=1f-3)
    N, _, B = size(raw)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        rplus = copy(raw)
        rminus = copy(raw)
        @inbounds for b in 1:B, i in 1:N
            rplus[i, c, b] += eps_raw
            rminus[i, c, b] -= eps_raw
        end
        ep = structured_entries_host(model, rplus, stats, device)
        em = structured_entries_host(model, rminus, stats, device)
        cols_p = (ep.m11, ep.m21, ep.m31, ep.m12, ep.m22, ep.m32, ep.m13, ep.m23, ep.m33)
        cols_m = (em.m11, em.m21, em.m31, em.m12, em.m22, em.m32, em.m13, em.m23, em.m33)
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function structured_action_noise_divergence(model::StructuredMobilityNN,
        raw::Array{Float32, 3}, score_raw::Array{Float32, 3}, stats::DataStats,
        device::ExecutionDevice, rng::AbstractRNG)
    N, _, B = size(raw)
    host = structured_entries_host(model, raw, stats, device)
    Q = N * B
    xflat = reshape(permutedims(raw, (2, 1, 3)), 3, Q)
    sflat = reshape(permutedims(score_raw, (2, 1, 3)), 3, Q)
    drift_flat = Array{Float32}(undef, 3, Q)
    noise_flat = Array{Float32}(undef, 3, Q)
    z = randn(rng, Float32, 3, Q)
    @inbounds for q in 1:Q
        s1, s2, s3 = sflat[1, q], sflat[2, q], sflat[3, q]
        drift_flat[1, q] = host.m11[q] * s1 + host.m12[q] * s2 + host.m13[q] * s3
        drift_flat[2, q] = host.m21[q] * s1 + host.m22[q] * s2 + host.m23[q] * s3
        drift_flat[3, q] = host.m31[q] * s1 + host.m32[q] * s2 + host.m33[q] * s3

        x1, x2, x3 = xflat[1, q], xflat[2, q], xflat[3, q]
        r2 = max(x1 * x1 + x2 * x2 + x3 * x3, 1f-6)
        invr = 1f0 / sqrt(r2)
        ux1, ux2, ux3 = x1 * invr, x2 * invr, x3 * invr
        dotuz = ux1 * z[1, q] + ux2 * z[2, q] + ux3 * z[3, q]
        sp = sqrt(max(host.lambda_perp[q], 0f0))
        sa = sqrt(max(host.lambda_para[q], 0f0))
        para1, para2, para3 = ux1 * dotuz, ux2 * dotuz, ux3 * dotuz
        noise_flat[1, q] = sp * (z[1, q] - para1) + sa * para1
        noise_flat[2, q] = sp * (z[2, q] - para2) + sa * para2
        noise_flat[3, q] = sp * (z[3, q] - para3) + sa * para3
    end
    drift_flat .+= structured_divergence_blocks(model, raw, stats, device)
    drift = permutedims(reshape(drift_flat, 3, N, B), (2, 1, 3))
    noise = permutedims(reshape(noise_flat, 3, N, B), (2, 1, 3))
    return drift, noise
end

function integrate_structured_forward(dm_cfg_path::AbstractString, phi_cfg_path::AbstractString;
        output_h5::AbstractString="", mobility_scale_override::Float64=NaN,
        dt_override::Float64=NaN)
    base = dirname(dm_cfg_path)
    dm_cfg = load_dm_config(dm_cfg_path)
    fcfg = load_structured_forward_config(phi_cfg_path)
    mobility_scale = isfinite(mobility_scale_override) ? mobility_scale_override : fcfg.mobility_scale
    dt = isfinite(dt_override) ? dt_override : fcfg.dt
    device = detect_spin_device(dm_cfg.device, dm_cfg.required_gpu_name)
    activate_and_describe_device!(device, dm_cfg.device, dm_cfg.required_gpu_name)
    data_h5 = resolve_path(base, dm_cfg.input_hdf5)
    score_path = resolve_path(base, dm_cfg.score_bson)
    sampler = build_cond_sampler(data_h5, dm_cfg.burnin_fraction,
        dm_cfg.tau_max_decorrelation_multiples, dm_cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    blob = BSON.load(resolve_path(base, dm_cfg.output_bson))
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    rng = MersenneTwister(dm_cfg.seed + 2500)
    z = initial_raw_batch_cond_structured(sampler, fcfg.ntraj, rng)
    nsteps = ceil(Int, fcfg.total_time / dt)
    burn_steps = floor(Int, fcfg.burnin_time / dt)
    save_every = max(1, round(Int, fcfg.save_dt / dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, fcfg.ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    progress = ProgressMeter.Progress(nsteps; desc="Forward structured $(basename(dm_cfg_path))")
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            saved[save_idx, :, :, :] .= Float32.(z)
            times[save_idx] = (step - burn_steps) * fcfg.dt
        end
        step == nsteps && break
        raw32 = Float32.(z)
        score_raw = evaluate_raw_score_local(score_model, raw32, stats, score_sigma,
            device; batch_size=fcfg.score_batch_size)
        clamp!(score_raw, -fcfg.score_clip, fcfg.score_clip)
        drift, noise = structured_action_noise_divergence(model, raw32, score_raw, stats, device, rng)
        if mobility_scale != 1.0
            drift .*= Float32(mobility_scale)
            noise .*= Float32(sqrt(mobility_scale))
        end
        @. z = z + dt * Float64(drift) + sqrt(2.0 * dt) * Float64(noise)
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    isempty(output_h5) && (output_h5 = joinpath(@__DIR__, "..", "data",
        replace(splitext(basename(dm_cfg_path))[1], "fit_dM_" => "forward_dM_") * ".h5"))
    ensure_parent_dir(output_h5)
    h5open(output_h5, "w") do f
        f["/trajectories/time"] = times[1:save_idx]
        f["/trajectories/states"] = saved[1:save_idx, :, :, :]
        f["/metadata/model"] = "structured coefficient mobility NN score-based Langevin"
        f["/metadata/source_config"] = dm_cfg_path
        f["/metadata/source_model"] = resolve_path(base, dm_cfg.output_bson)
        f["/metadata/divergence"] = "raw-coordinate central finite differences"
        f["/metadata/mobility_scale"] = mobility_scale
        f["/metadata/dt"] = dt
    end
    @printf("Saved structured learned-M forward trajectories to %s\n", output_h5)
    return output_h5
end

function main()
    dm_cfg = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "fit_dM_struct_gpu2_vS1.toml")
    phi_cfg = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "configs", "fit_Phi.toml")
    out = length(ARGS) >= 3 ? ARGS[3] : ""
    scale = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
    dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : NaN
    integrate_structured_forward(dm_cfg, phi_cfg; output_h5=out,
        mobility_scale_override=scale, dt_override=dt)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
