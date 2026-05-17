#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi.jl"))

using HDF5
using Printf

function trueM_action_noise(raw::Array{Float32, 3}, score_raw::Array{Float32, 3},
        p::SpinParams, rng::AbstractRNG)
    N, _, B = size(raw)
    drift = Array{Float32}(undef, N, 3, B)
    noise = Array{Float32}(undef, N, 3, B)
    div_coeff = p.theta * (4.0 * p.alpha_parallel - 2.0 * p.alpha_perp)
    sqtheta = sqrt(p.theta)
    @inbounds for b in 1:B, i in 1:N
        x1 = Float64(raw[i, 1, b])
        x2 = Float64(raw[i, 2, b])
        x3 = Float64(raw[i, 3, b])
        s1 = Float64(score_raw[i, 1, b])
        s2 = Float64(score_raw[i, 2, b])
        s3 = Float64(score_raw[i, 3, b])
        a1, a2, a3 = apply_A(x1, x2, x3, s1, s2, s3, p)
        c1, c2, c3 = cross3(x1, x2, x3, s1, s2, s3)
        drift[i, 1, b] = Float32(p.theta * a1 - p.gamma * p.theta * c1 + div_coeff * x1)
        drift[i, 2, b] = Float32(p.theta * a2 - p.gamma * p.theta * c2 + div_coeff * x2)
        drift[i, 3, b] = Float32(p.theta * a3 - p.gamma * p.theta * c3 + div_coeff * x3)
        z1, z2, z3 = randn(rng), randn(rng), randn(rng)
        n1, n2, n3 = apply_sqrt_A(x1, x2, x3, z1, z2, z3, p)
        noise[i, 1, b] = Float32(sqtheta * n1)
        noise[i, 2, b] = Float32(sqtheta * n2)
        noise[i, 3, b] = Float32(sqtheta * n3)
    end
    return drift, noise
end

function integrate_trueM_score(phi_cfg_path::AbstractString, output_h5::AbstractString;
        dt_override::Float64=NaN, device_request::AbstractString="",
        required_gpu_name::AbstractString="", score_bson_override::AbstractString="")
    base = dirname(phi_cfg_path)
    params = load_config(phi_cfg_path)
    dt = isfinite(dt_override) ? dt_override : params.forward_dt
    request = isempty(device_request) ? params.device : String(device_request)
    required = isempty(required_gpu_name) ? params.required_gpu_name : String(required_gpu_name)
    device = detect_spin_device(request, required)
    activate_and_describe_device!(device, request, required)
    data_h5 = resolve_path(base, params.input_hdf5)
    p = load_phys(data_h5)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_path = isempty(score_bson_override) ?
        resolve_path(base, params.score_bson) : resolve_path(base, score_bson_override)
    score_model, stats, sigma, _ = load_score_checkpoint(score_path, device)
    Flux.testmode!(score_model)

    rng = MersenneTwister(params.seed + 2300 + round(Int, 1_000_000 * dt))
    z = initial_raw_batch(sampler, params.forward_ntraj, rng)
    nsteps = ceil(Int, params.forward_total_time / dt)
    burn_steps = floor(Int, params.forward_burnin_time / dt)
    save_every = max(1, round(Int, params.forward_save_dt / dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, params.forward_ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    progress = ProgressMeter.Progress(nsteps; desc="Forward trueM+score dt=$(dt)")
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            saved[save_idx, :, :, :] .= Float32.(z)
            times[save_idx] = (step - burn_steps) * dt
        end
        step == nsteps && break
        raw32 = Float32.(z)
        score_raw = evaluate_raw_score(score_model, raw32, stats, sigma, device;
            batch_size=params.score_batch_size)
        clamp!(score_raw, -params.forward_score_clip, params.forward_score_clip)
        drift, noise = trueM_action_noise(raw32, score_raw, p, rng)
        @. z = z + dt * Float64(drift) + sqrt(2.0 * dt) * Float64(noise)
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    ensure_parent_dir(output_h5)
    h5open(output_h5, "w") do f
        f["/trajectories/time"] = times[1:save_idx]
        f["/trajectories/states"] = saved[1:save_idx, :, :, :]
        f["/metadata/model"] = "true mobility with learned stationary score, ex-post diagnostic"
        f["/metadata/source_config"] = phi_cfg_path
        f["/metadata/score_bson"] = score_path
        f["/metadata/dt"] = dt
        f["/metadata/no_cheating_audit"] = "True M is used only in this ex-post forward diagnostic, not in training."
    end
    @printf("Saved true-M learned-score forward trajectories to %s\n", output_h5)
    return output_h5
end

function main()
    phi_cfg = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE
    out = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(@__DIR__, "..", "data", "forward_trueM_score.h5")
    dt = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : NaN
    device = length(ARGS) >= 4 ? ARGS[4] : ""
    required = length(ARGS) >= 5 ? ARGS[5] : ""
    score_bson = length(ARGS) >= 6 ? ARGS[6] : ""
    integrate_trueM_score(phi_cfg, out; dt_override=dt,
        device_request=device, required_gpu_name=required, score_bson_override=score_bson)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
