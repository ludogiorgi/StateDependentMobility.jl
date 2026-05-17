#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using HDF5
using KernelDensity
using Printf
using Statistics

Base.@kwdef struct ForwardConfig
    dt::Float64
    total_time::Float64
    burnin_time::Float64
    save_dt::Float64
    ntraj::Int
    score_clip::Float32
    score_batch_size::Int
    mobility_scale::Float64
    delta_scale::Float64
    skew_scale::Float64
end

function load_forward_config(path::AbstractString)
    raw = TOML.parsefile(path)
    fwd = raw["forward"]
    eval = raw["evaluation"]
    return ForwardConfig(
        dt=Float64(get(fwd, "dt", 0.003)),
        total_time=Float64(get(fwd, "total_time", 180.0)),
        burnin_time=Float64(get(fwd, "burnin_time", 30.0)),
        save_dt=Float64(get(fwd, "save_dt", 0.04)),
        ntraj=Int(get(fwd, "ntrajectories", 72)),
        score_clip=Float32(get(fwd, "score_clip", 80.0)),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
        mobility_scale=Float64(get(fwd, "mobility_scale", 1.0)),
        delta_scale=Float64(get(fwd, "delta_scale", 1.0)),
        skew_scale=Float64(get(fwd, "skew_scale", 1.0)),
    )
end

function feature_matrix_from_xn(xn::Array{Float32, 3}, mode::Symbol)
    feats = feature_tensor(xn, mode)
    F = size(feats, 2)
    return reshape(permutedims(feats, (2, 1, 3)), F, size(xn, 1) * size(xn, 3))
end

function params_to_blocks(y, model::LocalMobilityNN; skew_gain=one(eltype(y)))
    sscale = eltype(y)(model.sym_scale)
    kscale = eltype(y)(model.skew_scale)
    floor = eltype(y)(model.sym_floor)
    l11 = NNlib.softplus.(y[1, :]) .* sscale .+ floor
    l21 = y[2, :] .* sscale
    l22 = NNlib.softplus.(y[3, :]) .* sscale .+ floor
    l31 = y[4, :] .* sscale
    l32 = y[5, :] .* sscale
    l33 = NNlib.softplus.(y[6, :]) .* sscale .+ floor
    k1 = y[7, :] .* kscale .* skew_gain
    k2 = y[8, :] .* kscale .* skew_gain
    k3 = y[9, :] .* kscale .* skew_gain
    s11 = l11 .* l11
    s12 = l11 .* l21
    s13 = l11 .* l31
    s22 = l21 .* l21 .+ l22 .* l22
    s23 = l21 .* l31 .+ l22 .* l32
    s33 = l31 .* l31 .+ l32 .* l32 .+ l33 .* l33
    return (; l11, l21, l22, l31, l32, l33, k1, k2, k3,
        m11=s11, m12=s12 .- k3, m13=s13 .+ k2,
        m21=s12 .+ k3, m22=s22, m23=s23 .- k1,
        m31=s13 .- k2, m32=s23 .+ k1, m33=s33)
end

function block_params_from_features(model::LocalMobilityNN, features_dev; skew_gain=1.0)
    y = model.mlp(features_dev)
    return params_to_blocks(y, model; skew_gain=eltype(y)(skew_gain))
end

function central_feature_row(mode::Symbol, c::Int)
    mode == :local && return c
    mode == :local_r2 && return c
    mode == :neighbor && return 3 + c
    mode == :neighbor_r2 && return 3 + c
    mode == :neighbor_all_r2 && return 3 + c
    error("Unknown feature mode $(mode)")
end

function std_for_flat(stats::DataStats, N::Int, B::Int, c::Int)
    out = Vector{Float32}(undef, N * B)
    @inbounds for b in 1:B, i in 1:N
        out[(b - 1) * N + i] = stats.std[c, i]
    end
    return out
end

function normalized_component_flat(xn::Array{Float32, 3}, c::Int)
    N, _, B = size(xn)
    out = Vector{Float32}(undef, N * B)
    @inbounds for b in 1:B, i in 1:N
        out[(b - 1) * N + i] = xn[i, c, b]
    end
    return out
end

function divergence_blocks(model::LocalMobilityNN, features::Matrix{Float32},
        xn::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice;
        eps_raw::Float32=1f-3, skew_gain::Float64=1.0)
    N, _, B = size(xn)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        stdv = std_for_flat(stats, N, B, c)
        epsn = eps_raw ./ stdv
        row = central_feature_row(model.feature_mode, c)
        fplus = copy(features)
        fminus = copy(features)
        xnc = normalized_component_flat(xn, c)
        @inbounds for q in 1:Q
            fplus[row, q] += epsn[q]
            fminus[row, q] -= epsn[q]
            if model.feature_mode == :local_r2
                fplus[4, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[4, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            elseif model.feature_mode == :neighbor_r2
                fplus[10, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[10, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            elseif model.feature_mode == :neighbor_all_r2
                fplus[11, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[11, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            end
        end
        bp = block_params_from_features(model, move_array(fplus, device); skew_gain=skew_gain)
        bm = block_params_from_features(model, move_array(fminus, device); skew_gain=skew_gain)
        cols_p = (Array(bp.m11), Array(bp.m21), Array(bp.m31),
            Array(bp.m12), Array(bp.m22), Array(bp.m32),
            Array(bp.m13), Array(bp.m23), Array(bp.m33))
        cols_m = (Array(bm.m11), Array(bm.m21), Array(bm.m31),
            Array(bm.m12), Array(bm.m22), Array(bm.m32),
            Array(bm.m13), Array(bm.m23), Array(bm.m33))
        # Add derivative of column c to every output row.
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function divergence_blocks(model::EquivariantMobilityNN, xn::Array{Float32, 3},
        stats::DataStats, device::ExecutionDevice; eps_raw::Float32=1f-3,
        skew_gain::Float64=1.0)
    N, _, B = size(xn)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        epsn = reshape(eps_raw ./ stats.std[c, :], N, 1, 1)
        xplus = copy(xn)
        xminus = copy(xn)
        @views xplus[:, c:c, :] .+= epsn
        @views xminus[:, c:c, :] .-= epsn
        bp = block_params(model, move_array(xplus, device); skew_gain=skew_gain)
        bm = block_params(model, move_array(xminus, device); skew_gain=skew_gain)
        cols_p = (Array(bp.l11 .* bp.l11),
            Array(bp.l11 .* bp.l21 .+ bp.k3),
            Array(bp.l11 .* bp.l31 .- bp.k2),
            Array(bp.l11 .* bp.l21 .- bp.k3),
            Array(bp.l21 .* bp.l21 .+ bp.l22 .* bp.l22),
            Array(bp.l21 .* bp.l31 .+ bp.l22 .* bp.l32 .+ bp.k1),
            Array(bp.l11 .* bp.l31 .+ bp.k2),
            Array(bp.l21 .* bp.l31 .+ bp.l22 .* bp.l32 .- bp.k1),
            Array(bp.l31 .* bp.l31 .+ bp.l32 .* bp.l32 .+ bp.l33 .* bp.l33))
        cols_m = (Array(bm.l11 .* bm.l11),
            Array(bm.l11 .* bm.l21 .+ bm.k3),
            Array(bm.l11 .* bm.l31 .- bm.k2),
            Array(bm.l11 .* bm.l21 .- bm.k3),
            Array(bm.l21 .* bm.l21 .+ bm.l22 .* bm.l22),
            Array(bm.l21 .* bm.l31 .+ bm.l22 .* bm.l32 .+ bm.k1),
            Array(bm.l11 .* bm.l31 .+ bm.k2),
            Array(bm.l21 .* bm.l31 .+ bm.l22 .* bm.l32 .- bm.k1),
            Array(bm.l31 .* bm.l31 .+ bm.l32 .* bm.l32 .+ bm.l33 .* bm.l33))
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function action_noise_divergence(model::LocalMobilityNN, raw::Array{Float32, 3},
        score_raw::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice,
        rng::AbstractRNG; phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        sqrt_phi_block::Union{Nothing, Matrix{Float32}}=nothing, delta_scale::Float64=1.0,
        skew_gain::Float64=1.0)
    N, _, B = size(raw)
    xn = apply_stats_tensor(raw, stats)
    features = feature_matrix_from_xn(xn, model.feature_mode)
    bp = block_params_from_features(model, move_array(features, device); skew_gain=skew_gain)
    host = (;
        l11=Array(bp.l11), l21=Array(bp.l21), l22=Array(bp.l22),
        l31=Array(bp.l31), l32=Array(bp.l32), l33=Array(bp.l33),
        m11=Array(bp.m11), m12=Array(bp.m12), m13=Array(bp.m13),
        m21=Array(bp.m21), m22=Array(bp.m22), m23=Array(bp.m23),
        m31=Array(bp.m31), m32=Array(bp.m32), m33=Array(bp.m33))
    Q = N * B
    sflat = reshape(permutedims(score_raw, (2, 1, 3)), 3, Q)
    drift_flat = Array{Float32}(undef, 3, Q)
    noise_flat = Array{Float32}(undef, 3, Q)
    z = randn(rng, Float32, 3, Q)
    beta = Float32(delta_scale)
    @inbounds for q in 1:Q
        s1, s2, s3 = sflat[1, q], sflat[2, q], sflat[3, q]
        drift_flat[1, q] = host.m11[q] * s1 + host.m12[q] * s2 + host.m13[q] * s3
        drift_flat[2, q] = host.m21[q] * s1 + host.m22[q] * s2 + host.m23[q] * s3
        drift_flat[3, q] = host.m31[q] * s1 + host.m32[q] * s2 + host.m33[q] * s3
        # Noise factor is the Cholesky factor of the symmetric PSD part.
        noise_flat[1, q] = host.l11[q] * z[1, q]
        noise_flat[2, q] = host.l21[q] * z[1, q] + host.l22[q] * z[2, q]
        noise_flat[3, q] = host.l31[q] * z[1, q] + host.l32[q] * z[2, q] + host.l33[q] * z[3, q]
    end
    div = divergence_blocks(model, features, xn, stats, device; skew_gain=skew_gain)
    drift_flat .+= div
    if delta_scale != 1.0
        require_condition(phi_block !== nothing && sqrt_phi_block !== nothing,
            "delta_scale requires Phi block and square-root Phi block.")
        require_condition(0.0 <= delta_scale <= 1.0,
            "delta_scale currently supports only the PSD-preserving interval [0, 1].")
        phi = phi_block::Matrix{Float32}
        phi_drift = phi * sflat
        @. drift_flat = beta * drift_flat + (1f0 - beta) * phi_drift
        zphi = randn(rng, Float32, 3, Q)
        phi_noise = (sqrt_phi_block::Matrix{Float32}) * zphi
        @. noise_flat = sqrt(beta) * noise_flat + sqrt(1f0 - beta) * phi_noise
    end
    drift = permutedims(reshape(drift_flat, 3, N, B), (2, 1, 3))
    noise = permutedims(reshape(noise_flat, 3, N, B), (2, 1, 3))
    return drift, noise
end

function action_noise_divergence(model::EquivariantMobilityNN, raw::Array{Float32, 3},
        score_raw::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice,
        rng::AbstractRNG; phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        sqrt_phi_block::Union{Nothing, Matrix{Float32}}=nothing, delta_scale::Float64=1.0,
        skew_gain::Float64=1.0)
    N, _, B = size(raw)
    xn = apply_stats_tensor(raw, stats)
    bp = block_params(model, move_array(xn, device); skew_gain=skew_gain)
    host = (;
        l11=Array(bp.l11), l21=Array(bp.l21), l22=Array(bp.l22),
        l31=Array(bp.l31), l32=Array(bp.l32), l33=Array(bp.l33),
        k1=Array(bp.k1), k2=Array(bp.k2), k3=Array(bp.k3))
    Q = N * B
    sflat = reshape(permutedims(score_raw, (2, 1, 3)), 3, Q)
    drift_flat = Array{Float32}(undef, 3, Q)
    noise_flat = Array{Float32}(undef, 3, Q)
    z = randn(rng, Float32, 3, Q)
    beta = Float32(delta_scale)
    @inbounds for q in 1:Q
        s11 = host.l11[q] * host.l11[q]
        s12 = host.l11[q] * host.l21[q]
        s13 = host.l11[q] * host.l31[q]
        s22 = host.l21[q] * host.l21[q] + host.l22[q] * host.l22[q]
        s23 = host.l21[q] * host.l31[q] + host.l22[q] * host.l32[q]
        s33 = host.l31[q] * host.l31[q] + host.l32[q] * host.l32[q] + host.l33[q] * host.l33[q]
        m12 = s12 - host.k3[q]
        m13 = s13 + host.k2[q]
        m21 = s12 + host.k3[q]
        m23 = s23 - host.k1[q]
        m31 = s13 - host.k2[q]
        m32 = s23 + host.k1[q]
        s1, s2, s3 = sflat[1, q], sflat[2, q], sflat[3, q]
        drift_flat[1, q] = s11 * s1 + m12 * s2 + m13 * s3
        drift_flat[2, q] = m21 * s1 + s22 * s2 + m23 * s3
        drift_flat[3, q] = m31 * s1 + m32 * s2 + s33 * s3
        noise_flat[1, q] = host.l11[q] * z[1, q]
        noise_flat[2, q] = host.l21[q] * z[1, q] + host.l22[q] * z[2, q]
        noise_flat[3, q] = host.l31[q] * z[1, q] + host.l32[q] * z[2, q] + host.l33[q] * z[3, q]
    end
    div = divergence_blocks(model, xn, stats, device; skew_gain=skew_gain)
    drift_flat .+= div
    if delta_scale != 1.0
        require_condition(phi_block !== nothing && sqrt_phi_block !== nothing,
            "delta_scale requires Phi block and square-root Phi block.")
        require_condition(0.0 <= delta_scale <= 1.0,
            "delta_scale currently supports only the PSD-preserving interval [0, 1].")
        phi = phi_block::Matrix{Float32}
        phi_drift = phi * sflat
        @. drift_flat = beta * drift_flat + (1f0 - beta) * phi_drift
        zphi = randn(rng, Float32, 3, Q)
        phi_noise = (sqrt_phi_block::Matrix{Float32}) * zphi
        @. noise_flat = sqrt(beta) * noise_flat + sqrt(1f0 - beta) * phi_noise
    end
    drift = permutedims(reshape(drift_flat, 3, N, B), (2, 1, 3))
    noise = permutedims(reshape(noise_flat, 3, N, B), (2, 1, 3))
    return drift, noise
end

function initial_raw_batch_cond(sampler::CondPairSampler, ntraj::Int, rng::AbstractRNG)
    raw = sample_raw_states_cond(sampler, ntraj, rng)
    return Float64.(raw)
end

function integrate_dm_forward(dm_cfg_path::AbstractString, phi_cfg_path::AbstractString;
        output_h5::AbstractString="", mobility_scale_override::Float64=NaN,
        dt_override::Float64=NaN, delta_scale_override::Float64=NaN,
        skew_scale_override::Float64=NaN)
    base = dirname(dm_cfg_path)
    dm_cfg = load_dm_config(dm_cfg_path)
    fcfg = load_forward_config(phi_cfg_path)
    mobility_scale = isfinite(mobility_scale_override) ? mobility_scale_override : fcfg.mobility_scale
    delta_scale = isfinite(delta_scale_override) ? delta_scale_override : fcfg.delta_scale
    skew_scale = isfinite(skew_scale_override) ? skew_scale_override : fcfg.skew_scale
    dt = isfinite(dt_override) ? dt_override : fcfg.dt
    device = detect_spin_device(dm_cfg.device, dm_cfg.required_gpu_name)
    activate_and_describe_device!(device, dm_cfg.device, dm_cfg.required_gpu_name)
    data_h5 = resolve_path(base, dm_cfg.input_hdf5)
    score_path = resolve_path(base, dm_cfg.score_bson)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, dm_cfg.burnin_fraction,
        dm_cfg.tau_max_decorrelation_multiples, dm_cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    blob = BSON.load(resolve_path(base, dm_cfg.output_bson))
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    phi_block = nothing
    sqrt_phi_block = nothing
    if delta_scale != 1.0
        target = BSON.load(resolve_path(base, dm_cfg.target_artifact_bson))
        phi_block = Matrix{Float32}(target[:Phi_block])
        sym_phi = 0.5f0 .* (phi_block .+ transpose(phi_block))
        sqrt_phi_block = Matrix{Float32}(cholesky(Symmetric(Matrix{Float64}(sym_phi))).L)
    end
    rng = MersenneTwister(dm_cfg.seed + 1500)
    z = initial_raw_batch_cond(sampler, fcfg.ntraj, rng)
    nsteps = ceil(Int, fcfg.total_time / dt)
    burn_steps = floor(Int, fcfg.burnin_time / dt)
    save_every = max(1, round(Int, fcfg.save_dt / dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, fcfg.ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    progress = ProgressMeter.Progress(nsteps; desc="Forward $(basename(dm_cfg_path))")
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
        if delta_scale != 1.0
            drift, noise = action_noise_divergence(model, raw32, score_raw, stats, device, rng;
                phi_block=phi_block, sqrt_phi_block=sqrt_phi_block,
                delta_scale=delta_scale, skew_gain=skew_scale)
        else
            drift, noise = action_noise_divergence(model, raw32, score_raw, stats, device, rng;
                skew_gain=skew_scale)
        end
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
        f["/metadata/model"] = "learned local mobility NN score-based Langevin"
        f["/metadata/source_config"] = dm_cfg_path
        f["/metadata/source_model"] = resolve_path(base, dm_cfg.output_bson)
        f["/metadata/divergence"] = "feature-space central finite differences"
        f["/metadata/mobility_scale"] = mobility_scale
        f["/metadata/delta_scale"] = delta_scale
        f["/metadata/skew_scale"] = skew_scale
        f["/metadata/dt"] = dt
    end
    @printf("Saved learned-M forward trajectories to %s\n", output_h5)
    return output_h5
end

function main()
    dm_cfg = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "fit_dM_gpu2_vC.toml")
    phi_cfg = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "configs", "fit_Phi.toml")
    out = length(ARGS) >= 3 ? ARGS[3] : ""
    scale = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
    dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : NaN
    delta_scale = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : NaN
    skew_scale = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : NaN
    integrate_dm_forward(dm_cfg, phi_cfg; output_h5=out,
        mobility_scale_override=scale, dt_override=dt, delta_scale_override=delta_scale,
        skew_scale_override=skew_scale)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
