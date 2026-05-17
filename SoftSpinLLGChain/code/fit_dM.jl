#!/usr/bin/env julia

include(joinpath(@__DIR__, "search_nonlinear_observables.jl"))
include(joinpath(@__DIR__, "joint_score.jl"))

using LinearAlgebra
using Printf
using Statistics

const DM_DEFAULT_CONFIG = normpath(joinpath(@__DIR__, "..", "configs", "fit_dM_gpu0_vA.toml"))

Base.@kwdef struct DMConfig
    input_hdf5::String
    score_bson::String
    cond_score_bson::String
    phi_artifact_bson::String
    retained_channels_toml::String
    target_artifact_bson::String
    burnin_fraction::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    max_lags::Int
    target_pairs_per_lag::Int
    target_mean_samples::Int
    batch_pairs::Int
    epochs::Int
    batches_per_epoch::Int
    learning_rate::Float64
    weight_decay::Float64
    mean_penalty_weight::Float64
    mean_penalty_samples::Int
    eval_pairs_per_lag::Int
    eval_every::Int
    hidden_width::Int
    hidden_depth::Int
    feature_mode::Symbol
    sym_scale::Float32
    skew_scale::Float32
    sym_floor::Float32
    seed::Int
    output_bson::String
    metrics_txt::String
    figure_png::String
    device::String
    required_gpu_name::String
    prepare_targets::Bool
    train::Bool
    evaluate::Bool
    verbose::Bool
end

if !isdefined(@__MODULE__, :RetainedChannel)
    struct RetainedChannel
        observable::String
        target_component::Int
        data_rms::Float64
    end
end

struct LocalMobilityNN{M}
    mlp::M
    feature_mode::Symbol
    sym_scale::Float32
    skew_scale::Float32
    sym_floor::Float32
end

Flux.@functor LocalMobilityNN (mlp,)

struct EquivariantMobilityNN{M,T}
    mlp::M
    mean::T
    std::T
    sym_scale::Float32
    skew_scale::Float32
    sym_floor::Float32
end

Flux.@functor EquivariantMobilityNN
Flux.trainable(model::EquivariantMobilityNN) = (; mlp=model.mlp)

struct DMTrainingCache
    lag_to_cache::Dict{Int, Int}
    x0n::Vector{Array{Float32, 3}}
    rraw::Vector{Array{Float32, 3}}
    obsflat::Vector{Matrix{Float32}}
    npairs::Int
end

function load_dm_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    targets = raw["targets"]
    model = raw["model"]
    train = raw["training"]
    out = raw["output"]
    run = raw["run"]
    return DMConfig(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        cond_score_bson=String(data["cond_score_bson"]),
        phi_artifact_bson=String(data["phi_artifact_bson"]),
        retained_channels_toml=String(targets["retained_channels_toml"]),
        target_artifact_bson=String(targets["target_artifact_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 0.60)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        max_lags=Int(get(targets, "max_lags", 24)),
        target_pairs_per_lag=Int(get(targets, "target_pairs_per_lag", 40000)),
        target_mean_samples=Int(get(targets, "target_mean_samples", 160000)),
        batch_pairs=Int(get(train, "batch_pairs", 4096)),
        epochs=Int(get(train, "epochs", 80)),
        batches_per_epoch=Int(get(train, "batches_per_epoch", 96)),
        learning_rate=Float64(get(train, "learning_rate", 1e-3)),
        weight_decay=Float64(get(train, "weight_decay", 1e-6)),
        mean_penalty_weight=Float64(get(train, "mean_penalty_weight", 0.05)),
        mean_penalty_samples=Int(get(train, "mean_penalty_samples", 4096)),
        eval_pairs_per_lag=Int(get(train, "eval_pairs_per_lag", 12000)),
        eval_every=Int(get(train, "eval_every", 10)),
        hidden_width=Int(get(model, "hidden_width", 96)),
        hidden_depth=Int(get(model, "hidden_depth", 3)),
        feature_mode=Symbol(get(model, "feature_mode", "local")),
        sym_scale=Float32(get(model, "sym_scale", 0.12)),
        skew_scale=Float32(get(model, "skew_scale", 1.0)),
        sym_floor=Float32(get(model, "sym_floor", 1f-4)),
        seed=Int(get(train, "seed", 20260511)),
        output_bson=String(out["model_bson"]),
        metrics_txt=String(out["metrics_txt"]),
        figure_png=String(out["figure_png"]),
        device=String(get(run, "device", "GPU:0")),
        required_gpu_name=String(get(run, "required_gpu_name", "2080ti")),
        prepare_targets=Bool(get(run, "prepare_targets", false)),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
        verbose=Bool(get(run, "verbose", true)),
    )
end

function configured_init_bson(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    return String(get(raw["model"], "init_bson", ""))
end

function configured_cond_score_config(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    return String(get(raw["data"], "cond_score_config", "cond_score_gpu0_vA.toml"))
end

function configured_cond_score_kind(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    return Symbol(lowercase(String(get(raw["data"], "cond_score_kind", "conditional_residual"))))
end

function configured_target_kind(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    return Symbol(lowercase(String(get(raw["targets"], "target_kind", "data_only"))))
end

function configured_target_scale_source(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    return Symbol(lowercase(String(get(raw["targets"], "scale_source", "retained_channel_rms"))))
end

function load_transition_source(cond_kind::Symbol, cfg_path::AbstractString,
        cond_path::AbstractString, base::AbstractString, device::ExecutionDevice)
    cond_cfg_path = resolve_path(base, configured_cond_score_config(cfg_path))
    if cond_kind in (:conditional_residual, :cond_score, :residual)
        cond_cfg = load_config(cond_cfg_path)
    elseif cond_kind == :joint_score
        cond_cfg = load_joint_config(cond_cfg_path)
    else
        error("Unsupported cond_score_kind=$(cond_kind). Use conditional_residual or joint_score.")
    end
    cond_blob = BSON.load(cond_path)
    cond_model = move_model(cond_blob[:host_model], device)
    Flux.testmode!(cond_model)
    return cond_model, cond_cfg
end

function evaluate_transition_norm(cond_kind::Symbol, cond_model, x0, xt, tau_norm,
        stats::DataStats, cond_params, device::ExecutionDevice; batch_size::Int,
        score_model=nothing, score_sigma::Float32=0f0)
    if cond_kind in (:conditional_residual, :cond_score, :residual)
        return evaluate_residual_norm(cond_model, x0, xt, tau_norm, stats, cond_params, device;
            batch_size=batch_size, score_model=score_model, score_sigma=score_sigma)
    elseif cond_kind == :joint_score
        return evaluate_joint_transition_norm(cond_model, x0, xt, tau_norm, stats, cond_params, device;
            batch_size=batch_size, score_model=score_model, score_sigma=score_sigma)
    end
    error("Unsupported cond_score_kind=$(cond_kind).")
end

function load_retained_channels(path::AbstractString)
    parsed = TOML.parsefile(path)
    raw_channels = get(parsed, "channels", Any[])
    channels = RetainedChannel[]
    for ch in raw_channels
        target = String(ch["target_component"])
        tindex = findfirst(==(target), ["mx", "my", "mz"])
        tindex === nothing && error("Unknown target component $(target)")
        push!(channels, RetainedChannel(String(ch["observable"]), tindex, Float64(ch["data_rms"])))
    end
    return channels
end

function unique_observable_names(channels::Vector{RetainedChannel})
    return unique([ch.observable for ch in channels])
end

function selected_linear_indices(channels::Vector{RetainedChannel}, obs_index::Dict{String, Int}, N::Int)
    inds = Int[]
    channel_ids = Int[]
    for (cid, ch) in enumerate(channels)
        a = obs_index[ch.observable]
        c = ch.target_component
        for i in 1:N, j in 1:N
            row = (i - 1) * length(obs_index) + a
            col = (j - 1) * 3 + c
            push!(inds, row + (col - 1) * (N * length(obs_index)))
            push!(channel_ids, cid)
        end
    end
    return inds, channel_ids
end

function build_mobility_model(cfg::DMConfig, rng::AbstractRNG, stats::Union{Nothing, DataStats}=nothing)
    feature_dim = cfg.feature_mode == :local ? 3 :
        cfg.feature_mode == :local_r2 ? 4 :
        cfg.feature_mode == :neighbor ? 9 :
        cfg.feature_mode == :neighbor_r2 ? 10 :
        cfg.feature_mode == :neighbor_all_r2 ? 12 :
        cfg.feature_mode == :equivariant_r2 ? 1 :
        error("Unknown feature_mode $(cfg.feature_mode)")
    output_dim = cfg.feature_mode == :equivariant_r2 ? 3 : 9
    layers = Any[Dense(feature_dim => cfg.hidden_width, swish)]
    for _ in 2:cfg.hidden_depth
        push!(layers, Dense(cfg.hidden_width => cfg.hidden_width, swish))
    end
    push!(layers, Dense(cfg.hidden_width => output_dim))
    model = Chain(layers...)
    if cfg.feature_mode == :equivariant_r2
        stats === nothing && error("equivariant_r2 mobility model requires normalization statistics.")
        mean = Float32.(stats.mean)
        std = Float32.(stats.std)
        return EquivariantMobilityNN(model, mean, std, cfg.sym_scale, cfg.skew_scale, cfg.sym_floor)
    end
    return LocalMobilityNN(model, cfg.feature_mode, cfg.sym_scale, cfg.skew_scale, cfg.sym_floor)
end

function feature_tensor(xn, mode::Symbol)
    if mode == :local
        return xn
    elseif mode == :local_r2
        r2 = sum(abs2, xn; dims=2)
        return cat(xn, r2; dims=2)
    elseif mode == :neighbor
        xm = circshift(xn, (1, 0, 0))
        xp = circshift(xn, (-1, 0, 0))
        return cat(xm, xn, xp; dims=2)
    elseif mode == :neighbor_r2
        xm = circshift(xn, (1, 0, 0))
        xp = circshift(xn, (-1, 0, 0))
        r2 = sum(abs2, xn; dims=2)
        return cat(xm, xn, xp, r2; dims=2)
    elseif mode == :neighbor_all_r2
        xm = circshift(xn, (1, 0, 0))
        xp = circshift(xn, (-1, 0, 0))
        r2m = sum(abs2, xm; dims=2)
        r2 = sum(abs2, xn; dims=2)
        r2p = sum(abs2, xp; dims=2)
        return cat(xm, xn, xp, r2m, r2, r2p; dims=2)
    else
        error("Unknown feature mode $(mode)")
    end
end

function block_params(model::LocalMobilityNN, xn)
    N, _, B = size(xn)
    feats = feature_tensor(xn, model.feature_mode)
    F = size(feats, 2)
    flat = reshape(permutedims(feats, (2, 1, 3)), F, N * B)
    y = model.mlp(flat)
    sscale = eltype(y)(model.sym_scale)
    kscale = eltype(y)(model.skew_scale)
    floor = eltype(y)(model.sym_floor)
    l11 = NNlib.softplus.(y[1, :]) .* sscale .+ floor
    l21 = y[2, :] .* sscale
    l22 = NNlib.softplus.(y[3, :]) .* sscale .+ floor
    l31 = y[4, :] .* sscale
    l32 = y[5, :] .* sscale
    l33 = NNlib.softplus.(y[6, :]) .* sscale .+ floor
    k1 = y[7, :] .* kscale
    k2 = y[8, :] .* kscale
    k3 = y[9, :] .* kscale
    return (; N, B, l11, l21, l22, l31, l32, l33, k1, k2, k3)
end

function raw_from_normalized(model::EquivariantMobilityNN, xn)
    N, C, _ = size(xn)
    mean_tensor = reshape(permutedims(model.mean, (2, 1)), N, C, 1)
    std_tensor = reshape(permutedims(model.std, (2, 1)), N, C, 1)
    return xn .* std_tensor .+ mean_tensor
end

function block_params(model::EquivariantMobilityNN, xn; skew_gain=1.0)
    N, _, B = size(xn)
    raw = raw_from_normalized(model, xn)
    r2 = sum(abs2, raw; dims=2)
    feats = reshape(r2, 1, N * B)
    y = model.mlp(feats)
    sscale = eltype(y)(model.sym_scale)
    kscale = eltype(y)(model.skew_scale * skew_gain)
    floor = eltype(y)(model.sym_floor)
    lambda_perp = NNlib.softplus.(y[1, :]) .* sscale .+ floor
    lambda_para = NNlib.softplus.(y[2, :]) .* sscale .+ floor
    kappa = y[3, :] .* kscale
    xflat = reshape(permutedims(raw, (2, 1, 3)), 3, N * B)
    x1, x2, x3 = xflat[1, :], xflat[2, :], xflat[3, :]
    invr2 = one(eltype(y)) ./ max.(reshape(r2, N * B), eltype(y)(1f-8))
    anis = (lambda_para .- lambda_perp) .* invr2
    s11 = lambda_perp .+ anis .* x1 .* x1
    s22 = lambda_perp .+ anis .* x2 .* x2
    s33 = lambda_perp .+ anis .* x3 .* x3
    s12 = anis .* x1 .* x2
    s13 = anis .* x1 .* x3
    s23 = anis .* x2 .* x3
    tiny = eltype(y)(1f-8)
    l11 = sqrt.(max.(s11, tiny))
    l21 = s12 ./ l11
    l31 = s13 ./ l11
    l22 = sqrt.(max.(s22 .- l21 .* l21, tiny))
    l32 = (s23 .- l21 .* l31) ./ l22
    l33 = sqrt.(max.(s33 .- l31 .* l31 .- l32 .* l32, tiny))
    k1 = kappa .* x1
    k2 = kappa .* x2
    k3 = kappa .* x3
    return (; N, B, l11, l21, l22, l31, l32, l33, k1, k2, k3)
end

function mobility_action(model, xn, rraw)
    bp = block_params(model, xn)
    N, B = bp.N, bp.B
    rflat = reshape(permutedims(rraw, (2, 1, 3)), 3, N * B)
    r1, r2, r3 = rflat[1, :], rflat[2, :], rflat[3, :]
    s11 = bp.l11 .* bp.l11
    s12 = bp.l11 .* bp.l21
    s13 = bp.l11 .* bp.l31
    s22 = bp.l21 .* bp.l21 .+ bp.l22 .* bp.l22
    s23 = bp.l21 .* bp.l31 .+ bp.l22 .* bp.l32
    s33 = bp.l31 .* bp.l31 .+ bp.l32 .* bp.l32 .+ bp.l33 .* bp.l33
    a1 = s11 .* r1 .+ s12 .* r2 .+ s13 .* r3 .+
        bp.k3 .* r2 .- bp.k2 .* r3
    a2 = s12 .* r1 .+ s22 .* r2 .+ s23 .* r3 .-
        bp.k3 .* r1 .+ bp.k1 .* r3
    a3 = s13 .* r1 .+ s23 .* r2 .+ s33 .* r3 .+
        bp.k2 .* r1 .- bp.k1 .* r2
    out = reshape(vcat(reshape(a1, 1, :), reshape(a2, 1, :), reshape(a3, 1, :)), 3, N, B)
    return permutedims(out, (2, 1, 3))
end

function mean_block(model, xn)
    bp = block_params(model, xn)
    s11 = bp.l11 .* bp.l11
    s12 = bp.l11 .* bp.l21
    s13 = bp.l11 .* bp.l31
    s22 = bp.l21 .* bp.l21 .+ bp.l22 .* bp.l22
    s23 = bp.l21 .* bp.l31 .+ bp.l22 .* bp.l32
    s33 = bp.l31 .* bp.l31 .+ bp.l32 .* bp.l32 .+ bp.l33 .* bp.l33
    vals = [mean(s11), mean(s12 .- bp.k3), mean(s13 .+ bp.k2),
        mean(s12 .+ bp.k3), mean(s22), mean(s23 .- bp.k1),
        mean(s13 .- bp.k2), mean(s23 .+ bp.k1), mean(s33)]
    return reshape(reduce(vcat, vals), 3, 3)
end

function mean_block_penalty(model, xn, phi_vals::NTuple{9, Float32}, phi_norm::Float32)
    bp = block_params(model, xn)
    s11 = bp.l11 .* bp.l11
    s12 = bp.l11 .* bp.l21
    s13 = bp.l11 .* bp.l31
    s22 = bp.l21 .* bp.l21 .+ bp.l22 .* bp.l22
    s23 = bp.l21 .* bp.l31 .+ bp.l22 .* bp.l32
    s33 = bp.l31 .* bp.l31 .+ bp.l32 .* bp.l32 .+ bp.l33 .* bp.l33
    invnorm = 1f0 / phi_norm
    terms = (
        (mean(s11) - phi_vals[1]) * invnorm,
        (mean(s12 .- bp.k3) - phi_vals[2]) * invnorm,
        (mean(s13 .+ bp.k2) - phi_vals[3]) * invnorm,
        (mean(s12 .+ bp.k3) - phi_vals[4]) * invnorm,
        (mean(s22) - phi_vals[5]) * invnorm,
        (mean(s23 .- bp.k1) - phi_vals[6]) * invnorm,
        (mean(s13 .- bp.k2) - phi_vals[7]) * invnorm,
        (mean(s23 .+ bp.k1) - phi_vals[8]) * invnorm,
        (mean(s33) - phi_vals[9]) * invnorm,
    )
    return sum(t -> t * t, terms) / 9
end

function sample_raw_states_cond(sampler::CondPairSampler, nsamples::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(sampler.states)
    raw = Array{Float32}(undef, N, 3, nsamples)
    @inbounds for b in 1:nsamples
        t = rand(rng, sampler.start_idx:nt)
        tr = rand(rng, 1:ntraj)
        raw[:, :, b] .= sampler.states[t, :, :, tr]
    end
    return raw
end

function evaluate_raw_score_local(model, raw::Array{Float32, 3}, stats::DataStats,
        sigma::Float32, device::ExecutionDevice; batch_size::Int)
    normed = apply_stats_tensor(raw, stats)
    sn = evaluate_score_norm(model, normed, sigma, device; batch_size)
    return normalized_score_to_raw(sn, stats)
end

function prepare_dm_targets(cfg::DMConfig, base::AbstractString, device::ExecutionDevice,
        cfg_path::AbstractString)
    data_h5 = resolve_path(base, cfg.input_hdf5)
    score_path = resolve_path(base, cfg.score_bson)
    phi_path = resolve_path(base, cfg.phi_artifact_bson)
    retained_path = resolve_path(base, cfg.retained_channels_toml)
    out_path = resolve_path(base, cfg.target_artifact_bson)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    phi_blob = BSON.load(phi_path)
    Phi = Matrix{Float64}(phi_blob[:Phi])
    channels = load_retained_channels(retained_path)
    names = unique_observable_names(channels)
    lib = NonlinearLibrary(names)
    obs_index = Dict(name => i for (i, name) in enumerate(names))
    selected_inds, selected_channel_ids = selected_linear_indices(channels, obs_index, sampler.N)
    means = estimate_nonlinear_means(sampler, p, lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 11))
    lags = sampler.lag_steps[1:min(cfg.max_lags, length(sampler.lag_steps))]
    nobs = length(names)
    Cdata = Array{Float64}(undef, length(lags), sampler.N, nobs, sampler.D)
    Cphi = similar(Cdata)
    mu = Float64.(mean_flat(stats))
    rng = MersenneTwister(cfg.seed + 12)
    for (li, lag) in enumerate(lags)
        x0, xt, xp, xm, _ = sample_fixed_lag_window(sampler, lag, cfg.target_pairs_per_lag, rng)
        x0f = Float64.(flatten_batch(x0))
        x0f .-= mu
        obsp = nonlinear_observables(xp, p, lib)
        obsm = nonlinear_observables(xm, p, lib)
        deriv = (obsp .- obsm) ./ Float32(2.0 * sampler.save_dt)
        deriv_flat = reshape(deriv, sampler.N * nobs, cfg.target_pairs_per_lag)
        Cdata[li, :, :, :] .= reshape(Matrix{Float64}(deriv_flat) * transpose(x0f) ./ cfg.target_pairs_per_lag,
            sampler.N, nobs, sampler.D)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * nobs, cfg.target_pairs_per_lag)
        sraw = evaluate_raw_score_local(score_model, x0, stats, score_sigma, device; batch_size=4096)
        action_phi = transpose(Phi) * Matrix{Float64}(flatten_batch(sraw))
        Cphi[li, :, :, :] .= reshape(Matrix{Float64}(obs_flat) * transpose(action_phi) ./ cfg.target_pairs_per_lag,
            sampler.N, nobs, sampler.D)
        @printf("Prepared dM target lag %.5g (%d/%d), %d observables, %d pairs\n",
            lag * sampler.save_dt, li, length(lags), nobs, cfg.target_pairs_per_lag)
    end
    A_target = Cdata .- Cphi
    target_vec = Array{Float32}(undef, length(lags), length(selected_inds))
    scale_vec = Array{Float32}(undef, length(selected_inds))
    for li in eachindex(lags)
        flat = vec(A_target[li, :, :, :])
        target_vec[li, :] .= Float32.(flat[selected_inds])
    end
    scale_source = configured_target_scale_source(cfg_path)
    if scale_source == :data_target
        for k in axes(target_vec, 2)
            scale_vec[k] = Float32(max(sqrt(mean(abs2, @view target_vec[:, k])), 1f-6))
        end
    elseif scale_source == :data_channel
        for cid in unique(selected_channel_ids)
            inds = findall(==(cid), selected_channel_ids)
            scale = Float32(max(sqrt(mean(abs2, @view target_vec[:, inds])), 1f-6))
            scale_vec[inds] .= scale
        end
    elseif scale_source == :retained_channel_rms
        for (k, cid) in enumerate(selected_channel_ids)
            scale_vec[k] = Float32(max(channels[cid].data_rms, 1f-6))
        end
    else
        error("Unsupported [targets].scale_source=$(scale_source).")
    end
    Phi_block = haskey(phi_blob, :Phi_projected_block) ?
        Matrix{Float32}(phi_blob[:Phi_projected_block]) :
        Matrix{Float32}(Phi[1:3, 1:3])
    ensure_parent_dir(out_path)
    BSON.bson(out_path, Dict(:names => names, :channels => channels,
        :observable_means => means, :lags => lags, :taus => lags .* sampler.save_dt,
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Cdot_data => Cdata, :Cdot_phi => Cphi, :A_target => A_target,
        :Phi => Matrix{Float32}(Phi), :Phi_block => Phi_block,
        :scale_source => scale_source,
        :save_dt => sampler.save_dt, :N => sampler.N, :D => sampler.D,
        :no_cheating_audit => "Cdot_data was estimated from trajectory finite differences. Cdot_phi used learned stationary score and data-only Phi. True mobility was not used in these targets."))
    @printf("Saved dM target artifact to %s\n", out_path)
    return BSON.load(out_path)
end

function load_or_prepare_targets(cfg::DMConfig, base::AbstractString, device::ExecutionDevice,
        cfg_path::AbstractString)
    path = resolve_path(base, cfg.target_artifact_bson)
    if isfile(path) && !cfg.prepare_targets
        return BSON.load(path)
    end
    target_kind = configured_target_kind(cfg_path)
    if target_kind == :oracle_truem
        error("Oracle true-M target artifact is missing or prepare_targets=true. Run prepare_oracle_trueM_dM_targets.jl first.")
    end
    return prepare_dm_targets(cfg, base, device, cfg_path)
end

function selected_prediction(model, x0, xt, tau_norm, score_model, stats, score_sigma,
        cond_model, cond_params, cond_kind::Symbol, target, p, cfg, device)
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    B = size(x0, 3)
    rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats, cond_params, device;
        batch_size=min(cond_params.batch_size, B), score_model=score_model, score_sigma=score_sigma)
    rraw = normalized_residual_to_raw(rnorm, stats)
    obs = nonlinear_observables(xt, p, lib)
    center_observables!(obs, means)
    obs_dev = move_array(reshape(obs, size(obs, 1) * size(obs, 2), B), device)
    r_dev = move_array(rraw, device)
    x0n_dev = move_array(apply_stats_tensor(x0, stats), device)
    model_action = mobility_action(model, x0n_dev, r_dev)
    phi = move_array(Matrix{Float32}(target[:Phi]), device)
    phi_action = phi' * reshape(permutedims(r_dev, (2, 1, 3)), size(phi, 1), B)
    model_flat = reshape(permutedims(model_action, (2, 1, 3)), size(phi, 1), B)
    delta_action = model_flat .- phi_action
    mat = -(obs_dev * transpose(delta_action)) ./ eltype(delta_action)(B)
    sel = target[:selected_indices]
    return mat[sel]
end

function selected_prediction_precomputed(model, x0n_dev, r_dev, obs_dev, phi_dev, selected_indices)
    B = size(r_dev, 3)
    model_action = mobility_action(model, x0n_dev, r_dev)
    phi_action = phi_dev' * reshape(permutedims(r_dev, (2, 1, 3)), size(phi_dev, 1), B)
    model_flat = reshape(permutedims(model_action, (2, 1, 3)), size(phi_dev, 1), B)
    delta_action = model_flat .- phi_action
    mat = -(obs_dev * transpose(delta_action)) ./ eltype(delta_action)(B)
    return mat[selected_indices]
end

function precompute_batch_inputs(x0, xt, tau_norm, score_model, stats, score_sigma,
        cond_model, cond_params, cond_kind::Symbol, target, p, cfg, device)
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    B = size(x0, 3)
    rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats, cond_params, device;
        batch_size=min(cond_params.batch_size, B), score_model=score_model, score_sigma=score_sigma)
    rraw = normalized_residual_to_raw(rnorm, stats)
    obs = nonlinear_observables(xt, p, lib)
    center_observables!(obs, means)
    obs_dev = move_array(reshape(obs, size(obs, 1) * size(obs, 2), B), device)
    r_dev = move_array(rraw, device)
    x0n_dev = move_array(apply_stats_tensor(x0, stats), device)
    return x0n_dev, r_dev, obs_dev
end

function agreement(pred::AbstractVector, target::AbstractVector)
    p = Float64.(Array(pred))
    t = Float64.(Array(target))
    rel = sqrt(mean((p .- t) .^ 2)) / max(sqrt(mean(t .^ 2)), eps(Float64))
    corr = dot(p, t) / max(norm(p) * norm(t), eps(Float64))
    return rel, corr
end

function evaluate_A(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_params, cond_kind::Symbol, target, p, device; lag_indices=nothing)
    rng = MersenneTwister(cfg.seed + 700)
    preds = Vector{Float32}[]
    refs = Vector{Float32}[]
    lags_all = Vector{Int}(target[:lags])
    active = lag_indices === nothing ? collect(eachindex(lags_all)) : collect(lag_indices)
    for li in active
        lag = lags_all[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, cfg.eval_pairs_per_lag, rng)
        pred = selected_prediction(model, x0, xt, tau_norm, score_model, stats, score_sigma,
            cond_model, cond_params, cond_kind, target, p, cfg, device)
        push!(preds, Float32.(Array(pred)))
        push!(refs, vec(Array{Float32}(target[:target_vec][li, :])))
    end
    pred = reduce(vcat, preds)
    ref = reduce(vcat, refs)
    rel, corr = agreement(pred, ref)
    return (; relative_rmse=rel, correlation=corr, pred=pred, target=ref)
end

function configured_lag_indices(cfg_path::AbstractString, nlags::Int)
    raw = TOML.parsefile(cfg_path)
    train = raw["training"]
    first = Int(get(train, "first_lag_index", 1))
    last = Int(get(train, "last_lag_index", 0))
    last = last <= 0 ? nlags : min(last, nlags)
    require_condition(1 <= first <= last <= nlags,
        "Invalid active lag window first_lag_index=$(first), last_lag_index=$(last), nlags=$(nlags).")
    return collect(first:last)
end

function configured_cache_pairs(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    train = raw["training"]
    return Int(get(train, "cache_pairs_per_lag", 0))
end

function build_training_cache(cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_cfg, cond_kind::Symbol, target, p, device, active_lag_indices, npairs::Int)
    npairs <= 0 && return nothing
    rng = MersenneTwister(cfg.seed + 2500)
    lags = Vector{Int}(target[:lags])
    x0n_cache = Array{Float32, 3}[]
    rraw_cache = Array{Float32, 3}[]
    obs_cache = Matrix{Float32}[]
    lag_to_cache = Dict{Int, Int}()
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    for (cache_idx, li) in enumerate(active_lag_indices)
        lag = lags[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, npairs, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats, cond_cfg, device;
            batch_size=min(cond_cfg.batch_size, npairs), score_model=score_model,
            score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        push!(x0n_cache, apply_stats_tensor(x0, stats))
        push!(rraw_cache, rraw)
        push!(obs_cache, reshape(obs, size(obs, 1) * size(obs, 2), npairs))
        lag_to_cache[li] = cache_idx
        cfg.verbose && @printf("Cached M-training tensors for lag %.5g (%d/%d), pairs=%d\n",
            lag * sampler.save_dt, cache_idx, length(active_lag_indices), npairs)
        GC.gc()
    end
    return DMTrainingCache(lag_to_cache, x0n_cache, rraw_cache, obs_cache, npairs)
end

function sample_cached_batch(cache::DMTrainingCache, li::Int, batch_pairs::Int,
        rng::AbstractRNG, device::ExecutionDevice)
    ci = cache.lag_to_cache[li]
    idx = rand(rng, 1:cache.npairs, batch_pairs)
    x0n_dev = move_array(copy(@view cache.x0n[ci][:, :, idx]), device)
    r_dev = move_array(copy(@view cache.rraw[ci][:, :, idx]), device)
    obs_dev = move_array(copy(@view cache.obsflat[ci][:, idx]), device)
    return x0n_dev, r_dev, obs_dev
end

function true_mobility_diagnostics(model, cfg, sampler, stats, target, p, device)
    raw = sample_raw_states_cond(sampler, 8000, MersenneTwister(cfg.seed + 800))
    xn = move_array(apply_stats_tensor(raw, stats), device)
    bp = block_params(model, xn)
    nb = length(Array(bp.l11))
    pred_blocks = Array{Float64}(undef, 3, 3, nb)
    l11, l21, l22 = Array(bp.l11), Array(bp.l21), Array(bp.l22)
    l31, l32, l33 = Array(bp.l31), Array(bp.l32), Array(bp.l33)
    k1, k2, k3 = Array(bp.k1), Array(bp.k2), Array(bp.k3)
    for q in 1:nb
        S = [l11[q]^2 l11[q]*l21[q] l11[q]*l31[q];
             l11[q]*l21[q] l21[q]^2+l22[q]^2 l21[q]*l31[q]+l22[q]*l32[q];
             l11[q]*l31[q] l21[q]*l31[q]+l22[q]*l32[q] l31[q]^2+l32[q]^2+l33[q]^2]
        K = [0.0 -k3[q] k2[q]; k3[q] 0.0 -k1[q]; -k2[q] k1[q] 0.0]
        pred_blocks[:, :, q] .= S .+ K
    end
    N, _, B = size(raw)
    true_blocks = Array{Float64}(undef, 3, 3, N * B)
    for b in 1:B, i in 1:N
        M = true_mobility_matrix(@view(raw[:, :, b]), p)
        rows = ((i - 1) * 3 + 1):(i * 3)
        true_blocks[:, :, (b - 1) * N + i] .= M[rows, rows]
    end
    rel = norm(vec(pred_blocks .- true_blocks)) / max(norm(vec(true_blocks)), eps(Float64))
    corr = dot(vec(pred_blocks), vec(true_blocks)) / max(norm(vec(pred_blocks)) * norm(vec(true_blocks)), eps(Float64))
    mean_pred = mean(pred_blocks; dims=3)[:, :, 1]
    mean_true = mean(true_blocks; dims=3)[:, :, 1]
    mean_phi = Matrix{Float64}(target[:Phi_block])
    return (; relative_rmse=rel, correlation=corr, mean_pred, mean_true, mean_phi,
        mean_phi_rel=norm(mean_pred - mean_phi) / max(norm(mean_phi), eps(Float64)))
end

function render_dm_figure(path, history, eval_metrics, true_metrics, cfg)
    fig = Figure(; size=(2600, 1800))
    Label(fig[0, 1:2], "SoftSpinLLGChain mobility NN diagnostics";
        fontsize=30, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Training loss", xlabel="epoch", ylabel="loss", yscale=log10)
    lines!(ax1, 1:length(history[:loss]), history[:loss]; color=STYLE_PRIMARY, linewidth=2)
    ax2 = Axis(fig[1, 2]; title="Validation A metrics", xlabel="eval", ylabel="metric")
    lines!(ax2, 1:length(history[:val_rel]), history[:val_rel]; color=STYLE_HIGHLIGHT, linewidth=2, label="rel.RMSE")
    lines!(ax2, 1:length(history[:val_corr]), history[:val_corr]; color=STYLE_PRIMARY, linewidth=2, label="corr")
    axislegend(ax2; position=:rb)
    mats = [true_metrics.mean_pred, true_metrics.mean_true, true_metrics.mean_phi,
        true_metrics.mean_pred - true_metrics.mean_true]
    titles = ["mean M_NN block", "mean M_true block", "Phi onsite block", "M_NN - M_true"]
    for j in 1:4
        ax = Axis(fig[2 + (j - 1) ÷ 2, 1 + (j - 1) % 2]; title=titles[j])
        heatmap!(ax, mats[j]; colormap=:balance)
    end
    Label(fig[4, 1:2],
        @sprintf("A val rel.RMSE %.4f corr %.4f; true-M ex-post rel.RMSE %.4f corr %.4f; mean-vs-Phi rel %.4f; feature=%s",
            eval_metrics.relative_rmse, eval_metrics.correlation,
            true_metrics.relative_rmse, true_metrics.correlation,
            true_metrics.mean_phi_rel, String(cfg.feature_mode));
        fontsize=22, tellwidth=false)
    save_figure_checked(path, fig)
end

function best_checkpoint_path(out_model::AbstractString)
    stem, ext = splitext(out_model)
    isempty(ext) && return stem * "_best.bson"
    return stem * "_best" * ext
end

function final_checkpoint_path(out_model::AbstractString)
    stem, ext = splitext(out_model)
    isempty(ext) && return stem * "_final.bson"
    return stem * "_final" * ext
end

function mobility_training_audit(target)
    if haskey(target, :target_kind) && Symbol(target[:target_kind]) == :oracle_trueM
        return "Oracle diagnostic: mobility loss used A targets generated with true M and Phi=<M_true> by explicit user request. This checkpoint is not data-only and must not be reported as a benchmark-success result."
    end
    return "Mobility loss used only data Cdot, data-only Phi GFDT with learned stationary score, learned conditional residual score, and mean Phi penalty. True M was used only for ex-post diagnostics."
end

function save_dm_checkpoint(path::AbstractString, model, cfg, history, epoch::Int,
        eval_metrics; final::Bool=false, audit_message::AbstractString="")
    ensure_parent_dir(path)
    isempty(audit_message) && (audit_message = "Mobility checkpoint selected only by data-driven A validation error. True M was not used for training or checkpoint selection.")
    BSON.bson(path, Dict(:host_model => Flux.fmap(cpu, model), :cfg => cfg,
        :history => history, :best_epoch => epoch, :eval_metrics => eval_metrics,
        :checkpoint_kind => final ? "final_epoch" : "best_validation",
        :target_artifact => cfg.target_artifact_bson,
        :no_cheating_audit => audit_message))
    return path
end

function train_dm(cfg_path::AbstractString)
    base = dirname(cfg_path)
    cfg = load_dm_config(cfg_path)
    device = detect_spin_device(cfg.device, cfg.required_gpu_name)
    activate_and_describe_device!(device, cfg.device, cfg.required_gpu_name)
    data_h5 = resolve_path(base, cfg.input_hdf5)
    score_path = resolve_path(base, cfg.score_bson)
    cond_path = resolve_path(base, cfg.cond_score_bson)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    cond_kind = configured_cond_score_kind(cfg_path)
    cond_model, cond_cfg = load_transition_source(cond_kind, cfg_path, cond_path, base, device)
    @printf("Using conditional transition source kind: %s\n", String(cond_kind))
    target = load_or_prepare_targets(cfg, base, device, cfg_path)
    audit_message = mobility_training_audit(target)
    rng = MersenneTwister(cfg.seed)
    init_bson = configured_init_bson(cfg_path)
    model = if isempty(strip(init_bson))
        move_model(build_mobility_model(cfg, rng, stats), device)
    else
        init_path = resolve_path(base, init_bson)
        require_condition(isfile(init_path), "Missing init_bson checkpoint $(init_path)")
        init_blob = BSON.load(init_path)
        @printf("Warm-starting mobility model from %s\n", init_path)
        move_model(init_blob[:host_model], device)
    end
    opt = Flux.setup(AdamW(cfg.learning_rate, (0.9, 0.999), cfg.weight_decay), model)
    history = Dict(:loss => Float64[], :fit_loss => Float64[], :mean_loss => Float64[],
        :val_rel => Float64[], :val_corr => Float64[])
    out_model = resolve_path(base, cfg.output_bson)
    best_model_host = nothing
    best_epoch = 0
    best_rel = Inf
    best_eval = nothing
    phi_block_cpu = Matrix{Float32}(target[:Phi_block])
    phi_vals = Tuple(vec(permutedims(phi_block_cpu, (2, 1))))
    phi_block_norm = Float32(max(norm(phi_block_cpu), 1f-6))
    phi_dev = move_array(Matrix{Float32}(target[:Phi]), device)
    selected_indices = target[:selected_indices]
    target_vec_dev = move_array(Array{Float32}(target[:target_vec]), device)
    scale_vec = move_array(Array{Float32}(target[:scale_vec]), device)
    lags = Vector{Int}(target[:lags])
    active_lag_indices = configured_lag_indices(cfg_path, length(lags))
    if cfg.verbose && active_lag_indices != collect(eachindex(lags))
        @printf("Active mobility-training lag indices: %d:%d of %d\n",
            first(active_lag_indices), last(active_lag_indices), length(lags))
    end
    cache_pairs = configured_cache_pairs(cfg_path)
    training_cache = build_training_cache(cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_cfg, cond_kind, target, p, device, active_lag_indices, cache_pairs)
    progress = ProgressMeter.Progress(cfg.epochs; desc="Training mobility NN")
    for epoch in 1:cfg.epochs
        losses = Float64[]
        fit_losses = Float64[]
        mean_losses = Float64[]
        for _ in 1:cfg.batches_per_epoch
            li = rand(rng, active_lag_indices)
            lag = lags[li]
            if training_cache === nothing
                x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, cfg.batch_pairs, rng)
                x0n_dev, r_dev, obs_dev = precompute_batch_inputs(x0, xt, tau_norm,
                    score_model, stats, score_sigma, cond_model, cond_cfg, cond_kind, target, p, cfg, device)
            else
                x0n_dev, r_dev, obs_dev = sample_cached_batch(training_cache, li,
                    cfg.batch_pairs, rng, device)
            end
            xmean = sample_raw_states_cond(sampler, cfg.mean_penalty_samples, rng)
            xmean_dev = move_array(apply_stats_tensor(xmean, stats), device)
            loss_val, grads = Flux.withgradient(model) do m
                pred = selected_prediction_precomputed(m, x0n_dev, r_dev, obs_dev, phi_dev, selected_indices)
                ref = @view target_vec_dev[li, :]
                fit_loss = mean(abs2, (pred .- ref) ./ scale_vec)
                mean_loss = mean_block_penalty(m, xmean_dev, phi_vals, phi_block_norm)
                fit_loss + Float32(cfg.mean_penalty_weight) * mean_loss
            end
            opt, model = Flux.update!(opt, model, grads[1])
            push!(losses, Float64(to_host(loss_val)))
        end
        push!(history[:loss], mean(losses))
        push!(history[:fit_loss], NaN)
        push!(history[:mean_loss], NaN)
        if epoch % cfg.eval_every == 0 || epoch == cfg.epochs
            ev = evaluate_A(model, cfg, sampler, score_model, stats, score_sigma,
                cond_model, cond_cfg, cond_kind, target, p, device; lag_indices=active_lag_indices)
            push!(history[:val_rel], ev.relative_rmse)
            push!(history[:val_corr], ev.correlation)
            @printf("epoch %d: loss %.6e, val A rel %.5f corr %.5f\n",
                epoch, history[:loss][end], ev.relative_rmse, ev.correlation)
            if ev.relative_rmse < best_rel
                best_rel = ev.relative_rmse
                best_epoch = epoch
                best_eval = ev
                best_model_host = Flux.fmap(cpu, model)
                save_dm_checkpoint(best_checkpoint_path(out_model), best_model_host,
                    cfg, history, best_epoch, best_eval; audit_message)
                @printf("  saved new best validation checkpoint at epoch %d (rel %.5f)\n",
                    best_epoch, best_rel)
            end
        end
        ProgressMeter.next!(progress; showvalues=[(:epoch, epoch), (:loss, history[:loss][end])])
    end
    ProgressMeter.finish!(progress)
    if best_model_host === nothing
        eval_metrics = evaluate_A(model, cfg, sampler, score_model, stats, score_sigma,
            cond_model, cond_cfg, cond_kind, target, p, device)
        best_model_host = Flux.fmap(cpu, model)
        best_eval = eval_metrics
        best_epoch = cfg.epochs
    else
        eval_metrics = best_eval
        if best_epoch != cfg.epochs
            save_dm_checkpoint(final_checkpoint_path(out_model), model, cfg, history,
                cfg.epochs, evaluate_A(model, cfg, sampler, score_model, stats, score_sigma,
                    cond_model, cond_cfg, cond_kind, target, p, device; lag_indices=active_lag_indices);
                final=true, audit_message)
            model = move_model(best_model_host, device)
        end
    end
    true_metrics = true_mobility_diagnostics(model, cfg, sampler, stats, target, p, device)
    ensure_parent_dir(out_model)
    BSON.bson(out_model, Dict(:host_model => best_model_host, :cfg => cfg,
        :history => history, :eval_metrics => eval_metrics,
        :true_metrics => true_metrics, :best_epoch => best_epoch,
        :target_artifact => cfg.target_artifact_bson,
        :no_cheating_audit => audit_message * " No Langevin forward validation was run by fit_dM.jl."))
    metrics_path = resolve_path(base, cfg.metrics_txt)
    open(metrics_path, "w") do io
        println(io, "SoftSpinLLGChain Step 3 mobility NN metrics")
        println(io, "config = $(basename(cfg_path))")
        println(io, "feature_mode = $(cfg.feature_mode)")
        println(io, "best_epoch = $(best_epoch)")
        println(io, "active_lag_indices = $(first(active_lag_indices)):$(last(active_lag_indices)) / $(length(lags))")
        println(io, @sprintf("A validation rel.RMSE = %.8e", eval_metrics.relative_rmse))
        println(io, @sprintf("A validation corr = %.8e", eval_metrics.correlation))
        println(io, @sprintf("true M block rel.RMSE ex-post = %.8e", true_metrics.relative_rmse))
        println(io, @sprintf("true M block corr ex-post = %.8e", true_metrics.correlation))
        println(io, @sprintf("mean M_NN vs Phi onsite rel.RMSE = %.8e", true_metrics.mean_phi_rel))
        println(io, "No Langevin equation was run.")
        println(io, "Audit: $(audit_message)")
    end
    render_dm_figure(resolve_path(base, cfg.figure_png), history, eval_metrics, true_metrics, cfg)
    @printf("Saved mobility model to %s\n", out_model)
    @printf("Saved metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_dm(length(ARGS) >= 1 ? ARGS[1] : DM_DEFAULT_CONFIG)
end
