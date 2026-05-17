#!/usr/bin/env julia

include(joinpath(@__DIR__, "prepare_oracle_trueM_dM_targets.jl"))
include(joinpath(@__DIR__, "right_observables.jl"))

using LinearAlgebra
using Printf
using Statistics

const DEFAULT_RIGHTOBS_DM_CONFIG = normpath(joinpath(@__DIR__, "..", "configs",
    "fit_dM_phys_pC_oracle_trueM_rightobs_v1_gpu0_equiv.toml"))

struct RightRetainedChannel
    observable::String
    right_observable::String
    data_rms::Float64
end

struct DMRightTrainingCache
    lag_to_cache::Dict{Int, Int}
    x0n::Vector{Array{Float32, 3}}
    rraw::Vector{Array{Float32, 3}}
    obsflat::Vector{Matrix{Float32}}
    right_grad::Vector{Array{Float32, 4}}
    npairs::Int
end

function load_right_retained_channels(path::AbstractString)
    parsed = TOML.parsefile(path)
    raw_channels = get(parsed, "channels", Any[])
    channels = RightRetainedChannel[]
    for ch in raw_channels
        rob = String(get(ch, "right_observable", get(ch, "target_observable", "")))
        isempty(rob) && error("Right-observable retained channel is missing right_observable.")
        push!(channels, RightRetainedChannel(String(ch["observable"]), rob,
            Float64(ch["data_rms"])))
    end
    return channels
end

unique_left_names(channels::Vector{RightRetainedChannel}) =
    unique([ch.observable for ch in channels])

unique_right_names(channels::Vector{RightRetainedChannel}) =
    unique([ch.right_observable for ch in channels])

function selected_right_indices(channels::Vector{RightRetainedChannel},
        left_index::Dict{String, Int}, right_index::Dict{String, Int}, N::Int)
    inds = Int[]
    channel_ids = Int[]
    nleft = length(left_index)
    nright = length(right_index)
    for (cid, ch) in enumerate(channels)
        a = left_index[ch.observable]
        b = right_index[ch.right_observable]
        for i in 1:N, j in 1:N
            row = (i - 1) * nleft + a
            col = (j - 1) * nright + b
            push!(inds, row + (col - 1) * (N * nleft))
            push!(channel_ids, cid)
        end
    end
    return inds, channel_ids
end

function full_action_to_local(action::AbstractMatrix{<:Real}, N::Int, B::Int)
    out = Array{Float32}(undef, N, 3, B)
    @inbounds for b in 1:B, i in 1:N, c in 1:3
        out[i, c, b] = Float32(action[(i - 1) * 3 + c, b])
    end
    return out
end

function rightobs_target_scales(channels::Vector{RightRetainedChannel},
        selected_channel_ids::Vector{Int})
    scale_vec = Array{Float32}(undef, length(selected_channel_ids))
    for (k, cid) in enumerate(selected_channel_ids)
        scale_vec[k] = Float32(max(channels[cid].data_rms, 1f-6))
    end
    return scale_vec
end

function prepare_oracle_trueM_rightobs_targets(cfg::DMConfig, base::AbstractString,
        device::ExecutionDevice, cfg_path::AbstractString)
    target_kind = configured_target_kind(cfg_path)
    require_condition(target_kind in (:oracle_truem_rightobs, :oracle_truem_phin),
        "Expected [targets] target_kind = \"oracle_trueM_rightobs\" in $(cfg_path).")

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
    @printf("Oracle right-observable target transition source kind: %s\n", String(cond_kind))

    channels = load_right_retained_channels(retained_path)
    left_names = unique_left_names(channels)
    right_names = unique_right_names(channels)
    left_lib = NonlinearLibrary(left_names)
    right_lib = RightObservableLibrary(right_names)
    left_index = Dict(name => i for (i, name) in enumerate(left_names))
    right_index = Dict(name => i for (i, name) in enumerate(right_names))
    selected_inds, selected_channel_ids = selected_right_indices(channels, left_index,
        right_index, sampler.N)
    left_means = estimate_nonlinear_means(sampler, p, left_lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 211))
    right_means = estimate_right_means(sampler, right_lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 212))
    Phi_true, Phi_block = oracle_true_mean_mobility(sampler, p, cfg.target_mean_samples,
        cfg.seed + 240)

    lags = sampler.lag_steps[1:min(cfg.max_lags, length(sampler.lag_steps))]
    nleft = length(left_names)
    nright = length(right_names)
    Ctrue_selected = Array{Float32}(undef, length(lags), length(selected_inds))
    Cphi_selected = similar(Ctrue_selected)
    target_vec = similar(Ctrue_selected)
    rng = MersenneTwister(cfg.seed + 220)
    for (li, lag) in enumerate(lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, cfg.target_pairs_per_lag, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats,
            cond_cfg, device; batch_size=min(cond_cfg.batch_size, cfg.target_pairs_per_lag),
            score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        left = nonlinear_observables(xt, p, left_lib)
        center_observables!(left, left_means)
        left_flat = reshape(left, sampler.N * nleft, cfg.target_pairs_per_lag)
        _, right_grad, _ = right_observable_value_grad_hess(x0, right_lib)
        true_action = oracle_true_action_batch(x0, rraw, p)
        phi_action = full_action_to_local(transpose(Phi_true) *
            Matrix{Float64}(flatten_batch(rraw)), sampler.N, cfg.target_pairs_per_lag)
        right_true = right_action_from_site_action(true_action, right_grad)
        right_phi = right_action_from_site_action(phi_action, right_grad)
        true_flat = reshape(right_true, sampler.N * nright, cfg.target_pairs_per_lag)
        phi_flat = reshape(right_phi, sampler.N * nright, cfg.target_pairs_per_lag)
        mat_true = -Matrix{Float64}(left_flat) * transpose(Matrix{Float64}(true_flat)) /
            cfg.target_pairs_per_lag
        mat_phi = -Matrix{Float64}(left_flat) * transpose(Matrix{Float64}(phi_flat)) /
            cfg.target_pairs_per_lag
        Ctrue_selected[li, :] .= Float32.(vec(mat_true)[selected_inds])
        Cphi_selected[li, :] .= Float32.(vec(mat_phi)[selected_inds])
        target_vec[li, :] .= Ctrue_selected[li, :] .- Cphi_selected[li, :]
        @printf("Prepared oracle right-observable target lag %.5g (%d/%d), left=%d right=%d pairs=%d\n",
            lag * sampler.save_dt, li, length(lags), nleft, nright, cfg.target_pairs_per_lag)
        GC.gc()
    end

    scale_vec = rightobs_target_scales(channels, selected_channel_ids)

    ensure_parent_dir(out_path)
    BSON.bson(out_path, Dict(:target_kind => :oracle_trueM_rightobs,
        :names => left_names, :right_names => right_names, :channels => channels,
        :observable_means => left_means, :right_observable_means => right_means,
        :lags => lags, :taus => lags .* sampler.save_dt,
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Cdot_trueM_cond_selected => Ctrue_selected,
        :Cdot_phi_trueMmean_cond_selected => Cphi_selected,
        :Cdot_data_selected => Ctrue_selected, :Cdot_phi_selected => Cphi_selected,
        :A_target_selected => target_vec,
        :Phi => Matrix{Float32}(Phi_true), :Phi_block => Matrix{Float32}(Phi_block),
        :save_dt => sampler.save_dt, :N => sampler.N, :D => sampler.D,
        :conditional_source_kind => cond_kind,
        :conditional_source_bson => cfg.cond_score_bson,
        :audit => "Oracle diagnostic target with generalized right observables: true M and Phi=<M_true> intentionally entered Cdot and A by explicit user request. Not data-only."))
    @printf("Saved oracle right-observable target artifact to %s\n", out_path)
    return BSON.load(out_path)
end

function prepare_data_rightobs_targets(cfg::DMConfig, base::AbstractString,
        device::ExecutionDevice, cfg_path::AbstractString)
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
    Phi_block = haskey(phi_blob, :Phi_projected_block) ?
        Matrix{Float32}(phi_blob[:Phi_projected_block]) :
        Matrix{Float32}(Phi[1:3, 1:3])

    channels = load_right_retained_channels(retained_path)
    left_names = unique_left_names(channels)
    right_names = unique_right_names(channels)
    left_lib = NonlinearLibrary(left_names)
    right_lib = RightObservableLibrary(right_names)
    left_index = Dict(name => i for (i, name) in enumerate(left_names))
    right_index = Dict(name => i for (i, name) in enumerate(right_names))
    selected_inds, selected_channel_ids = selected_right_indices(channels, left_index,
        right_index, sampler.N)
    left_means = estimate_nonlinear_means(sampler, p, left_lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 311))
    right_means = estimate_right_means(sampler, right_lib, cfg.target_mean_samples,
        MersenneTwister(cfg.seed + 312))

    lags = sampler.lag_steps[1:min(cfg.max_lags, length(sampler.lag_steps))]
    nleft = length(left_names)
    nright = length(right_names)
    Cdata_selected = Array{Float32}(undef, length(lags), length(selected_inds))
    Cphi_selected = similar(Cdata_selected)
    target_vec = similar(Cdata_selected)
    rng = MersenneTwister(cfg.seed + 320)
    for (li, lag) in enumerate(lags)
        x0, xt, xp, xm, _ = sample_fixed_lag_window(sampler, lag,
            cfg.target_pairs_per_lag, rng)
        right_vals, right_grad, right_hess = right_observable_value_grad_hess(x0,
            right_lib)
        center_right_values!(right_vals, right_means)
        right_flat = reshape(right_vals, sampler.N * nright, cfg.target_pairs_per_lag)

        leftp = nonlinear_observables(xp, p, left_lib)
        leftm = nonlinear_observables(xm, p, left_lib)
        left_deriv = (leftp .- leftm) ./ Float32(2.0 * sampler.save_dt)
        deriv_flat = reshape(left_deriv, sampler.N * nleft, cfg.target_pairs_per_lag)
        mat_data = Matrix{Float64}(deriv_flat) * transpose(Matrix{Float64}(right_flat)) /
            cfg.target_pairs_per_lag

        left = nonlinear_observables(xt, p, left_lib)
        center_observables!(left, left_means)
        left_flat = reshape(left, sampler.N * nleft, cfg.target_pairs_per_lag)
        sraw = evaluate_raw_score_local(score_model, x0, stats, score_sigma, device;
            batch_size=4096)
        Bphi = right_phi_generator_terms(Phi, sraw, right_grad, right_hess)
        Bphi_flat = reshape(Bphi, sampler.N * nright, cfg.target_pairs_per_lag)
        mat_phi = Matrix{Float64}(left_flat) * transpose(Matrix{Float64}(Bphi_flat)) /
            cfg.target_pairs_per_lag
        Cdata_selected[li, :] .= Float32.(vec(mat_data)[selected_inds])
        Cphi_selected[li, :] .= Float32.(vec(mat_phi)[selected_inds])
        target_vec[li, :] .= Cdata_selected[li, :] .- Cphi_selected[li, :]
        @printf("Prepared data-only right-observable target lag %.5g (%d/%d), left=%d right=%d pairs=%d\n",
            lag * sampler.save_dt, li, length(lags), nleft, nright, cfg.target_pairs_per_lag)
        GC.gc()
    end

    scale_vec = rightobs_target_scales(channels, selected_channel_ids)

    ensure_parent_dir(out_path)
    BSON.bson(out_path, Dict(:target_kind => :data_only_rightobs,
        :names => left_names, :right_names => right_names, :channels => channels,
        :observable_means => left_means, :right_observable_means => right_means,
        :lags => lags, :taus => lags .* sampler.save_dt,
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Cdot_data_selected => Cdata_selected,
        :Cdot_phi_selected => Cphi_selected,
        :A_target_selected => target_vec,
        :Phi => Matrix{Float32}(Phi), :Phi_block => Phi_block,
        :save_dt => sampler.save_dt, :N => sampler.N, :D => sampler.D,
        :audit => "Data-only generalized-right-observable target: Cdot_data used trajectory finite differences; Cdot_phi used learned stationary score, data-only Phi, and analytic gradients/Hessians of chosen observables only. True M did not enter this artifact."))
    @printf("Saved data-only right-observable target artifact to %s\n", out_path)
    return BSON.load(out_path)
end

function load_or_prepare_right_targets(cfg::DMConfig, base::AbstractString,
        device::ExecutionDevice, cfg_path::AbstractString)
    path = resolve_path(base, cfg.target_artifact_bson)
    if isfile(path) && !cfg.prepare_targets
        return BSON.load(path)
    end
    target_kind = configured_target_kind(cfg_path)
    if target_kind in (:oracle_truem_rightobs, :oracle_truem_phin)
        return prepare_oracle_trueM_rightobs_targets(cfg, base, device, cfg_path)
    elseif target_kind in (:data_only_rightobs, :data_rightobs)
        return prepare_data_rightobs_targets(cfg, base, device, cfg_path)
    end
    error("Unsupported right-observable target_kind=$(target_kind).")
end

function precompute_right_batch_inputs(x0, xt, tau_norm, score_model, stats, score_sigma,
        cond_model, cond_params, cond_kind::Symbol, target, p, cfg, device)
    left_lib = NonlinearLibrary(Vector{String}(target[:names]))
    right_lib = RightObservableLibrary(Vector{String}(target[:right_names]))
    left_means = Vector{Float64}(target[:observable_means])
    B = size(x0, 3)
    rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats,
        cond_params, device; batch_size=min(cond_params.batch_size, B),
        score_model=score_model, score_sigma=score_sigma)
    rraw = normalized_residual_to_raw(rnorm, stats)
    left = nonlinear_observables(xt, p, left_lib)
    center_observables!(left, left_means)
    _, right_grad, _ = right_observable_value_grad_hess(x0, right_lib)
    x0n_dev = move_array(apply_stats_tensor(x0, stats), device)
    r_dev = move_array(rraw, device)
    left_dev = move_array(reshape(left, size(left, 1) * size(left, 2), B), device)
    grad_dev = move_array(right_grad, device)
    return x0n_dev, r_dev, left_dev, grad_dev
end

function precompute_right_batch_inputs_host(x0, xt, tau_norm, score_model, stats,
        score_sigma, cond_model, cond_params, cond_kind::Symbol, target, p, cfg, device)
    left_lib = NonlinearLibrary(Vector{String}(target[:names]))
    right_lib = RightObservableLibrary(Vector{String}(target[:right_names]))
    left_means = Vector{Float64}(target[:observable_means])
    B = size(x0, 3)
    rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats,
        cond_params, device; batch_size=min(cond_params.batch_size, B),
        score_model=score_model, score_sigma=score_sigma)
    rraw = normalized_residual_to_raw(rnorm, stats)
    left = nonlinear_observables(xt, p, left_lib)
    center_observables!(left, left_means)
    _, right_grad, _ = right_observable_value_grad_hess(x0, right_lib)
    return apply_stats_tensor(x0, stats), rraw,
        reshape(left, size(left, 1) * size(left, 2), B), right_grad
end

function selected_prediction_precomputed_right(model, x0n_dev, r_dev, left_dev,
        grad_dev, phi_dev, selected_indices)
    N = size(r_dev, 1)
    B = size(r_dev, 3)
    model_action = mobility_action(model, x0n_dev, r_dev)
    phi_action = phi_dev' * reshape(permutedims(r_dev, (2, 1, 3)), size(phi_dev, 1), B)
    model_flat = reshape(permutedims(model_action, (2, 1, 3)), size(phi_dev, 1), B)
    delta_flat = model_flat .- phi_action
    delta = permutedims(reshape(delta_flat, 3, N, B), (2, 1, 3))
    q = dropdims(sum(reshape(delta, N, 1, 3, B) .* grad_dev; dims=3); dims=3)
    right_flat = reshape(q, size(q, 1) * size(q, 2), B)
    mat = -(left_dev * transpose(right_flat)) ./ eltype(delta_flat)(B)
    return mat[selected_indices]
end

function sample_cached_right_batch(cache::DMRightTrainingCache, li::Int,
        batch_pairs::Int, rng::AbstractRNG, device::ExecutionDevice)
    ci = cache.lag_to_cache[li]
    idx = rand(rng, 1:cache.npairs, batch_pairs)
    x0n_dev = move_array(copy(@view cache.x0n[ci][:, :, idx]), device)
    r_dev = move_array(copy(@view cache.rraw[ci][:, :, idx]), device)
    obs_dev = move_array(copy(@view cache.obsflat[ci][:, idx]), device)
    grad_dev = move_array(copy(@view cache.right_grad[ci][:, :, :, idx]), device)
    return x0n_dev, r_dev, obs_dev, grad_dev
end

function build_right_training_cache(cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_cfg, cond_kind::Symbol, target, p, device, active_lag_indices,
        npairs::Int)
    npairs <= 0 && return nothing
    rng = MersenneTwister(cfg.seed + 3500)
    lags = Vector{Int}(target[:lags])
    x0n_cache = Array{Float32, 3}[]
    rraw_cache = Array{Float32, 3}[]
    obs_cache = Matrix{Float32}[]
    grad_cache = Array{Float32, 4}[]
    lag_to_cache = Dict{Int, Int}()
    for (cache_idx, li) in enumerate(active_lag_indices)
        lag = lags[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, npairs, rng)
        x0n, rraw, obsflat, grad = precompute_right_batch_inputs_host(x0, xt, tau_norm,
            score_model, stats, score_sigma, cond_model, cond_cfg, cond_kind, target,
            p, cfg, device)
        push!(x0n_cache, x0n)
        push!(rraw_cache, rraw)
        push!(obs_cache, obsflat)
        push!(grad_cache, grad)
        lag_to_cache[li] = cache_idx
        cfg.verbose && @printf("Cached right-observable M tensors for lag %.5g (%d/%d), pairs=%d\n",
            lag * sampler.save_dt, cache_idx, length(active_lag_indices), npairs)
        GC.gc()
    end
    return DMRightTrainingCache(lag_to_cache, x0n_cache, rraw_cache, obs_cache,
        grad_cache, npairs)
end

function evaluate_A_right(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_params, cond_kind::Symbol, target, p, device; lag_indices=nothing)
    rng = MersenneTwister(cfg.seed + 1700)
    preds = Vector{Float32}[]
    refs = Vector{Float32}[]
    lags_all = Vector{Int}(target[:lags])
    active = lag_indices === nothing ? collect(eachindex(lags_all)) : collect(lag_indices)
    phi_dev = move_array(Matrix{Float32}(target[:Phi]), device)
    selected_indices = target[:selected_indices]
    for li in active
        lag = lags_all[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag,
            cfg.eval_pairs_per_lag, rng)
        x0n_dev, r_dev, left_dev, grad_dev = precompute_right_batch_inputs(x0, xt,
            tau_norm, score_model, stats, score_sigma, cond_model, cond_params,
            cond_kind, target, p, cfg, device)
        pred = selected_prediction_precomputed_right(model, x0n_dev, r_dev, left_dev,
            grad_dev, phi_dev, selected_indices)
        push!(preds, Float32.(Array(pred)))
        push!(refs, vec(Array{Float32}(target[:target_vec][li, :])))
    end
    pred = reduce(vcat, preds)
    ref = reduce(vcat, refs)
    rel, corr = agreement(pred, ref)
    return (; relative_rmse=rel, correlation=corr, pred=pred, target=ref)
end

function train_dm_rightobs(cfg_path::AbstractString)
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
    cond_model, cond_cfg = load_transition_source(cond_kind, cfg_path, cond_path, base,
        device)
    @printf("Using conditional transition source kind: %s\n", String(cond_kind))
    target = load_or_prepare_right_targets(cfg, base, device, cfg_path)
    audit_message = mobility_training_audit(target)
    if Symbol(target[:target_kind]) == :oracle_trueM_rightobs
        audit_message = "Oracle diagnostic: mobility loss used generalized-right-observable A targets generated with true M and Phi=<M_true> by explicit user request. This checkpoint is not data-only."
    elseif Symbol(target[:target_kind]) == :data_only_rightobs
        audit_message = "Mobility loss used only trajectory finite-difference Cdot, data-only Phi GFDT with learned stationary score, learned conditional residual score, and mean Phi penalty. True M was used only for ex-post diagnostics."
    end
    if !cfg.train
        @printf("Prepared/loaded right-observable target and stopped because [run].train=false.\n")
        return nothing
    end

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
    cache_pairs = configured_cache_pairs(cfg_path)
    training_cache = build_right_training_cache(cfg, sampler, score_model, stats,
        score_sigma, cond_model, cond_cfg, cond_kind, target, p, device,
        active_lag_indices, cache_pairs)

    progress = ProgressMeter.Progress(cfg.epochs; desc="Training right-observable mobility NN")
    for epoch in 1:cfg.epochs
        losses = Float64[]
        for _ in 1:cfg.batches_per_epoch
            li = rand(rng, active_lag_indices)
            lag = lags[li]
            if training_cache === nothing
                x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag,
                    cfg.batch_pairs, rng)
                x0n_dev, r_dev, left_dev, grad_dev = precompute_right_batch_inputs(x0,
                    xt, tau_norm, score_model, stats, score_sigma, cond_model, cond_cfg,
                    cond_kind, target, p, cfg, device)
            else
                x0n_dev, r_dev, left_dev, grad_dev = sample_cached_right_batch(
                    training_cache, li, cfg.batch_pairs, rng, device)
            end
            xmean = sample_raw_states_cond(sampler, cfg.mean_penalty_samples, rng)
            xmean_dev = move_array(apply_stats_tensor(xmean, stats), device)
            loss_val, grads = Flux.withgradient(model) do m
                pred = selected_prediction_precomputed_right(m, x0n_dev, r_dev, left_dev,
                    grad_dev, phi_dev, selected_indices)
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
            ev = evaluate_A_right(model, cfg, sampler, score_model, stats, score_sigma,
                cond_model, cond_cfg, cond_kind, target, p, device;
                lag_indices=active_lag_indices)
            push!(history[:val_rel], ev.relative_rmse)
            push!(history[:val_corr], ev.correlation)
            @printf("epoch %d: loss %.6e, rightobs val A rel %.5f corr %.5f\n",
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
        best_eval = evaluate_A_right(model, cfg, sampler, score_model, stats, score_sigma,
            cond_model, cond_cfg, cond_kind, target, p, device;
            lag_indices=active_lag_indices)
        best_model_host = Flux.fmap(cpu, model)
        best_epoch = cfg.epochs
    else
        if best_epoch != cfg.epochs
            save_dm_checkpoint(final_checkpoint_path(out_model), model, cfg, history,
                cfg.epochs, evaluate_A_right(model, cfg, sampler, score_model, stats,
                    score_sigma, cond_model, cond_cfg, cond_kind, target, p, device;
                    lag_indices=active_lag_indices); final=true, audit_message)
            model = move_model(best_model_host, device)
        end
    end
    true_metrics = true_mobility_diagnostics(model, cfg, sampler, stats, target, p, device)
    ensure_parent_dir(out_model)
    BSON.bson(out_model, Dict(:host_model => best_model_host, :cfg => cfg,
        :history => history, :eval_metrics => best_eval,
        :true_metrics => true_metrics, :best_epoch => best_epoch,
        :target_artifact => cfg.target_artifact_bson,
        :right_observable_training => true,
        :no_cheating_audit => audit_message * " No Langevin forward validation was run by fit_dM_rightobs.jl."))

    metrics_path = resolve_path(base, cfg.metrics_txt)
    open(metrics_path, "w") do io
        println(io, "SoftSpinLLGChain generalized right-observable mobility NN metrics")
        println(io, "config = $(basename(cfg_path))")
        println(io, "feature_mode = $(cfg.feature_mode)")
        println(io, "target_kind = $(target[:target_kind])")
        println(io, "left_observables = $(length(target[:names]))")
        println(io, "right_observables = $(length(target[:right_names]))")
        println(io, "best_epoch = $(best_epoch)")
        println(io, "active_lag_indices = $(first(active_lag_indices)):$(last(active_lag_indices)) / $(length(lags))")
        println(io, @sprintf("A validation rel.RMSE = %.8e", best_eval.relative_rmse))
        println(io, @sprintf("A validation corr = %.8e", best_eval.correlation))
        println(io, @sprintf("true M block rel.RMSE ex-post = %.8e", true_metrics.relative_rmse))
        println(io, @sprintf("true M block corr ex-post = %.8e", true_metrics.correlation))
        println(io, @sprintf("mean M_NN vs Phi onsite rel.RMSE = %.8e", true_metrics.mean_phi_rel))
        println(io, "Audit: $(audit_message)")
    end
    render_dm_figure(resolve_path(base, cfg.figure_png), history, best_eval, true_metrics, cfg)
    @printf("Saved right-observable mobility model to %s\n", out_model)
    @printf("Saved metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_dm_rightobs(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RIGHTOBS_DM_CONFIG)
end
