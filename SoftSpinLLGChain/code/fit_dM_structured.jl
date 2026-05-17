#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using LinearAlgebra
using Printf
using Statistics

const STRUCT_DEFAULT_CONFIG = normpath(joinpath(@__DIR__, "..", "configs", "fit_dM_struct_gpu2_vS1.toml"))

struct StructuredMobilityNN{M}
    mlp::M
    feature_mode::Symbol
    sym_scale::Float32
    skew_scale::Float32
    sym_floor::Float32
end

Flux.@functor StructuredMobilityNN (mlp,)

function structured_feature_tensor(raw, stats::DataStats, mode::Symbol)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(raw, 1), size(raw, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(raw, 1), size(raw, 2), 1)
    if raw isa CUDA.CuArray
        mean_tensor = cu(mean_tensor)
        std_tensor = cu(std_tensor)
    end
    xn = (raw .- mean_tensor) ./ std_tensor
    r2 = sum(abs2, raw; dims=2)
    if mode == :radial
        return r2
    elseif mode == :local
        return xn
    elseif mode == :local_r2
        return cat(xn, r2; dims=2)
    elseif mode == :neighbor_r2
        xm = circshift(xn, (1, 0, 0))
        xp = circshift(xn, (-1, 0, 0))
        return cat(xm, xn, xp, r2; dims=2)
    else
        error("Unknown structured feature mode $(mode)")
    end
end

function build_structured_model(cfg::DMConfig, rng::AbstractRNG)
    feature_dim = cfg.feature_mode == :radial ? 1 :
        cfg.feature_mode == :local ? 3 :
        cfg.feature_mode == :local_r2 ? 4 :
        cfg.feature_mode == :neighbor_r2 ? 10 :
        error("Unknown structured feature_mode $(cfg.feature_mode)")
    layers = Any[Dense(feature_dim => cfg.hidden_width, swish)]
    for _ in 2:cfg.hidden_depth
        push!(layers, Dense(cfg.hidden_width => cfg.hidden_width, swish))
    end
    push!(layers, Dense(cfg.hidden_width => 3))
    return StructuredMobilityNN(Chain(layers...), cfg.feature_mode,
        cfg.sym_scale, cfg.skew_scale, cfg.sym_floor)
end

function structured_coefficients(model::StructuredMobilityNN, raw, stats::DataStats)
    N, _, B = size(raw)
    feats = structured_feature_tensor(raw, stats, model.feature_mode)
    F = size(feats, 2)
    flat = reshape(permutedims(feats, (2, 1, 3)), F, N * B)
    y = model.mlp(flat)
    sscale = eltype(y)(model.sym_scale)
    kscale = eltype(y)(model.skew_scale)
    floor = eltype(y)(model.sym_floor)
    lambda_perp = NNlib.softplus.(y[1, :]) .* sscale .+ floor
    lambda_para = NNlib.softplus.(y[2, :]) .* sscale .+ floor
    beta = y[3, :] .* kscale
    return (; N, B, lambda_perp, lambda_para, beta)
end

function structured_mobility_action(model::StructuredMobilityNN, raw, stats::DataStats, rraw)
    coeff = structured_coefficients(model, raw, stats)
    N, B = coeff.N, coeff.B
    x = reshape(permutedims(raw, (2, 1, 3)), 3, N * B)
    r = reshape(permutedims(rraw, (2, 1, 3)), 3, N * B)
    x1, x2, x3 = x[1, :], x[2, :], x[3, :]
    r1, r2, r3 = r[1, :], r[2, :], r[3, :]
    rsq = x1 .* x1 .+ x2 .* x2 .+ x3 .* x3
    inv_rsq = 1 ./ max.(rsq, eltype(rsq)(1f-6))
    dotxr = x1 .* r1 .+ x2 .* r2 .+ x3 .* r3
    anis = (coeff.lambda_para .- coeff.lambda_perp) .* dotxr .* inv_rsq
    c1 = x2 .* r3 .- x3 .* r2
    c2 = x3 .* r1 .- x1 .* r3
    c3 = x1 .* r2 .- x2 .* r1
    a1 = coeff.lambda_perp .* r1 .+ anis .* x1 .+ coeff.beta .* c1
    a2 = coeff.lambda_perp .* r2 .+ anis .* x2 .+ coeff.beta .* c2
    a3 = coeff.lambda_perp .* r3 .+ anis .* x3 .+ coeff.beta .* c3
    out = reshape(vcat(reshape(a1, 1, :), reshape(a2, 1, :), reshape(a3, 1, :)), 3, N, B)
    return permutedims(out, (2, 1, 3))
end

function structured_block_entries(model::StructuredMobilityNN, raw, stats::DataStats)
    coeff = structured_coefficients(model, raw, stats)
    N, B = coeff.N, coeff.B
    x = reshape(permutedims(raw, (2, 1, 3)), 3, N * B)
    x1, x2, x3 = x[1, :], x[2, :], x[3, :]
    rsq = x1 .* x1 .+ x2 .* x2 .+ x3 .* x3
    inv_rsq = 1 ./ max.(rsq, eltype(rsq)(1f-6))
    anis = (coeff.lambda_para .- coeff.lambda_perp) .* inv_rsq
    s11 = coeff.lambda_perp .+ anis .* x1 .* x1
    s12 = anis .* x1 .* x2
    s13 = anis .* x1 .* x3
    s22 = coeff.lambda_perp .+ anis .* x2 .* x2
    s23 = anis .* x2 .* x3
    s33 = coeff.lambda_perp .+ anis .* x3 .* x3
    return (; lambda_perp=coeff.lambda_perp, lambda_para=coeff.lambda_para, beta=coeff.beta,
        m11=s11, m12=s12 .- coeff.beta .* x3, m13=s13 .+ coeff.beta .* x2,
        m21=s12 .+ coeff.beta .* x3, m22=s22, m23=s23 .- coeff.beta .* x1,
        m31=s13 .- coeff.beta .* x2, m32=s23 .+ coeff.beta .* x1, m33=s33)
end

function structured_mean_block(model::StructuredMobilityNN, raw, stats::DataStats)
    e = structured_block_entries(model, raw, stats)
    vals = [mean(e.m11), mean(e.m12), mean(e.m13),
        mean(e.m21), mean(e.m22), mean(e.m23),
        mean(e.m31), mean(e.m32), mean(e.m33)]
    return reshape(reduce(vcat, vals), 3, 3)
end

function structured_mean_penalty(model::StructuredMobilityNN, raw, stats::DataStats,
        phi_vals::NTuple{9, Float32}, phi_norm::Float32)
    e = structured_block_entries(model, raw, stats)
    invnorm = 1f0 / phi_norm
    terms = (
        (mean(e.m11) - phi_vals[1]) * invnorm,
        (mean(e.m12) - phi_vals[2]) * invnorm,
        (mean(e.m13) - phi_vals[3]) * invnorm,
        (mean(e.m21) - phi_vals[4]) * invnorm,
        (mean(e.m22) - phi_vals[5]) * invnorm,
        (mean(e.m23) - phi_vals[6]) * invnorm,
        (mean(e.m31) - phi_vals[7]) * invnorm,
        (mean(e.m32) - phi_vals[8]) * invnorm,
        (mean(e.m33) - phi_vals[9]) * invnorm,
    )
    return sum(t -> t * t, terms) / 9
end

function selected_prediction_structured(model, raw_dev, r_dev, obs_dev, phi_dev, stats,
        selected_indices)
    B = size(r_dev, 3)
    model_action = structured_mobility_action(model, raw_dev, stats, r_dev)
    phi_action = phi_dev' * reshape(permutedims(r_dev, (2, 1, 3)), size(phi_dev, 1), B)
    model_flat = reshape(permutedims(model_action, (2, 1, 3)), size(phi_dev, 1), B)
    delta_action = model_flat .- phi_action
    mat = -(obs_dev * transpose(delta_action)) ./ eltype(delta_action)(B)
    return mat[selected_indices]
end

function target_rms_scale(target)
    t = Array{Float32}(target[:target_vec])
    scale = vec(sqrt.(mean(abs2, t; dims=1)))
    return max.(scale, Float32(1f-5))
end

function evaluate_A_structured(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_params, target, p, device; lag_indices=nothing)
    rng = MersenneTwister(cfg.seed + 700)
    phi_dev = move_array(Matrix{Float32}(target[:Phi]), device)
    selected_indices = target[:selected_indices]
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    preds = Vector{Float32}[]
    refs = Vector{Float32}[]
    lags_all = Vector{Int}(target[:lags])
    active = lag_indices === nothing ? collect(eachindex(lags_all)) : collect(lag_indices)
    for li in active
        lag = lags_all[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, cfg.eval_pairs_per_lag, rng)
        B = size(x0, 3)
        rnorm = evaluate_residual_norm(cond_model, x0, xt, tau_norm, stats, cond_params, device;
            batch_size=min(cond_params.batch_size, B), score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_dev = move_array(reshape(obs, size(obs, 1) * size(obs, 2), B), device)
        pred = selected_prediction_structured(model, move_array(x0, device), move_array(rraw, device),
            obs_dev, phi_dev, stats, selected_indices)
        push!(preds, Float32.(Array(pred)))
        push!(refs, vec(Array{Float32}(target[:target_vec][li, :])))
    end
    pred = reduce(vcat, preds)
    ref = reduce(vcat, refs)
    rel, corr = agreement(pred, ref)
    return (; relative_rmse=rel, correlation=corr, pred=pred, target=ref)
end

function structured_true_diagnostics(model, cfg, sampler, stats, target, p, device; nsamples::Int=12000)
    raw = sample_raw_states_cond(sampler, nsamples, MersenneTwister(cfg.seed + 800))
    e = structured_block_entries(model, move_array(raw, device), stats)
    nb = length(Array(e.m11))
    pred_blocks = Array{Float64}(undef, 3, 3, nb)
    vals = (Array(e.m11), Array(e.m12), Array(e.m13), Array(e.m21), Array(e.m22),
        Array(e.m23), Array(e.m31), Array(e.m32), Array(e.m33))
    @inbounds for q in 1:nb
        pred_blocks[:, :, q] .= [vals[1][q] vals[2][q] vals[3][q];
            vals[4][q] vals[5][q] vals[6][q]; vals[7][q] vals[8][q] vals[9][q]]
    end
    N, _, B = size(raw)
    true_blocks = Array{Float64}(undef, 3, 3, N * B)
    for b in 1:B, i in 1:N
        M = true_mobility_matrix(@view(raw[:, :, b]), p)
        rows = ((i - 1) * 3 + 1):(i * 3)
        true_blocks[:, :, (b - 1) * N + i] .= M[rows, rows]
    end
    rel = norm(vec(pred_blocks .- true_blocks)) / max(norm(vec(true_blocks)), eps(Float64))
    corr = dot(vec(pred_blocks), vec(true_blocks)) /
        max(norm(vec(pred_blocks)) * norm(vec(true_blocks)), eps(Float64))
    mean_pred = mean(pred_blocks; dims=3)[:, :, 1]
    mean_true = mean(true_blocks; dims=3)[:, :, 1]
    mean_phi = Matrix{Float64}(target[:Phi_block])
    return (; relative_rmse=rel, correlation=corr, mean_pred, mean_true, mean_phi,
        mean_phi_rel=norm(mean_pred - mean_phi) / max(norm(mean_phi), eps(Float64)))
end

function train_structured_dm(cfg_path::AbstractString)
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
    cond_cfg = load_config(resolve_path(base, "cond_score_gpu0_vA.toml"))
    cond_blob = BSON.load(cond_path)
    cond_model = move_model(cond_blob[:host_model], device)
    Flux.testmode!(cond_model)
    target = load_or_prepare_targets(cfg, base, device)
    rng = MersenneTwister(cfg.seed)
    model = move_model(build_structured_model(cfg, rng), device)
    opt = Flux.setup(AdamW(cfg.learning_rate, (0.9, 0.999), cfg.weight_decay), model)
    history = Dict(:loss => Float64[], :val_rel => Float64[], :val_corr => Float64[],
        :best_epoch => Int[], :best_rel => Float64[])
    phi_block_cpu = Matrix{Float32}(target[:Phi_block])
    phi_vals = Tuple(vec(permutedims(phi_block_cpu, (2, 1))))
    phi_block_norm = Float32(max(norm(phi_block_cpu), 1f-6))
    phi_dev = move_array(Matrix{Float32}(target[:Phi]), device)
    selected_indices = target[:selected_indices]
    target_vec_dev = move_array(Array{Float32}(target[:target_vec]), device)
    scale_vec = move_array(target_rms_scale(target), device)
    lags = Vector{Int}(target[:lags])
    active_lag_indices = configured_lag_indices(cfg_path, length(lags))
    if cfg.verbose && active_lag_indices != collect(eachindex(lags))
        @printf("Active structured mobility-training lag indices: %d:%d of %d\n",
            first(active_lag_indices), last(active_lag_indices), length(lags))
    end
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    best_rel = Inf
    best_epoch = 0
    best_model_cpu = nothing
    progress = ProgressMeter.Progress(cfg.epochs; desc="Training structured mobility NN")
    for epoch in 1:cfg.epochs
        losses = Float64[]
        for _ in 1:cfg.batches_per_epoch
            li = rand(rng, active_lag_indices)
            lag = lags[li]
            x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, cfg.batch_pairs, rng)
            B = size(x0, 3)
            rnorm = evaluate_residual_norm(cond_model, x0, xt, tau_norm, stats, cond_cfg, device;
                batch_size=min(cond_cfg.batch_size, B), score_model=score_model, score_sigma=score_sigma)
            rraw = normalized_residual_to_raw(rnorm, stats)
            obs = nonlinear_observables(xt, p, lib)
            center_observables!(obs, means)
            raw_dev = move_array(x0, device)
            r_dev = move_array(rraw, device)
            obs_dev = move_array(reshape(obs, size(obs, 1) * size(obs, 2), B), device)
            xmean = move_array(sample_raw_states_cond(sampler, cfg.mean_penalty_samples, rng), device)
            loss_val, grads = Flux.withgradient(model) do m
                pred = selected_prediction_structured(m, raw_dev, r_dev, obs_dev,
                    phi_dev, stats, selected_indices)
                ref = @view target_vec_dev[li, :]
                fit_loss = mean(abs2, (pred .- ref) ./ scale_vec)
                mean_loss = structured_mean_penalty(m, xmean, stats, phi_vals, phi_block_norm)
                fit_loss + Float32(cfg.mean_penalty_weight) * mean_loss
            end
            opt, model = Flux.update!(opt, model, grads[1])
            push!(losses, Float64(to_host(loss_val)))
        end
        push!(history[:loss], mean(losses))
        if epoch % cfg.eval_every == 0 || epoch == cfg.epochs
            ev = evaluate_A_structured(model, cfg, sampler, score_model, stats, score_sigma,
                cond_model, cond_cfg, target, p, device; lag_indices=active_lag_indices)
            push!(history[:val_rel], ev.relative_rmse)
            push!(history[:val_corr], ev.correlation)
            if ev.relative_rmse < best_rel
                best_rel = ev.relative_rmse
                best_epoch = epoch
                best_model_cpu = Flux.fmap(cpu, model)
            end
            push!(history[:best_epoch], best_epoch)
            push!(history[:best_rel], best_rel)
            @printf("epoch %d: loss %.6e, val A rel %.5f corr %.5f, best %.5f @ %d\n",
                epoch, history[:loss][end], ev.relative_rmse, ev.correlation, best_rel, best_epoch)
        end
        ProgressMeter.next!(progress; showvalues=[(:epoch, epoch), (:loss, history[:loss][end])])
    end
    ProgressMeter.finish!(progress)
    best_model_cpu === nothing && (best_model_cpu = Flux.fmap(cpu, model))
    model = move_model(best_model_cpu, device)
    eval_metrics = evaluate_A_structured(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_cfg, target, p, device; lag_indices=active_lag_indices)
    true_metrics = structured_true_diagnostics(model, cfg, sampler, stats, target, p, device)
    out_model = resolve_path(base, cfg.output_bson)
    ensure_parent_dir(out_model)
    BSON.bson(out_model, Dict(:host_model => best_model_cpu, :cfg => cfg,
        :model_kind => "structured_coefficients",
        :history => history, :eval_metrics => eval_metrics, :true_metrics => true_metrics,
        :target_artifact => cfg.target_artifact_bson,
        :best_epoch => best_epoch, :best_rel => best_rel,
        :loss_scale => "per-selected-channel target RMS across lags",
        :no_cheating_audit => "Structured mobility loss used only data Cdot, data-only Phi GFDT with learned stationary score, learned conditional residual score, and mean Phi penalty. Analytic mobility structure was used only as an architectural support/equivariance constraint; true M was used only for ex-post diagnostics."))
    metrics_path = resolve_path(base, cfg.metrics_txt)
    open(metrics_path, "w") do io
        println(io, "SoftSpinLLGChain Step 3 structured mobility NN metrics")
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
        println(io, "No-cheating audit: true M was used only after training for diagnostics.")
    end
    render_dm_figure(resolve_path(base, cfg.figure_png), history, eval_metrics, true_metrics, cfg)
    @printf("Saved structured mobility model to %s\n", out_model)
    @printf("Saved metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_structured_dm(length(ARGS) >= 1 ? ARGS[1] : STRUCT_DEFAULT_CONFIG)
end
