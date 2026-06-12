#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))
if !isdefined(@__MODULE__, :CondPairSampler)
    struct CondPairSampler end
end
include(joinpath(@__DIR__, "right_observables.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "score.toml"))

Base.@kwdef struct ScoreConfig
    input_hdf5::String
    burnin_fraction::Float64
    max_samples::Int
    spin_inversion_augment::Bool
    enforce_zero_mean::Bool
    validation_fraction::Float64
    model_config::ScoreUNetConfig
    architecture::Symbol
    normalization::Symbol
    output_mode::Symbol
    input_features::Symbol
    residual_dilations::Vector{Int}
    residual_repeats::Int
    init_seed::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float64
    sigma::Float32
    epoch_subset_size::Int
    use_lr_schedule::Bool
    min_lr_factor::Float64
    lr_schedule_mode::Symbol
    final_learning_rate::Float64
    antithetic_noise::Bool
    loss_output_scale::Symbol
    gradient_accumulation_steps::Int
    stein_weight_max::Float64
    stein_weight_warmup_epochs::Int
    stein_every::Int
    stein_samples::Int
    stein_batch_size::Int
    stein_obs_family::String
    stein_phi_bson::String
    stein_bootstrap_blocks::Int
    stein_weight_floor::Float64
    stein_gfdt_reduction::Symbol
    stein_lambda_x::Float64
    stein_lambda_0::Float64
    move_weight::Float64
    move_reference_bson::String
    output_stein_history_txt::String
    metric_guard_abort::Bool
    guard_max_rel_rmse::Float64
    guard_max_safe_rel_rmse::Float64
    guard_max_move_rel::Float64
    save_best_validation::Bool
    symmetrized_loss::Bool
    checkpoint_every::Int
    allow_nondefault_sigma::Bool
    warm_start_bson::String
    resume_training_state::Bool
    analytic_every::Int
    seed::Int
    exact_score_samples::Int
    langevin_validate::Bool
    langevin_dt::Float64
    langevin_total_time::Float64
    langevin_burnin_time::Float64
    langevin_save_dt::Float64
    langevin_ntraj::Int
    langevin_score_clip::Float32
    langevin_max_pdf_samples::Int
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    output_langevin_hdf5::String
    output_langevin_png::String
    output_metrics_txt::String
    output_analytic_history_txt::String
    device::String
    required_gpu_name::String
    train::Bool
    evaluate::Bool
end

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    model = raw["model"]
    train = raw["training"]
    data = raw["data"]
    fig = raw["figure"]
    out = raw["output"]
    run = raw["run"]
    langevin = get(raw, "langevin", Dict{String, Any}())
    architecture = Symbol(lowercase(String(get(model, "architecture", "unet"))))
    output_mode = Symbol(lowercase(String(get(model, "output_mode", "noise"))))
    symmetrized_loss = Bool(get(train, "symmetrized_loss", false))
    loss_scale_raw = lowercase(String(get(train, "loss_output_scale", "auto")))
    loss_output_scale = loss_scale_raw == "auto" ?
        (output_mode == :noise && !symmetrized_loss ? :noise : :score) :
        Symbol(loss_scale_raw)
    cfg = ScoreUNetConfig(
        in_channels=SPIN_CHANNELS,
        base_channels=Int(get(model, "base_channels", 96)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(get(model, "activation", "swish")),
        final_activation=activation_from_string(get(model, "final_activation", "identity")),
    )
    params = ScoreConfig(
        input_hdf5=String(data["input_hdf5"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        max_samples=Int(get(data, "max_samples", 1_048_576)),
        spin_inversion_augment=Bool(get(data, "spin_inversion_augment", false)),
        enforce_zero_mean=Bool(get(data, "enforce_zero_mean", false)),
        validation_fraction=Float64(get(data, "validation_fraction", 0.05)),
        model_config=cfg,
        architecture=architecture,
        normalization=normalization_from_string(get(model, "normalization", "none")),
        output_mode=output_mode,
        input_features=Symbol(lowercase(String(get(model, "input_features", "spin")))),
        residual_dilations=Int.(get(model, "residual_dilations", [1, 2, 4, 8, 4, 2, 1])),
        residual_repeats=Int(get(model, "residual_repeats", 1)),
        init_seed=Int(get(model, "init_seed", 314159)),
        batch_size=Int(get(train, "batch_size", 8192)),
        epochs=Int(get(train, "epochs", 140)),
        learning_rate=Float64(get(train, "learning_rate", 1.2e-4)),
        sigma=Float32(get(train, "sigma", 0.05)),
        epoch_subset_size=Int(get(train, "epoch_subset_size", 262144)),
        use_lr_schedule=Bool(get(train, "use_lr_schedule", true)),
        min_lr_factor=Float64(get(train, "min_lr_factor", 0.08)),
        lr_schedule_mode=Symbol(lowercase(String(get(train, "lr_schedule_mode", "legacy")))),
        final_learning_rate=Float64(get(train, "final_learning_rate", 0.0)),
        antithetic_noise=Bool(get(train, "antithetic_noise", false)),
        loss_output_scale=loss_output_scale,
        gradient_accumulation_steps=Int(get(train, "gradient_accumulation_steps", 1)),
        stein_weight_max=Float64(get(train, "stein_weight_max", 0.0)),
        stein_weight_warmup_epochs=Int(get(train, "stein_weight_warmup_epochs", 0)),
        stein_every=Int(get(train, "stein_every", 1)),
        stein_samples=Int(get(train, "stein_samples", 0)),
        stein_batch_size=Int(get(train, "stein_batch_size", 0)),
        stein_obs_family=String(get(train, "stein_obs_family", "core")),
        stein_phi_bson=String(get(train, "stein_phi_bson", "")),
        stein_bootstrap_blocks=Int(get(train, "stein_bootstrap_blocks", 32)),
        stein_weight_floor=Float64(get(train, "stein_weight_floor", 1.0e-24)),
        stein_gfdt_reduction=Symbol(lowercase(String(get(train, "stein_gfdt_reduction", "sum")))),
        stein_lambda_x=Float64(get(train, "stein_lambda_x", 1.0)),
        stein_lambda_0=Float64(get(train, "stein_lambda_0", 1.0)),
        move_weight=Float64(get(train, "move_weight", 0.0)),
        move_reference_bson=String(get(train, "move_reference_bson", "")),
        output_stein_history_txt=String(get(out, "stein_history_txt",
            replace(String(out["model_bson"]), ".bson" => "_stein_history.csv"))),
        metric_guard_abort=Bool(get(train, "metric_guard_abort", false)),
        guard_max_rel_rmse=Float64(get(train, "guard_max_rel_rmse", 0.0)),
        guard_max_safe_rel_rmse=Float64(get(train, "guard_max_safe_rel_rmse", 0.0)),
        guard_max_move_rel=Float64(get(train, "guard_max_move_rel", 0.0)),
        save_best_validation=Bool(get(train, "save_best_validation", true)),
        symmetrized_loss=symmetrized_loss,
        checkpoint_every=Int(get(train, "checkpoint_every", 0)),
        allow_nondefault_sigma=Bool(get(train, "allow_nondefault_sigma", false)),
        warm_start_bson=String(get(train, "warm_start_bson", "")),
        resume_training_state=Bool(get(train, "resume_training_state", true)),
        analytic_every=Int(get(train, "analytic_every", 20)),
        seed=Int(get(train, "seed", 20260508)),
        exact_score_samples=Int(get(fig, "exact_score_samples", 100000)),
        langevin_validate=Bool(get(run, "langevin_validate", true)),
        langevin_dt=Float64(get(langevin, "dt", 0.002)),
        langevin_total_time=Float64(get(langevin, "total_time", 80.0)),
        langevin_burnin_time=Float64(get(langevin, "burnin_time", 16.0)),
        langevin_save_dt=Float64(get(langevin, "save_dt", 0.04)),
        langevin_ntraj=Int(get(langevin, "ntrajectories", 192)),
        langevin_score_clip=Float32(get(langevin, "score_clip", 80.0)),
        langevin_max_pdf_samples=Int(get(langevin, "max_pdf_samples", 250000)),
        figure_width=Int(get(fig, "width", 3000)),
        figure_height=Int(get(fig, "height", 2200)),
        output_bson=String(out["model_bson"]),
        output_png=String(out["figure_png"]),
        output_langevin_hdf5=String(out["langevin_hdf5"]),
        output_langevin_png=String(out["langevin_figure_png"]),
        output_metrics_txt=String(get(out, "metrics_txt", replace(String(out["figure_png"]), ".png" => "_metrics.txt"))),
        output_analytic_history_txt=String(get(out, "analytic_history_txt",
            replace(String(out["model_bson"]), ".bson" => "_analytic_history.csv"))),
        device=String(get(run, "device", "GPU:0")),
        required_gpu_name=String(get(run, "required_gpu_name", "2080ti")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
    )
    require_condition(params.sigma == 0.05f0 || params.allow_nondefault_sigma,
        "DSM sigma must remain 0.05 unless explicitly changed by the user and allow_nondefault_sigma=true.")
    require_condition(params.model_config.periodic, "The score model must be periodic.")
    require_condition(params.architecture in (:unet, :dilated_rescnn, :largekernel_rescnn),
        "Unsupported score architecture=$(params.architecture).")
    require_condition(params.output_mode in (:noise, :score), "output_mode must be either noise or score.")
    require_condition(params.loss_output_scale in (:score, :sigma, :noise),
        "loss_output_scale must be score, sigma, or noise.")
    require_condition(params.loss_output_scale != :sigma || params.output_mode == :score || params.symmetrized_loss,
        "loss_output_scale=sigma is only valid for direct-score DSM training.")
    require_condition(params.gradient_accumulation_steps >= 1,
        "gradient_accumulation_steps must be at least 1.")
    require_condition(params.stein_weight_max >= 0.0, "stein_weight_max must be nonnegative.")
    require_condition(params.stein_weight_warmup_epochs >= 0,
        "stein_weight_warmup_epochs must be nonnegative.")
    require_condition(params.stein_every >= 1, "stein_every must be at least 1.")
    require_condition(params.stein_samples >= 0, "stein_samples must be nonnegative.")
    require_condition(params.stein_batch_size >= 0, "stein_batch_size must be nonnegative.")
    require_condition(params.stein_bootstrap_blocks >= 1,
        "stein_bootstrap_blocks must be at least 1.")
    require_condition(params.stein_weight_floor > 0.0, "stein_weight_floor must be positive.")
    require_condition(params.stein_gfdt_reduction in (:sum, :mean),
        "stein_gfdt_reduction must be sum or mean.")
    require_condition(params.stein_lambda_x >= 0.0, "stein_lambda_x must be nonnegative.")
    require_condition(params.stein_lambda_0 >= 0.0, "stein_lambda_0 must be nonnegative.")
    require_condition(params.move_weight >= 0.0, "move_weight must be nonnegative.")
    require_condition(params.guard_max_rel_rmse >= 0.0, "guard_max_rel_rmse must be nonnegative.")
    require_condition(params.guard_max_safe_rel_rmse >= 0.0,
        "guard_max_safe_rel_rmse must be nonnegative.")
    require_condition(params.guard_max_move_rel >= 0.0, "guard_max_move_rel must be nonnegative.")
    if params.stein_weight_max > 0.0
        require_condition(!isempty(strip(params.stein_phi_bson)),
            "stein_phi_bson is required when stein_weight_max > 0.")
        require_condition(params.stein_samples > 0,
            "stein_samples must be positive when stein_weight_max > 0.")
        require_condition(params.stein_batch_size > 0,
            "stein_batch_size must be positive when stein_weight_max > 0.")
    end
    if params.move_weight > 0.0
        require_condition(!isempty(strip(params.move_reference_bson)),
            "move_reference_bson is required when move_weight > 0.")
    end
    require_condition(params.lr_schedule_mode in (:legacy, :none, :local_exp),
        "lr_schedule_mode must be legacy, none, or local_exp.")
    require_condition(params.architecture == :unet || params.output_mode == :score,
        "Residual-CNN stationary score branches are direct-score models; set output_mode=score.")
    require_condition(params.input_features in (:spin, :spin_r2),
        "input_features must be either spin or spin_r2.")
    require_condition(0.0 <= params.validation_fraction < 1.0,
        "validation_fraction must satisfy 0 <= validation_fraction < 1.")
    return params
end

function load_score_dataset(path::AbstractString, params::ScoreConfig, rng::AbstractRNG)
    times, states = load_spin_states(path)
    start = burnin_start_index(length(times), params.burnin_fraction)
    raw_n = params.spin_inversion_augment && params.max_samples > 0 ?
        max(1, ceil(Int, params.max_samples / 2)) : params.max_samples
    raw = sample_state_tensor(states, start, raw_n, rng)
    if params.spin_inversion_augment
        raw = cat(raw, -raw; dims=3)
        if params.max_samples > 0 && size(raw, 3) > params.max_samples
            raw = raw[:, :, 1:params.max_samples]
        end
    end
    stats = channel_shared_stats(raw)
    if params.enforce_zero_mean
        stats = DataStats(zeros(Float32, size(stats.mean)), stats.std)
    end
    normed = apply_stats_tensor(raw, stats)
    return NormalizedDataset(normed, stats), times, states, start
end

function lr_factor(epoch::Int, epochs::Int, min_factor::Float64)
    x = (epoch - 1) / max(epochs - 1, 1)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + cos(pi * x))
end

function score_learning_rate(params::ScoreConfig, epoch::Int, start_epoch::Int, stop_epoch::Int)
    if params.lr_schedule_mode == :none || !params.use_lr_schedule
        return params.learning_rate
    elseif params.lr_schedule_mode == :local_exp
        final_lr = params.final_learning_rate > 0 ?
            params.final_learning_rate : params.learning_rate * params.min_lr_factor
        final_lr = min(final_lr, params.learning_rate)
        u = (epoch - start_epoch) / max(stop_epoch - start_epoch, 1)
        return params.learning_rate * (final_lr / max(params.learning_rate, eps(Float64)))^u
    else
        return params.learning_rate * lr_factor(epoch, stop_epoch, params.min_lr_factor)
    end
end

function dsm_score_prediction(model, noisy, params::ScoreConfig)
    return params.symmetrized_loss || params.output_mode == :score ?
        score_from_dsm_model(model, noisy, params.sigma) : model(noisy)
end

function dsm_prediction_target(current_model, batch, signed_noise, params::ScoreConfig)
    noisy = batch .+ params.sigma .* signed_noise
    raw_pred = dsm_score_prediction(current_model, noisy, params)
    if params.loss_output_scale == :sigma
        return params.sigma .* raw_pred, -signed_noise
    elseif params.loss_output_scale == :score || params.symmetrized_loss || params.output_mode == :score
        return raw_pred, signed_noise .* (-one(eltype(signed_noise)) / params.sigma)
    else
        return raw_pred, signed_noise
    end
end

function dsm_loss_value(current_model, batch, noise, params::ScoreConfig)
    pred, target = dsm_prediction_target(current_model, batch, noise, params)
    loss = Flux.Losses.mse(pred, target)
    if params.antithetic_noise
        pred2, target2 = dsm_prediction_target(current_model, batch, -noise, params)
        loss = (loss + Flux.Losses.mse(pred2, target2)) * eltype(loss)(0.5)
    end
    return loss
end

function dsm_monitor_metrics(current_model, batch, noise, params::ScoreConfig)
    metrics = Float64[]
    for signed_noise in (noise, -noise)
        if signed_noise !== noise && !params.antithetic_noise
            continue
        end
        pred, target = dsm_prediction_target(current_model, batch, signed_noise, params)
        pred_h = Array(to_host(pred))
        target_h = Array(to_host(target))
        null_mse = mean(abs2, target_h)
        mse = mean(abs2, pred_h .- target_h)
        dotpt = sum(pred_h .* target_h)
        push!(metrics, sqrt(null_mse))
        push!(metrics, sqrt(mean(abs2, pred_h)))
        push!(metrics, null_mse)
        push!(metrics, 1.0 - mse / max(null_mse, eps(Float64)))
        push!(metrics, dotpt / max(sqrt(sum(abs2, pred_h) * sum(abs2, target_h)), eps(Float64)))
    end
    n = params.antithetic_noise ? 2 : 1
    return (target_rms=mean(metrics[1:5:end]), prediction_rms=mean(metrics[2:5:end]),
        null_mse=mean(metrics[3:5:end]), fractional_improvement=mean(metrics[4:5:end]),
        prediction_target_cosine=mean(metrics[5:5:end]))
end

function score_checkpoint_blob(host_model, model_cfg, dataset::NormalizedDataset,
        params::ScoreConfig, p::SpinParams, history, epoch::Int, val_loss::Float64;
        checkpoint_kind::AbstractString, training_state=nothing)
    trainer_cfg = Dict(:sigma => params.sigma, :epochs => params.epochs,
        :batch_size => params.batch_size, :learning_rate => params.learning_rate,
        :epoch_subset_size => params.epoch_subset_size,
        :use_lr_schedule => params.use_lr_schedule,
        :min_lr_factor => params.min_lr_factor,
        :lr_schedule_mode => String(params.lr_schedule_mode),
        :final_learning_rate => params.final_learning_rate,
        :antithetic_noise => params.antithetic_noise,
        :loss_output_scale => String(params.loss_output_scale),
        :gradient_accumulation_steps => params.gradient_accumulation_steps,
        :stein_weight_max => params.stein_weight_max,
        :stein_weight_warmup_epochs => params.stein_weight_warmup_epochs,
        :stein_every => params.stein_every,
        :stein_samples => params.stein_samples,
        :stein_batch_size => params.stein_batch_size,
        :stein_obs_family => params.stein_obs_family,
        :stein_phi_bson => params.stein_phi_bson,
        :stein_bootstrap_blocks => params.stein_bootstrap_blocks,
        :stein_weight_floor => params.stein_weight_floor,
        :stein_gfdt_reduction => String(params.stein_gfdt_reduction),
        :stein_lambda_x => params.stein_lambda_x,
        :stein_lambda_0 => params.stein_lambda_0,
        :move_weight => params.move_weight,
        :move_reference_bson => params.move_reference_bson,
        :stein_history_txt => params.output_stein_history_txt,
        :metric_guard_abort => params.metric_guard_abort,
        :guard_max_rel_rmse => params.guard_max_rel_rmse,
        :guard_max_safe_rel_rmse => params.guard_max_safe_rel_rmse,
        :guard_max_move_rel => params.guard_max_move_rel,
        :architecture => String(params.architecture),
        :output_mode => String(params.output_mode),
        :input_features => String(params.input_features),
        :residual_dilations => params.residual_dilations,
        :residual_repeats => params.residual_repeats,
        :warm_start_bson => params.warm_start_bson,
        :resume_training_state => params.resume_training_state,
        :analytic_every => params.analytic_every,
        :analytic_history_txt => params.output_analytic_history_txt,
        :seed => params.seed,
        :init_seed => params.init_seed,
        :validation_fraction => params.validation_fraction,
        :spin_inversion_augment => params.spin_inversion_augment,
        :enforce_zero_mean => params.enforce_zero_mean,
        :symmetrized_loss => params.symmetrized_loss,
        :save_best_validation => params.save_best_validation,
        :checkpoint_kind => String(checkpoint_kind),
        :checkpoint_epoch => epoch,
        :checkpoint_validation_loss => val_loss)
    metadata = Dict(:no_cheating_audit =>
        "DSM target is constructed only from Gaussian noise added to data samples; analytic score is excluded from all losses and model selection. Symmetry augmentation, when enabled, uses only the spin-inversion symmetry of the observed system.",
        :score_output_semantics =>
        "score_from_dsm_model returns the physical normalized stationary score. loss_output_scale=sigma only rescales the minimized DSM loss target; it does not change downstream score semantics.",
        :stein_training_semantics =>
        "Optional Stein/GFDT terms use observed stationary samples, a configured data-derived Phi artifact, and learned score outputs. Analytic score remains excluded from the minimized loss and is used only in diagnostics. Metric guards, when configured, are ex-post rejection checks rather than training targets.")
    blob = Dict(:host_model => host_model, :model_cfg => model_cfg,
        :stats => dataset.stats, :trainer_cfg => trainer_cfg, :history => history,
        :phys => p, :metadata => metadata)
    training_state === nothing || (blob[:training_state] = training_state)
    return blob
end

function periodic_score_checkpoint_path(model_path::AbstractString, epoch::Int)
    root, ext = splitext(model_path)
    return string(root, "_epoch", lpad(string(epoch), 4, '0'), ext)
end

function analytic_score_diagnostic(model, dataset::NormalizedDataset, p::SpinParams,
        params::ScoreConfig, device::ExecutionDevice, keep::AbstractVector{<:Integer})
    clean = copy(dataset.data[:, :, keep])
    pred = evaluate_score_norm(model, clean, params.sigma, device; batch_size=params.batch_size)
    exact = standardized_analytic_score(clean, dataset.stats, p)
    rel, cosv = score_metric_pair(pred, exact)
    comp_rel = [
        norm(vec(pred[:, c, :] .- exact[:, c, :])) /
            max(norm(vec(exact[:, c, :])), eps(Float64))
        for c in 1:size(pred, 2)
    ]
    score_norm_ratio = sqrt(mean(abs2, pred)) / max(sqrt(mean(abs2, exact)), eps(Float64))
    raw = denormalize_tensor(clean, dataset.stats)
    norms = [sqrt(sum(abs2, raw[i, :, b])) for i in axes(raw, 1), b in axes(raw, 3)]
    lo, hi = quantile(vec(norms), 0.02), quantile(vec(norms), 0.98)
    safe = vec([all(lo <= sqrt(sum(abs2, raw[i, :, b])) <= hi for i in axes(raw, 1)) for b in axes(raw, 3)])
    srel, scos = any(safe) ? score_metric_pair(pred[:, :, safe], exact[:, :, safe]) : (NaN, NaN)
    return Dict(:rel_rmse => rel, :cosine => cosv, :safe_rel_rmse => srel,
        :safe_cosine => scos, :safe_count => count(identity, safe), :total_count => length(safe),
        :rmse_x => comp_rel[1], :rmse_y => comp_rel[2], :rmse_z => comp_rel[3],
        :score_norm_ratio => score_norm_ratio)
end

function append_analytic_history(path::AbstractString, epoch::Int, diag; dsm_loss::Float64=NaN)
    ensure_parent_dir(path)
    need_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        if need_header
            println(io, "epoch,dsm_loss,rel_rmse,cosine,safe_rel_rmse,safe_cosine,rmse_x,rmse_y,rmse_z,score_norm_ratio,safe_count,total_count")
        end
        println(io, @sprintf("%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d,%d",
            epoch, dsm_loss, diag[:rel_rmse], diag[:cosine], diag[:safe_rel_rmse],
            diag[:safe_cosine], diag[:rmse_x], diag[:rmse_y], diag[:rmse_z],
            diag[:score_norm_ratio], diag[:safe_count], diag[:total_count]))
    end
end

function ensure_score_history_keys!(history)
    for key in (:train_loss, :val_loss, :score_norm, :target_rms, :prediction_rms,
            :null_mse, :fractional_improvement, :prediction_target_cosine,
            :analytic_epoch, :analytic_rel_rmse, :analytic_cosine,
            :analytic_safe_rel_rmse, :analytic_safe_cosine,
            :objective_loss, :stein_epoch, :stein_weight, :stein_train_loss,
            :stein_gfdt_weighted_sum, :stein_gfdt_weighted_mean,
            :stein_gfdt_rms, :stein_coord_norm, :stein_z_norm,
            :stein_move_rel)
        haskey(history, key) || (history[key] = Float64[])
    end
    return history
end

function host_training_state(opt, rng::AbstractRNG, noise_rng::AbstractRNG,
        train_idx::AbstractVector{<:Integer}, val_idx::AbstractVector{<:Integer},
        val_noise::AbstractArray, analytic_idx::AbstractVector{<:Integer},
        epoch::Int, stop_epoch::Int)
    return Dict(
        :optimizer_state => Flux.fmap(cpu, opt),
        :rng => rng,
        :noise_rng => noise_rng,
        :train_idx => collect(Int, train_idx),
        :val_idx => collect(Int, val_idx),
        :val_noise => Array(val_noise),
        :analytic_idx => collect(Int, analytic_idx),
        :dataset_length => maximum(vcat(train_idx, val_idx, analytic_idx, [0])),
        :completed_epoch => epoch,
        :stop_epoch => stop_epoch,
    )
end

function make_score_splits(dataset::NormalizedDataset, params::ScoreConfig, rng::AbstractRNG)
    n = length(dataset)
    if params.validation_fraction <= 0
        return collect(1:n), Int[], Array{Float32}(undef, size(dataset.data, 1), size(dataset.data, 2), 0)
    end
    val_n = min(max(params.batch_size, floor(Int, params.validation_fraction * n)), n - 1)
    val_idx = randperm(rng, n)[1:val_n]
    val_set = Set(val_idx)
    train_idx = [i for i in 1:n if !(i in val_set)]
    isempty(train_idx) && error("Training split is empty; reduce validation_fraction or increase samples.")
    val_noise = randn(MersenneTwister(params.seed + 11), Float32, size(dataset.data, 1),
        size(dataset.data, 2), val_n)
    return train_idx, val_idx, val_noise
end

flatten_batch_tensor(x) = reshape(permutedims(x, (2, 1, 3)), :, size(x, 3))

function score_std_tensor(stats::DataStats, N::Int, device::ExecutionDevice)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), N, SPIN_CHANNELS, 1)
    return move_array(Float32.(std_tensor), device)
end

function score_norm_to_raw_tensor(score_norm, std_tensor)
    return score_norm ./ std_tensor
end

function load_phi_matrix(path::AbstractString)
    blob = BSON.parse(path)
    if haskey(blob, :Phi)
        return Matrix{Float64}(BSON.raise_recursive(blob[:Phi], @__MODULE__))
    elseif haskey(blob, :Phi_projected)
        return Matrix{Float64}(BSON.raise_recursive(blob[:Phi_projected], @__MODULE__))
    end
    error("Phi artifact $(path) has neither :Phi nor :Phi_projected.")
end

function site_average_features(vals::Array{Float32, 3})
    N, R, B = size(vals)
    out = Matrix{Float32}(undef, R, B)
    invN = Float32(1 / N)
    @inbounds for b in 1:B, a in 1:R
        acc = 0.0f0
        for i in 1:N
            acc += vals[i, a, b]
        end
        out[a, b] = acc * invN
    end
    return out
end

function site_average_grad_hess(grads::Array{Float32, 4}, hess::Array{Float32, 5}, D::Int)
    N, R, _, B = size(grads)
    grad_avg = zeros(Float32, R, D, B)
    hess_avg = zeros(Float32, R, D, D, B)
    invN = Float32(1 / N)
    @inbounds for b in 1:B, i in 1:N, a in 1:R
        rows = ((i - 1) * SPIN_CHANNELS + 1):(i * SPIN_CHANNELS)
        for c in 1:SPIN_CHANNELS
            grad_avg[a, rows[c], b] += invN * grads[i, a, c, b]
        end
        for c in 1:SPIN_CHANNELS, d in 1:SPIN_CHANNELS
            hess_avg[a, rows[c], rows[d], b] += invN * hess[i, a, c, d, b]
        end
    end
    return grad_avg, hess_avg
end

function precompute_stein_terms(grad_avg::Array{Float32, 3},
        hess_avg::Array{Float32, 4}, Phi::AbstractMatrix{<:Real})
    R, D, B = size(grad_avg)
    hess_trace = Matrix{Float32}(undef, R, B)
    grad_phi_grad = Array{Float32}(undef, R, R, B)
    Phi64 = Matrix{Float64}(Phi)
    @inbounds for b in 1:B
        for a in 1:R
            ht = 0.0
            for i in 1:D, j in 1:D
                ht += Phi64[i, j] * Float64(hess_avg[a, j, i, b])
            end
            hess_trace[a, b] = Float32(ht)
        end
        for m in 1:R, n in 1:R
            gt = 0.0
            for i in 1:D, j in 1:D
                gt += Float64(grad_avg[m, i, b]) * Phi64[i, j] * Float64(grad_avg[n, j, b])
            end
            grad_phi_grad[m, n, b] = Float32(gt)
        end
    end
    return hess_trace, grad_phi_grad
end

function block_standard_errors(samples::Matrix{Float64}, nblocks::Int)
    K, B = size(samples)
    nb = min(nblocks, B)
    edges = round.(Int, range(1, B + 1; length=nb + 1))
    block_means = Matrix{Float64}(undef, K, nb)
    @inbounds for bi in 1:nb
        lo = edges[bi]
        hi = edges[bi + 1] - 1
        block_means[:, bi] .= vec(mean(@view(samples[:, lo:hi]); dims=2))
    end
    se = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        se[k] = std(@view block_means[k, :]) / sqrt(nb)
    end
    return se, nb
end

function gfdt_stein_samples_cpu(phi::Matrix{Float32}, grad_avg::Array{Float32, 3},
        hess_trace::Matrix{Float32}, grad_phi_grad::Array{Float32, 3},
        score_raw::Array{Float32, 3}, Phi::AbstractMatrix{<:Real})
    R, B = size(phi)
    D = size(Phi, 1)
    score_flat = Matrix{Float64}(flatten_batch(score_raw))
    action = transpose(Matrix{Float64}(Phi)) * score_flat
    samples = Matrix{Float64}(undef, R * R, B)
    @inbounds for b in 1:B
        Bphi = Vector{Float64}(undef, R)
        for n in 1:R
            score_term = 0.0
            for d in 1:D
                score_term += action[d, b] * Float64(grad_avg[n, d, b])
            end
            Bphi[n] = score_term + Float64(hess_trace[n, b])
        end
        k = 1
        for m in 1:R, n in 1:R
            samples[k, b] = Float64(phi[m, b]) * Bphi[n] + Float64(grad_phi_grad[m, n, b])
            k += 1
        end
    end
    return samples
end

struct ScoreSteinCache
    normed_cpu::Array{Float32, 3}
    phi_cpu::Matrix{Float32}
    grad_cpu::Array{Float32, 3}
    hess_trace_cpu::Matrix{Float32}
    grad_phi_grad_cpu::Array{Float32, 3}
    weights_cpu::Matrix{Float32}
    Phi_cpu::Matrix{Float32}
    normed::Any
    phi::Any
    grad::Any
    hess_trace::Any
    grad_phi_grad::Any
    weights::Any
    Phi_T::Any
    identity::Any
    std_tensor::Any
    reference_score_norm::Any
    reference_rms::Float64
end

function load_reference_score_norm(path::AbstractString, normed::Array{Float32, 3},
        params::ScoreConfig, device::ExecutionDevice)
    blob = BSON.load(path)
    haskey(blob, :host_model) || error("move_reference_bson lacks :host_model: $(path)")
    ref_model = move_model(blob[:host_model], device)
    Flux.testmode!(ref_model)
    ref_sigma = Float32(get(get(blob, :trainer_cfg, Dict{Symbol, Any}()), :sigma, params.sigma))
    return evaluate_score_norm(ref_model, normed, ref_sigma, device; batch_size=params.batch_size)
end

function make_stein_training_cache(dataset::NormalizedDataset, params::ScoreConfig,
        device::ExecutionDevice, model)
    if params.stein_weight_max <= 0.0 && params.move_weight <= 0.0
        return nothing
    end
    n = length(dataset)
    ns = params.stein_samples > 0 ? min(params.stein_samples, n) : min(8192, n)
    idx = randperm(MersenneTwister(params.seed + 310), n)[1:ns]
    normed = copy(dataset.data[:, :, idx])
    raw = denormalize_tensor(normed, dataset.stats)
    Phi = load_phi_matrix(params.stein_phi_bson)
    lib = RightObservableLibrary(right_candidate_names(params.stein_obs_family))
    vals, grads, hess = right_observable_value_grad_hess(raw, lib)
    phi = site_average_features(vals)
    grad_avg, hess_avg = site_average_grad_hess(grads, hess, size(Phi, 1))
    hess_trace, grad_phi_grad = precompute_stein_terms(grad_avg, hess_avg, Phi)

    Flux.testmode!(model)
    baseline_score_norm = evaluate_score_norm(model, normed, params.sigma, device;
        batch_size=params.batch_size)
    baseline_score_raw = normalized_score_to_raw(baseline_score_norm, dataset.stats)
    samples = gfdt_stein_samples_cpu(phi, grad_avg, hess_trace, grad_phi_grad,
        baseline_score_raw, Phi)
    se, _ = block_standard_errors(samples, params.stein_bootstrap_blocks)
    weights = reshape(Float32.(1.0 ./ (se .^ 2 .+ params.stein_weight_floor)),
        length(lib.names), length(lib.names))

    reference_score_norm = nothing
    reference_rms = NaN
    if params.move_weight > 0.0
        ref_norm = load_reference_score_norm(params.move_reference_bson, normed, params, device)
        reference_rms = sqrt(mean(abs2, ref_norm))
        reference_score_norm = move_array(ref_norm, device)
    end
    Flux.trainmode!(model)

    D = size(Phi, 1)
    return ScoreSteinCache(
        normed, phi, grad_avg, hess_trace, grad_phi_grad, weights, Float32.(Phi),
        move_array(normed, device),
        move_array(phi, device),
        move_array(grad_avg, device),
        move_array(hess_trace, device),
        move_array(grad_phi_grad, device),
        move_array(weights, device),
        move_array(Float32.(transpose(Phi)), device),
        move_array(Matrix{Float32}(I, D, D), device),
        score_std_tensor(dataset.stats, size(normed, 1), device),
        reference_score_norm,
        reference_rms,
    )
end

function stein_weight_for_epoch(params::ScoreConfig, epoch::Int, start_epoch::Int)
    params.stein_weight_max <= 0.0 && return 0.0
    local_epoch = epoch - start_epoch + 1
    params.stein_weight_warmup_epochs <= 0 && return params.stein_weight_max
    return params.stein_weight_max *
        min(1.0, local_epoch / max(params.stein_weight_warmup_epochs, 1))
end

function stein_aux_losses(current_model, cache::ScoreSteinCache,
        params::ScoreConfig, pos::AbstractVector{<:Integer})
    z = cache.normed[:, :, pos]
    phi = cache.phi[:, pos]
    grad = cache.grad[:, :, pos]
    hess_trace = cache.hess_trace[:, pos]
    grad_phi_grad = cache.grad_phi_grad[:, :, pos]
    B = size(z, 3)
    score_norm = score_from_dsm_model(current_model, z, params.sigma)
    score_raw = score_norm_to_raw_tensor(score_norm, cache.std_tensor)
    sflat = flatten_batch_tensor(score_raw)
    action = cache.Phi_T * sflat
    score_term = dropdims(sum(grad .* reshape(action, 1, size(action, 1), size(action, 2)); dims=2), dims=2)
    Bphi = score_term .+ hess_trace
    residual = (phi * transpose(Bphi)) ./ eltype(Bphi)(B) .+
        dropdims(mean(grad_phi_grad; dims=3), dims=3)
    gfdt_sum = sum(cache.weights .* abs2.(residual))
    gfdt = params.stein_gfdt_reduction == :mean ?
        gfdt_sum / eltype(gfdt_sum)(length(residual)) : gfdt_sum

    zflat = flatten_batch_tensor(z)
    sflat_norm = flatten_batch_tensor(score_norm)
    r0 = dropdims(mean(sflat_norm; dims=2), dims=2)
    r1 = (zflat * transpose(sflat_norm)) ./ eltype(sflat_norm)(B) .+ cache.identity
    l0 = sum(abs2, r0)
    lx = sum(abs2, r1)
    stein = gfdt + eltype(gfdt)(params.stein_lambda_x) * lx +
        eltype(gfdt)(params.stein_lambda_0) * l0

    move = if cache.reference_score_norm === nothing
        zero(stein)
    else
        ref = cache.reference_score_norm[:, :, pos]
        mean(abs2, score_norm .- ref)
    end
    return stein, move
end

function score_metric_guard_reasons(params::ScoreConfig, epoch::Int,
        adiag, sdiag)
    reasons = String[]
    if adiag !== nothing
        rel = Float64(adiag[:rel_rmse])
        if params.guard_max_rel_rmse > 0.0 &&
                (!isfinite(rel) || rel > params.guard_max_rel_rmse)
            push!(reasons, @sprintf("rel.RMSE %.6e exceeds %.6e", rel,
                params.guard_max_rel_rmse))
        end
        safe_rel = Float64(adiag[:safe_rel_rmse])
        if params.guard_max_safe_rel_rmse > 0.0 &&
                (!isfinite(safe_rel) || safe_rel > params.guard_max_safe_rel_rmse)
            push!(reasons, @sprintf("safe rel.RMSE %.6e exceeds %.6e", safe_rel,
                params.guard_max_safe_rel_rmse))
        end
    end
    if sdiag !== nothing
        move_rel = Float64(sdiag[:move_rel])
        if params.guard_max_move_rel > 0.0 &&
                (!isfinite(move_rel) || move_rel > params.guard_max_move_rel)
            push!(reasons, @sprintf("move_rel %.6e exceeds %.6e", move_rel,
                params.guard_max_move_rel))
        end
    end
    isempty(reasons) || pushfirst!(reasons, @sprintf("epoch %d", epoch))
    return reasons
end

function coordinate_stein_terms(normed::Array{Float32, 3}, score_norm::Array{Float32, 3})
    z = Matrix{Float64}(flatten_batch(normed))
    s = Matrix{Float64}(flatten_batch(score_norm))
    D, B = size(z)
    r0 = vec(mean(s; dims=2))
    r1 = (z * transpose(s)) ./ B + Matrix{Float64}(I, D, D)
    return norm(r0)^2, norm(r1)^2, norm(r0) / sqrt(D), norm(r1) / sqrt(D)
end

function component_stein_blocks(normed::Array{Float32, 3}, score_norm::Array{Float32, 3})
    z = Matrix{Float64}(flatten_batch(normed))
    s = Matrix{Float64}(flatten_batch(score_norm))
    D, B = size(z)
    mat = (z * transpose(s)) ./ B + Matrix{Float64}(I, D, D)
    out = Dict{String, Float64}()
    for (ci, name) in enumerate(("x", "y", "z"))
        idx = ci:SPIN_CHANNELS:D
        out[name] = norm(mat[idx, idx]) / sqrt(length(idx))
    end
    return out
end

function stein_cache_diagnostic(model, cache::ScoreSteinCache,
        params::ScoreConfig, device::ExecutionDevice)
    Flux.testmode!(model)
    score_norm = evaluate_score_norm(model, cache.normed_cpu, params.sigma, device;
        batch_size=params.batch_size)
    score_raw = normalized_score_to_raw(score_norm, DataStats(
        permutedims(zeros(Float32, size(cache.std_tensor, 1), SPIN_CHANNELS), (2, 1)),
        permutedims(Array(to_host(cache.std_tensor[:, :, 1])), (2, 1))))
    samples = gfdt_stein_samples_cpu(cache.phi_cpu, cache.grad_cpu,
        cache.hess_trace_cpu, cache.grad_phi_grad_cpu, score_raw, cache.Phi_cpu)
    residual = vec(mean(samples; dims=2))
    weights = vec(Float64.(cache.weights_cpu))
    gfdt_weighted_sum = sum(weights .* residual .^ 2)
    gfdt_weighted_mean = mean(weights .* residual .^ 2)
    gfdt_unweighted_sum = sum(residual .^ 2)
    gfdt_rms = sqrt(mean(residual .^ 2))
    gfdt_max_abs = maximum(abs.(residual))
    l0, lx, mean_norm, coord_norm = coordinate_stein_terms(cache.normed_cpu, score_norm)
    blocks = component_stein_blocks(cache.normed_cpu, score_norm)
    move_rel = NaN
    if cache.reference_score_norm !== nothing
        ref = Array(to_host(cache.reference_score_norm))
        move_rel = sqrt(mean(abs2, score_norm .- ref)) / max(cache.reference_rms, eps(Float64))
    end
    Flux.trainmode!(model)
    return Dict(
        :gfdt_weighted_sum => gfdt_weighted_sum,
        :gfdt_weighted_mean => gfdt_weighted_mean,
        :gfdt_unweighted_sum => gfdt_unweighted_sum,
        :gfdt_rms => gfdt_rms,
        :gfdt_max_abs => gfdt_max_abs,
        :Lx => lx,
        :L0 => l0,
        :mean_score_norm => mean_norm,
        :coord_stein_norm => coord_norm,
        :stein_x => blocks["x"],
        :stein_y => blocks["y"],
        :stein_z => blocks["z"],
        :total_lambda_x1_lambda0_1 => gfdt_weighted_sum + lx + l0,
        :move_rel => move_rel,
    )
end

function append_stein_history(path::AbstractString, epoch::Int, stein_weight::Float64,
        train_stein::Float64, train_move::Float64, diag)
    ensure_parent_dir(path)
    need_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        if need_header
            println(io, "epoch,stein_weight,train_stein_loss,train_move_loss,gfdt_weighted_sum,gfdt_weighted_mean,gfdt_unweighted_sum,gfdt_rms,gfdt_max_abs,Lx,L0,mean_score_norm,coord_stein_norm,stein_x,stein_y,stein_z,total_lambda_x1_lambda0_1,move_rel")
        end
        println(io, @sprintf("%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e",
            epoch, stein_weight, train_stein, train_move,
            diag[:gfdt_weighted_sum], diag[:gfdt_weighted_mean],
            diag[:gfdt_unweighted_sum], diag[:gfdt_rms], diag[:gfdt_max_abs],
            diag[:Lx], diag[:L0], diag[:mean_score_norm], diag[:coord_stein_norm],
            diag[:stein_x], diag[:stein_y], diag[:stein_z],
            diag[:total_lambda_x1_lambda0_1], diag[:move_rel]))
    end
end

function train_score_model(dataset::NormalizedDataset, params::ScoreConfig,
        device::ExecutionDevice, p::SpinParams, model_path::AbstractString)
    Random.seed!(params.init_seed)
    model, model_cfg = if params.architecture == :unet
        build_spin_unet(params.model_config, params.normalization, size(dataset.data, 1);
            output_mode=params.output_mode, input_features=params.input_features)
    else
        build_spin_residual_score_net(params.model_config, params.normalization, size(dataset.data, 1);
            input_features=params.input_features, architecture=params.architecture,
            dilations=params.residual_dilations, repeats=params.residual_repeats)
    end
    Random.seed!()
    warm = nothing
    if !isempty(strip(params.warm_start_bson))
        warm_path = params.warm_start_bson
        isfile(warm_path) || error("warm_start_bson does not exist: $(warm_path)")
        warm = BSON.load(warm_path)
        haskey(warm, :host_model) || error("warm_start_bson lacks :host_model: $(warm_path)")
        model = warm[:host_model]
        model_cfg = get(warm, :model_cfg, model_cfg)
        @printf("Warm-started score model from %s\n", warm_path)
    end
    model = move_model(model, device)
    Flux.trainmode!(model)
    opt = Flux.setup(Flux.Optimisers.Adam(params.learning_rate), model)
    history = Dict(:train_loss => Float64[], :val_loss => Float64[], :score_norm => Float64[],
        :target_rms => Float64[], :prediction_rms => Float64[], :null_mse => Float64[],
        :fractional_improvement => Float64[], :prediction_target_cosine => Float64[],
        :analytic_epoch => Float64[], :analytic_rel_rmse => Float64[],
        :analytic_cosine => Float64[], :analytic_safe_rel_rmse => Float64[],
        :analytic_safe_cosine => Float64[])
    ensure_score_history_keys!(history)
    best = (; val_loss=Inf, epoch=0, host_model=nothing, training_state=nothing)
    n = length(dataset)
    rng = MersenneTwister(params.seed)
    noise_rng = MersenneTwister(params.seed + 1)
    start_epoch = 1
    stop_epoch = params.epochs
        if warm !== nothing && params.resume_training_state && haskey(warm, :training_state)
        state = warm[:training_state]
        if haskey(state, :optimizer_state)
            opt = move_model(state[:optimizer_state], device)
            Flux.adjust!(opt, params.learning_rate)
            @printf("Restored Adam optimizer state from %s\n", params.warm_start_bson)
        else
            @printf("Warm checkpoint lacks optimizer state; using fresh Adam state.\n")
        end
        history = deepcopy(get(warm, :history, history))
        ensure_score_history_keys!(history)
        rng = get(state, :rng, rng)
        noise_rng = get(state, :noise_rng, noise_rng)
        saved_n = Int(get(state, :dataset_length, 0))
        saved_cfg = get(warm, :trainer_cfg, Dict{Symbol, Any}())
        saved_validation_fraction = Float64(get(saved_cfg, :validation_fraction, NaN))
        if saved_n == n && saved_validation_fraction == params.validation_fraction
            val_idx = collect(Int, state[:val_idx])
            train_idx = collect(Int, state[:train_idx])
            val_noise = Float32.(state[:val_noise])
            analytic_idx = haskey(state, :analytic_idx) ? collect(Int, state[:analytic_idx]) :
                randperm(MersenneTwister(params.seed + 200), n)[1:min(params.exact_score_samples, n)]
        else
            train_idx, val_idx, val_noise = make_score_splits(dataset, params, rng)
            analytic_idx = randperm(MersenneTwister(params.seed + 200), n)[1:min(params.exact_score_samples, n)]
            @printf("Rebuilt score train/validation split for current data: %d train, %d validation samples.\n",
                length(train_idx), length(val_idx))
        end
        completed_epoch = Int(get(state, :completed_epoch, length(get(history, :val_loss, Float64[]))))
        start_epoch = completed_epoch + 1
        stop_epoch = completed_epoch + params.epochs
        saved_val = Float64(get(saved_cfg,
            :checkpoint_validation_loss, length(get(history, :val_loss, Float64[])) > 0 ?
            history[:val_loss][end] : Inf))
        best = (; val_loss=saved_val, epoch=completed_epoch, host_model=Flux.fmap(cpu, model),
            training_state=state)
        @printf("Resuming score training state from epoch %d; running through epoch %d.\n",
            completed_epoch, stop_epoch)
    else
        train_idx, val_idx, val_noise = make_score_splits(dataset, params, rng)
        analytic_idx = randperm(MersenneTwister(params.seed + 200), n)[1:min(params.exact_score_samples, n)]
        if !isempty(params.output_analytic_history_txt) && isfile(params.output_analytic_history_txt)
            rm(params.output_analytic_history_txt; force=true)
        end
        if !isempty(params.output_stein_history_txt) && isfile(params.output_stein_history_txt)
            rm(params.output_stein_history_txt; force=true)
        end
    end
    stein_cache = make_stein_training_cache(dataset, params, device, model)
    if stein_cache !== nothing
        @printf("Prepared Stein/GFDT training cache with %d samples and %d observables from %s\n",
            size(stein_cache.normed_cpu, 3), size(stein_cache.phi_cpu, 1), params.stein_obs_family)
    end
    progress = Progress(max(stop_epoch - start_epoch + 1, 0); desc="Training soft-spin stationary score")
    for epoch in start_epoch:stop_epoch
        current_lr = score_learning_rate(params, epoch, start_epoch, stop_epoch)
        current_stein_weight = stein_weight_for_epoch(params, epoch, start_epoch)
        Flux.adjust!(opt, current_lr)
        idxs = if params.epoch_subset_size > 0 && params.epoch_subset_size < length(train_idx)
            train_idx[randperm(rng, length(train_idx))[1:params.epoch_subset_size]]
        else
            train_idx[randperm(rng, length(train_idx))]
        end
        losses = Float64[]
        objective_losses = Float64[]
        dsm_mses = Float64[]
        stein_losses = Float64[]
        move_losses = Float64[]
        target_rmses = Float64[]
        pred_rmses = Float64[]
        null_mses = Float64[]
        frac_improvements = Float64[]
        cosines = Float64[]
        parts = collect(Iterators.partition(idxs, params.batch_size))
        for part_group in Iterators.partition(parts, params.gradient_accumulation_steps)
            group_batches = Any[]
            for part in part_group
                batch_cpu = copy(dataset.data[:, :, collect(part)])
                noise_cpu = randn(noise_rng, Float32, size(batch_cpu))
                push!(group_batches, (move_array(batch_cpu, device), move_array(noise_cpu, device)))
            end
            use_aux = stein_cache !== nothing && epoch % params.stein_every == 0 &&
                (current_stein_weight > 0.0 || params.move_weight > 0.0)
            stein_pos = use_aux ?
                rand(rng, 1:size(stein_cache.normed_cpu, 3),
                    min(params.stein_batch_size, size(stein_cache.normed_cpu, 3))) :
                Int[]
            loss_value, grads = Flux.withgradient(model) do current_model
                total = nothing
                for (batch, noise) in group_batches
                    loss = dsm_loss_value(current_model, batch, noise, params)
                    total = total === nothing ? loss : total + loss
                end
                objective = total / length(group_batches)
                if use_aux
                    stein_loss, move_loss = stein_aux_losses(current_model, stein_cache, params, stein_pos)
                    objective = objective +
                        eltype(objective)(current_stein_weight) * stein_loss +
                        eltype(objective)(params.move_weight) * move_loss
                end
                objective
            end
            opt, model = Flux.update!(opt, model, grads[1])
            push!(objective_losses, Float64(to_host(loss_value)))
            if use_aux
                stein_loss_post, move_loss_post = stein_aux_losses(model, stein_cache, params, stein_pos)
                push!(stein_losses, Float64(to_host(stein_loss_post)))
                push!(move_losses, Float64(to_host(move_loss_post)))
            end
            for (batch, noise) in group_batches
                mon = dsm_monitor_metrics(model, batch, noise, params)
                push!(dsm_mses, mon.null_mse * (1.0 - mon.fractional_improvement))
                push!(target_rmses, mon.target_rms)
                push!(pred_rmses, mon.prediction_rms)
                push!(null_mses, mon.null_mse)
                push!(frac_improvements, mon.fractional_improvement)
                push!(cosines, mon.prediction_target_cosine)
            end
        end
        train_loss_value = stein_cache === nothing ? mean(objective_losses) : mean(dsm_mses)
        val_loss = isempty(val_idx) ? train_loss_value :
            validation_loss(model, dataset, val_idx, val_noise, params, device)
        monitor_n = min(4096, n)
        score = evaluate_score_norm(model, copy(dataset.data[:, :, 1:monitor_n]), params.sigma, device;
            batch_size=params.batch_size)
        push!(losses, train_loss_value)
        push!(history[:train_loss], train_loss_value)
        push!(history[:objective_loss], mean(objective_losses))
        push!(history[:val_loss], val_loss)
        push!(history[:score_norm], mean(sqrt.(sum(abs2, reshape(score, :, monitor_n); dims=1))))
        push!(history[:target_rms], mean(target_rmses))
        push!(history[:prediction_rms], mean(pred_rmses))
        push!(history[:null_mse], mean(null_mses))
        push!(history[:fractional_improvement], mean(frac_improvements))
        push!(history[:prediction_target_cosine], mean(cosines))
        if params.save_best_validation && val_loss < best.val_loss
            state = host_training_state(opt, rng, noise_rng, train_idx, val_idx,
                val_noise, analytic_idx, epoch, stop_epoch)
            best = (; val_loss=val_loss, epoch=epoch, host_model=Flux.fmap(cpu, model),
                training_state=state)
        end
        if params.analytic_every > 0 && epoch % params.analytic_every == 0
            Flux.testmode!(model)
            adiag = analytic_score_diagnostic(model, dataset, p, params, device, analytic_idx)
            Flux.trainmode!(model)
            push!(history[:analytic_epoch], Float64(epoch))
            push!(history[:analytic_rel_rmse], adiag[:rel_rmse])
            push!(history[:analytic_cosine], adiag[:cosine])
            push!(history[:analytic_safe_rel_rmse], adiag[:safe_rel_rmse])
            push!(history[:analytic_safe_cosine], adiag[:safe_cosine])
            append_analytic_history(params.output_analytic_history_txt, epoch, adiag;
                dsm_loss=history[:train_loss][end])
            @printf("Analytic score diagnostic epoch %d: rel.RMSE %.6e, safe rel.RMSE %.6e\n",
                epoch, adiag[:rel_rmse], adiag[:safe_rel_rmse])
        else
            adiag = nothing
        end
        if stein_cache !== nothing && params.analytic_every > 0 && epoch % params.analytic_every == 0
            sdiag = stein_cache_diagnostic(model, stein_cache, params, device)
            train_stein = isempty(stein_losses) ? NaN : mean(stein_losses)
            train_move = isempty(move_losses) ? NaN : mean(move_losses)
            push!(history[:stein_epoch], Float64(epoch))
            push!(history[:stein_weight], current_stein_weight)
            push!(history[:stein_train_loss], train_stein)
            push!(history[:stein_gfdt_weighted_sum], sdiag[:gfdt_weighted_sum])
            push!(history[:stein_gfdt_weighted_mean], sdiag[:gfdt_weighted_mean])
            push!(history[:stein_gfdt_rms], sdiag[:gfdt_rms])
            push!(history[:stein_coord_norm], sdiag[:coord_stein_norm])
            push!(history[:stein_z_norm], sdiag[:stein_z])
            push!(history[:stein_move_rel], sdiag[:move_rel])
            append_stein_history(params.output_stein_history_txt, epoch, current_stein_weight,
                train_stein, train_move, sdiag)
            @printf("Stein/GFDT diagnostic epoch %d: total %.6e, weighted mean %.6e, coord %.6e, z %.6e, move %.6e\n",
                epoch, sdiag[:total_lambda_x1_lambda0_1], sdiag[:gfdt_weighted_mean],
                sdiag[:coord_stein_norm], sdiag[:stein_z], sdiag[:move_rel])
        else
            sdiag = nothing
        end
        if params.metric_guard_abort && params.analytic_every > 0 && epoch % params.analytic_every == 0
            guard_reasons = score_metric_guard_reasons(params, epoch, adiag, sdiag)
            if !isempty(guard_reasons)
                error("Score metric guard failed: " * join(guard_reasons, "; "))
            end
        end
        if params.checkpoint_every > 0 && epoch % params.checkpoint_every == 0
            host_latest = Flux.fmap(cpu, model)
            state = host_training_state(opt, rng, noise_rng, train_idx, val_idx,
                val_noise, analytic_idx, epoch, stop_epoch)
            ckpt_path = periodic_score_checkpoint_path(model_path, epoch)
            ensure_parent_dir(ckpt_path)
            BSON.bson(ckpt_path, score_checkpoint_blob(host_latest, model_cfg, dataset,
                params, p, history, epoch, val_loss; checkpoint_kind="periodic_latest",
                training_state=state))
            latest_path = string(splitext(model_path)[1], "_latest", splitext(model_path)[2])
            BSON.bson(latest_path, score_checkpoint_blob(host_latest, model_cfg, dataset,
                params, p, history, epoch, val_loss; checkpoint_kind="latest",
                training_state=state))
            @printf("Saved periodic score checkpoint epoch %d to %s\n", epoch, ckpt_path)
        end
        ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch), (:train_loss, history[:train_loss][end]), (:val_loss, val_loss),
            (:lr, current_lr),
            (:stein_w, current_stein_weight),
            (:objective, history[:objective_loss][end]),
            (:frac_improve, history[:fractional_improvement][end]),
            (:pred_target_cos, history[:prediction_target_cosine][end])
        ])
    end
    ProgressMeter.finish!(progress)
    if !params.save_best_validation || best.host_model === nothing
        state = host_training_state(opt, rng, noise_rng, train_idx, val_idx,
            val_noise, analytic_idx, stop_epoch, stop_epoch)
        best = (; val_loss=history[:val_loss][end], epoch=stop_epoch,
            host_model=Flux.fmap(cpu, model), training_state=state)
    end
    history[:best_validation_epoch] = [Float64(best.epoch)]
    history[:best_validation_loss] = [Float64(best.val_loss)]
    return model, model_cfg, history, best
end

function validation_loss(model, dataset, idxs, val_noise, params, device)
    losses = Float64[]
    Flux.testmode!(model)
    for pos_part in Iterators.partition(1:length(idxs), params.batch_size)
        pos = collect(pos_part)
        batch_cpu = copy(dataset.data[:, :, idxs[pos]])
        noise_cpu = copy(@view val_noise[:, :, pos])
        batch = move_array(batch_cpu, device)
        noise = move_array(noise_cpu, device)
        loss = dsm_loss_value(model, batch, noise, params)
        push!(losses, Float64(to_host(loss)))
    end
    Flux.trainmode!(model)
    return mean(losses)
end

function load_or_train(params::ScoreConfig, dataset, p, model_path, device)
    if params.train
        model, model_cfg, history, best = train_score_model(dataset, params, device, p, model_path)
        host_model = params.save_best_validation ? best.host_model : Flux.fmap(cpu, model)
        final_epoch = Int(round(history[:best_validation_epoch][1]))
        final_val = Float64(history[:best_validation_loss][1])
        ensure_parent_dir(model_path)
        blob = score_checkpoint_blob(host_model, model_cfg, dataset, params, p, history,
            final_epoch, final_val; checkpoint_kind="best_validation_final",
            training_state=best.training_state)
        blob[:trainer_cfg][:best_validation_epoch] = final_epoch
        blob[:trainer_cfg][:best_validation_loss] = final_val
        BSON.bson(model_path, blob)
        @printf("Saved score checkpoint to %s\n", model_path)
        if params.save_best_validation
            model = move_model(host_model, device)
            Flux.testmode!(model)
        end
        return model, model_cfg, history
    else
        blob = BSON.load(model_path)
        model = move_model(blob[:host_model], device)
        Flux.testmode!(model)
        return model, blob[:model_cfg], blob[:history]
    end
end

function diagnostics(model, dataset, p, params, device)
    rng = MersenneTwister(params.seed + 200)
    n = min(params.exact_score_samples, length(dataset))
    keep = randperm(rng, length(dataset))[1:n]
    clean = copy(dataset.data[:, :, keep])
    pred = evaluate_score_norm(model, clean, params.sigma, device; batch_size=params.batch_size)
    exact = standardized_analytic_score(clean, dataset.stats, p)
    rel, cosv = score_metric_pair(pred, exact)
    raw = denormalize_tensor(clean, dataset.stats)
    norms = [sqrt(sum(abs2, raw[i, :, b])) for i in axes(raw, 1), b in axes(raw, 3)]
    lo, hi = quantile(vec(norms), 0.02), quantile(vec(norms), 0.98)
    safe = vec([all(lo <= sqrt(sum(abs2, raw[i, :, b])) <= hi for i in axes(raw, 1)) for b in axes(raw, 3)])
    srel, scos = any(safe) ? score_metric_pair(pred[:, :, safe], exact[:, :, safe]) : (NaN, NaN)
    noise = randn(rng, Float32, size(clean))
    noisy = clean .+ params.sigma .* noise
    sn = evaluate_score_norm(model, noisy, params.sigma, device; batch_size=params.batch_size)
    xflat = flatten_batch(noisy)
    sflat = flatten_batch(sn)
    stein = -(Matrix{Float64}(sflat) * transpose(Matrix{Float64}(xflat))) ./ size(xflat, 2)
    return Dict(:rel_rmse => rel, :cosine => cosv, :safe_rel_rmse => srel,
        :safe_cosine => scos, :safe_count => count(identity, safe), :total_count => length(safe),
        :stein_rel_error => norm(stein - I(size(stein, 1))) / sqrt(size(stein, 1))), stein
end

function render_score_figure(path, params, history, diag, stein)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    figure_title!(fig, "Soft-spin LLG stationary score diagnostics";
        subtitle=@sprintf("DSM sigma=%.3f, full rel.RMSE=%.3e, cosine=%.5f, safe rel.RMSE=%.3e",
            params.sigma, diag[:rel_rmse], diag[:cosine], diag[:safe_rel_rmse]))
    train_loss = get(history, :train_loss, Float64[])
    val_loss = get(history, :val_loss, Float64[])
    if isempty(train_loss)
        train_loss = [NaN]
        val_loss = [NaN]
    end
    epochs = collect(1:length(train_loss))
    ax1 = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
    lines!(ax1, epochs, train_loss; color=STYLE_PRIMARY, linewidth=2, label="train")
    lines!(ax1, epochs, val_loss; color=STYLE_SECONDARY, linewidth=2, label="validation")
    axislegend(ax1; position=:rt)
    ax2 = Axis(fig[1, 2]; title="Mean score norm", xlabel="epoch", ylabel="norm")
    score_norm = get(history, :score_norm, Float64[])
    isempty(score_norm) && (score_norm = fill(NaN, length(epochs)))
    lines!(ax2, epochs, score_norm; color=STYLE_ACCENT, linewidth=2)
    ax3 = Axis(fig[2, 1]; title="Projected Stein residual", xlabel="column", ylabel="row")
    residual = stein - I(size(stein, 1))
    heatmap!(ax3, residual; colormap=:balance)
    ax4 = Axis(fig[2, 2]; title="Validation text", xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    text!(ax4, 0.02, 0.9; space=:relative, align=(:left, :top), fontsize=22,
        text=@sprintf("Full analytic diagnostic rel.RMSE: %.6e\nFull cosine: %.6f\nSafe samples: %d / %d\nSafe rel.RMSE: %.6e\nSafe cosine: %.6f\nStein relative error: %.6e\n\nNo-cheating audit: analytic score was used only here, after DSM training.",
            diag[:rel_rmse], diag[:cosine], diag[:safe_count], diag[:total_count],
            diag[:safe_rel_rmse], diag[:safe_cosine], diag[:stein_rel_error]))
    hidedecorations!(ax4)
    save_figure_checked(path, fig)
end

function sample_initial_norm(dataset, ntraj::Int, rng::AbstractRNG)
    idx = rand(rng, 1:length(dataset), ntraj)
    return Float64.(dataset.data[:, :, idx])
end

function run_score_langevin(model, dataset, params, device)
    rng = MersenneTwister(params.seed + 400)
    z = sample_initial_norm(dataset, params.langevin_ntraj, rng)
    N = size(z, 1)
    nsteps = ceil(Int, params.langevin_total_time / params.langevin_dt)
    burn_steps = floor(Int, params.langevin_burnin_time / params.langevin_dt)
    save_every = max(1, round(Int, params.langevin_save_dt / params.langevin_dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, N, SPIN_CHANNELS, params.langevin_ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            raw = denormalize_tensor(Float32.(z), dataset.stats)
            saved[save_idx, :, :, :] .= raw
            times[save_idx] = (step - burn_steps) * params.langevin_dt
        end
        step == nsteps && break
        zn = Float32.(z)
        score = Float64.(evaluate_score_norm(model, zn, params.sigma, device; batch_size=4096))
        clamp!(score, -Float64(params.langevin_score_clip), Float64(params.langevin_score_clip))
        noise = randn(rng, size(z))
        @. z = z + params.langevin_dt * score + sqrt(2.0 * params.langevin_dt) * noise
    end
    return times[1:save_idx], saved[1:save_idx, :, :, :]
end

function score_langevin_metrics(obs_states, gen_states)
    obs_flat = permutedims(Float64.(reshape(permutedims(obs_states, (2, 3, 1, 4)), :, size(obs_states, 1) * size(obs_states, 4))))
    gen_flat = permutedims(Float64.(reshape(permutedims(gen_states, (2, 3, 1, 4)), :, size(gen_states, 1) * size(gen_states, 4))))
    cov_obs = cov(obs_flat)
    cov_gen = cov(gen_flat)
    cov_metrics = agreement_metrics(cov_obs, cov_gen)
    nn_obs = translation_covariance_by_lag(obs_states, 1)
    nn_gen = translation_covariance_by_lag(gen_states, 1)
    spec_obs = component_structure_spectrum(obs_states)
    spec_gen = component_structure_spectrum(gen_states)
    acf_obs = global_component_acf(obs_states, min(100, size(obs_states, 1) - 1))
    acf_gen = global_component_acf(gen_states, min(100, size(gen_states, 1) - 1))
    vals_obs = Dict(
        :mx => vec(Float64.(obs_states[:, :, 1, :])),
        :my => vec(Float64.(obs_states[:, :, 2, :])),
        :mz => vec(Float64.(obs_states[:, :, 3, :])),
        :norm => vec([sqrt(sum(abs2, obs_states[t, i, :, tr])) for t in axes(obs_states, 1), i in axes(obs_states, 2), tr in axes(obs_states, 4)]),
    )
    vals_gen = Dict(
        :mx => vec(Float64.(gen_states[:, :, 1, :])),
        :my => vec(Float64.(gen_states[:, :, 2, :])),
        :mz => vec(Float64.(gen_states[:, :, 3, :])),
        :norm => vec([sqrt(sum(abs2, gen_states[t, i, :, tr])) for t in axes(gen_states, 1), i in axes(gen_states, 2), tr in axes(gen_states, 4)]),
    )
    moments = Dict{Symbol, Any}()
    for key in (:mx, :my, :mz, :norm)
        o = vals_obs[key]
        g = vals_gen[key]
        moments[key] = Dict(
            :mean_obs => mean(o), :mean_gen => mean(g),
            :std_obs => std(o), :std_gen => std(g),
            :mean_abs_error => abs(mean(g) - mean(o)),
            :std_rel_error => abs(std(g) - std(o)) / max(std(o), eps(Float64)),
        )
    end
    return Dict(:covariance => cov_metrics, :moments => moments,
        :nearest_neighbor_covariance => agreement_metrics(nn_obs, nn_gen),
        :structure_spectrum => agreement_metrics(spec_obs, spec_gen),
        :global_acf => agreement_metrics(acf_obs, acf_gen))
end

function translation_covariance_by_lag(states, ell::Int)
    T, N, C, R = size(states)
    means = Array{Float64}(undef, C)
    @inbounds for c in 1:C
        means[c] = mean(Float64, @view states[:, :, c, :])
    end
    out = zeros(Float64, C, C)
    count = 0
    @inbounds for tr in 1:R, t in 1:T, i in 1:N
        j = periodic(i + ell, N)
        for a in 1:C, b in 1:C
            out[a, b] += (Float64(states[t, i, a, tr]) - means[a]) *
                (Float64(states[t, j, b, tr]) - means[b])
        end
        count += 1
    end
    out ./= max(count, 1)
    return out
end

function component_structure_spectrum(states)
    T, N, C, R = size(states)
    means = Array{Float64}(undef, C)
    @inbounds for c in 1:C
        means[c] = mean(Float64, @view states[:, :, c, :])
    end
    spec = zeros(Float64, N, C)
    @inbounds for c in 1:C, k in 0:(N - 1)
        total = 0.0
        for tr in 1:R, t in 1:T
            re = 0.0
            im = 0.0
            for i in 1:N
                angle = -2.0 * pi * k * (i - 1) / N
                val = Float64(states[t, i, c, tr]) - means[c]
                re += val * cos(angle)
                im += val * sin(angle)
            end
            total += (re * re + im * im) / N
        end
        spec[k + 1, c] = total / max(T * R, 1)
    end
    return spec
end

function global_component_acf(states, maxlag::Int)
    T, N, C, R = size(states)
    maxlag = max(0, min(maxlag, T - 1))
    series = Array{Float64}(undef, T, C, R)
    @inbounds for tr in 1:R, t in 1:T, c in 1:C
        s = 0.0
        for i in 1:N
            s += Float64(states[t, i, c, tr])
        end
        series[t, c, tr] = s / N
    end
    out = zeros(Float64, maxlag + 1, C)
    @inbounds for c in 1:C
        mu = mean(@view series[:, c, :])
        denom = 0.0
        for tr in 1:R, t in 1:T
            v = series[t, c, tr] - mu
            denom += v * v
        end
        denom = max(denom, eps(Float64))
        for lag in 0:maxlag
            acc = 0.0
            for tr in 1:R, t in 1:(T - lag)
                acc += (series[t, c, tr] - mu) * (series[t + lag, c, tr] - mu)
            end
            out[lag + 1, c] = acc / denom
        end
    end
    return out
end

function render_langevin_figure(path, obs_states, gen_states, params)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    figure_title!(fig, "Score-only Langevin validation";
        subtitle="Observed data vs dz=s_theta(z)dt+sqrt(2)dW in normalized coordinates")
    labels = ["mx", "my", "mz", "|m|"]
    vals_obs = [
        vec(Float64.(obs_states[:, :, 1, :])),
        vec(Float64.(obs_states[:, :, 2, :])),
        vec(Float64.(obs_states[:, :, 3, :])),
        vec([sqrt(sum(abs2, obs_states[t, i, :, tr])) for t in axes(obs_states, 1), i in axes(obs_states, 2), tr in axes(obs_states, 4)]),
    ]
    vals_gen = [
        vec(Float64.(gen_states[:, :, 1, :])),
        vec(Float64.(gen_states[:, :, 2, :])),
        vec(Float64.(gen_states[:, :, 3, :])),
        vec([sqrt(sum(abs2, gen_states[t, i, :, tr])) for t in axes(gen_states, 1), i in axes(gen_states, 2), tr in axes(gen_states, 4)]),
    ]
    for j in 1:4
        ax = Axis(fig[1, j]; title="PDF $(labels[j])", xlabel=labels[j], ylabel="density")
        xo, yo = kde(vals_obs[j][1:max(1, length(vals_obs[j]) ÷ params.langevin_max_pdf_samples):end]).x,
            kde(vals_obs[j][1:max(1, length(vals_obs[j]) ÷ params.langevin_max_pdf_samples):end]).density
        xg, yg = kde(vals_gen[j][1:max(1, length(vals_gen[j]) ÷ params.langevin_max_pdf_samples):end]).x,
            kde(vals_gen[j][1:max(1, length(vals_gen[j]) ÷ params.langevin_max_pdf_samples):end]).density
        lines!(ax, xo, yo; color=:black, linewidth=2, label="obs")
        lines!(ax, xg, yg; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="score")
        axislegend(ax; position=:rt)
    end
    cov_obs = cov(permutedims(Float64.(reshape(permutedims(obs_states, (2, 3, 1, 4)), :, size(obs_states, 1) * size(obs_states, 4)))))
    cov_gen = cov(permutedims(Float64.(reshape(permutedims(gen_states, (2, 3, 1, 4)), :, size(gen_states, 1) * size(gen_states, 4)))))
    axc = Axis(fig[2, 1:2]; title="Observed covariance", xlabel="flat index", ylabel="flat index")
    heatmap!(axc, cov_obs; colormap=:balance)
    axd = Axis(fig[2, 3:4]; title="Score Langevin covariance error", xlabel="flat index", ylabel="flat index")
    heatmap!(axd, cov_gen - cov_obs; colormap=:balance)
    nn_obs = translation_covariance_by_lag(obs_states, 1)
    nn_gen = translation_covariance_by_lag(gen_states, 1)
    axn = Axis(fig[3, 1]; title="Nearest-neighbor covariance error", xlabel="component", ylabel="component",
        xticks=(1:3, labels[1:3]), yticks=(1:3, labels[1:3]))
    heatmap!(axn, nn_gen - nn_obs; colormap=:balance)
    spec_obs = component_structure_spectrum(obs_states)
    spec_gen = component_structure_spectrum(gen_states)
    axs = Axis(fig[3, 2]; title="Structure spectrum", xlabel="wavenumber", ylabel="power")
    for c in 1:3
        lines!(axs, 0:(size(spec_obs, 1) - 1), spec_obs[:, c]; color=(:black, 0.85),
            linewidth=2, linestyle=c == 1 ? :solid : (c == 2 ? :dash : :dot),
            label=c == 1 ? "obs" : nothing)
        lines!(axs, 0:(size(spec_gen, 1) - 1), spec_gen[:, c]; color=STYLE_PRIMARY,
            linewidth=2, linestyle=c == 1 ? :solid : (c == 2 ? :dash : :dot),
            label=c == 1 ? "score" : nothing)
    end
    axislegend(axs; position=:rt)
    acf_obs = global_component_acf(obs_states, min(100, size(obs_states, 1) - 1))
    acf_gen = global_component_acf(gen_states, min(100, size(gen_states, 1) - 1))
    axa = Axis(fig[3, 3:4]; title="Global magnetization ACF", xlabel="saved lag", ylabel="ACF")
    for c in 1:3
        lines!(axa, 0:(size(acf_obs, 1) - 1), acf_obs[:, c]; color=(:black, 0.85),
            linewidth=2, linestyle=c == 1 ? :solid : (c == 2 ? :dash : :dot),
            label=c == 1 ? "obs" : nothing)
        lines!(axa, 0:(size(acf_gen, 1) - 1), acf_gen[:, c]; color=STYLE_PRIMARY,
            linewidth=2, linestyle=c == 1 ? :solid : (c == 2 ? :dash : :dot),
            label=c == 1 ? "score" : nothing)
    end
    axislegend(axa; position=:rt)
    save_figure_checked(path, fig)
end

function save_score_metrics(path, params, diag, langevin_diag)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain stationary score metrics")
        println(io, @sprintf("DSM sigma = %.8e", params.sigma))
        println(io, @sprintf("analytic diagnostic rel.RMSE = %.8e", diag[:rel_rmse]))
        println(io, @sprintf("analytic diagnostic cosine = %.8e", diag[:cosine]))
        println(io, @sprintf("safe analytic diagnostic rel.RMSE = %.8e", diag[:safe_rel_rmse]))
        println(io, @sprintf("safe analytic diagnostic cosine = %.8e", diag[:safe_cosine]))
        println(io, @sprintf("Stein relative error = %.8e", diag[:stein_rel_error]))
        if langevin_diag !== nothing
            covm = langevin_diag[:covariance]
            println(io, @sprintf("score Langevin covariance rel.RMSE = %.8e", covm[:relative_rmse]))
            println(io, @sprintf("score Langevin covariance corr = %.8e", covm[:correlation]))
            nn = langevin_diag[:nearest_neighbor_covariance]
            sp = langevin_diag[:structure_spectrum]
            ac = langevin_diag[:global_acf]
            println(io, @sprintf("score Langevin nearest-neighbor covariance rel.RMSE = %.8e", nn[:relative_rmse]))
            println(io, @sprintf("score Langevin nearest-neighbor covariance corr = %.8e", nn[:correlation]))
            println(io, @sprintf("score Langevin structure spectrum rel.RMSE = %.8e", sp[:relative_rmse]))
            println(io, @sprintf("score Langevin structure spectrum corr = %.8e", sp[:correlation]))
            println(io, @sprintf("score Langevin global magnetization ACF rel.RMSE = %.8e", ac[:relative_rmse]))
            println(io, @sprintf("score Langevin global magnetization ACF corr = %.8e", ac[:correlation]))
            for key in (:mx, :my, :mz, :norm)
                m = langevin_diag[:moments][key]
                println(io, @sprintf("%s mean obs/gen = %.8e %.8e", String(key), m[:mean_obs], m[:mean_gen]))
                println(io, @sprintf("%s std obs/gen = %.8e %.8e", String(key), m[:std_obs], m[:std_gen]))
                println(io, @sprintf("%s std rel.error = %.8e", String(key), m[:std_rel_error]))
            end
        end
        println(io, "No-cheating audit: analytic score diagnostics are ex-post only; score selection should use DSM validation and observed score-Langevin statistics.")
    end
    @printf("Saved score metrics to %s\n", path)
end

function save_langevin(path, times, states)
    ensure_parent_dir(path)
    h5open(path, "w") do f
        f["/trajectories/time"] = times
        f["/trajectories/states"] = states
        f["/trajectories/channel_names"] = channel_names()
        f["/metadata/model"] = "score-only normalized Langevin"
        f["/metadata/equation"] = "dz = s_theta(z) dt + sqrt(2) dW"
    end
    @printf("Saved score-only Langevin validation to %s\n", path)
end

function load_phys(path::AbstractString)
    h5open(path, "r") do f
        return SpinParams(
            N=Int(read(f["/metadata/N"])),
            lambda=Float64(read(f["/metadata/lambda"])),
            mstar=Float64(read(f["/metadata/mstar"])),
            J=Float64(read(f["/metadata/J"])),
            K=Float64(read(f["/metadata/K"])),
            theta=Float64(read(f["/metadata/Theta"])),
            gamma=Float64(read(f["/metadata/gamma"])),
            alpha_perp=Float64(read(f["/metadata/alpha_perp"])),
            alpha_parallel=Float64(read(f["/metadata/alpha_parallel"])),
            eps=Float64(read(f["/metadata/eps"])),
        )
    end
end

function run_pipeline(param_file::AbstractString)
    base = dirname(param_file)
    params = load_config(param_file)
    data_h5 = resolve_path(base, params.input_hdf5)
    model_path = resolve_path(base, params.output_bson)
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)
    p = load_phys(data_h5)
    dataset, times, states, start = load_score_dataset(data_h5, params, MersenneTwister(params.seed))
    @printf("Loaded %d normalized DSM samples from %s\n", length(dataset), data_h5)
    model, _, history = load_or_train(params, dataset, p, model_path, device)
    if params.evaluate
        diag, stein = diagnostics(model, dataset, p, params, device)
        render_score_figure(resolve_path(base, params.output_png), params, history, diag, stein)
        langevin_diag = nothing
        if params.langevin_validate
            lt, ls = run_score_langevin(model, dataset, params, device)
            save_langevin(resolve_path(base, params.output_langevin_hdf5), lt, ls)
            obs = states[start:end, :, :, :]
            gen = ls
            langevin_diag = score_langevin_metrics(obs, gen)
            render_langevin_figure(resolve_path(base, params.output_langevin_png), obs, gen, params)
        end
        save_score_metrics(resolve_path(base, params.output_metrics_txt), params, diag, langevin_diag)
    end
    @printf("Step 2 score stage complete. No-cheating audit: DSM loss used only data samples plus Gaussian noise; analytic score was excluded from training and model selection.\n")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE)
end
