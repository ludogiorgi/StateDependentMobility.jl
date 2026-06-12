#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "cond_score.toml"))

Base.@kwdef struct PhiConfig
    input_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    max_fit_lags::Int
    pairs_per_lag_phi::Int
    pairs_per_lag_cdot::Int
    phi_projected_fit::Bool
    phi_projected_pairs_per_lag::Int
    phi_projected_chunk_size::Int
    phi_projected_sampling::String
    phi_fit_max_lag::Int
    phi_fit_degree::Int
    phi_include_zero_lag::Bool
    phi_zero_lag_samples::Int
    phi_shortlag_run::Bool
    phi_shortlag_samples::Int
    phi_shortlag_dt::Float64
    phi_shortlag_steps::Vector{Int}
    score_batch_size::Int
    true_mobility_samples::Int
    forward_run::Bool
    forward_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_save_dt::Float64
    forward_ntraj::Int
    forward_score_clip::Float32
    figure_width::Int
    figure_height::Int
    artifact_bson::String
    metrics_txt::String
    cdot_figure_png::String
    phi_figure_png::String
    forward_stats_png::String
    forward_cmn_png::String
    forward_hdf5::String
    device::String
    required_gpu_name::String
    seed::Int
    verbose::Bool
end

Base.@kwdef struct CondScoreConfig
    input_hdf5::String
    score_bson::String
    phi_artifact_bson::String
    burnin_fraction::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    model_config::ScoreUNetConfig
    architecture::Symbol
    normalization::Symbol
    input_features::Symbol
    include_delta_input::Bool
    include_tau_scalar::Bool
    time_fourier_frequencies::Int
    residual_dilations::Vector{Int}
    residual_repeats::Int
    physical_hidden_width::Int
    physical_hidden_depth::Int
    physical_output_scale::Float32
    init_seed::Int
    sigma::Float32
    epochs::Int
    batches_per_epoch::Int
    batch_size::Int
    learning_rate::Float64
    use_lr_schedule::Bool
    warmup_steps::Int
    min_lr_factor::Float64
    gradient_clip_value::Float64
    target_clip_value::Float64
    mean_score_weight::Float64
    stein_weight::Float64
    lag_sampling_power::Float64
    train_min_lag_step::Int
    train_max_lag_step::Int
    score_target::Symbol
    residual_output_scale::Symbol
    checkpoint_every::Int
    resume::Bool
    seed::Int
    eval_tau_count::Int
    eval_pairs_per_lag::Int
    operator_pairs_per_lag::Int
    operator_lag_count::Int
    operator_aux_target_bson::String
    operator_aux_weight::Float64
    operator_aux_batch_size::Int
    operator_aux_lag_first::Int
    operator_aux_lag_last::Int
    operator_aux_scale_floor::Float64
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    metrics_txt::String
    device::String
    required_gpu_name::String
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

struct CondOperatorAuxTarget
    path::String
    target_phi_group::Matrix{Float32}
    lags::Vector{Int}
    taus::Vector{Float64}
    names::Vector{String}
    means::Vector{Float64}
    Phi::Matrix{Float32}
    group_index_matrix::Matrix{Int}
    scale_vec::Vector{Float32}
    score_radial_scale::Float64
    radial_basis::Matrix{Float64}
end

struct CondPairSampler
    times::Vector{Float64}
    states::Array{Float32, 4}
    start_idx::Int
    save_dt::Float64
    N::Int
    D::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    tau_max::Float64
    tD::Float64
end

function cond_extra_feature_channels(input_features::Symbol)
    input_features == :basic && return 0
    input_features == :spin_r2_dot && return 3
    error("Unsupported conditional input_features=$(input_features).")
end

function cond_input_channels(nfreq::Int; include_delta_input::Bool,
        include_tau_scalar::Bool=false, input_features::Symbol=:basic)
    return 2SPIN_CHANNELS + (include_delta_input ? SPIN_CHANNELS : 0) +
        cond_extra_feature_channels(input_features) +
        (include_tau_scalar ? 1 : 0) + 2nfreq
end

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    train = raw["training"]
    diag = raw["diagnostics"]
    opaux = get(raw, "operator_aux", Dict{String, Any}())
    fig = raw["figure"]
    out = raw["output"]
    run = raw["run"]
    nfreq = Int(get(model, "time_fourier_frequencies", 8))
    include_delta = Bool(get(model, "include_delta_input", true))
    include_tau = Bool(get(model, "include_tau_scalar", false))
    architecture = Symbol(lowercase(String(get(model, "architecture", "unet"))))
    input_features = Symbol(lowercase(String(get(model, "input_features", "basic"))))
    cfg = ScoreUNetConfig(
        in_channels=cond_input_channels(nfreq; include_delta_input=include_delta,
            include_tau_scalar=include_tau, input_features=input_features),
        base_channels=Int(get(model, "base_channels", 128)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(get(model, "activation", "swish")),
        final_activation=activation_from_string(get(model, "final_activation", "identity")),
    )
    params = CondScoreConfig(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        phi_artifact_bson=String(data["phi_artifact_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 0.6)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        model_config=cfg,
        architecture=architecture,
        normalization=normalization_from_string(get(model, "normalization", "none")),
        input_features=input_features,
        include_delta_input=include_delta,
        include_tau_scalar=include_tau,
        time_fourier_frequencies=nfreq,
        residual_dilations=Int.(get(model, "residual_dilations", [1, 2, 4, 8, 4, 2, 1])),
        residual_repeats=Int(get(model, "residual_repeats", 1)),
        physical_hidden_width=Int(get(model, "physical_hidden_width", 160)),
        physical_hidden_depth=Int(get(model, "physical_hidden_depth", 4)),
        physical_output_scale=Float32(get(model, "physical_output_scale", 1.0)),
        init_seed=Int(get(model, "init_seed", 20260510)),
        sigma=Float32(get(train, "sigma", 0.05)),
        epochs=Int(get(train, "epochs", 120)),
        batches_per_epoch=Int(get(train, "batches_per_epoch", 256)),
        batch_size=Int(get(train, "batch_size", 4096)),
        learning_rate=Float64(get(train, "learning_rate", 1.5e-4)),
        use_lr_schedule=Bool(get(train, "use_lr_schedule", true)),
        warmup_steps=Int(get(train, "warmup_steps", 500)),
        min_lr_factor=Float64(get(train, "min_lr_factor", 0.08)),
        gradient_clip_value=Float64(get(train, "gradient_clip_value", 0.0)),
        target_clip_value=Float64(get(train, "target_clip_value", 0.0)),
        mean_score_weight=Float64(get(train, "mean_score_weight", 0.0)),
        stein_weight=Float64(get(train, "stein_weight", 0.0)),
        lag_sampling_power=Float64(get(train, "lag_sampling_power", 1.0)),
        train_min_lag_step=Int(get(train, "train_min_lag_step", 0)),
        train_max_lag_step=Int(get(train, "train_max_lag_step", 0)),
        score_target=Symbol(lowercase(String(get(train, "score_target", "residual")))),
        residual_output_scale=Symbol(lowercase(String(get(train, "residual_output_scale", "raw")))),
        checkpoint_every=Int(get(train, "checkpoint_every", 10)),
        resume=Bool(get(train, "resume", true)),
        seed=Int(get(train, "seed", 20260510)),
        eval_tau_count=Int(get(diag, "eval_tau_count", 8)),
        eval_pairs_per_lag=Int(get(diag, "eval_pairs_per_lag", 50000)),
        operator_pairs_per_lag=Int(get(diag, "operator_pairs_per_lag", 40000)),
        operator_lag_count=Int(get(diag, "operator_lag_count", 24)),
        operator_aux_target_bson=String(get(opaux, "target_bson", "")),
        operator_aux_weight=Float64(get(opaux, "weight", 0.0)),
        operator_aux_batch_size=Int(get(opaux, "batch_size", 1024)),
        operator_aux_lag_first=Int(get(opaux, "lag_first", 7)),
        operator_aux_lag_last=Int(get(opaux, "lag_last", 24)),
        operator_aux_scale_floor=Float64(get(opaux, "scale_floor", 1.0e-4)),
        figure_width=Int(get(fig, "width", 3400)),
        figure_height=Int(get(fig, "height", 2600)),
        output_bson=String(out["model_bson"]),
        output_png=String(out["figure_png"]),
        metrics_txt=String(out["metrics_txt"]),
        device=String(get(run, "device", "GPU:1")),
        required_gpu_name=String(get(run, "required_gpu_name", "5070")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
        verbose=Bool(get(run, "verbose", true)),
    )
    allow_nondefault_sigma = Bool(get(train, "allow_nondefault_sigma", false))
    require_condition(params.sigma == 0.05f0 || allow_nondefault_sigma,
        "Conditional DSM sigma must remain 0.05 unless training.allow_nondefault_sigma=true.")
    require_condition(params.model_config.periodic, "Conditional score model must be periodic.")
    require_condition(params.normalization != :batchnorm, "BatchNorm is forbidden.")
    require_condition(params.time_fourier_frequencies >= 1, "Use Fourier time features, not a scalar lag only.")
    require_condition(params.input_features in (:basic, :spin_r2_dot),
        "Conditional input_features must be basic or spin_r2_dot.")
    require_condition(params.architecture in (:unet, :physical_mlp, :largekernel_rescnn,
        :rescnn, :largekernel_cnn, :cnn),
        "Conditional architecture must be unet, physical_mlp, largekernel_rescnn, or largekernel_cnn.")
    require_condition(params.lag_sampling_power >= 1.0, "lag_sampling_power must be >= 1.")
    require_condition(params.train_min_lag_step >= 0, "train_min_lag_step must be nonnegative.")
    require_condition(params.train_max_lag_step >= 0, "train_max_lag_step must be nonnegative.")
    require_condition(params.train_max_lag_step == 0 ||
        params.train_min_lag_step <= params.train_max_lag_step,
        "train_min_lag_step must be <= train_max_lag_step when both are set.")
    require_condition(params.score_target in (:residual, :posterior),
        "score_target must be residual or posterior.")
    require_condition(params.residual_output_scale in (:raw, :sigma),
        "residual_output_scale must be raw or sigma.")
    require_condition(params.score_target == :residual || params.residual_output_scale == :raw,
        "residual_output_scale=sigma is currently implemented only for score_target=residual.")
    require_condition(params.operator_aux_weight >= 0.0,
        "operator_aux.weight must be nonnegative.")
    require_condition(params.operator_aux_batch_size > 0,
        "operator_aux.batch_size must be positive.")
    require_condition(params.operator_aux_lag_first >= 1,
        "operator_aux.lag_first must use one-based target lag indices.")
    require_condition(params.operator_aux_lag_last >= params.operator_aux_lag_first,
        "operator_aux.lag_last must be >= operator_aux.lag_first.")
    require_condition(params.operator_aux_scale_floor > 0,
        "operator_aux.scale_floor must be positive.")
    return params
end

function build_cond_sampler(path::AbstractString, burnin_fraction::Float64,
        tau_max_decorrelation_multiples::Float64, lag_stride::Int)
    times, states = load_spin_states(path)
    start = burnin_start_index(length(times), burnin_fraction)
    save_dt = times[2] - times[1]
    tD = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    tau_max = min(tau_max_decorrelation_multiples * tD, times[end] - times[start])
    max_lag = min(length(times) - start - 1, floor(Int, tau_max / save_dt))
    require_condition(max_lag >= 1, "No conditional-score lags available.")
    lag_steps = collect(1:lag_stride:max_lag)
    return CondPairSampler(times, states, start, save_dt, size(states, 2), 3size(states, 2),
        lag_steps, lag_steps .* save_dt, tau_max, tD)
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

function lr_schedule(step::Int, total_steps::Int, params::CondScoreConfig)
    if params.warmup_steps > 0 && step <= params.warmup_steps
        return params.learning_rate * step / params.warmup_steps
    end
    x = (step - params.warmup_steps) / max(total_steps - params.warmup_steps, 1)
    factor = params.min_lr_factor + 0.5 * (1 - params.min_lr_factor) * (1 + cos(pi * clamp(x, 0, 1)))
    return params.learning_rate * factor
end

function build_cond_model(params::CondScoreConfig, N::Int)
    Random.seed!(params.init_seed)
    if params.architecture == :physical_mlp
        feature_dim = physical_cond_feature_dim(params.time_fourier_frequencies;
            include_tau_scalar=params.include_tau_scalar)
        layers = Any[Dense(feature_dim => params.physical_hidden_width, swish)]
        for _ in 2:params.physical_hidden_depth
            push!(layers, Dense(params.physical_hidden_width => params.physical_hidden_width, swish))
        end
        push!(layers, Dense(params.physical_hidden_width => SPIN_CHANNELS))
        cfg = Dict(:architecture => "physical_mlp",
            :feature_dim => feature_dim,
            :hidden_width => params.physical_hidden_width,
            :hidden_depth => params.physical_hidden_depth,
            :time_fourier_frequencies => params.time_fourier_frequencies,
            :include_delta_input => params.include_delta_input,
            :include_tau_scalar => params.include_tau_scalar,
            :output_scale => params.physical_output_scale)
        model = PhysicalCondResidualMLP(Chain(layers...),
            params.time_fourier_frequencies, params.include_delta_input,
            params.include_tau_scalar, params.physical_output_scale)
    elseif params.architecture == :unet
        cfg = adjust_model_config_for_length(params.model_config, N)
        model = SpinConditionalResidualUNet(build_unet(cfg; normalization=params.normalization))
    elseif params.architecture == :largekernel_rescnn || params.architecture == :rescnn
        kernel = max(params.model_config.kernel_size, 9)
        cfg = ScoreUNetConfig(
            in_channels=params.model_config.in_channels,
            base_channels=params.model_config.base_channels,
            channel_multipliers=params.model_config.channel_multipliers,
            kernel_size=kernel,
            periodic=params.model_config.periodic,
            activation=params.model_config.activation,
                final_activation=params.model_config.final_activation,
        )
        backbone = build_periodic_residual_score_net(cfg.in_channels, SPIN_CHANNELS;
            base_channels=cfg.base_channels,
            kernel=kernel,
            dilations=params.residual_dilations,
            repeats=params.residual_repeats,
            activation=cfg.activation,
            normalization=params.normalization)
        model = SpinConditionalResidualUNet(backbone)
    elseif params.architecture == :largekernel_cnn || params.architecture == :cnn
        kernel = max(params.model_config.kernel_size, 9)
        cfg = ScoreUNetConfig(
            in_channels=params.model_config.in_channels,
            base_channels=params.model_config.base_channels,
            channel_multipliers=params.model_config.channel_multipliers,
            kernel_size=kernel,
            periodic=params.model_config.periodic,
            activation=params.model_config.activation,
            final_activation=params.model_config.final_activation,
        )
        backbone = build_periodic_cnn_score_net(cfg.in_channels, SPIN_CHANNELS;
            base_channels=cfg.base_channels,
            kernel=kernel,
            dilations=params.residual_dilations,
            repeats=params.residual_repeats,
            activation=cfg.activation,
            normalization=params.normalization)
        model = SpinConditionalResidualUNet(backbone)
    else
        error("Unsupported conditional-score architecture=$(params.architecture). " *
              "Use unet, physical_mlp, largekernel_rescnn, or largekernel_cnn.")
    end
    Random.seed!()
    return model, cfg
end

function time_features!(input::Array{Float32, 3}, offset::Int, tau_norm::Vector{Float32}, nfreq::Int)
    N = size(input, 1)
    B = size(input, 3)
    @inbounds for b in 1:B
        t = tau_norm[b]
        for k in 1:nfreq
            input[:, offset + 2k - 1, b] .= sinpi(2f0^Float32(k - 1) * t)
            input[:, offset + 2k, b] .= cospi(2f0^Float32(k - 1) * t)
        end
    end
    return nothing
end

function encode_cond_input!(input::Array{Float32, 3}, x0n::Array{Float32, 3},
        xtn::Array{Float32, 3}, tau_norm::Vector{Float32}, params::CondScoreConfig)
    input[:, 1:3, :] .= x0n
    input[:, 4:6, :] .= xtn
    offset = 6
    if params.include_delta_input
        input[:, 7:9, :] .= xtn .- x0n
        offset = 9
    end
    if params.input_features == :spin_r2_dot
        @views begin
            input[:, offset + 1:offset + 1, :] .= sum(abs2, x0n; dims=2)
            input[:, offset + 2:offset + 2, :] .= sum(abs2, xtn; dims=2)
            input[:, offset + 3:offset + 3, :] .= sum(x0n .* xtn; dims=2)
        end
        offset += 3
    end
    if params.include_tau_scalar
        @inbounds for b in 1:size(input, 3)
            input[:, offset + 1, b] .= tau_norm[b]
        end
        offset += 1
    end
    time_features!(input, offset, tau_norm, params.time_fourier_frequencies)
    return input
end

function refresh_cond_derived_features!(input, params::CondScoreConfig)
    offset = 6
    if params.include_delta_input
        @views input[:, 7:9, :] .= input[:, 4:6, :] .- input[:, 1:3, :]
        offset = 9
    end
    if params.input_features == :spin_r2_dot
        @views begin
            input[:, offset + 1:offset + 1, :] .= sum(abs2, input[:, 1:3, :]; dims=2)
            input[:, offset + 2:offset + 2, :] .= sum(abs2, input[:, 4:6, :]; dims=2)
            input[:, offset + 3:offset + 3, :] .= sum(input[:, 1:3, :] .* input[:, 4:6, :]; dims=2)
        end
    end
    return input
end

function training_lag_steps(sampler::CondPairSampler, params::CondScoreConfig)
    lo = params.train_min_lag_step <= 0 ? first(sampler.lag_steps) : params.train_min_lag_step
    hi = params.train_max_lag_step <= 0 ? last(sampler.lag_steps) : params.train_max_lag_step
    steps = [lag for lag in sampler.lag_steps if lo <= lag <= hi]
    require_condition(!isempty(steps),
        "No conditional-score training lags remain after applying train_min_lag_step=$(params.train_min_lag_step), train_max_lag_step=$(params.train_max_lag_step).")
    return steps
end

function sample_lag_step(lag_steps::AbstractVector{<:Integer}, rng::AbstractRNG,
        lag_sampling_power::Float64)
    L = length(lag_steps)
    if lag_sampling_power == 1.0
        return lag_steps[rand(rng, 1:L)]
    end
    idx = clamp(1 + floor(Int, rand(rng)^lag_sampling_power * L), 1, L)
    return lag_steps[idx]
end

function sample_pair_batch!(x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, sampler::CondPairSampler, rng::AbstractRNG,
        lag_steps::AbstractVector{<:Integer}, lag_sampling_power::Float64)
    nt, _, _, ntraj = size(sampler.states)
    B = size(x0, 3)
    @inbounds for b in 1:B
        lag = sample_lag_step(lag_steps, rng, lag_sampling_power)
        upper = nt - lag
        t = rand(rng, sampler.start_idx:upper)
        tr = rand(rng, 1:ntraj)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        tau_norm[b] = Float32((lag * sampler.save_dt) / sampler.tau_max)
    end
    return nothing
end

function sample_fixed_lag_pairs(sampler::CondPairSampler, lag::Int, npairs::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(sampler.states)
    x0 = Array{Float32}(undef, N, 3, npairs)
    xt = Array{Float32}(undef, N, 3, npairs)
    upper = nt - lag
    @inbounds for b in 1:npairs
        t = rand(rng, sampler.start_idx:upper)
        tr = rand(rng, 1:ntraj)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
    end
    return x0, xt, fill(Float32(lag * sampler.save_dt / sampler.tau_max), npairs)
end

function load_stationary_checkpoint(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    stats_obj = blob[:stats]
    stats = stats_obj isa DataStats ? stats_obj : DataStats(Float32.(stats_obj[:mean]), Float32.(stats_obj[:std]))
    sigma = Float32(blob[:trainer_cfg][:sigma])
    return model, stats, sigma, blob
end

function residual_from_model(model, input, params::CondScoreConfig)
    pred_pos = model(input)
    C = size(input, 2)
    if params.include_delta_input && C >= 9
        inv = C > 9 ?
            cat(-input[:, 1:6, :], -input[:, 7:9, :], input[:, 10:C, :]; dims=2) :
            cat(-input[:, 1:6, :], -input[:, 7:9, :]; dims=2)
    elseif C > 6
        inv = cat(-input[:, 1:6, :], input[:, 7:C, :]; dims=2)
    else
        inv = -input[:, 1:6, :]
    end
    pred_neg = model(inv)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function transition_residual_from_model(model, input, params::CondScoreConfig,
        score_model, score_sigma::Float32)
    pred = residual_from_model(model, input, params)
    if params.score_target == :residual && params.residual_output_scale == :sigma
        return pred ./ params.sigma
    end
    params.score_target == :residual && return pred
    stat_score = score_from_dsm_model(score_model, @view(input[:, 1:3, :]), score_sigma)
    return pred .- stat_score
end

function residual_training_target(q_target, stat_score, params::CondScoreConfig)
    if params.score_target == :posterior
        return q_target
    end
    target = q_target .- stat_score
    params.residual_output_scale == :sigma && (target = target .* params.sigma)
    return target
end

function physical_residual_prediction(pred, stat_score, params::CondScoreConfig)
    if params.score_target == :posterior
        return pred .- stat_score
    elseif params.residual_output_scale == :sigma
        return pred ./ params.sigma
    else
        return pred
    end
end

function inverted_cond_input(input, params::CondScoreConfig)
    inv = copy(input)
    inv[:, 1:6, :] .*= -one(eltype(inv))
    params.include_delta_input && (inv[:, 7:9, :] .*= -one(eltype(inv)))
    return inv
end

function maybe_clip_tree(grads, clip_value::Float64)
    clip_value <= 0 && return grads
    c = Float32(clip_value)
    return Functors.fmap(grads) do g
        g isa AbstractArray ? clamp.(g, -c, c) : g
    end
end

function periodic_cond_checkpoint_path(model_path::AbstractString, epoch::Int)
    root, ext = splitext(model_path)
    return string(root, "_epoch", lpad(string(epoch), 4, '0'), ext)
end

function latest_cond_checkpoint_path(model_path::AbstractString)
    root, ext = splitext(model_path)
    return string(root, "_latest", ext)
end

function cond_training_state(opt, rng::AbstractRNG, noise_rng::AbstractRNG,
        global_step::Int, completed_epoch::Int, total_steps::Int,
        train_lags::AbstractVector{<:Integer})
    return Dict(
        :optimizer_state => Flux.fmap(cpu, opt),
        :rng => rng,
        :noise_rng => noise_rng,
        :global_step => global_step,
        :completed_epoch => completed_epoch,
        :total_steps => total_steps,
        :train_lags => collect(Int, train_lags),
    )
end

function train_cond_score(sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, p::SpinParams, params::CondScoreConfig, device::ExecutionDevice;
        model_path::AbstractString="", operator_aux_target_path::AbstractString=params.operator_aux_target_bson,
        initial_model=nothing, initial_model_cfg=nothing, initial_history=nothing,
        initial_state=nothing, start_epoch::Int=1)
    if initial_model === nothing
        model, model_cfg = build_cond_model(params, sampler.N)
        model = move_model(model, device)
    else
        model = initial_model
        model_cfg = initial_model_cfg
    end
    Flux.trainmode!(model)
    Flux.testmode!(score_model)
    opt = Flux.setup(Flux.Optimisers.Adam(params.learning_rate), model)
    rng = MersenneTwister(params.seed)
    noise_rng = MersenneTwister(params.seed + 1)
    op_rng = MersenneTwister(params.seed + 2)
    train_lags = training_lag_steps(sampler, params)
    params.verbose && @printf("Conditional DSM training lags: %d:%d (%d steps)\n",
        first(train_lags), last(train_lags), length(train_lags))
    aux = params.operator_aux_weight > 0 ? load_operator_aux_target(operator_aux_target_path) : nothing
    aux_lag_indices = Int[]
    if aux !== nothing
        max_li = min(length(aux.lags), params.operator_aux_lag_last)
        aux_lag_indices = [li for li in params.operator_aux_lag_first:max_li
            if aux.lags[li] in train_lags]
        require_condition(!isempty(aux_lag_indices),
            "operator_aux has no lag indices compatible with the conditional training lags.")
        params.verbose && @printf("Conditional operator auxiliary: %d target lags, weight %.4g, batch %d, target %s\n",
            length(aux_lag_indices), params.operator_aux_weight, params.operator_aux_batch_size, aux.path)
    elseif params.operator_aux_weight > 0
        error("operator_aux.weight > 0 but operator_aux.target_bson is empty.")
    end
    B = params.batch_size
    C = params.model_config.in_channels
    x0 = Array{Float32}(undef, sampler.N, 3, B)
    xt = similar(x0)
    x0n = similar(x0)
    xtn = similar(x0)
    input = Array{Float32}(undef, sampler.N, C, B)
    tau_norm = Vector{Float32}(undef, B)
    history = initial_history === nothing ?
        Dict(:train_loss => Float64[], :residual_norm => Float64[],
            :mean_residual_norm => Float64[], :stein_norm => Float64[],
            :target_rms => Float64[], :prediction_rms => Float64[],
            :null_mse => Float64[], :fractional_improvement => Float64[],
            :prediction_target_cosine => Float64[], :operator_aux_loss => Float64[]) :
        initial_history
    total_steps = params.epochs * params.batches_per_epoch
    global_step = max(0, start_epoch - 1) * params.batches_per_epoch
    if initial_state !== nothing
        if haskey(initial_state, :optimizer_state)
            opt = move_model(initial_state[:optimizer_state], device)
            @printf("Restored conditional Adam optimizer state.\n")
        end
        rng = get(initial_state, :rng, rng)
        noise_rng = get(initial_state, :noise_rng, noise_rng)
        global_step = Int(get(initial_state, :global_step, global_step))
    end
    progress = Progress(max(0, params.epochs - start_epoch + 1);
        desc="Training soft-spin conditional residual score")
    for epoch in start_epoch:params.epochs
        losses = Float64[]
        residual_norms = Float64[]
        mean_norms = Float64[]
        stein_norms = Float64[]
        target_rms = Float64[]
        prediction_rms = Float64[]
        null_mses = Float64[]
        frac_improvements = Float64[]
        target_cosines = Float64[]
        op_aux_losses = Float64[]
        for _ in 1:params.batches_per_epoch
            global_step += 1
            params.use_lr_schedule && Flux.adjust!(opt, lr_schedule(global_step, total_steps, params))
            sample_pair_batch!(x0, xt, tau_norm, sampler, rng, train_lags,
                params.lag_sampling_power)
            x0n .= apply_stats_tensor(x0, stats)
            xtn .= apply_stats_tensor(xt, stats)
            encode_cond_input!(input, x0n, xtn, tau_norm, params)
            noise_cpu = randn(noise_rng, Float32, sampler.N, 3, B)
            input_dev = move_array(input, device)
            noise = move_array(noise_cpu, device)
            @views input_dev[:, 1:3, :] .+= params.sigma .* noise
            refresh_cond_derived_features!(input_dev, params)
            stat_score = score_from_dsm_model(score_model, @view(input_dev[:, 1:3, :]), score_sigma)
            q_target = noise .* (-one(eltype(noise)) / params.sigma)
            target = residual_training_target(q_target, stat_score, params)
            if params.target_clip_value > 0
                c = Float32(params.target_clip_value)
                target = clamp.(target, -c, c)
            end
            inv_input_dev = inverted_cond_input(input_dev, params)
            x0_flat = params.stein_weight > 0 ? reshape(copy(@view input_dev[:, 1:3, :]), sampler.D, B) : nothing
            aux_li = aux === nothing ? 0 : rand(op_rng, aux_lag_indices)
            aux_batch = aux === nothing ? nothing :
                prepare_operator_aux_batch(aux, sampler, score_model, stats, score_sigma,
                    p, params, device, op_rng, aux_li)
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = (current_model(input_dev) .- current_model(inv_input_dev)) .* eltype(input_dev)(0.5)
                loss = Flux.Losses.mse(pred, target)
                residual_pred = physical_residual_prediction(pred, stat_score, params)
                if params.mean_score_weight > 0
                    mu = dropdims(sum(residual_pred; dims=3) ./ Float32(B); dims=3)
                    loss += Float32(params.mean_score_weight) * mean(abs2, mu)
                end
                if params.stein_weight > 0
                    score_flat = reshape(residual_pred, sampler.D, B)
                    stein = (score_flat * transpose(x0_flat)) ./ Float32(B)
                    loss += Float32(params.stein_weight) * mean(abs2, stein)
                end
                if aux !== nothing
                    op_loss = grouped_phi_operator_loss_from_batch(current_model, aux_batch,
                        stats, params, score_model, score_sigma, device)
                    loss += Float32(params.operator_aux_weight) * op_loss
                end
                loss
            end
            opt, model = Flux.update!(opt, model, maybe_clip_tree(grads[1], params.gradient_clip_value))
            push!(losses, Float64(to_host(loss_value)))
            if aux !== nothing && global_step % 20 == 0
                op_loss_value = grouped_phi_operator_loss_from_batch(model, aux_batch,
                    stats, params, score_model, score_sigma, device)
                push!(op_aux_losses, Float64(to_host(op_loss_value)))
            end
            if global_step % 20 == 0
                pred_train = residual_from_model(model, input_dev, params)
                pred = physical_residual_prediction(pred_train, stat_score, params)
                pred_h = Array(to_host(pred))
                pred_train_h = Array(to_host(pred_train))
                target_h = Array(to_host(target))
                push!(residual_norms, sqrt(mean(abs2, pred_h)))
                pred_flat = reshape(pred_h, sampler.D, B)
                push!(mean_norms, norm(vec(mean(pred_flat; dims=2))) / sqrt(sampler.D))
                push!(stein_norms, params.stein_weight > 0 ?
                    norm((pred_flat * transpose(Float32.(Array(to_host(x0_flat))))) ./ B) / sqrt(sampler.D) : NaN)
                null_mse = mean(abs2, target_h)
                model_mse = mean(abs2, pred_train_h .- target_h)
                push!(target_rms, sqrt(null_mse))
                push!(prediction_rms, sqrt(mean(abs2, pred_train_h)))
                push!(null_mses, null_mse)
                push!(frac_improvements, (null_mse - model_mse) / max(null_mse, eps(Float32)))
                dotpt = sum(Float64.(pred_train_h) .* Float64.(target_h))
                npred = sqrt(sum(abs2, Float64.(pred_train_h)))
                ntarg = sqrt(sum(abs2, Float64.(target_h)))
                push!(target_cosines, dotpt / max(npred * ntarg, eps(Float64)))
            end
        end
        push!(history[:train_loss], mean(losses))
        push!(history[:residual_norm], isempty(residual_norms) ? NaN : mean(residual_norms))
        push!(history[:mean_residual_norm], isempty(mean_norms) ? NaN : mean(mean_norms))
        push!(history[:stein_norm], isempty(stein_norms) ? NaN : mean(skipmissing(stein_norms)))
        push!(history[:target_rms], isempty(target_rms) ? NaN : mean(target_rms))
        push!(history[:prediction_rms], isempty(prediction_rms) ? NaN : mean(prediction_rms))
        push!(history[:null_mse], isempty(null_mses) ? NaN : mean(null_mses))
        push!(history[:fractional_improvement], isempty(frac_improvements) ? NaN : mean(frac_improvements))
        push!(history[:prediction_target_cosine], isempty(target_cosines) ? NaN : mean(target_cosines))
        if !haskey(history, :operator_aux_loss)
            history[:operator_aux_loss] = Float64[]
        end
        push!(history[:operator_aux_loss], isempty(op_aux_losses) ? NaN : mean(op_aux_losses))
        ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch), (:loss, history[:train_loss][end]),
            (:residual_norm, history[:residual_norm][end]),
            (:frac_improve, history[:fractional_improvement][end]),
            (:pred_target_cos, history[:prediction_target_cosine][end]),
            (:op_aux, history[:operator_aux_loss][end])
        ])
        if !isempty(model_path) && params.checkpoint_every > 0 &&
                (epoch % params.checkpoint_every == 0 || epoch == params.epochs)
            state = cond_training_state(opt, rng, noise_rng, global_step, epoch,
                total_steps, train_lags)
            ckpt_path = periodic_cond_checkpoint_path(model_path, epoch)
            save_cond_model(ckpt_path, model, model_cfg, stats, params, sampler,
                history, Dict{Symbol, Any}(); completed_epoch=epoch,
                training_state=state)
            save_cond_model(latest_cond_checkpoint_path(model_path), model, model_cfg,
                stats, params, sampler, history, Dict{Symbol, Any}();
                completed_epoch=epoch, training_state=state)
            save_cond_model(model_path, model, model_cfg, stats, params, sampler,
                history, Dict{Symbol, Any}(); completed_epoch=epoch,
                training_state=state)
        end
    end
    ProgressMeter.finish!(progress)
    Flux.testmode!(model)
    return model, model_cfg, history
end

function evaluate_residual_norm(model, x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, stats::DataStats, params::CondScoreConfig,
        device::ExecutionDevice; batch_size::Int=4096, score_model=nothing,
        score_sigma::Float32=0f0)
    x0n = apply_stats_tensor(x0, stats)
    xtn = apply_stats_tensor(xt, stats)
    out = Array{Float32}(undef, size(x0))
    C = params.model_config.in_channels
    for lo in 1:batch_size:size(x0, 3)
        hi = min(lo + batch_size - 1, size(x0, 3))
        input = Array{Float32}(undef, size(x0, 1), C, hi - lo + 1)
        encode_cond_input!(input, x0n[:, :, lo:hi], xtn[:, :, lo:hi], tau_norm[lo:hi], params)
        input_dev = move_array(input, device)
        pred = transition_residual_from_model(model, input_dev, params, score_model, score_sigma)
        out[:, :, lo:hi] .= to_host(pred)
    end
    return out
end

function normalized_residual_to_raw(residual_norm::Array{Float32, 3}, stats::DataStats)
    return normalized_score_to_raw(residual_norm, stats)
end

function load_operator_aux_target(path::AbstractString)
    isempty(path) && return nothing
    target = BSON.load(path, @__MODULE__)
    names = Vector{String}(target[:names])
    means = Vector{Float64}(target[:observable_means])
    Phi = Matrix{Float32}(target[:Phi])
    group_index_matrix = Matrix{Int}(target[:group_index_matrix])
    lags = Vector{Int}(target[:lags])
    taus = Vector{Float64}(target[:taus])
    Cdot_phi = Array{Float64}(target[:Cdot_phi])
    G = size(group_index_matrix, 2)
    target_phi_group = Matrix{Float32}(undef, size(Cdot_phi, 1), G)
    @inbounds for li in 1:size(Cdot_phi, 1)
        flat = vec(@view Cdot_phi[li, :, :, :])
        for g in 1:G
            acc = 0.0
            for i in axes(group_index_matrix, 1)
                acc += flat[group_index_matrix[i, g]]
            end
            target_phi_group[li, g] = Float32(acc / size(group_index_matrix, 1))
        end
    end
    scale_vec = haskey(target, :scale_vec) ?
        Vector{Float32}(target[:scale_vec]) :
        fill(Float32(max(sqrt(mean(abs2, target_phi_group)), 1.0e-6)), G)
    radial_basis = haskey(target, :expanded_radial_basis) ?
        Matrix{Float64}(target[:expanded_radial_basis]) :
        Matrix{Float64}(I, 3, 3)
    return CondOperatorAuxTarget(
        abspath(path), target_phi_group, lags, taus, names, means, Phi,
        group_index_matrix, scale_vec,
        haskey(target, :score_radial_scale) ? Float64(target[:score_radial_scale]) : 1.0,
        radial_basis)
end

function flatten_batch_device(x)
    N, _, B = size(x)
    return reshape(permutedims(x, (2, 1, 3)), 3N, B)
end

function operator_aux_observables(raw::Array{Float32, 3}, score_raw::Array{Float32, 3},
        names::Vector{String}, means::Vector{Float64})
    N, _, B = size(raw)
    obs = Array{Float32}(undef, N, length(names), B)
    @inbounds for b in 1:B, i in 1:N
        x1, x2, x3 = raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]
        s1, s2, s3 = score_raw[i, 1, b], score_raw[i, 2, b], score_raw[i, 3, b]
        r2 = x1 * x1 + x2 * x2 + x3 * x3
        xdot_s = x1 * s1 + x2 * s2 + x3 * s3
        Ts = (r2 * s1 - x1 * xdot_s,
              r2 * s2 - x2 * xdot_s,
              r2 * s3 - x3 * xdot_s)
        Ps = (x1 * xdot_s, x2 * xdot_s, x3 * xdot_s)
        Cs = (x2 * s3 - x3 * s2,
              x3 * s1 - x1 * s3,
              x1 * s2 - x2 * s1)
        vals = Dict(
            "mc_m_mx" => x1, "mc_m_my" => x2, "mc_m_mz" => x3,
            "mc_s_mx" => s1, "mc_s_my" => s2, "mc_s_mz" => s3,
            "mc_Ts_mx" => Ts[1], "mc_Ts_my" => Ts[2], "mc_Ts_mz" => Ts[3],
            "mc_Ps_mx" => Ps[1], "mc_Ps_my" => Ps[2], "mc_Ps_mz" => Ps[3],
            "mc_Cs_mx" => Cs[1], "mc_Cs_my" => Cs[2], "mc_Cs_mz" => Cs[3],
        )
        for (a, name) in pairs(names)
            haskey(vals, name) ||
                error("operator_aux currently supports only mobility-channel 5-family names; got $(name)")
            obs[i, a, b] = Float32(vals[name] - means[a])
        end
    end
    return obs
end

function prepare_operator_aux_batch(aux::CondOperatorAuxTarget,
        sampler::CondPairSampler, score_model, stats::DataStats, score_sigma::Float32,
        p::SpinParams, params::CondScoreConfig, device::ExecutionDevice,
        rng::AbstractRNG, lag_index::Int)
    B = params.operator_aux_batch_size
    lag = aux.lags[lag_index]
    x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, B, rng)
    x0n = apply_stats_tensor(x0, stats)
    xtn = apply_stats_tensor(xt, stats)
    input = Array{Float32}(undef, sampler.N, params.model_config.in_channels, B)
    encode_cond_input!(input, x0n, xtn, tau_norm, params)

    score_xt_norm = evaluate_score_norm(score_model, xtn, score_sigma, device;
        batch_size=params.batch_size)
    score_xt_raw = normalized_score_to_raw(score_xt_norm, stats)
    obs = operator_aux_observables(xt, score_xt_raw, aux.names, aux.means)

    input_dev = move_array(input, device)
    obs_flat = move_array(reshape(obs, sampler.N * length(aux.names), B), device)
    target = move_array(collect(@view(aux.target_phi_group[lag_index, :])), device)
    scale = move_array(max.(abs.(collect(@view aux.target_phi_group[lag_index, :])),
        max.(aux.scale_vec, Float32(params.operator_aux_scale_floor))), device)
    Phi_dev = move_array(aux.Phi, device)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), sampler.N, 3, 1)
    std_dev = move_array(std_tensor, device)
    return (; input_dev, obs_flat, target, scale, Phi_dev, std_dev,
        group_index_matrix=aux.group_index_matrix, batch_size=B)
end

function grouped_phi_operator_loss_from_batch(model, batch, stats::DataStats,
        params::CondScoreConfig, score_model, score_sigma::Float32, device::ExecutionDevice)
    rnorm = transition_residual_from_model(model, batch.input_dev, params, score_model, score_sigma)
    rraw = rnorm ./ batch.std_dev
    rflat = flatten_batch_device(rraw)
    action = transpose(batch.Phi_dev) * rflat
    mat = -(batch.obs_flat * transpose(action)) ./ Float32(batch.batch_size)
    pred_group = dropdims(mean(reshape(vec(mat)[vec(batch.group_index_matrix)],
        size(batch.group_index_matrix, 1), size(batch.group_index_matrix, 2)); dims=1); dims=1)
    return mean(abs2, (pred_group .- batch.target) ./ batch.scale)
end

function observable_values_cond(raw::Array{Float32, 3}, p::SpinParams)
    N, _, B = size(raw)
    names = ["mx", "my", "mz", "r2", "mperp2", "mz2", "local_U"]
    obs = Array{Float32}(undef, N, length(names), B)
    @inbounds for b in 1:B
        for i in 1:N
            ip = periodic(i + 1, N)
            x1, x2, x3 = Float64(raw[i, 1, b]), Float64(raw[i, 2, b]), Float64(raw[i, 3, b])
            r2 = x1 * x1 + x2 * x2 + x3 * x3
            diff2 = sum(abs2, Float64.(raw[ip, :, b]) .- Float64.(raw[i, :, b]))
            obs[i, 1, b] = raw[i, 1, b]
            obs[i, 2, b] = raw[i, 2, b]
            obs[i, 3, b] = raw[i, 3, b]
            obs[i, 4, b] = Float32(r2)
            obs[i, 5, b] = Float32(x1 * x1 + x2 * x2)
            obs[i, 6, b] = Float32(x3 * x3)
            obs[i, 7, b] = Float32(0.25 * p.lambda * (r2 - p.mstar^2)^2 + 0.5 * p.J * diff2 - 0.5 * p.K * x3^2)
        end
    end
    return obs, names
end

function center_observables!(obs::Array{Float32, 3}, means::Vector{Float64})
    @inbounds for j in 1:length(means)
        obs[:, j, :] .-= Float32(means[j])
    end
    return obs
end

function true_action_batch(x0::Array{Float32, 3}, rraw::Array{Float32, 3}, p::SpinParams)
    N, _, B = size(x0)
    out = Array{Float32}(undef, N, 3, B)
    @inbounds for b in 1:B
        M = true_mobility_matrix(@view(x0[:, :, b]), p)
        action = transpose(M) * Float64.(flatten_state(@view rraw[:, :, b]))
        for i in 1:N, c in 1:3
            out[i, c, b] = Float32(action[(i - 1) * 3 + c])
        end
    end
    return out
end

function conditional_diagnostics(model, sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, p::SpinParams, params::CondScoreConfig, phi_blob,
        device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 300)
    eval_lag_idxs = round.(Int, range(1, length(sampler.lag_steps), length=params.eval_tau_count))
    mean_norm = Float64[]
    stein_norm = Float64[]
    posterior_mse = Float64[]
    taus_eval = Float64[]
    for idx in eval_lag_idxs
        lag = sampler.lag_steps[idx]
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, params.eval_pairs_per_lag, rng)
        rnorm = evaluate_residual_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        x0n = apply_stats_tensor(x0, stats)
        stat = evaluate_score_norm(score_model, x0n, score_sigma, device; batch_size=params.batch_size)
        qnorm = rnorm .+ stat
        flat_r = Float64.(flatten_batch(rnorm))
        flat_x = Float64.(flatten_batch(x0n))
        push!(mean_norm, norm(vec(mean(flat_r; dims=2))) / sqrt(sampler.D))
        push!(stein_norm, norm((flat_r * transpose(flat_x)) ./ size(flat_r, 2)) / sqrt(sampler.D))
        noise = randn(rng, Float32, size(x0))
        noisy = x0n .+ params.sigma .* noise
        input = Array{Float32}(undef, sampler.N, params.model_config.in_channels, size(x0, 3))
        encode_cond_input!(input, noisy, apply_stats_tensor(xt, stats), tau_norm, params)
        stat_noisy = evaluate_score_norm(score_model, noisy, score_sigma, device; batch_size=params.batch_size)
        pred_raw = Array(to_host(transition_residual_from_model(model, move_array(input, device),
            params, score_model, score_sigma)))
        q_noisy = pred_raw .+ stat_noisy
        target_q = noise .* (-1f0 / params.sigma)
        push!(posterior_mse, mean(abs2, q_noisy .- target_q))
        push!(taus_eval, lag * sampler.save_dt)
    end

    names = Vector{String}(phi_blob[:observable_names])
    means = Vector{Float64}(phi_blob[:observable_means])
    Cdot_data = Array{Float64}(phi_blob[:Cdot_data])
    Cdot_phi = Array{Float64}(phi_blob[:Cdot_phi])
    Phi = Matrix{Float64}(phi_blob[:Phi])
    operator_lags = phi_blob[:lags][1:min(params.operator_lag_count, length(phi_blob[:lags]))]
    Ctrue = Array{Float64}(undef, length(operator_lags), sampler.N, length(names), sampler.D)
    Cphi_cond = Array{Float64}(undef, length(operator_lags), sampler.N, length(names), sampler.D)
    for (li, lag) in enumerate(operator_lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, Int(lag), params.operator_pairs_per_lag, rng)
        rnorm = evaluate_residual_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        rflat = Matrix{Float64}(flatten_batch(rraw))
        phi_action_flat = transpose(Phi) * rflat
        action = true_action_batch(x0, rraw, p)
        obs, _ = observable_values_cond(xt, p)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * length(names), params.operator_pairs_per_lag)
        mat_phi = -Matrix{Float64}(obs_flat) * transpose(phi_action_flat) / params.operator_pairs_per_lag
        Cphi_cond[li, :, :, :] .= reshape(mat_phi, sampler.N, length(names), sampler.D)
        action_flat = flatten_batch(action)
        mat = -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(action_flat)) / params.operator_pairs_per_lag
        Ctrue[li, :, :, :] .= reshape(mat, sampler.N, length(names), sampler.D)
        params.verbose && @printf("Conditional true-M operator lag %.5g (%d/%d)\n",
            lag * sampler.save_dt, li, length(operator_lags))
    end
    Cref = Cdot_data[1:length(operator_lags), :, :, :]
    Cphi_ref = Cdot_phi[1:length(operator_lags), :, :, :]
    op_metrics = agreement_metrics(Cref, Ctrue)
    phi_metrics = agreement_metrics(Cphi_ref, Cphi_cond)
    return Dict(:taus_eval => taus_eval, :mean_norm => mean_norm, :stein_norm => stein_norm,
        :posterior_mse => posterior_mse, :operator_lags => operator_lags .* sampler.save_dt,
        :operator_trueM => Ctrue, :operator_metrics => op_metrics,
        :operator_phi_cond => Cphi_cond, :operator_phi_metrics => phi_metrics)
end

function render_cond_figure(path, params::CondScoreConfig, history, diag, phi_blob)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    op = diag[:operator_metrics]
    phi_op = diag[:operator_phi_metrics]
    figure_title!(fig, "Soft-spin LLG conditional residual score diagnostics";
        subtitle=@sprintf("Phi operator rel.RMSE %.4e, corr %.5f; true-M diagnostic rel.RMSE %.4e",
            phi_op[:relative_rmse], phi_op[:correlation], op[:relative_rmse]))
    epochs = collect(1:length(history[:train_loss]))
    ax1 = Axis(fig[1, 1]; title="DSM residual loss", xlabel="epoch", ylabel="loss", yscale=log10)
    lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=2)
    ax2 = Axis(fig[1, 2]; title="Scaled DSM fit", xlabel="epoch", ylabel="value")
    if haskey(history, :fractional_improvement)
        lines!(ax2, epochs, history[:fractional_improvement]; color=STYLE_PRIMARY, linewidth=2, label="improvement")
    end
    if haskey(history, :prediction_target_cosine)
        lines!(ax2, epochs, history[:prediction_target_cosine]; color=STYLE_SECONDARY, linewidth=2, label="cosine")
    end
    axislegend(ax2; position=:rb)
    ax2b = Axis(fig[1, 3]; title="Residual score norms", xlabel="epoch", ylabel="norm")
    if haskey(history, :target_rms)
        lines!(ax2b, epochs, history[:target_rms]; color=STYLE_REFERENCE, linewidth=2, label="target rms")
    end
    if haskey(history, :prediction_rms)
        lines!(ax2b, epochs, history[:prediction_rms]; color=STYLE_HIGHLIGHT, linewidth=2, label="prediction rms")
    end
    axislegend(ax2b; position=:rb)
    ax3 = Axis(fig[1, 4]; title="Lagwise residual diagnostics", xlabel="tau", ylabel="norm")
    lines!(ax3, diag[:taus_eval], diag[:mean_norm]; color=STYLE_PRIMARY, linewidth=2, label="||E[r]||/sqrt(D)")
    lines!(ax3, diag[:taus_eval], diag[:stein_norm]; color=STYLE_SECONDARY, linewidth=2, label="||E[r x0']||/sqrt(D)")
    axislegend(ax3; position=:rt)
    ax4 = Axis(fig[1, 5]; title="Posterior reconstruction", xlabel="tau", ylabel="DSM MSE", yscale=log10)
    lines!(ax4, diag[:taus_eval], diag[:posterior_mse]; color=STYLE_HIGHLIGHT, linewidth=2)

    ax5 = Axis(fig[1, 6]; title="Physical residual norm", xlabel="epoch", ylabel="norm")
    lines!(ax5, epochs, history[:residual_norm]; color=STYLE_ACCENT, linewidth=2, label="r rms")
    lines!(ax5, epochs, history[:mean_residual_norm]; color=STYLE_SECONDARY, linewidth=2, label="mean r")
    axislegend(ax5; position=:rt)

    Cdata = Array{Float64}(phi_blob[:Cdot_data])
    Cphi_ref = Array{Float64}(phi_blob[:Cdot_phi])
    Cphi_cond = diag[:operator_phi_cond]
    Ctrue = diag[:operator_trueM]
    names = Vector{String}(phi_blob[:observable_names])
    taus = diag[:operator_lags]
    panels = min(12, size(Cdata, 2) * min(3, size(Cdata, 3)))
    for panel in 1:panels
        site = 1 + (panel - 1) % size(Cdata, 2)
        obs = 1 + ((panel - 1) ÷ size(Cdata, 2)) % min(3, size(Cdata, 3))
        col = (site - 1) * 3 + obs
        ax = Axis(fig[2 + (panel - 1) ÷ 4, 1 + (panel - 1) % 4];
            title="site $(site) $(names[obs]) vs x$(col)", xlabel="tau", ylabel="Cdot")
        lines!(ax, taus, Cdata[1:length(taus), site, obs, col]; color=:black, linewidth=2, label="data")
        lines!(ax, taus, Cphi_ref[1:length(taus), site, obs, col]; color=STYLE_SECONDARY, linewidth=2, label="Phi GFDT")
        lines!(ax, taus, Cphi_cond[:, site, obs, col]; color=STYLE_ACCENT, linewidth=2, linestyle=:dot, label="Phi with r")
        lines!(ax, taus, Ctrue[:, site, obs, col]; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="true-M with r")
        panel == 1 && axislegend(ax; position=:rt)
    end
    save_figure_checked(path, fig)
end

function save_cond_model(path, model, model_cfg, stats, params, sampler, history, diag;
        completed_epoch::Int=length(history[:train_loss]), training_state=nothing)
    ensure_parent_dir(path)
    host_model = Flux.fmap(cpu, model)
    trainer_cfg = Dict(:sigma => params.sigma, :epochs => params.epochs,
        :batch_size => params.batch_size, :learning_rate => params.learning_rate,
        :batches_per_epoch => params.batches_per_epoch,
        :architecture => String(params.architecture),
        :normalization => String(params.normalization),
        :input_features => String(params.input_features),
        :include_tau_scalar => params.include_tau_scalar,
        :time_fourier_frequencies => params.time_fourier_frequencies,
        :residual_dilations => params.residual_dilations,
        :residual_repeats => params.residual_repeats,
        :mean_score_weight => params.mean_score_weight,
        :stein_weight => params.stein_weight,
        :train_min_lag_step => params.train_min_lag_step,
        :train_max_lag_step => params.train_max_lag_step,
        :score_target => String(params.score_target),
        :residual_output_scale => String(params.residual_output_scale),
        :operator_aux_target_bson => params.operator_aux_target_bson,
        :operator_aux_weight => params.operator_aux_weight,
        :operator_aux_batch_size => params.operator_aux_batch_size,
        :operator_aux_lag_first => params.operator_aux_lag_first,
        :operator_aux_lag_last => params.operator_aux_lag_last,
        :network_output => params.residual_output_scale == :sigma ?
            "sigma_times_transition_residual" : "transition_residual_raw",
        :operator_output => "transition_residual_raw",
        :score_type => "transition_residual_direct")
    metadata = Dict(:tau_max => sampler.tau_max, :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times, :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
        :include_tau_scalar => params.include_tau_scalar,
        :input_features => String(params.input_features),
        :architecture => String(params.architecture),
        :score_target => String(params.score_target),
        :train_min_lag_step => params.train_min_lag_step,
        :train_max_lag_step => params.train_max_lag_step,
        :completed_epoch => completed_epoch,
        :requested_device => params.device,
        :required_gpu_name => params.required_gpu_name,
        :conditioning_smoothed => false,
        :stationary_score_bson => params.score_bson,
        :residual_output_scale => String(params.residual_output_scale),
        :operator_aux_target_bson => params.operator_aux_target_bson,
        :operator_aux_weight => params.operator_aux_weight,
        :no_cheating_audit => "Conditional DSM target used only x0 Gaussian noise, clean xt conditioning, and the learned stationary score to form the residual target. No analytic score or true mobility entered training.")
    payload = Dict(:host_model => host_model, :model_cfg => model_cfg,
        :stats => stats, :trainer_cfg => trainer_cfg, :history => history,
        :metadata => metadata, :diagnostics => diag)
    training_state !== nothing && (payload[:training_state] = training_state)
    BSON.bson(path, payload)
    @printf("Saved conditional residual score checkpoint to %s\n", path)
end

function run_pipeline(param_file::AbstractString)
    base = dirname(param_file)
    params = load_config(param_file)
    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    phi_path = resolve_path(base, params.phi_artifact_bson)
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    model_path = resolve_path(base, params.output_bson)
    diag = Dict{Symbol, Any}()
    if params.train
        if params.resume && isfile(model_path)
            blob = BSON.load(model_path)
            model = move_model(blob[:host_model], device)
            model_cfg = blob[:model_cfg]
            history = blob[:history]
            state = get(blob, :training_state, nothing)
            completed = state === nothing ?
                Int(get(blob[:metadata], :completed_epoch, length(history[:train_loss]))) :
                Int(get(state, :completed_epoch, get(blob[:metadata], :completed_epoch, length(history[:train_loss]))))
            if completed >= params.epochs
                @printf("Checkpoint %s already has %d/%d epochs; skipping training.\n",
                    model_path, completed, params.epochs)
                Flux.testmode!(model)
            else
                @printf("Resuming %s from epoch %d/%d.\n", model_path, completed + 1, params.epochs)
                model, model_cfg, history = train_cond_score(sampler, score_model, stats, score_sigma,
                    p, params, device; model_path=model_path,
                    operator_aux_target_path=resolve_path(base, params.operator_aux_target_bson),
                    initial_model=model,
                    initial_model_cfg=model_cfg, initial_history=history,
                    initial_state=state, start_epoch=completed + 1)
            end
        else
            model, model_cfg, history = train_cond_score(sampler, score_model, stats, score_sigma,
                p, params, device; model_path=model_path,
                operator_aux_target_path=resolve_path(base, params.operator_aux_target_bson))
        end
        if !isfile(model_path)
            save_cond_model(model_path, model, model_cfg, stats, params, sampler, history, diag)
        end
    else
        blob = BSON.load(model_path)
        model = move_model(blob[:host_model], device)
        Flux.testmode!(model)
        model_cfg = blob[:model_cfg]
        history = blob[:history]
    end
    if params.evaluate
        phi_blob = BSON.load(phi_path)
        diag = conditional_diagnostics(model, sampler, score_model, stats, score_sigma, p, params, phi_blob, device)
        render_cond_figure(resolve_path(base, params.output_png), params, history, diag, phi_blob)
        metrics_path = resolve_path(base, params.metrics_txt)
        ensure_parent_dir(metrics_path)
        open(metrics_path, "w") do io
            println(io, "SoftSpinLLGChain Step 3 conditional score metrics")
            println(io, @sprintf("Phi conditional-vs-GFDT rel.RMSE = %.8e", diag[:operator_phi_metrics][:relative_rmse]))
            println(io, @sprintf("Phi conditional-vs-GFDT corr = %.8e", diag[:operator_phi_metrics][:correlation]))
            println(io, @sprintf("true-M operator rel.RMSE = %.8e", diag[:operator_metrics][:relative_rmse]))
            println(io, @sprintf("true-M operator corr = %.8e", diag[:operator_metrics][:correlation]))
            if haskey(history, :target_rms)
                println(io, @sprintf("last DSM target RMS = %.8e", history[:target_rms][end]))
            end
            if haskey(history, :prediction_rms)
                println(io, @sprintf("last DSM prediction RMS = %.8e", history[:prediction_rms][end]))
            end
            if haskey(history, :null_mse)
                println(io, @sprintf("last DSM null MSE = %.8e", history[:null_mse][end]))
            end
            if haskey(history, :fractional_improvement)
                println(io, @sprintf("last DSM fractional improvement = %.8e", history[:fractional_improvement][end]))
            end
            if haskey(history, :prediction_target_cosine)
                println(io, @sprintf("last DSM prediction-target cosine = %.8e", history[:prediction_target_cosine][end]))
            end
            println(io, @sprintf("mean lagwise ||E[r]||/sqrt(D) = %.8e", mean(diag[:mean_norm])))
            println(io, @sprintf("mean lagwise ||E[r x0']||/sqrt(D) = %.8e", mean(diag[:stein_norm])))
            println(io, @sprintf("mean posterior reconstruction DSM MSE = %.8e", mean(diag[:posterior_mse])))
            println(io, "No-cheating audit: true mobility was used only for this ex-post operator diagnostic.")
        end
        existing_state = isfile(model_path) ? get(BSON.load(model_path), :training_state, nothing) : nothing
        save_cond_model(model_path, model, model_cfg, stats, params, sampler, history, diag;
            training_state=existing_state)
    end
    @printf("Conditional score stage complete. No-cheating audit: no analytic score or true mobility entered conditional DSM training.\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE)
end
