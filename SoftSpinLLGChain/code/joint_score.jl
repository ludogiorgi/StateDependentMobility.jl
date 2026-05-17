#!/usr/bin/env julia

if !isdefined(@__MODULE__, :CondScoreConfig)
    include(joinpath(@__DIR__, "cond_score.jl"))
end
if !isdefined(@__MODULE__, :NonlinearLibrary)
    include(joinpath(@__DIR__, "search_nonlinear_observables.jl"))
end

const DEFAULT_JOINT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "joint_score_phys_pC_gpu2_vA.toml"))
const JOINT_SCORE_CHANNELS = 2SPIN_CHANNELS

if !isdefined(@__MODULE__, :RetainedChannel)
    struct RetainedChannel
        observable::String
        target_component::Int
        data_rms::Float64
    end
end

Base.@kwdef struct JointScoreConfig
    input_hdf5::String
    score_bson::String
    phi_artifact_bson::String
    burnin_fraction::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    model_config::ScoreUNetConfig
    normalization::Symbol
    feature_mode::Symbol
    include_delta_input::Bool
    include_tau_scalar::Bool
    time_fourier_frequencies::Int
    init_seed::Int
    sigma::Float32
    endpoint_noise::Symbol
    score_target::Symbol
    epochs::Int
    batches_per_epoch::Int
    batch_size::Int
    learning_rate::Float64
    use_lr_schedule::Bool
    warmup_steps::Int
    min_lr_factor::Float64
    gradient_clip_value::Float64
    target_clip_value::Float64
    initial_block_weight::Float64
    terminal_block_weight::Float64
    mean_score_weight::Float64
    stein_weight::Float64
    lag_sampling_mode::Symbol
    lag_sampling_power::Float64
    active_lag_first::Int
    active_lag_last::Int
    checkpoint_every::Int
    resume::Bool
    seed::Int
    eval_tau_count::Int
    eval_pairs_per_lag::Int
    operator_pairs_per_lag::Int
    operator_lag_count::Int
    retained_target_bson::String
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

struct JointScoreUNet{M}
    backbone::M
end

Functors.@functor JointScoreUNet (backbone,)

(model::JointScoreUNet)(x) = @view model.backbone(x)[:, 1:JOINT_SCORE_CHANNELS, :]

function joint_input_channels(feature_mode::Symbol, nfreq::Int; include_delta_input::Bool,
        include_tau_scalar::Bool)
    base = 2SPIN_CHANNELS
    delta = include_delta_input ? SPIN_CHANNELS : 0
    physical = feature_mode == :basic ? 0 :
        feature_mode == :physical_aug ? (2 + 3SPIN_CHANNELS) :
        feature_mode == :physical_full ? (2 + 7SPIN_CHANNELS) :
        error("Unknown joint feature_mode $(feature_mode).")
    return base + delta + physical + (include_tau_scalar ? 1 : 0) + 2nfreq
end

function load_joint_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    train = raw["training"]
    diag = raw["diagnostics"]
    fig = raw["figure"]
    out = raw["output"]
    run = raw["run"]
    nfreq = Int(get(model, "time_fourier_frequencies", 8))
    feature_mode = Symbol(lowercase(String(get(model, "feature_mode", "basic"))))
    include_delta = Bool(get(model, "include_delta_input", true))
    include_tau = Bool(get(model, "include_tau_scalar", false))
    score_target_raw = lowercase(String(get(train, "score_target", "residualized")))
    score_target = score_target_raw in ("residual", "residualized") ? :residualized :
        score_target_raw == "raw" ? :raw :
        error("score_target must be residualized/residual or raw.")
    endpoint_noise = Symbol(lowercase(String(get(train, "endpoint_noise", "both"))))
    lag_sampling_mode = Symbol(lowercase(String(get(train, "lag_sampling_mode", "uniform"))))
    cfg = ScoreUNetConfig(
        in_channels=joint_input_channels(feature_mode, nfreq;
            include_delta_input=include_delta, include_tau_scalar=include_tau),
        base_channels=Int(get(model, "base_channels", 128)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(get(model, "activation", "swish")),
        final_activation=activation_from_string(get(model, "final_activation", "identity")),
    )
    params = JointScoreConfig(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        phi_artifact_bson=String(data["phi_artifact_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 0.6)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        model_config=cfg,
        normalization=normalization_from_string(get(model, "normalization", "none")),
        feature_mode=feature_mode,
        include_delta_input=include_delta,
        include_tau_scalar=include_tau,
        time_fourier_frequencies=nfreq,
        init_seed=Int(get(model, "init_seed", 20260514)),
        sigma=Float32(get(train, "sigma", 0.05)),
        endpoint_noise=endpoint_noise,
        score_target=score_target,
        epochs=Int(get(train, "epochs", 120)),
        batches_per_epoch=Int(get(train, "batches_per_epoch", 256)),
        batch_size=Int(get(train, "batch_size", 1024)),
        learning_rate=Float64(get(train, "learning_rate", 1.0e-4)),
        use_lr_schedule=Bool(get(train, "use_lr_schedule", true)),
        warmup_steps=Int(get(train, "warmup_steps", 500)),
        min_lr_factor=Float64(get(train, "min_lr_factor", 0.08)),
        gradient_clip_value=Float64(get(train, "gradient_clip_value", 0.0)),
        target_clip_value=Float64(get(train, "target_clip_value", 0.0)),
        initial_block_weight=Float64(get(train, "initial_block_weight", 1.0)),
        terminal_block_weight=Float64(get(train, "terminal_block_weight", 1.0)),
        mean_score_weight=Float64(get(train, "mean_score_weight", 0.0)),
        stein_weight=Float64(get(train, "stein_weight", 0.0)),
        lag_sampling_mode=lag_sampling_mode,
        lag_sampling_power=Float64(get(train, "lag_sampling_power", 1.0)),
        active_lag_first=Int(get(train, "active_lag_first", 1)),
        active_lag_last=Int(get(train, "active_lag_last", 0)),
        checkpoint_every=Int(get(train, "checkpoint_every", 10)),
        resume=Bool(get(train, "resume", true)),
        seed=Int(get(train, "seed", 20260514)),
        eval_tau_count=Int(get(diag, "eval_tau_count", 8)),
        eval_pairs_per_lag=Int(get(diag, "eval_pairs_per_lag", 50000)),
        operator_pairs_per_lag=Int(get(diag, "operator_pairs_per_lag", 40000)),
        operator_lag_count=Int(get(diag, "operator_lag_count", 24)),
        retained_target_bson=String(get(diag, "retained_target_bson", "")),
        figure_width=Int(get(fig, "width", 3400)),
        figure_height=Int(get(fig, "height", 2600)),
        output_bson=String(out["model_bson"]),
        output_png=String(out["figure_png"]),
        metrics_txt=String(out["metrics_txt"]),
        device=String(get(run, "device", "GPU:2")),
        required_gpu_name=String(get(run, "required_gpu_name", "5070")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
        verbose=Bool(get(run, "verbose", true)),
    )
    require_condition(params.model_config.periodic, "Joint score model must be periodic.")
    require_condition(params.normalization != :batchnorm, "BatchNorm is forbidden.")
    require_condition(params.time_fourier_frequencies >= 1, "Use Fourier time features for joint score.")
    require_condition(params.endpoint_noise in (:both, :initial_only),
        "endpoint_noise must be both or initial_only for the implemented DSM targets.")
    require_condition(params.score_target in (:residualized, :raw),
        "score_target must be residualized or raw.")
    require_condition(params.lag_sampling_mode in (:uniform, :power, :short, :active),
        "lag_sampling_mode must be uniform, power, short, or active.")
    return params
end

function build_joint_model(params::JointScoreConfig, N::Int)
    Random.seed!(params.init_seed)
    cfg = adjust_model_config_for_length(params.model_config, N)
    model = JointScoreUNet(build_unet(cfg; normalization=params.normalization))
    Random.seed!()
    return model, cfg
end

function normalized_cross_features_cpu(x0::Array{Float32, 3}, xt::Array{Float32, 3})
    c1 = x0[:, 2:2, :] .* xt[:, 3:3, :] .- x0[:, 3:3, :] .* xt[:, 2:2, :]
    c2 = x0[:, 3:3, :] .* xt[:, 1:1, :] .- x0[:, 1:1, :] .* xt[:, 3:3, :]
    c3 = x0[:, 1:1, :] .* xt[:, 2:2, :] .- x0[:, 2:2, :] .* xt[:, 1:1, :]
    return cat(c1, c2, c3; dims=2)
end

function encode_joint_input!(input::Array{Float32, 3}, x0n::Array{Float32, 3},
        xtn::Array{Float32, 3}, tau_norm::Vector{Float32}, params::JointScoreConfig)
    offset = 0
    input[:, (offset + 1):(offset + 3), :] .= x0n
    offset += 3
    input[:, (offset + 1):(offset + 3), :] .= xtn
    offset += 3
    delta = xtn .- x0n
    if params.include_delta_input
        input[:, (offset + 1):(offset + 3), :] .= delta
        offset += 3
    end
    if params.feature_mode in (:physical_aug, :physical_full)
        r20 = sum(abs2, x0n; dims=2)
        r2t = sum(abs2, xtn; dims=2)
        lap0 = periodic_laplacian_sites(x0n)
        lapt = periodic_laplacian_sites(xtn)
        cross = normalized_cross_features_cpu(x0n, xtn)
        input[:, (offset + 1):(offset + 1), :] .= r20
        offset += 1
        input[:, (offset + 1):(offset + 1), :] .= r2t
        offset += 1
        input[:, (offset + 1):(offset + 3), :] .= lap0
        offset += 3
        input[:, (offset + 1):(offset + 3), :] .= lapt
        offset += 3
        input[:, (offset + 1):(offset + 3), :] .= cross
        offset += 3
        if params.feature_mode == :physical_full
            lapd = periodic_laplacian_sites(delta)
            dr2 = r2t .- r20
            input[:, (offset + 1):(offset + 3), :] .= lapd
            offset += 3
            input[:, (offset + 1):(offset + 3), :] .= x0n .* r20
            offset += 3
            input[:, (offset + 1):(offset + 3), :] .= xtn .* r2t
            offset += 3
            input[:, (offset + 1):(offset + 3), :] .= delta .* dr2
            offset += 3
        end
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

function joint_lag_step(sampler::CondPairSampler, rng::AbstractRNG, params::JointScoreConfig)
    L = length(sampler.lag_steps)
    if params.lag_sampling_mode == :uniform
        return sampler.lag_steps[rand(rng, 1:L)]
    elseif params.lag_sampling_mode in (:power, :short)
        idx = clamp(1 + floor(Int, rand(rng)^params.lag_sampling_power * L), 1, L)
        return sampler.lag_steps[idx]
    elseif params.lag_sampling_mode == :active
        first = clamp(params.active_lag_first, 1, L)
        last = params.active_lag_last <= 0 ? L : clamp(params.active_lag_last, first, L)
        return sampler.lag_steps[rand(rng, first:last)]
    end
    error("Unsupported lag_sampling_mode $(params.lag_sampling_mode).")
end

function sample_joint_pair_batch!(x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, sampler::CondPairSampler, rng::AbstractRNG,
        params::JointScoreConfig)
    nt, _, _, ntraj = size(sampler.states)
    B = size(x0, 3)
    @inbounds for b in 1:B
        lag = joint_lag_step(sampler, rng, params)
        upper = nt - lag
        t = rand(rng, sampler.start_idx:upper)
        tr = rand(rng, 1:ntraj)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        tau_norm[b] = Float32((lag * sampler.save_dt) / sampler.tau_max)
    end
    return nothing
end

function make_noisy_joint_inputs(x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, stats::DataStats, params::JointScoreConfig,
        noise_rng::AbstractRNG)
    x0n = apply_stats_tensor(x0, stats)
    xtn = apply_stats_tensor(xt, stats)
    noise0 = randn(noise_rng, Float32, size(x0n))
    noiset = params.endpoint_noise == :both ?
        randn(noise_rng, Float32, size(xtn)) :
        zeros(Float32, size(xtn))
    x0_noisy = x0n .+ params.sigma .* noise0
    xt_noisy = params.endpoint_noise == :both ? xtn .+ params.sigma .* noiset : xtn
    C = params.model_config.in_channels
    input = Array{Float32}(undef, size(x0, 1), C, size(x0, 3))
    input_inv = similar(input)
    encode_joint_input!(input, x0_noisy, xt_noisy, tau_norm, params)
    encode_joint_input!(input_inv, -x0_noisy, -xt_noisy, tau_norm, params)
    return input, input_inv, noise0, noiset, x0_noisy, xt_noisy
end

function joint_prediction(model, input_dev, input_inv_dev)
    pred_pos = model(input_dev)
    pred_neg = model(input_inv_dev)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function joint_transition_residual_from_prediction(pred, input_dev, params::JointScoreConfig,
        score_model, score_sigma::Float32)
    pred0 = @view pred[:, 1:3, :]
    params.score_target == :residualized && return pred0
    stat0 = score_from_dsm_model(score_model, @view(input_dev[:, 1:3, :]), score_sigma)
    return pred0 .- stat0
end

function joint_q_blocks_from_prediction(pred, input_dev, params::JointScoreConfig,
        score_model, score_sigma::Float32)
    pred0 = @view pred[:, 1:3, :]
    predt = @view pred[:, 4:6, :]
    if params.score_target == :raw
        return pred0, predt
    end
    stat0 = score_from_dsm_model(score_model, @view(input_dev[:, 1:3, :]), score_sigma)
    statt = score_from_dsm_model(score_model, @view(input_dev[:, 4:6, :]), score_sigma)
    return pred0 .+ stat0, predt .+ statt
end

function joint_lr_schedule(step::Int, total_steps::Int, params::JointScoreConfig)
    if params.warmup_steps > 0 && step <= params.warmup_steps
        return params.learning_rate * step / params.warmup_steps
    end
    x = (step - params.warmup_steps) / max(total_steps - params.warmup_steps, 1)
    factor = params.min_lr_factor + 0.5 * (1 - params.min_lr_factor) * (1 + cos(pi * clamp(x, 0, 1)))
    return params.learning_rate * factor
end

function train_joint_score(sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, params::JointScoreConfig, device::ExecutionDevice;
        model_path::AbstractString="", initial_model=nothing, initial_model_cfg=nothing,
        initial_history=nothing, start_epoch::Int=1)
    if initial_model === nothing
        model, model_cfg = build_joint_model(params, sampler.N)
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
    B = params.batch_size
    x0 = Array{Float32}(undef, sampler.N, SPIN_CHANNELS, B)
    xt = similar(x0)
    tau_norm = Vector{Float32}(undef, B)
    history = initial_history === nothing ?
        Dict(:train_loss => Float64[], :joint_norm => Float64[],
            :transition_norm => Float64[], :mean_transition_norm => Float64[],
            :stein_norm => Float64[]) :
        initial_history
    total_steps = params.epochs * params.batches_per_epoch
    global_step = max(0, start_epoch - 1) * params.batches_per_epoch
    progress = Progress(max(0, params.epochs - start_epoch + 1);
        desc="Training soft-spin joint score")
    for epoch in start_epoch:params.epochs
        losses = Float64[]
        joint_norms = Float64[]
        transition_norms = Float64[]
        mean_norms = Float64[]
        stein_norms = Float64[]
        for _ in 1:params.batches_per_epoch
            global_step += 1
            params.use_lr_schedule && Flux.adjust!(opt, joint_lr_schedule(global_step, total_steps, params))
            sample_joint_pair_batch!(x0, xt, tau_norm, sampler, rng, params)
            input, input_inv, noise0, noiset, _, _ = make_noisy_joint_inputs(x0, xt, tau_norm, stats, params, noise_rng)
            input_dev = move_array(input, device)
            input_inv_dev = move_array(input_inv, device)
            noise0_dev = move_array(noise0, device)
            noiset_dev = move_array(noiset, device)
            stat0 = score_from_dsm_model(score_model, @view(input_dev[:, 1:3, :]), score_sigma)
            statt = score_from_dsm_model(score_model, @view(input_dev[:, 4:6, :]), score_sigma)
            target0 = noise0_dev .* (-one(eltype(noise0_dev)) / params.sigma)
            targett = noiset_dev .* (-one(eltype(noiset_dev)) / params.sigma)
            if params.score_target == :residualized
                target0 = target0 .- stat0
                targett = targett .- statt
            end
            if params.target_clip_value > 0
                c = Float32(params.target_clip_value)
                target0 = clamp.(target0, -c, c)
                targett = clamp.(targett, -c, c)
            end
            x0_flat = params.stein_weight > 0 ?
                reshape(copy(@view input_dev[:, 1:3, :]), sampler.D, B) : nothing
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = joint_prediction(current_model, input_dev, input_inv_dev)
                pred0 = @view pred[:, 1:3, :]
                predt = @view pred[:, 4:6, :]
                loss = Float32(params.initial_block_weight) * Flux.Losses.mse(pred0, target0)
                if params.terminal_block_weight > 0
                    loss += Float32(params.terminal_block_weight) * Flux.Losses.mse(predt, targett)
                end
                transition_pred = params.score_target == :residualized ? pred0 : pred0 .- stat0
                if params.mean_score_weight > 0
                    mu = dropdims(sum(transition_pred; dims=3) ./ Float32(B); dims=3)
                    loss += Float32(params.mean_score_weight) * mean(abs2, mu)
                end
                if params.stein_weight > 0
                    score_flat = reshape(copy(transition_pred), sampler.D, B)
                    stein = (score_flat * transpose(x0_flat)) ./ Float32(B)
                    loss += Float32(params.stein_weight) * mean(abs2, stein)
                end
                loss
            end
            opt, model = Flux.update!(opt, model, maybe_clip_tree(grads[1], params.gradient_clip_value))
            push!(losses, Float64(to_host(loss_value)))
            if global_step % 20 == 0
                pred = joint_prediction(model, input_dev, input_inv_dev)
                transition = joint_transition_residual_from_prediction(pred, input_dev, params, score_model, score_sigma)
                pred_h = Array(to_host(pred))
                transition_h = Array(to_host(transition))
                push!(joint_norms, sqrt(mean(abs2, pred_h)))
                push!(transition_norms, sqrt(mean(abs2, transition_h)))
                pred_flat = reshape(transition_h, sampler.D, B)
                push!(mean_norms, norm(vec(mean(pred_flat; dims=2))) / sqrt(sampler.D))
                push!(stein_norms, params.stein_weight > 0 ?
                    norm((pred_flat * transpose(Float32.(Array(to_host(x0_flat))))) ./ B) / sqrt(sampler.D) : NaN)
            end
        end
        push!(history[:train_loss], mean(losses))
        push!(history[:joint_norm], isempty(joint_norms) ? NaN : mean(joint_norms))
        push!(history[:transition_norm], isempty(transition_norms) ? NaN : mean(transition_norms))
        push!(history[:mean_transition_norm], isempty(mean_norms) ? NaN : mean(mean_norms))
        push!(history[:stein_norm], isempty(stein_norms) ? NaN : mean(skipmissing(stein_norms)))
        ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch), (:loss, history[:train_loss][end]),
            (:transition_norm, history[:transition_norm][end])
        ])
        if !isempty(model_path) && params.checkpoint_every > 0 &&
                (epoch % params.checkpoint_every == 0 || epoch == params.epochs)
            save_joint_model(model_path, model, model_cfg, stats, params, sampler,
                history, Dict{Symbol, Any}(); completed_epoch=epoch)
        end
    end
    ProgressMeter.finish!(progress)
    Flux.testmode!(model)
    return model, model_cfg, history
end

function evaluate_joint_transition_norm(model, x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, stats::DataStats, params::JointScoreConfig,
        device::ExecutionDevice; batch_size::Int=4096, score_model=nothing,
        score_sigma::Float32=0f0)
    x0n = apply_stats_tensor(x0, stats)
    xtn = apply_stats_tensor(xt, stats)
    out = Array{Float32}(undef, size(x0))
    C = params.model_config.in_channels
    for lo in 1:batch_size:size(x0, 3)
        hi = min(lo + batch_size - 1, size(x0, 3))
        input = Array{Float32}(undef, size(x0, 1), C, hi - lo + 1)
        input_inv = similar(input)
        encode_joint_input!(input, x0n[:, :, lo:hi], xtn[:, :, lo:hi], tau_norm[lo:hi], params)
        encode_joint_input!(input_inv, -x0n[:, :, lo:hi], -xtn[:, :, lo:hi], tau_norm[lo:hi], params)
        input_dev = move_array(input, device)
        pred = joint_prediction(model, input_dev, move_array(input_inv, device))
        r = joint_transition_residual_from_prediction(pred, input_dev, params, score_model, score_sigma)
        out[:, :, lo:hi] .= to_host(r)
    end
    return out
end

function joint_score_diagnostics(model, sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, p::SpinParams, params::JointScoreConfig, phi_blob,
        device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 300)
    noise_rng = MersenneTwister(params.seed + 301)
    eval_lag_idxs = round.(Int, range(1, length(sampler.lag_steps), length=params.eval_tau_count))
    mean_norm = Float64[]
    stein_norm = Float64[]
    posterior_mse = Float64[]
    terminal_mse = Float64[]
    taus_eval = Float64[]
    for idx in eval_lag_idxs
        lag = sampler.lag_steps[idx]
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, params.eval_pairs_per_lag, rng)
        rnorm = evaluate_joint_transition_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        x0n = apply_stats_tensor(x0, stats)
        flat_r = Float64.(flatten_batch(rnorm))
        flat_x = Float64.(flatten_batch(x0n))
        push!(mean_norm, norm(vec(mean(flat_r; dims=2))) / sqrt(sampler.D))
        push!(stein_norm, norm((flat_r * transpose(flat_x)) ./ size(flat_r, 2)) / sqrt(sampler.D))

        input, input_inv, noise0, noiset, _, _ =
            make_noisy_joint_inputs(x0, xt, tau_norm, stats, params, noise_rng)
        input_dev = move_array(input, device)
        pred = joint_prediction(model, input_dev, move_array(input_inv, device))
        q0, qt = joint_q_blocks_from_prediction(pred, input_dev, params, score_model, score_sigma)
        target0 = move_array(noise0 .* (-1f0 / params.sigma), device)
        targett = move_array(noiset .* (-1f0 / params.sigma), device)
        push!(posterior_mse, Float64(to_host(mean(abs2, q0 .- target0))))
        push!(terminal_mse, params.endpoint_noise == :both ?
            Float64(to_host(mean(abs2, qt .- targett))) : NaN)
        push!(taus_eval, lag * sampler.save_dt)
    end

    names = Vector{String}(phi_blob[:observable_names])
    means = Vector{Float64}(phi_blob[:observable_means])
    Cdot_data = Array{Float64}(phi_blob[:Cdot_data])
    operator_lags = phi_blob[:lags][1:min(params.operator_lag_count, length(phi_blob[:lags]))]
    Ctrue = Array{Float64}(undef, length(operator_lags), sampler.N, length(names), sampler.D)
    for (li, lag) in enumerate(operator_lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, Int(lag), params.operator_pairs_per_lag, rng)
        rnorm = evaluate_joint_transition_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        action = true_action_batch(x0, rraw, p)
        obs, _ = observable_values_cond(xt, p)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * length(names), params.operator_pairs_per_lag)
        action_flat = flatten_batch(action)
        mat = -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(action_flat)) / params.operator_pairs_per_lag
        Ctrue[li, :, :, :] .= reshape(mat, sampler.N, length(names), sampler.D)
        params.verbose && @printf("Joint-score true-M operator lag %.5g (%d/%d)\n",
            lag * sampler.save_dt, li, length(operator_lags))
    end
    Cref = Cdot_data[1:length(operator_lags), :, :, :]
    op_metrics = agreement_metrics(Cref, Ctrue)
    active_first = clamp(params.active_lag_first, 1, length(operator_lags))
    active_last = params.active_lag_last <= 0 ? length(operator_lags) :
        clamp(params.active_lag_last, active_first, length(operator_lags))
    active_metrics = agreement_metrics(
        Array(@view Cref[active_first:active_last, :, :, :]),
        Array(@view Ctrue[active_first:active_last, :, :, :]))
    return Dict(:taus_eval => taus_eval, :mean_norm => mean_norm,
        :stein_norm => stein_norm, :posterior_mse => posterior_mse,
        :terminal_mse => terminal_mse, :operator_lags => operator_lags .* sampler.save_dt,
        :operator_trueM => Ctrue, :operator_metrics => op_metrics,
        :operator_active_metrics => active_metrics,
        :operator_active_indices => (active_first, active_last))
end

function retained_operator_diagnostics(model, sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, p::SpinParams, params::JointScoreConfig, target_blob,
        device::ExecutionDevice)
    names = Vector{String}(target_blob[:names])
    means = Vector{Float64}(target_blob[:observable_means])
    selected_indices = Vector{Int}(target_blob[:selected_indices])
    Cdot_data = Array{Float64}(target_blob[:Cdot_data])
    lags = Int.(target_blob[:lags][1:min(params.operator_lag_count, length(target_blob[:lags]))])
    lib = NonlinearLibrary(names)
    retained_data = Matrix{Float64}(undef, length(lags), length(selected_indices))
    retained_trueM = similar(retained_data)
    for li in eachindex(lags)
        retained_data[li, :] .= vec(Cdot_data[li, :, :, :])[selected_indices]
    end
    rng = MersenneTwister(params.seed + 700)
    for (li, lag) in enumerate(lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, params.operator_pairs_per_lag, rng)
        rnorm = evaluate_joint_transition_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        action = true_action_batch(x0, rraw, p)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * length(names), params.operator_pairs_per_lag)
        action_flat = flatten_batch(action)
        mat = -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(action_flat)) /
            params.operator_pairs_per_lag
        retained_trueM[li, :] .= vec(mat)[selected_indices]
        params.verbose && @printf("Joint-score retained true-M operator lag %.5g (%d/%d)\n",
            lag * sampler.save_dt, li, length(lags))
    end
    active_first = clamp(params.active_lag_first, 1, length(lags))
    active_last = params.active_lag_last <= 0 ? length(lags) :
        clamp(params.active_lag_last, active_first, length(lags))
    all_metrics = agreement_metrics(retained_data, retained_trueM)
    active_metrics = agreement_metrics(
        @view(retained_data[active_first:active_last, :]),
        @view(retained_trueM[active_first:active_last, :]))
    return Dict(:lags => lags .* sampler.save_dt,
        :selected_count => length(selected_indices),
        :retained_data => retained_data,
        :retained_trueM => retained_trueM,
        :all_metrics => all_metrics,
        :active_metrics => active_metrics,
        :active_indices => (active_first, active_last))
end

function render_joint_figure(path, params::JointScoreConfig, history, diag, phi_blob)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    op = diag[:operator_metrics]
    figure_title!(fig, "Soft-spin LLG joint-score diagnostics";
        subtitle=@sprintf("true-M operator rel.RMSE %.4e, corr %.5f, target=%s, features=%s",
            op[:relative_rmse], op[:correlation], String(params.score_target),
            String(params.feature_mode)))
    epochs = collect(1:length(history[:train_loss]))
    ax1 = Axis(fig[1, 1]; title="Joint DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
    lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=2)
    ax2 = Axis(fig[1, 2]; title="Score norms", xlabel="epoch", ylabel="norm")
    lines!(ax2, epochs, history[:joint_norm]; color=STYLE_ACCENT, linewidth=2, label="joint rms")
    lines!(ax2, epochs, history[:transition_norm]; color=STYLE_PRIMARY, linewidth=2, label="r rms")
    lines!(ax2, epochs, history[:mean_transition_norm]; color=STYLE_SECONDARY, linewidth=2, label="mean r")
    axislegend(ax2; position=:rt)
    ax3 = Axis(fig[1, 3]; title="Lagwise transition diagnostics", xlabel="tau", ylabel="norm")
    lines!(ax3, diag[:taus_eval], diag[:mean_norm]; color=STYLE_PRIMARY, linewidth=2, label="||E[r]||/sqrt(D)")
    lines!(ax3, diag[:taus_eval], diag[:stein_norm]; color=STYLE_SECONDARY, linewidth=2, label="||E[r x0']||/sqrt(D)")
    axislegend(ax3; position=:rt)
    ax4 = Axis(fig[1, 4]; title="Joint posterior reconstruction", xlabel="tau", ylabel="DSM MSE", yscale=log10)
    lines!(ax4, diag[:taus_eval], diag[:posterior_mse]; color=STYLE_HIGHLIGHT, linewidth=2, label="x0 block")
    if any(isfinite, diag[:terminal_mse])
        lines!(ax4, diag[:taus_eval], diag[:terminal_mse]; color=STYLE_ACCENT, linewidth=2, label="xt block")
        axislegend(ax4; position=:rt)
    end

    Cdata = Array{Float64}(phi_blob[:Cdot_data])
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
        lines!(ax, taus, Ctrue[:, site, obs, col]; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash,
            label="true-M with joint r")
        panel == 1 && axislegend(ax; position=:rt)
    end
    save_figure_checked(path, fig)
end

function save_joint_model(path, model, model_cfg, stats, params, sampler, history, diag;
        completed_epoch::Int=length(history[:train_loss]))
    ensure_parent_dir(path)
    host_model = Flux.fmap(cpu, model)
    trainer_cfg = Dict(:sigma => params.sigma, :epochs => params.epochs,
        :batch_size => params.batch_size, :learning_rate => params.learning_rate,
        :batches_per_epoch => params.batches_per_epoch,
        :mean_score_weight => params.mean_score_weight,
        :stein_weight => params.stein_weight,
        :score_target => String(params.score_target),
        :endpoint_noise => String(params.endpoint_noise),
        :score_type => "joint_score")
    metadata = Dict(:tau_max => sampler.tau_max, :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times, :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
        :include_tau_scalar => params.include_tau_scalar,
        :feature_mode => String(params.feature_mode),
        :completed_epoch => completed_epoch,
        :requested_device => params.device,
        :required_gpu_name => params.required_gpu_name,
        :stationary_score_bson => params.score_bson,
        :operational_transition_score => "first joint-score block minus stationary score for raw targets; first residualized block for residualized targets",
        :no_cheating_audit => "Joint DSM target used trajectory endpoint pairs and Gaussian DSM noise only. The learned stationary score entered only to residualize the DSM target. True mobility appears only in ex-post diagnostics.")
    BSON.bson(path, Dict(:host_model => host_model, :model_cfg => model_cfg,
        :stats => stats, :trainer_cfg => trainer_cfg, :history => history,
        :metadata => metadata, :diagnostics => diag))
    @printf("Saved joint-score checkpoint to %s\n", path)
end

function run_joint_pipeline(param_file::AbstractString)
    base = dirname(param_file)
    params = load_joint_config(param_file)
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
            completed = Int(get(blob[:metadata], :completed_epoch, length(history[:train_loss])))
            if completed >= params.epochs
                @printf("Checkpoint %s already has %d/%d epochs; skipping training.\n",
                    model_path, completed, params.epochs)
                Flux.testmode!(model)
            else
                @printf("Resuming %s from epoch %d/%d.\n", model_path, completed + 1, params.epochs)
                model, model_cfg, history = train_joint_score(sampler, score_model, stats, score_sigma,
                    params, device; model_path=model_path, initial_model=model,
                    initial_model_cfg=model_cfg, initial_history=history, start_epoch=completed + 1)
            end
        else
            model, model_cfg, history = train_joint_score(sampler, score_model, stats, score_sigma,
                params, device; model_path=model_path)
        end
        save_joint_model(model_path, model, model_cfg, stats, params, sampler, history, diag)
    else
        blob = BSON.load(model_path)
        model = move_model(blob[:host_model], device)
        Flux.testmode!(model)
        model_cfg = blob[:model_cfg]
        history = blob[:history]
    end
    if params.evaluate
        phi_blob = BSON.load(phi_path)
        diag = joint_score_diagnostics(model, sampler, score_model, stats, score_sigma, p, params, phi_blob, device)
        if !isempty(params.retained_target_bson)
            retained_path = resolve_path(base, params.retained_target_bson)
            if isfile(retained_path)
                diag[:retained_operator] = retained_operator_diagnostics(model, sampler, score_model,
                    stats, score_sigma, p, params, BSON.load(retained_path), device)
            else
                @warn "retained_target_bson was configured but file does not exist" retained_path
            end
        end
        render_joint_figure(resolve_path(base, params.output_png), params, history, diag, phi_blob)
        metrics_path = resolve_path(base, params.metrics_txt)
        ensure_parent_dir(metrics_path)
        open(metrics_path, "w") do io
            println(io, "SoftSpinLLGChain Step 3 structured joint-score metrics")
            println(io, "config = $(basename(param_file))")
            println(io, "feature_mode = $(params.feature_mode)")
            println(io, "score_target = $(params.score_target)")
            println(io, "endpoint_noise = $(params.endpoint_noise)")
            println(io, @sprintf("true-M operator rel.RMSE = %.8e", diag[:operator_metrics][:relative_rmse]))
            println(io, @sprintf("true-M operator corr = %.8e", diag[:operator_metrics][:correlation]))
            ai = diag[:operator_active_indices]
            println(io, "active operator lag indices = $(ai[1]):$(ai[2])")
            println(io, @sprintf("active-lag true-M operator rel.RMSE = %.8e",
                diag[:operator_active_metrics][:relative_rmse]))
            println(io, @sprintf("active-lag true-M operator corr = %.8e",
                diag[:operator_active_metrics][:correlation]))
            if haskey(diag, :retained_operator)
                rop = diag[:retained_operator]
                ri = rop[:active_indices]
                println(io, @sprintf("retained selected channels = %d", rop[:selected_count]))
                println(io, @sprintf("retained true-M operator rel.RMSE = %.8e",
                    rop[:all_metrics][:relative_rmse]))
                println(io, @sprintf("retained true-M operator corr = %.8e",
                    rop[:all_metrics][:correlation]))
                println(io, "retained active operator lag indices = $(ri[1]):$(ri[2])")
                println(io, @sprintf("retained active-lag true-M operator rel.RMSE = %.8e",
                    rop[:active_metrics][:relative_rmse]))
                println(io, @sprintf("retained active-lag true-M operator corr = %.8e",
                    rop[:active_metrics][:correlation]))
            end
            println(io, @sprintf("mean lagwise ||E[r]||/sqrt(D) = %.8e", mean(diag[:mean_norm])))
            println(io, @sprintf("mean lagwise ||E[r x0']||/sqrt(D) = %.8e", mean(diag[:stein_norm])))
            println(io, @sprintf("mean initial-block posterior DSM MSE = %.8e", mean(diag[:posterior_mse])))
            if any(isfinite, diag[:terminal_mse])
                println(io, @sprintf("mean terminal-block posterior DSM MSE = %.8e", mean(filter(isfinite, diag[:terminal_mse]))))
            end
            println(io, "Gate reference old conditional vA_cont2 true-M operator rel.RMSE = 3.78631217e-01")
            println(io, "Gate reference old conditional vA_cont2 true-M operator corr = 9.34871794e-01")
            println(io, "No-cheating audit: true mobility was used only for this ex-post operator diagnostic.")
        end
        save_joint_model(model_path, model, model_cfg, stats, params, sampler, history, diag)
    end
    @printf("Joint-score stage complete. No-cheating audit: no analytic score or true mobility entered joint DSM training.\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_joint_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_JOINT_PARAM_FILE)
end
