#!/usr/bin/env julia

# Reverse conditional score for the complex-amplitude chain:
#     score(x0 | xt, tau)
#
# The input has normalized x0 q/p, clean normalized xt q/p, and either scalar
# or Fourier lag features.  Only x0 is noised during denoising training.
# The target has only the two x0 noise channels.  This avoids smoothing the
# conditioning variable xt, which is the bias source tested in fit_dM.jl.

include(joinpath(@__DIR__, "joint_score.jl"))

const REVERSE_DEFAULT_PARAM_FILE = joinpath(@__DIR__, "reverse_cond_score.toml")
const REVERSE_OUTPUT_CHANNELS = 2
const REVERSE_SCORE_TYPE_POSTERIOR = "reverse_conditional_x0_given_xt"
const REVERSE_SCORE_TYPE_RESIDUAL = "reverse_transition_residual_x0_given_xt"

Base.@kwdef struct ChainPotentialParams
    alpha::Float64
    beta::Float64
    kappa::Float64
end

Base.@kwdef mutable struct LangevinConfig
    dt::Float64 = 1e-3
    sample_dt::Float64 = 2e-2
    nsteps::Int = 40_000
    resolution::Int = 20
    n_ensembles::Int = 256
    burn_in::Int = 4_000
    sigma::Float32 = 0.1f0
    seed::Int = 21
    progress::Bool = false
end

struct ReverseConditionalScoreUNet{M}
    backbone::M
end

Functors.@functor ReverseConditionalScoreUNet (backbone,)

function (model::ReverseConditionalScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:REVERSE_OUTPUT_CHANNELS, :]
end

function score_from_reverse_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function mean_reverse_score_norm(model, batch, sigma::Float32)
    scores = score_from_reverse_model(model, batch, sigma)
    arr = to_host(scores)
    return sqrt(mean(abs2, arr))
end

function bson_get(d, key::Symbol)
    haskey(d, key) && return d[key]
    skey = String(key)
    haskey(d, skey) && return d[skey]
    error("Key $(key) not found in BSON payload.")
end

function train_reverse_model(sampler::JointPairSampler, stats::DataStats,
        params::L96JointScoreParams, device::ExecutionDevice;
        score_type::String=REVERSE_SCORE_TYPE_POSTERIOR,
        stationary_model=nothing,
        stationary_sigma::Union{Nothing, Float32}=nothing)
    residual_mode = score_type == REVERSE_SCORE_TYPE_RESIDUAL
    if residual_mode
        stationary_model === nothing && error("Residual reverse conditional training requires a stationary score model.")
        stationary_sigma === nothing && error("Residual reverse conditional training requires stationary_sigma.")
        Flux.testmode!(stationary_model)
    end
    Random.seed!(params.model_init_seed)
    model_cfg = adjust_model_config_for_length(params.model_config, size(sampler.states, 2))
    model = ReverseConditionalScoreUNet(build_unet(model_cfg; normalization=params.model_normalization))
    Random.seed!()
    model = move_model(model, device)
    Flux.trainmode!(model)

    opt_state = Flux.setup(Flux.Optimisers.Adam(params.trainer_config.lr), model)
    lr_scheduler = params.trainer_config.use_lr_schedule ?
        create_lr_schedule(params.trainer_config, params.batches_per_epoch) : nothing
    thread_rngs = seed_thread_rngs(params.trainer_config.seed)
    global_step = 0

    K = size(sampler.states, 2)
    B = params.trainer_config.batch_size
    sigma = params.trainer_config.sigma
    state_cpu = Array{Float32}(undef, K, JOINT_STATE_CHANNELS, B)
    norm_cpu = Array{Float32}(undef, K, JOINT_STATE_CHANNELS, B)
    input_cpu = Array{Float32}(undef, K, params.model_config.in_channels, B)
    tnorm_cpu = Vector{Float32}(undef, B)
    noise_cpu = Array{Float32}(undef, K, REVERSE_OUTPUT_CHANNELS, B)
    stein_dim = K * REVERSE_OUTPUT_CHANNELS
    stein_eye_device = move_array(Matrix{Float32}(I, stein_dim, stein_dim), device)
    stein_zero_device = move_array(zeros(Float32, stein_dim, stein_dim), device)

    noisy_device = device isa GPUDevice ? CUDA.CuArray{Float32}(undef, K, params.model_config.in_channels, B) :
        Array{Float32}(undef, K, params.model_config.in_channels, B)
    noise_device = device isa GPUDevice ? CUDA.CuArray{Float32}(undef, K, REVERSE_OUTPUT_CHANNELS, B) :
        noise_cpu

    history = Dict(
        :train_loss => Float64[],
        :score_norm => Float64[],
        :param_norm => Float64[],
        :epoch_time => Float64[],
    )

    progress = params.trainer_config.progress ?
        Progress(params.trainer_config.epochs;
            desc=residual_mode ? "Training reverse transition residual score" : "Training reverse conditional score") : nothing

    for epoch in 1:params.trainer_config.epochs
        epoch_t0 = time_ns()
        epoch_losses = Float64[]
        accumulated_grads = nothing
        accum_count = 0

        for _ in 1:params.batches_per_epoch
            global_step += 1
            lr_scheduler !== nothing && Flux.adjust!(opt_state, lr_scheduler(global_step))

            sample_pair_batch!(state_cpu, tnorm_cpu, sampler, thread_rngs)
            norm_cpu .= apply_pair_stats(state_cpu, stats)
            encode_joint_input!(input_cpu, norm_cpu, tnorm_cpu;
                time_features=params.time_features,
                time_fourier_frequencies=params.time_fourier_frequencies,
                include_delta_input=params.include_delta_input)

            if device isa GPUDevice
                CUDA.@allowscalar copyto!(noisy_device, input_cpu)
                fill_gpu_noise!(noise_device)
                @views noisy_device[:, 1:REVERSE_OUTPUT_CHANNELS, :] .+= sigma .* noise_device
            else
                noisy_device .= input_cpu
                Threads.@threads for sample_idx in 1:B
                    rng = thread_rngs[Threads.threadid()]
                    @inbounds for idx in 1:(K * REVERSE_OUTPUT_CHANNELS)
                        noise_cpu[idx + (sample_idx - 1) * K * REVERSE_OUTPUT_CHANNELS] = randn(rng, Float32)
                    end
                end
                @views noisy_device[:, 1:REVERSE_OUTPUT_CHANNELS, :] .+= sigma .* noise_cpu
            end
            params.include_delta_input && refresh_delta_input_channels!(noisy_device)

            target_noise = noise_device
            if residual_mode
                stat_score = score_from_model(stationary_model,
                    @view(noisy_device[:, 1:REVERSE_OUTPUT_CHANNELS, :]), stationary_sigma)
                target_noise = noise_device .+ sigma .* stat_score
            end

            loss_value, grads = Flux.withgradient(model) do current_model
                pred = current_model(noisy_device)
                loss = Flux.Losses.mse(pred, target_noise)
                if params.stein_weight > 0.0
                    score_x0 = (-pred ./ sigma) .* 1.0f0
                    noisy_x0 = copy(noisy_device[:, 1:REVERSE_OUTPUT_CHANNELS, :])
                    score_flat = reshape(score_x0, stein_dim, B)
                    x_flat = reshape(noisy_x0, stein_dim, B)
                    stein_mat = (score_flat * transpose(x_flat)) ./ Float32(B)
                    stein_target = residual_mode ? stein_zero_device : -stein_eye_device
                    loss += Float32(params.stein_weight) * Flux.Losses.mse(stein_mat, stein_target)
                end
                if params.mean_score_weight > 0.0
                    score_x0 = (-pred ./ sigma) .* 1.0f0
                    score_mean = dropdims(sum(score_x0; dims=3) ./ Float32(B); dims=3)
                    loss += Float32(params.mean_score_weight) * mean(abs2, score_mean)
                end
                loss / params.trainer_config.accumulation_steps
            end

            accumulated_grads = accumulate_trees(accumulated_grads, grads[1])
            accum_count += 1
            push!(epoch_losses, Float64(to_host(loss_value)) * params.trainer_config.accumulation_steps)

            if accum_count >= params.trainer_config.accumulation_steps
                opt_state, model = Flux.update!(opt_state, model, accumulated_grads)
                accumulated_grads = nothing
                accum_count = 0
            end
        end

        if accumulated_grads !== nothing && accum_count > 0
            opt_state, model = Flux.update!(opt_state, model, accumulated_grads)
        end

        sample_pair_batch!(state_cpu, tnorm_cpu, sampler, thread_rngs)
        norm_cpu .= apply_pair_stats(state_cpu, stats)
        encode_joint_input!(input_cpu, norm_cpu, tnorm_cpu;
            time_features=params.time_features,
            time_fourier_frequencies=params.time_fourier_frequencies,
            include_delta_input=params.include_delta_input)
        monitor_batch = move_array(input_cpu, device)

        push!(history[:train_loss], mean(epoch_losses))
        push!(history[:score_norm], mean_reverse_score_norm(model, monitor_batch, sigma))
        push!(history[:param_norm], parameter_norm(model))
        push!(history[:epoch_time], (time_ns() - epoch_t0) / 1e9)

        if progress !== nothing
            ProgressMeter.next!(progress; showvalues=[
                (:epoch, epoch),
                (:loss, history[:train_loss][end]),
                (:score_norm, history[:score_norm][end]),
            ])
        end
    end

    progress !== nothing && ProgressMeter.finish!(progress)
    return model, history
end

function save_reverse_model(path::AbstractString, model, stats::DataStats,
        params::L96JointScoreParams, sampler::JointPairSampler, history;
        score_type::String=REVERSE_SCORE_TYPE_POSTERIOR,
        stationary_score_bson::String="")
    host_model = cpu(model)
    metadata = Dict(
        :tau_min => sampler.tau_min,
        :tau_max => sampler.tau_max,
        :decorrelation_time => sampler.decorrelation_time,
        :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times,
        :shared_normalization => params.shared_normalization,
        :model_normalization => String(params.model_normalization),
        :time_features => params.time_features,
        :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
        :score_type => score_type,
        :stationary_score_bson => stationary_score_bson,
        :conditioning_smoothed => false,
    )
    diagnostics = Dict{Symbol, Any}()
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history diagnostics
    return nothing
end

function run_reverse_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    raw = TOML.parsefile(param_file)
    training_cfg = raw["training"]
    data_cfg = raw["data"]
    score_target = lowercase(String(get(training_cfg, "score_target", "posterior")))
    score_type = score_target in ("residual", "transition_residual", "conditional_residual") ?
        REVERSE_SCORE_TYPE_RESIDUAL : REVERSE_SCORE_TYPE_POSTERIOR
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    output_bson = resolve_path(base_dir, params.output_bson)
    stationary_score_bson = String(get(data_cfg, "score_bson", ""))
    ensure_parent_dir(output_bson)

    require_condition(isfile(input_hdf5), "Input HDF5 file not found: $(input_hdf5)")
    sampler = build_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    stats = compute_state_stats(sampler, params.shared_normalization)

    device = detect_device(params.device_name)
    activate_device!(device)
    @printf("Training device request: %s\n", params.device_name)
    @printf("Resolved execution device: %s\n", describe_device(device))
    @printf("Reverse conditional lag window: [%.3f, %.3f] with %d discrete lags\n",
        sampler.tau_min, sampler.tau_max, length(sampler.lag_steps))
    @printf("Observed decorrelation time: %.6f\n", sampler.decorrelation_time)
    @printf("Reverse score target: %s\n", score_type)

    stationary_model = nothing
    stationary_sigma = nothing
    if score_type == REVERSE_SCORE_TYPE_RESIDUAL
        require_condition(!isempty(stationary_score_bson),
            "Residual reverse conditional training requires [data].score_bson.")
        stationary_path = resolve_path(base_dir, stationary_score_bson)
        require_condition(isfile(stationary_path), "Stationary score BSON not found: $(stationary_path)")
        score_blob = BSON.load(stationary_path)
        stationary_model = move_model(bson_get(score_blob, :host_model), device)
        Flux.testmode!(stationary_model)
        stationary_sigma = Float32(bson_get(score_blob, :trainer_cfg).sigma)
        @printf("Loaded frozen stationary score model from %s (sigma=%.5f)\n",
            stationary_path, stationary_sigma)
    end

    model, history = train_reverse_model(sampler, stats, params, device;
        score_type=score_type, stationary_model=stationary_model,
        stationary_sigma=stationary_sigma)
    checkpoint_path = string(output_bson, ".checkpoint")
    @printf("Saving reverse conditional checkpoint to %s\n", checkpoint_path)
    save_reverse_model(checkpoint_path, model, stats, params, sampler, history;
        score_type=score_type, stationary_score_bson=stationary_score_bson)
    @printf("Saving reverse conditional model to %s\n", output_bson)
    save_reverse_model(output_bson, model, stats, params, sampler, history;
        score_type=score_type, stationary_score_bson=stationary_score_bson)
    rm(checkpoint_path; force=true)
    @printf("Done. Saved reverse conditional score model.\n")
    return history
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = length(ARGS) >= 1 ? ARGS[1] : REVERSE_DEFAULT_PARAM_FILE
    run_reverse_pipeline(param_file)
end
