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
    include_delta_input::Bool
    include_tau_scalar::Bool
    time_fourier_frequencies::Int
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
    score_target::Symbol
    checkpoint_every::Int
    resume::Bool
    seed::Int
    eval_tau_count::Int
    eval_pairs_per_lag::Int
    operator_pairs_per_lag::Int
    operator_lag_count::Int
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

function cond_input_channels(nfreq::Int; include_delta_input::Bool, include_tau_scalar::Bool=false)
    return 2SPIN_CHANNELS + (include_delta_input ? SPIN_CHANNELS : 0) +
        (include_tau_scalar ? 1 : 0) + 2nfreq
end

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    train = raw["training"]
    diag = raw["diagnostics"]
    fig = raw["figure"]
    out = raw["output"]
    run = raw["run"]
    nfreq = Int(get(model, "time_fourier_frequencies", 8))
    include_delta = Bool(get(model, "include_delta_input", true))
    include_tau = Bool(get(model, "include_tau_scalar", false))
    architecture = Symbol(lowercase(String(get(model, "architecture", "unet"))))
    cfg = ScoreUNetConfig(
        in_channels=cond_input_channels(nfreq; include_delta_input=include_delta,
            include_tau_scalar=include_tau),
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
        include_delta_input=include_delta,
        include_tau_scalar=include_tau,
        time_fourier_frequencies=nfreq,
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
        score_target=Symbol(lowercase(String(get(train, "score_target", "residual")))),
        checkpoint_every=Int(get(train, "checkpoint_every", 10)),
        resume=Bool(get(train, "resume", true)),
        seed=Int(get(train, "seed", 20260510)),
        eval_tau_count=Int(get(diag, "eval_tau_count", 8)),
        eval_pairs_per_lag=Int(get(diag, "eval_pairs_per_lag", 50000)),
        operator_pairs_per_lag=Int(get(diag, "operator_pairs_per_lag", 40000)),
        operator_lag_count=Int(get(diag, "operator_lag_count", 24)),
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
    require_condition(params.sigma == 0.05f0, "Conditional DSM sigma must remain 0.05 unless explicitly changed.")
    require_condition(params.model_config.periodic, "Conditional score model must be periodic.")
    require_condition(params.normalization != :batchnorm, "BatchNorm is forbidden.")
    require_condition(params.time_fourier_frequencies >= 1, "Use Fourier time features, not a scalar lag only.")
    require_condition(params.architecture in (:unet, :physical_mlp),
        "Conditional architecture must be unet or physical_mlp.")
    require_condition(params.lag_sampling_power >= 1.0, "lag_sampling_power must be >= 1.")
    require_condition(params.score_target in (:residual, :posterior),
        "score_target must be residual or posterior.")
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
    else
        cfg = adjust_model_config_for_length(params.model_config, N)
        model = SpinConditionalResidualUNet(build_unet(cfg; normalization=params.normalization))
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
    if params.include_tau_scalar
        @inbounds for b in 1:size(input, 3)
            input[:, offset + 1, b] .= tau_norm[b]
        end
        offset += 1
    end
    time_features!(input, offset, tau_norm, params.time_fourier_frequencies)
    return input
end

function sample_lag_step(sampler::CondPairSampler, rng::AbstractRNG, lag_sampling_power::Float64)
    L = length(sampler.lag_steps)
    if lag_sampling_power == 1.0
        return sampler.lag_steps[rand(rng, 1:L)]
    end
    idx = clamp(1 + floor(Int, rand(rng)^lag_sampling_power * L), 1, L)
    return sampler.lag_steps[idx]
end

function sample_pair_batch!(x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, sampler::CondPairSampler, rng::AbstractRNG,
        lag_sampling_power::Float64)
    nt, _, _, ntraj = size(sampler.states)
    B = size(x0, 3)
    @inbounds for b in 1:B
        lag = sample_lag_step(sampler, rng, lag_sampling_power)
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
    inv = copy(input)
    inv[:, 1:6, :] .*= -one(eltype(inv))
    params.include_delta_input && (inv[:, 7:9, :] .*= -one(eltype(inv)))
    pred_neg = model(inv)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function transition_residual_from_model(model, input, params::CondScoreConfig,
        score_model, score_sigma::Float32)
    pred = residual_from_model(model, input, params)
    params.score_target == :residual && return pred
    stat_score = score_from_dsm_model(score_model, @view(input[:, 1:3, :]), score_sigma)
    return pred .- stat_score
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

function train_cond_score(sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, params::CondScoreConfig, device::ExecutionDevice;
        model_path::AbstractString="", initial_model=nothing, initial_model_cfg=nothing,
        initial_history=nothing, start_epoch::Int=1)
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
            :mean_residual_norm => Float64[], :stein_norm => Float64[]) :
        initial_history
    total_steps = params.epochs * params.batches_per_epoch
    global_step = max(0, start_epoch - 1) * params.batches_per_epoch
    progress = Progress(max(0, params.epochs - start_epoch + 1);
        desc="Training soft-spin conditional residual score")
    for epoch in start_epoch:params.epochs
        losses = Float64[]
        residual_norms = Float64[]
        mean_norms = Float64[]
        stein_norms = Float64[]
        for _ in 1:params.batches_per_epoch
            global_step += 1
            params.use_lr_schedule && Flux.adjust!(opt, lr_schedule(global_step, total_steps, params))
            sample_pair_batch!(x0, xt, tau_norm, sampler, rng, params.lag_sampling_power)
            x0n .= apply_stats_tensor(x0, stats)
            xtn .= apply_stats_tensor(xt, stats)
            encode_cond_input!(input, x0n, xtn, tau_norm, params)
            noise_cpu = randn(noise_rng, Float32, sampler.N, 3, B)
            input_dev = move_array(input, device)
            noise = move_array(noise_cpu, device)
            @views input_dev[:, 1:3, :] .+= params.sigma .* noise
            params.include_delta_input && (@views input_dev[:, 7:9, :] .= input_dev[:, 4:6, :] .- input_dev[:, 1:3, :])
            stat_score = score_from_dsm_model(score_model, @view(input_dev[:, 1:3, :]), score_sigma)
            q_target = noise .* (-one(eltype(noise)) / params.sigma)
            target = params.score_target == :posterior ? q_target : q_target .- stat_score
            if params.target_clip_value > 0
                c = Float32(params.target_clip_value)
                target = clamp.(target, -c, c)
            end
            inv_input_dev = inverted_cond_input(input_dev, params)
            x0_flat = params.stein_weight > 0 ? reshape(copy(@view input_dev[:, 1:3, :]), sampler.D, B) : nothing
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = (current_model(input_dev) .- current_model(inv_input_dev)) .* eltype(input_dev)(0.5)
                loss = Flux.Losses.mse(pred, target)
                residual_pred = params.score_target == :posterior ? pred .- stat_score : pred
                if params.mean_score_weight > 0
                    mu = dropdims(sum(residual_pred; dims=3) ./ Float32(B); dims=3)
                    loss += Float32(params.mean_score_weight) * mean(abs2, mu)
                end
                if params.stein_weight > 0
                    score_flat = reshape(residual_pred, sampler.D, B)
                    stein = (score_flat * transpose(x0_flat)) ./ Float32(B)
                    loss += Float32(params.stein_weight) * mean(abs2, stein)
                end
                loss
            end
            opt, model = Flux.update!(opt, model, maybe_clip_tree(grads[1], params.gradient_clip_value))
            push!(losses, Float64(to_host(loss_value)))
            if global_step % 20 == 0
                pred = transition_residual_from_model(model, input_dev, params, score_model, score_sigma)
                pred_h = Array(to_host(pred))
                push!(residual_norms, sqrt(mean(abs2, pred_h)))
                pred_flat = reshape(pred_h, sampler.D, B)
                push!(mean_norms, norm(vec(mean(pred_flat; dims=2))) / sqrt(sampler.D))
                push!(stein_norms, params.stein_weight > 0 ?
                    norm((pred_flat * transpose(Float32.(Array(to_host(x0_flat))))) ./ B) / sqrt(sampler.D) : NaN)
            end
        end
        push!(history[:train_loss], mean(losses))
        push!(history[:residual_norm], isempty(residual_norms) ? NaN : mean(residual_norms))
        push!(history[:mean_residual_norm], isempty(mean_norms) ? NaN : mean(mean_norms))
        push!(history[:stein_norm], isempty(stein_norms) ? NaN : mean(skipmissing(stein_norms)))
        ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch), (:loss, history[:train_loss][end]),
            (:residual_norm, history[:residual_norm][end])
        ])
        if !isempty(model_path) && params.checkpoint_every > 0 &&
                (epoch % params.checkpoint_every == 0 || epoch == params.epochs)
            save_cond_model(model_path, model, model_cfg, stats, params, sampler,
                history, Dict{Symbol, Any}(); completed_epoch=epoch)
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
        pred = params.score_target == :posterior ?
            transition_residual_from_model(model, input_dev, params, score_model, score_sigma) :
            residual_from_model(model, input_dev, params)
        out[:, :, lo:hi] .= to_host(pred)
    end
    return out
end

function normalized_residual_to_raw(residual_norm::Array{Float32, 3}, stats::DataStats)
    return normalized_score_to_raw(residual_norm, stats)
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
        pred_raw = Array(to_host(residual_from_model(model, move_array(input, device), params)))
        q_noisy = params.score_target == :posterior ? pred_raw : pred_raw .+ stat_noisy
        target_q = noise .* (-1f0 / params.sigma)
        push!(posterior_mse, mean(abs2, q_noisy .- target_q))
        push!(taus_eval, lag * sampler.save_dt)
    end

    names = Vector{String}(phi_blob[:observable_names])
    means = Vector{Float64}(phi_blob[:observable_means])
    Cdot_data = Array{Float64}(phi_blob[:Cdot_data])
    operator_lags = phi_blob[:lags][1:min(params.operator_lag_count, length(phi_blob[:lags]))]
    Ctrue = Array{Float64}(undef, length(operator_lags), sampler.N, length(names), sampler.D)
    for (li, lag) in enumerate(operator_lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, Int(lag), params.operator_pairs_per_lag, rng)
        rnorm = evaluate_residual_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        action = true_action_batch(x0, rraw, p)
        obs, _ = observable_values_cond(xt, p)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * length(names), params.operator_pairs_per_lag)
        action_flat = flatten_batch(action)
        mat = -Matrix{Float64}(obs_flat) * transpose(Matrix{Float64}(action_flat)) / params.operator_pairs_per_lag
        Ctrue[li, :, :, :] .= reshape(mat, sampler.N, length(names), sampler.D)
        params.verbose && @printf("Conditional true-M operator lag %.5g (%d/%d)\n",
            lag * sampler.save_dt, li, length(operator_lags))
    end
    Cref = Cdot_data[1:length(operator_lags), :, :, :]
    op_metrics = agreement_metrics(Cref, Ctrue)
    return Dict(:taus_eval => taus_eval, :mean_norm => mean_norm, :stein_norm => stein_norm,
        :posterior_mse => posterior_mse, :operator_lags => operator_lags .* sampler.save_dt,
        :operator_trueM => Ctrue, :operator_metrics => op_metrics)
end

function render_cond_figure(path, params::CondScoreConfig, history, diag, phi_blob)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    op = diag[:operator_metrics]
    figure_title!(fig, "Soft-spin LLG conditional residual score diagnostics";
        subtitle=@sprintf("true-M operator rel.RMSE %.4e, corr %.5f",
            op[:relative_rmse], op[:correlation]))
    epochs = collect(1:length(history[:train_loss]))
    ax1 = Axis(fig[1, 1]; title="DSM residual loss", xlabel="epoch", ylabel="loss", yscale=log10)
    lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=2)
    ax2 = Axis(fig[1, 2]; title="Residual score norms", xlabel="epoch", ylabel="norm")
    lines!(ax2, epochs, history[:residual_norm]; color=STYLE_ACCENT, linewidth=2, label="r rms")
    lines!(ax2, epochs, history[:mean_residual_norm]; color=STYLE_SECONDARY, linewidth=2, label="mean r")
    axislegend(ax2; position=:rt)
    ax3 = Axis(fig[1, 3]; title="Lagwise residual diagnostics", xlabel="tau", ylabel="norm")
    lines!(ax3, diag[:taus_eval], diag[:mean_norm]; color=STYLE_PRIMARY, linewidth=2, label="||E[r]||/sqrt(D)")
    lines!(ax3, diag[:taus_eval], diag[:stein_norm]; color=STYLE_SECONDARY, linewidth=2, label="||E[r x0']||/sqrt(D)")
    axislegend(ax3; position=:rt)
    ax4 = Axis(fig[1, 4]; title="Posterior reconstruction", xlabel="tau", ylabel="DSM MSE", yscale=log10)
    lines!(ax4, diag[:taus_eval], diag[:posterior_mse]; color=STYLE_HIGHLIGHT, linewidth=2)

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
        lines!(ax, taus, Ctrue[:, site, obs, col]; color=STYLE_PRIMARY, linewidth=2, linestyle=:dash, label="true-M with r")
        panel == 1 && axislegend(ax; position=:rt)
    end
    save_figure_checked(path, fig)
end

function save_cond_model(path, model, model_cfg, stats, params, sampler, history, diag;
        completed_epoch::Int=length(history[:train_loss]))
    ensure_parent_dir(path)
    host_model = Flux.fmap(cpu, model)
    trainer_cfg = Dict(:sigma => params.sigma, :epochs => params.epochs,
        :batch_size => params.batch_size, :learning_rate => params.learning_rate,
        :batches_per_epoch => params.batches_per_epoch,
        :architecture => String(params.architecture),
        :mean_score_weight => params.mean_score_weight,
        :stein_weight => params.stein_weight,
        :score_target => String(params.score_target),
        :score_type => "transition_residual_direct")
    metadata = Dict(:tau_max => sampler.tau_max, :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times, :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
        :score_target => String(params.score_target),
        :completed_epoch => completed_epoch,
        :requested_device => params.device,
        :required_gpu_name => params.required_gpu_name,
        :conditioning_smoothed => false,
        :stationary_score_bson => params.score_bson,
        :no_cheating_audit => "Conditional DSM target used only x0 Gaussian noise, clean xt conditioning, and the learned stationary score to form the residual target. No analytic score or true mobility entered training.")
    BSON.bson(path, Dict(:host_model => host_model, :model_cfg => model_cfg,
        :stats => stats, :trainer_cfg => trainer_cfg, :history => history,
        :metadata => metadata, :diagnostics => diag))
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
            completed = Int(get(blob[:metadata], :completed_epoch, length(history[:train_loss])))
            if completed >= params.epochs
                @printf("Checkpoint %s already has %d/%d epochs; skipping training.\n",
                    model_path, completed, params.epochs)
                Flux.testmode!(model)
            else
                @printf("Resuming %s from epoch %d/%d.\n", model_path, completed + 1, params.epochs)
                model, model_cfg, history = train_cond_score(sampler, score_model, stats, score_sigma,
                    params, device; model_path=model_path, initial_model=model,
                    initial_model_cfg=model_cfg, initial_history=history, start_epoch=completed + 1)
            end
        else
            model, model_cfg, history = train_cond_score(sampler, score_model, stats, score_sigma,
                params, device; model_path=model_path)
        end
        save_cond_model(model_path, model, model_cfg, stats, params, sampler, history, diag)
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
            println(io, @sprintf("true-M operator rel.RMSE = %.8e", diag[:operator_metrics][:relative_rmse]))
            println(io, @sprintf("true-M operator corr = %.8e", diag[:operator_metrics][:correlation]))
            println(io, @sprintf("mean lagwise ||E[r]||/sqrt(D) = %.8e", mean(diag[:mean_norm])))
            println(io, @sprintf("mean lagwise ||E[r x0']||/sqrt(D) = %.8e", mean(diag[:stein_norm])))
            println(io, @sprintf("mean posterior reconstruction DSM MSE = %.8e", mean(diag[:posterior_mse])))
            println(io, "No-cheating audit: true mobility was used only for this ex-post operator diagnostic.")
        end
        save_cond_model(model_path, model, model_cfg, stats, params, sampler, history, diag)
    end
    @printf("Conditional score stage complete. No-cheating audit: no analytic score or true mobility entered conditional DSM training.\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE)
end
