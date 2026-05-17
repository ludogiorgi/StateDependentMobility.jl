#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "fhd_learning_common.jl"))

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "cond_score.toml")

Base.@kwdef struct FHDCondScoreParams
    input_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    tau_min::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    model_config::ScoreUNetConfig
    model_normalization::Symbol
    time_features::String
    time_fourier_frequencies::Int
    include_delta_input::Bool
    model_init_seed::Int
    warm_start_bson::String
    batch_size::Int
    epochs::Int
    batches_per_epoch::Int
    learning_rate::Float64
    sigma::Float32
    use_lr_schedule::Bool
    warmup_steps::Int
    min_lr_factor::Float64
    gradient_clip_value::Float64
    target_clip_value::Float64
    stein_weight::Float64
    mean_score_weight::Float64
    seed::Int
    progress::Bool
    eval_tau_count::Int
    eval_pairs_per_lag::Int
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    device_name::String
    required_gpu_name::String
    evaluate::Bool
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    training = raw["training"]
    figure = raw["figure"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    time_features = normalize_time_features(get(model, "time_features", "fourier"))
    nfreq = Int(get(model, "time_fourier_frequencies", 8))
    include_delta = Bool(get(model, "include_delta_input", true))
    time_features == "fourier" && require_condition(nfreq >= 1, "Fourier time features require at least one frequency.")
    in_channels = cond_input_channels(time_features, nfreq; include_delta_input=include_delta)
    model_cfg = ScoreUNetConfig(
        in_channels=in_channels,
        base_channels=Int(get(model, "base_channels", 96)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(get(model, "activation", "swish")),
        final_activation=activation_from_string(get(model, "final_activation", "identity")),
    )
    params = FHDCondScoreParams(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_min=Float64(get(data, "tau_min", 0.0)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 0.67)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        model_config=model_cfg,
        model_normalization=normalization_from_string(get(model, "normalization", "none")),
        time_features=time_features,
        time_fourier_frequencies=nfreq,
        include_delta_input=include_delta,
        model_init_seed=Int(get(model, "init_seed", 271830)),
        warm_start_bson=String(get(model, "warm_start_bson", "")),
        batch_size=Int(get(training, "batch_size", 4096)),
        epochs=Int(get(training, "epochs", 180)),
        batches_per_epoch=Int(get(training, "batches_per_epoch", 256)),
        learning_rate=Float64(get(training, "learning_rate", 1.8e-4)),
        sigma=Float32(get(training, "sigma", 0.05)),
        use_lr_schedule=Bool(get(training, "use_lr_schedule", true)),
        warmup_steps=Int(get(training, "warmup_steps", 500)),
        min_lr_factor=Float64(get(training, "min_lr_factor", 0.08)),
        gradient_clip_value=Float64(get(training, "gradient_clip_value", 0.0)),
        target_clip_value=Float64(get(training, "target_clip_value", 0.0)),
        stein_weight=Float64(get(training, "stein_weight", 0.0)),
        mean_score_weight=Float64(get(training, "mean_score_weight", 0.0)),
        seed=Int(get(training, "seed", 20260621)),
        progress=Bool(get(training, "progress", true)),
        eval_tau_count=Int(get(figure, "eval_tau_count", 6)),
        eval_pairs_per_lag=Int(get(figure, "eval_pairs_per_lag", 40000)),
        figure_width=Int(get(figure, "width", 3000)),
        figure_height=Int(get(figure, "height", 2200)),
        output_bson=String(output["model_bson"]),
        output_png=String(output["figure_png"]),
        device_name=String(get(run, "device", "GPU:1")),
        required_gpu_name=String(get(run, "required_gpu_name", "5070")),
        evaluate=Bool(get(run, "evaluate", true)),
    )
    require_condition(params.sigma == 0.05f0, "This experiment is configured to train DSM only at sigma=0.05.")
    require_condition(params.model_config.periodic, "FHDChain conditional score U-Net must be periodic.")
    require_condition(params.model_normalization != :batchnorm, "New FHDChain conditional models must not use BatchNorm.")
    return params
end

function clip_gradient_tree(grads, clip_value::Float64)
    clip_value <= 0.0 && return grads
    c = Float32(clip_value)
    return Functors.fmap(grads) do g
        g isa AbstractArray ? clamp.(g, -c, c) : g
    end
end

function load_stationary_checkpoint(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    stats = load_stats_from_bson(blob)
    trainer_cfg = dict_get(blob, :trainer_cfg)
    sigma = Float32(trainer_cfg.sigma)
    return model, stats, sigma, blob
end

function train_cond_model(sampler::FHDPairSampler, stationary_model, stats::DataStats,
        stationary_sigma::Float32, params::FHDCondScoreParams, device::ExecutionDevice,
        base_dir::AbstractString)
    if isempty(params.warm_start_bson)
        Random.seed!(params.model_init_seed)
        model, model_cfg = build_cond_unet(params.model_config, params.model_normalization)
        Random.seed!()
    else
        warm_path = resolve_path(base_dir, params.warm_start_bson)
        isfile(warm_path) || error("Warm-start conditional checkpoint not found: $(warm_path)")
        blob = BSON.load(warm_path)
        model = dict_get(blob, :host_model)
        model_cfg = dict_get(blob, :model_cfg)
        @printf("Warm-starting conditional residual model from %s\n", warm_path)
    end
    model = move_model(model, device)
    Flux.trainmode!(model)
    Flux.testmode!(stationary_model)

    opt_state = Flux.setup(Flux.Optimisers.Adam(params.learning_rate), model)
    trainer_cfg = ScoreTrainerConfig(batch_size=params.batch_size, epochs=params.epochs,
        lr=params.learning_rate, sigma=params.sigma, shuffle=true, seed=params.seed,
        progress=params.progress, use_lr_schedule=params.use_lr_schedule,
        warmup_steps=params.warmup_steps, min_lr_factor=params.min_lr_factor)
    lr_scheduler = params.use_lr_schedule ? create_lr_schedule(trainer_cfg, params.batches_per_epoch) : nothing
    rng = MersenneTwister(params.seed)
    noise_rng = MersenneTwister(params.seed + 1)
    K = sampler.K
    B = params.batch_size
    channels = params.model_config.in_channels
    x0 = Array{Float32}(undef, K, 2, B)
    xt = Array{Float32}(undef, K, 2, B)
    x0n = Array{Float32}(undef, K, 2, B)
    xtn = Array{Float32}(undef, K, 2, B)
    input = Array{Float32}(undef, K, channels, B)
    tnorm = Vector{Float32}(undef, B)
    noise_cpu = Array{Float32}(undef, K, 2, B)
    history = Dict(:train_loss => Float64[], :transition_score_norm => Float64[],
        :param_norm => Float64[], :epoch_time => Float64[])
    progress = params.progress ? Progress(params.epochs; desc="Training FHD reverse transition residual") : nothing
    global_step = 0

    for epoch in 1:params.epochs
        t0 = time_ns()
        losses = Float64[]
        for _ in 1:params.batches_per_epoch
            global_step += 1
            lr_scheduler !== nothing && Flux.adjust!(opt_state, lr_scheduler(global_step))
            sample_pair_batch!(x0, xt, tnorm, sampler, rng)
            x0n .= apply_fhd_stats(x0, stats)
            xtn .= apply_fhd_stats(xt, stats)
            encode_cond_input!(input, x0n, xtn, tnorm;
                time_features=params.time_features,
                time_fourier_frequencies=params.time_fourier_frequencies,
                include_delta_input=params.include_delta_input)
            fill_projected_noise!(noise_cpu, noise_rng)
            input_device = move_array(input, device)
            noise = move_array(noise_cpu, device)
            @views begin
                input_device[:, 1:2, :] .+= params.sigma .* noise
                project_zero_modes!(input_device[:, 1:2, :])
            end
            params.include_delta_input && refresh_delta_input_channels!(input_device)
            stat_score = score_from_projected_model(stationary_model,
                @view(input_device[:, 1:2, :]), stationary_sigma)
            target_noise = noise .+ params.sigma .* stat_score
            if params.target_clip_value > 0.0
                c = Float32(params.target_clip_value)
                target_noise = clamp.(target_noise, -c, c)
            end
            x_flat_for_stein = params.stein_weight > 0.0 ?
                reshape(copy(@view(input_device[:, 1:2, :])), K * 2, B) : nothing
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = current_model(input_device)
                loss = Flux.Losses.mse(pred, target_noise)
                if params.stein_weight > 0.0
                    transition_score = -pred ./ params.sigma
                    score_flat = reshape(transition_score, K * 2, B)
                    stein = (score_flat * transpose(x_flat_for_stein)) ./ Float32(B)
                    loss += Float32(params.stein_weight) * mean(abs2, stein)
                end
                if params.mean_score_weight > 0.0
                    transition_score = -pred ./ params.sigma
                    score_mean = dropdims(sum(transition_score; dims=3) ./ Float32(B); dims=3)
                    loss += Float32(params.mean_score_weight) * mean(abs2, score_mean)
                end
                loss
            end
            opt_state, model = Flux.update!(opt_state, model,
                clip_gradient_tree(grads[1], params.gradient_clip_value))
            push!(losses, Float64(to_host(loss_value)))
        end

        sample_pair_batch!(x0, xt, tnorm, sampler, rng)
        x0n .= apply_fhd_stats(x0, stats)
        xtn .= apply_fhd_stats(xt, stats)
        encode_cond_input!(input, x0n, xtn, tnorm;
            time_features=params.time_features,
            time_fourier_frequencies=params.time_fourier_frequencies,
            include_delta_input=params.include_delta_input)
        score = score_from_residual_model(model, move_array(input, device), params.sigma)
        flat = reshape(to_host(score), K * 2, B)
        push!(history[:train_loss], mean(losses))
        push!(history[:transition_score_norm], Float64(mean(sqrt.(sum(abs2, flat; dims=1)))))
        push!(history[:param_norm], parameter_norm(model))
        push!(history[:epoch_time], (time_ns() - t0) / 1e9)
        progress !== nothing && ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch),
            (:loss, history[:train_loss][end]),
            (:score_norm, history[:transition_score_norm][end]),
        ])
    end
    progress !== nothing && ProgressMeter.finish!(progress)
    return model, model_cfg, history
end

function choose_eval_lags(sampler::FHDPairSampler, count::Int)
    total = length(sampler.lag_steps)
    n = min(count, total)
    idxs = n == 1 ? [1] : unique(round.(Int, range(1, total, length=n)))
    return idxs
end

function cond_diagnostics(model, stationary_model, sampler::FHDPairSampler, stats::DataStats,
        stationary_sigma::Float32, params::FHDCondScoreParams, device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 200)
    idxs = choose_eval_lags(sampler, params.eval_tau_count)
    P = projection_matrix(sampler.K)
    tau = Float64[]
    mean_transition = Float64[]
    transition_x0 = Float64[]
    posterior_mean = Float64[]
    posterior_stein = Float64[]
    for idx in idxs
        lag = sampler.lag_steps[idx]
        x0, xt, _, _ = random_lag_pairs(sampler, lag, params.eval_pairs_per_lag, rng)
        x0n = apply_fhd_stats(x0, stats)
        trans = evaluate_transition_score_norm(model, x0, xt, sampler.lag_tnorm[idx], stats,
            params.sigma, device; batch_size=params.batch_size,
            time_features=params.time_features,
            time_fourier_frequencies=params.time_fourier_frequencies,
            include_delta_input=params.include_delta_input)
        stat = evaluate_stationary_score_norm(stationary_model, x0n, stationary_sigma,
            device; batch_size=params.batch_size)
        posterior = stat .+ trans
        rflat = Float64.(raw_flat_batch(trans))
        pflat = Float64.(raw_flat_batch(posterior))
        xflat = Float64.(raw_flat_batch(x0n))
        push!(tau, sampler.lag_times[idx])
        push!(mean_transition, norm(vec(mean(rflat; dims=2))) / sqrt(size(rflat, 1)))
        push!(transition_x0, norm((rflat * transpose(xflat)) ./ size(xflat, 2)) / norm(P))
        push!(posterior_mean, norm(vec(mean(pflat; dims=2))) / sqrt(size(pflat, 1)))
        stein = -(pflat * transpose(xflat)) ./ size(xflat, 2)
        push!(posterior_stein, norm(stein - P) / norm(P))
        @printf("Conditional diagnostics tau %.5g: ||E[r]||=%.3e, ||E[r x0']||=%.3e, posterior Stein=%.3e\n",
            tau[end], mean_transition[end], transition_x0[end], posterior_stein[end])
    end
    return Dict(
        :tau => tau,
        :mean_transition => mean_transition,
        :transition_x0 => transition_x0,
        :posterior_mean => posterior_mean,
        :posterior_stein => posterior_stein,
    )
end

function create_cond_figure(path::AbstractString, params::FHDCondScoreParams, history, diagnostics)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("residual DSM sigma=%.3f, final loss=%.3e", params.sigma, history[:train_loss][end])
        figure_title!(fig, "FHDChain reverse transition-score residual diagnostics"; subtitle=subtitle)
        epochs = collect(1:length(history[:train_loss]))
        ax1 = Axis(fig[1, 1]; title="Residual DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=curve_linewidth())
        ax2 = Axis(fig[1, 2]; title="Transition score norm", xlabel="epoch", ylabel="norm")
        lines!(ax2, epochs, history[:transition_score_norm]; color=STYLE_ACCENT, linewidth=curve_linewidth())
        tau = diagnostics[:tau]
        ax3 = Axis(fig[1, 3]; title="Transition self consistency", xlabel="tau", ylabel="norm")
        lines!(ax3, tau, diagnostics[:mean_transition]; color=STYLE_PRIMARY, label="||E[r]||")
        lines!(ax3, tau, diagnostics[:transition_x0]; color=STYLE_SECONDARY, label="||E[r x0']||")
        axislegend(ax3; position=:rt)
        ax4 = Axis(fig[2, 1]; title="Posterior diagnostics", xlabel="tau", ylabel="relative norm")
        lines!(ax4, tau, diagnostics[:posterior_mean]; color=STYLE_ACCENT, label="||E[q]||")
        lines!(ax4, tau, diagnostics[:posterior_stein]; color=STYLE_HIGHLIGHT, label="Stein error")
        axislegend(ax4; position=:rt)
        lines = String[
            "target = transition residual",
            "posterior score = frozen stationary score + residual",
            "operational transition score = residual only",
            "DSM noise only on x0 and zero-mode projected",
            @sprintf("tau range = [%.4g, %.4g]", minimum(tau), maximum(tau)),
            @sprintf("final DSM loss = %.6e", history[:train_loss][end]),
            @sprintf("min posterior Stein error = %.6e", minimum(diagnostics[:posterior_stein])),
            @sprintf("max ||E[r x0']||/||P|| = %.6e", maximum(diagnostics[:transition_x0])),
        ]
        text_panel!(fig[2, 2:3], lines; title="Summary")
        save_figure_checked(path, fig)
    end
    return nothing
end

function save_cond_model(path::AbstractString, model, model_cfg, stats::DataStats,
        params::FHDCondScoreParams, sampler::FHDPairSampler, history, diagnostics,
        stationary_score_bson::AbstractString)
    host_model = cpu(model)
    trainer_cfg = ScoreTrainerConfig(batch_size=params.batch_size, epochs=params.epochs,
        lr=params.learning_rate, sigma=params.sigma, seed=params.seed)
    stats_dict = Dict(:mean => stats.mean, :std => stats.std)
    metadata = Dict(
        :system => "fhd_chain",
        :score_type => "reverse_transition_residual_x0_given_xt",
        :stationary_score_bson => stationary_score_bson,
        :conditioning_smoothed => false,
        :tau_min => sampler.tau_min,
        :tau_max => sampler.tau_max,
        :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times,
        :lag_tnorm => sampler.lag_tnorm,
        :decorrelation_time => sampler.decorrelation_time,
        :time_features => params.time_features,
        :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
        :normalization => String(params.model_normalization),
        :analytic_score_used_for_training => false,
        :gradient_clip_value => params.gradient_clip_value,
        :target_clip_value => params.target_clip_value,
        :stein_weight => params.stein_weight,
        :mean_score_weight => params.mean_score_weight,
        :warm_start_bson => params.warm_start_bson,
    )
    BSON.@save path host_model model_cfg trainer_cfg stats_dict metadata history diagnostics
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    output_bson = resolve_path(base_dir, params.output_bson)
    output_png = resolve_path(base_dir, params.output_png)
    ensure_parent_dir(output_bson)
    ensure_parent_dir(output_png)
    require_condition(isfile(input_hdf5), "Input HDF5 not found: $(input_hdf5)")
    require_condition(isfile(score_bson), "Stationary score checkpoint not found: $(score_bson)")

    device = detect_fhd_device(params.device_name, params.required_gpu_name)
    activate_and_describe_device!(device, params.device_name, params.required_gpu_name)
    stationary_model, stats, stationary_sigma, _ = load_stationary_checkpoint(score_bson, device)
    @printf("Loaded frozen stationary score from %s (sigma=%.5f)\n", score_bson, stationary_sigma)

    sampler = build_fhd_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    @printf("Conditional lag window: [%.5g, %.5g], %d lags, tD=%.5g\n",
        sampler.tau_min, sampler.tau_max, length(sampler.lag_steps), sampler.decorrelation_time)
    model, model_cfg, history = train_cond_model(sampler, stationary_model, stats,
        stationary_sigma, params, device, base_dir)
    checkpoint = string(output_bson, ".checkpoint")
    save_cond_model(checkpoint, model, model_cfg, stats, params, sampler, history,
        Dict(:checkpoint => true), params.score_bson)
    if params.evaluate
        diagnostics = cond_diagnostics(model, stationary_model, sampler, stats,
            stationary_sigma, params, device)
        save_cond_model(output_bson, model, model_cfg, stats, params, sampler, history,
            diagnostics, params.score_bson)
        rm(checkpoint; force=true)
        create_cond_figure(output_png, params, history, diagnostics)
        @printf("Done. Residual DSM loss=%.6e, max ||E[r x0']||/||P||=%.6e\n",
            history[:train_loss][end], maximum(diagnostics[:transition_x0]))
    else
        save_cond_model(output_bson, model, model_cfg, stats, params, sampler, history,
            Dict(:diagnostics_skipped => true), params.score_bson)
        rm(checkpoint; force=true)
        @printf("Done. Residual DSM loss=%.6e\n", history[:train_loss][end])
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
