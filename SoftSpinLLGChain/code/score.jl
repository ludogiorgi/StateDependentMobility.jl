#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "score.toml"))

Base.@kwdef struct ScoreConfig
    input_hdf5::String
    burnin_fraction::Float64
    max_samples::Int
    spin_inversion_augment::Bool
    enforce_zero_mean::Bool
    model_config::ScoreUNetConfig
    normalization::Symbol
    output_mode::Symbol
    input_features::Symbol
    init_seed::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float64
    sigma::Float32
    epoch_subset_size::Int
    use_lr_schedule::Bool
    min_lr_factor::Float64
    save_best_validation::Bool
    symmetrized_loss::Bool
    checkpoint_every::Int
    allow_nondefault_sigma::Bool
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
        model_config=cfg,
        normalization=normalization_from_string(get(model, "normalization", "none")),
        output_mode=Symbol(lowercase(String(get(model, "output_mode", "noise")))),
        input_features=Symbol(lowercase(String(get(model, "input_features", "spin")))),
        init_seed=Int(get(model, "init_seed", 314159)),
        batch_size=Int(get(train, "batch_size", 8192)),
        epochs=Int(get(train, "epochs", 140)),
        learning_rate=Float64(get(train, "learning_rate", 1.2e-4)),
        sigma=Float32(get(train, "sigma", 0.05)),
        epoch_subset_size=Int(get(train, "epoch_subset_size", 262144)),
        use_lr_schedule=Bool(get(train, "use_lr_schedule", true)),
        min_lr_factor=Float64(get(train, "min_lr_factor", 0.08)),
        save_best_validation=Bool(get(train, "save_best_validation", true)),
        symmetrized_loss=Bool(get(train, "symmetrized_loss", false)),
        checkpoint_every=Int(get(train, "checkpoint_every", 0)),
        allow_nondefault_sigma=Bool(get(train, "allow_nondefault_sigma", false)),
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
        device=String(get(run, "device", "GPU:0")),
        required_gpu_name=String(get(run, "required_gpu_name", "2080ti")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
    )
    require_condition(params.sigma == 0.05f0 || params.allow_nondefault_sigma,
        "DSM sigma must remain 0.05 unless explicitly changed by the user and allow_nondefault_sigma=true.")
    require_condition(params.model_config.periodic, "The score model must be periodic.")
    require_condition(params.output_mode in (:noise, :score), "output_mode must be either noise or score.")
    require_condition(params.input_features in (:spin, :spin_r2),
        "input_features must be either spin or spin_r2.")
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

function score_checkpoint_blob(host_model, model_cfg, dataset::NormalizedDataset,
        params::ScoreConfig, p::SpinParams, history, epoch::Int, val_loss::Float64;
        checkpoint_kind::AbstractString)
    trainer_cfg = Dict(:sigma => params.sigma, :epochs => params.epochs,
        :batch_size => params.batch_size, :learning_rate => params.learning_rate,
        :output_mode => String(params.output_mode),
        :input_features => String(params.input_features),
        :spin_inversion_augment => params.spin_inversion_augment,
        :enforce_zero_mean => params.enforce_zero_mean,
        :symmetrized_loss => params.symmetrized_loss,
        :save_best_validation => params.save_best_validation,
        :checkpoint_kind => String(checkpoint_kind),
        :checkpoint_epoch => epoch,
        :checkpoint_validation_loss => val_loss)
    metadata = Dict(:no_cheating_audit =>
        "DSM target is constructed only from Gaussian noise added to data samples; analytic score is excluded from all losses and model selection. Symmetry augmentation, when enabled, uses only the spin-inversion symmetry of the observed system.")
    return Dict(:host_model => host_model, :model_cfg => model_cfg,
        :stats => dataset.stats, :trainer_cfg => trainer_cfg, :history => history,
        :phys => p, :metadata => metadata)
end

function periodic_score_checkpoint_path(model_path::AbstractString, epoch::Int)
    root, ext = splitext(model_path)
    return string(root, "_epoch", lpad(string(epoch), 4, '0'), ext)
end

function train_score_model(dataset::NormalizedDataset, params::ScoreConfig,
        device::ExecutionDevice, p::SpinParams, model_path::AbstractString)
    Random.seed!(params.init_seed)
    model, model_cfg = build_spin_unet(params.model_config, params.normalization, size(dataset.data, 1);
        output_mode=params.output_mode, input_features=params.input_features)
    Random.seed!()
    model = move_model(model, device)
    Flux.trainmode!(model)
    opt = Flux.setup(Flux.Optimisers.Adam(params.learning_rate), model)
    rng = MersenneTwister(params.seed)
    noise_rng = MersenneTwister(params.seed + 1)
    history = Dict(:train_loss => Float64[], :val_loss => Float64[], :score_norm => Float64[])
    best = (; val_loss=Inf, epoch=0, host_model=nothing)
    n = length(dataset)
    val_n = min(max(params.batch_size, n ÷ 20), n)
    val_idx = randperm(rng, n)[1:val_n]
    val_noise = randn(MersenneTwister(params.seed + 11), Float32, size(dataset.data, 1),
        size(dataset.data, 2), val_n)
    progress = Progress(params.epochs; desc="Training soft-spin stationary score")
    for epoch in 1:params.epochs
        if params.use_lr_schedule
            Flux.adjust!(opt, params.learning_rate * lr_factor(epoch, params.epochs, params.min_lr_factor))
        end
        idxs = if params.epoch_subset_size > 0 && params.epoch_subset_size < n
            randperm(rng, n)[1:params.epoch_subset_size]
        else
            randperm(rng, n)
        end
        losses = Float64[]
        for part in Iterators.partition(idxs, params.batch_size)
            batch_cpu = copy(dataset.data[:, :, collect(part)])
            noise_cpu = randn(noise_rng, Float32, size(batch_cpu))
            batch = move_array(batch_cpu, device)
            noise = move_array(noise_cpu, device)
            noisy = batch .+ params.sigma .* noise
            target = params.symmetrized_loss || params.output_mode == :score ?
                noise .* (-one(eltype(noise)) / params.sigma) : noise
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = params.symmetrized_loss ?
                    score_from_dsm_model(current_model, noisy, params.sigma) : current_model(noisy)
                Flux.Losses.mse(pred, target)
            end
            opt, model = Flux.update!(opt, model, grads[1])
            push!(losses, Float64(to_host(loss_value)))
        end
        val_loss = validation_loss(model, dataset, val_idx, val_noise, params, device)
        monitor_n = min(4096, n)
        score = evaluate_score_norm(model, copy(dataset.data[:, :, 1:monitor_n]), params.sigma, device;
            batch_size=params.batch_size)
        push!(history[:train_loss], mean(losses))
        push!(history[:val_loss], val_loss)
        push!(history[:score_norm], mean(sqrt.(sum(abs2, reshape(score, :, monitor_n); dims=1))))
        if params.save_best_validation && val_loss < best.val_loss
            best = (; val_loss=val_loss, epoch=epoch, host_model=Flux.fmap(cpu, model))
        end
        if params.checkpoint_every > 0 && epoch % params.checkpoint_every == 0
            host_latest = Flux.fmap(cpu, model)
            ckpt_path = periodic_score_checkpoint_path(model_path, epoch)
            ensure_parent_dir(ckpt_path)
            BSON.bson(ckpt_path, score_checkpoint_blob(host_latest, model_cfg, dataset,
                params, p, history, epoch, val_loss; checkpoint_kind="periodic_latest"))
            latest_path = string(splitext(model_path)[1], "_latest", splitext(model_path)[2])
            BSON.bson(latest_path, score_checkpoint_blob(host_latest, model_cfg, dataset,
                params, p, history, epoch, val_loss; checkpoint_kind="latest"))
            @printf("Saved periodic score checkpoint epoch %d to %s\n", epoch, ckpt_path)
        end
        ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch), (:train_loss, history[:train_loss][end]), (:val_loss, val_loss)
        ])
    end
    ProgressMeter.finish!(progress)
    if !params.save_best_validation || best.host_model === nothing
        best = (; val_loss=history[:val_loss][end], epoch=params.epochs,
            host_model=Flux.fmap(cpu, model))
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
        noise = move_array(noise_cpu, device)
        target = params.symmetrized_loss || params.output_mode == :score ?
            noise .* (-one(eltype(noise)) / params.sigma) : noise
        noisy = move_array(batch_cpu .+ params.sigma .* noise_cpu, device)
        pred = params.symmetrized_loss ?
            score_from_dsm_model(model, noisy, params.sigma) : model(noisy)
        loss = Flux.Losses.mse(pred, target)
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
            final_epoch, final_val; checkpoint_kind="best_validation_final")
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
    return Dict(:covariance => cov_metrics, :moments => moments)
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
