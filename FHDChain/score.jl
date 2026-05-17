#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "fhd_learning_common.jl"))

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "score.toml")

Base.@kwdef struct FHDScoreParams
    input_hdf5::String
    burnin_fraction::Float64
    max_samples::Int
    model_config::ScoreUNetConfig
    model_normalization::Symbol
    model_init_seed::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float64
    sigma::Float32
    epoch_subset_size::Int
    use_lr_schedule::Bool
    warmup_steps::Int
    min_lr_factor::Float64
    seed::Int
    progress::Bool
    stein_samples::Int
    exact_score_samples::Int
    safe_density_min::Float64
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    output_langevin_hdf5::String
    output_langevin_png::String
    device_name::String
    required_gpu_name::String
    train::Bool
    evaluate::Bool
    langevin_validate::Bool
    langevin_dt::Float64
    langevin_total_time::Float64
    langevin_burnin_time::Float64
    langevin_save_dt::Float64
    langevin_ntraj::Int
    langevin_score_clip::Float32
    langevin_max_pdf_samples::Int
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    training = raw["training"]
    figure = raw["figure"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    model_cfg = ScoreUNetConfig(
        in_channels=FHD_STATE_CHANNELS,
        base_channels=Int(get(model, "base_channels", 96)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(get(model, "activation", "swish")),
        final_activation=activation_from_string(get(model, "final_activation", "identity")),
    )
    params = FHDScoreParams(
        input_hdf5=String(data["input_hdf5"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        max_samples=Int(get(data, "max_samples", 1_048_576)),
        model_config=model_cfg,
        model_normalization=normalization_from_string(get(model, "normalization", "none")),
        model_init_seed=Int(get(model, "init_seed", 314159)),
        batch_size=Int(get(training, "batch_size", 8192)),
        epochs=Int(get(training, "epochs", 120)),
        learning_rate=Float64(get(training, "learning_rate", 1.5e-4)),
        sigma=Float32(get(training, "sigma", 0.05)),
        epoch_subset_size=Int(get(training, "epoch_subset_size", 262144)),
        use_lr_schedule=Bool(get(training, "use_lr_schedule", true)),
        warmup_steps=Int(get(training, "warmup_steps", 200)),
        min_lr_factor=Float64(get(training, "min_lr_factor", 0.08)),
        seed=Int(get(training, "seed", 20260620)),
        progress=Bool(get(training, "progress", true)),
        stein_samples=Int(get(figure, "stein_samples", 100_000)),
        exact_score_samples=Int(get(figure, "exact_score_samples", 100_000)),
        safe_density_min=Float64(get(figure, "safe_density_min", 0.05)),
        figure_width=Int(get(figure, "width", 3000)),
        figure_height=Int(get(figure, "height", 2200)),
        output_bson=String(output["model_bson"]),
        output_png=String(output["figure_png"]),
        output_langevin_hdf5=String(get(output, "langevin_hdf5", "outputs/score_sigma005_langevin.h5")),
        output_langevin_png=String(get(output, "langevin_figure_png", "outputs/score_sigma005_langevin_validation.png")),
        device_name=String(get(run, "device", "GPU:1")),
        required_gpu_name=String(get(run, "required_gpu_name", "5070")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
        langevin_validate=Bool(get(run, "langevin_validate", false)),
        langevin_dt=Float64(get(get(raw, "langevin", Dict{String, Any}()), "dt", 0.002)),
        langevin_total_time=Float64(get(get(raw, "langevin", Dict{String, Any}()), "total_time", 48.0)),
        langevin_burnin_time=Float64(get(get(raw, "langevin", Dict{String, Any}()), "burnin_time", 8.0)),
        langevin_save_dt=Float64(get(get(raw, "langevin", Dict{String, Any}()), "save_dt", 0.04)),
        langevin_ntraj=Int(get(get(raw, "langevin", Dict{String, Any}()), "ntrajectories", 192)),
        langevin_score_clip=Float32(get(get(raw, "langevin", Dict{String, Any}()), "score_clip", 80.0)),
        langevin_max_pdf_samples=Int(get(get(raw, "langevin", Dict{String, Any}()), "max_pdf_samples", 250000)),
    )
    require_condition(params.sigma == 0.05f0, "This experiment is configured to train DSM only at sigma=0.05.")
    require_condition(params.model_config.periodic, "FHDChain score U-Net must be periodic.")
    require_condition(params.model_normalization != :batchnorm, "New FHDChain score models must not use BatchNorm.")
    return params
end

function load_score_model_for_validation(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    stats = load_stats_from_bson(blob)
    trainer_cfg = dict_get(blob, :trainer_cfg)
    history = dict_get(blob, :history)
    diagnostics = haskey(blob, :diagnostics) ? blob[:diagnostics] : Dict{Symbol, Any}()
    phys = haskey(blob, :phys) ? blob[:phys] : nothing
    return model, stats, Float32(trainer_cfg.sigma), history, diagnostics, phys
end

function trainer_config(params::FHDScoreParams)
    return ScoreTrainerConfig(
        batch_size=params.batch_size,
        epochs=params.epochs,
        lr=params.learning_rate,
        sigma=params.sigma,
        shuffle=true,
        seed=params.seed,
        progress=params.progress,
        max_steps_per_epoch=nothing,
        accumulation_steps=1,
        use_lr_schedule=params.use_lr_schedule,
        warmup_steps=params.warmup_steps,
        min_lr_factor=params.min_lr_factor,
        epoch_subset_size=params.epoch_subset_size,
    )
end

function train_score_model(dataset::NormalizedDataset, params::FHDScoreParams,
        device::ExecutionDevice)
    Random.seed!(params.model_init_seed)
    model, model_cfg = build_score_unet(params.model_config, params.model_normalization)
    Random.seed!()
    model = move_model(model, device)
    Flux.trainmode!(model)

    cfg = trainer_config(params)
    opt_state = Flux.setup(Flux.Optimisers.Adam(cfg.lr), model)
    steps_per_epoch = cfg.epoch_subset_size > 0 ? ceil(Int, min(cfg.epoch_subset_size, length(dataset)) / cfg.batch_size) :
        ceil(Int, length(dataset) / cfg.batch_size)
    lr_scheduler = cfg.use_lr_schedule ? create_lr_schedule(cfg, steps_per_epoch) : nothing
    rng = MersenneTwister(cfg.seed)
    noise_rng = MersenneTwister(cfg.seed + 1)
    K, C, _ = size(dataset.data)
    history = Dict(:train_loss => Float64[], :score_norm => Float64[],
        :param_norm => Float64[], :epoch_time => Float64[])
    progress = cfg.progress ? Progress(cfg.epochs; desc="Training FHD stationary score") : nothing
    global_step = 0

    for epoch in 1:cfg.epochs
        t0 = time_ns()
        n = length(dataset)
        idxs = if cfg.epoch_subset_size > 0 && cfg.epoch_subset_size < n
            randperm(rng, n)[1:cfg.epoch_subset_size]
        else
            randperm(rng, n)
        end
        losses = Float64[]
        for part in Iterators.partition(idxs, cfg.batch_size)
            global_step += 1
            lr_scheduler !== nothing && Flux.adjust!(opt_state, lr_scheduler(global_step))
            batch_cpu = copy(dataset.data[:, :, collect(part)])
            B = size(batch_cpu, 3)
            noise_cpu = Array{Float32}(undef, K, C, B)
            fill_projected_noise!(noise_cpu, noise_rng)
            batch = move_array(batch_cpu, device)
            noise = move_array(noise_cpu, device)
            noisy = batch .+ cfg.sigma .* noise
            project_zero_modes!(noisy)
            loss_value, grads = Flux.withgradient(model) do current_model
                pred = current_model(noisy)
                Flux.Losses.mse(pred, noise)
            end
            opt_state, model = Flux.update!(opt_state, model, grads[1])
            push!(losses, Float64(to_host(loss_value)))
        end
        monitor_n = min(4096, length(dataset))
        monitor = move_array(copy(dataset.data[:, :, 1:monitor_n]), device)
        score = score_from_projected_model(model, monitor, cfg.sigma)
        score_norm = mean(sqrt.(sum(abs2, reshape(to_host(score), K * C, monitor_n); dims=1)))
        push!(history[:train_loss], mean(losses))
        push!(history[:score_norm], Float64(score_norm))
        push!(history[:param_norm], parameter_norm(model))
        push!(history[:epoch_time], (time_ns() - t0) / 1e9)
        progress !== nothing && ProgressMeter.next!(progress; showvalues=[
            (:epoch, epoch),
            (:loss, history[:train_loss][end]),
            (:score_norm, history[:score_norm][end]),
        ])
    end
    progress !== nothing && ProgressMeter.finish!(progress)
    return model, model_cfg, history
end

function score_diagnostics(model, dataset::NormalizedDataset, phys::FHDPhysicalParams,
        params::FHDScoreParams, device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 200)
    nstein = min(params.stein_samples, length(dataset))
    keep = randperm(rng, length(dataset))[1:nstein]
    clean = copy(dataset.data[:, :, keep])
    noise = randn(rng, Float32, size(clean))
    project_zero_modes!(noise)
    noisy = clean .+ params.sigma .* noise
    project_zero_modes!(noisy)
    scores = evaluate_stationary_score_norm(model, noisy, params.sigma, device; batch_size=params.batch_size)
    stein = stein_matrix_from_scores(scores, noisy)
    P = projection_matrix(size(clean, 1))
    stein_rel = norm(stein - P) / norm(P)
    exact = exact_score_metrics(model, dataset, phys, params.sigma, params.exact_score_samples,
        MersenneTwister(params.seed + 300), device; safe_density_min=params.safe_density_min,
        batch_size=params.batch_size)
    exact[:stein_relative_error] = stein_rel
    return exact, stein
end

function diagnostic_scatter_data(model, dataset::NormalizedDataset, phys::FHDPhysicalParams,
        params::FHDScoreParams, device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 400)
    n = min(12_000, length(dataset))
    keep = randperm(rng, length(dataset))[1:n]
    clean = copy(dataset.data[:, :, keep])
    pred = evaluate_stationary_score_norm(model, clean, params.sigma, device; batch_size=params.batch_size)
    exact = standardized_physical_score(clean, dataset.stats, phys; velocity_floor=phys.velocity_density_floor)
    return vec(Float64.(exact)), vec(Float64.(pred))
end

function create_score_figure(path::AbstractString, params::FHDScoreParams,
        dataset::NormalizedDataset, history, diagnostics, stein, scatter_exact, scatter_pred)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("DSM sigma=%.3f, train samples=%d, exact safe rel.RMSE=%.3e, safe cosine=%.5f",
            params.sigma, length(dataset), diagnostics[:sim_safe_rel_rmse], diagnostics[:sim_safe_cosine])
        figure_title!(fig, "FHDChain stationary score diagnostics"; subtitle=subtitle)

        epochs = collect(1:length(history[:train_loss]))
        ax1 = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=curve_linewidth())
        ax2 = Axis(fig[1, 2]; title="Mean score norm", xlabel="epoch", ylabel="norm")
        lines!(ax2, epochs, history[:score_norm]; color=STYLE_ACCENT, linewidth=curve_linewidth())
        ax3 = Axis(fig[1, 3]; title="Learned vs analytic score", xlabel="analytic score", ylabel="learned score")
        step = max(1, length(scatter_exact) ÷ 20_000)
        scatter!(ax3, scatter_exact[1:step:end], scatter_pred[1:step:end];
            markersize=3, color=(STYLE_PRIMARY, 0.22))
        lim = maximum(abs, vcat(scatter_exact[1:step:end], scatter_pred[1:step:end]))
        lines!(ax3, [-lim, lim], [-lim, lim]; color=STYLE_REFERENCE, linestyle=:dash)

        P = projection_matrix(size(dataset.data, 1))
        residual = stein - P
        clim = max(maximum(abs, residual), 1e-6)
        ax4 = Axis(fig[2, 1]; title="Projected Stein residual", xlabel="column", ylabel="row")
        hm = heatmap!(ax4, residual; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
        Colorbar(fig[2, 1, Right()], hm; label="Stein - P")

        ax5 = Axis(fig[2, 2]; title="Score error histogram", xlabel="learned - analytic", ylabel="density")
        hist!(ax5, scatter_pred .- scatter_exact; bins=120, normalization=:pdf, color=(STYLE_SECONDARY, 0.65))

        lines = String[
            @sprintf("normalization = channel-shared, no BatchNorm"),
            @sprintf("zero-mode-preserving DSM noise"),
            @sprintf("final DSM loss = %.6e", history[:train_loss][end]),
            @sprintf("Stein rel. error = %.6e", diagnostics[:stein_relative_error]),
            @sprintf("ideal all rel.RMSE / cosine = %.3e / %.5f", diagnostics[:ideal_rel_rmse], diagnostics[:ideal_cosine]),
            @sprintf("sim all rel.RMSE / cosine = %.3e / %.5f", diagnostics[:sim_rel_rmse], diagnostics[:sim_cosine]),
            @sprintf("ideal safe rel.RMSE / cosine = %.3e / %.5f", diagnostics[:ideal_safe_rel_rmse], diagnostics[:ideal_safe_cosine]),
            @sprintf("sim safe rel.RMSE / cosine = %.3e / %.5f", diagnostics[:sim_safe_rel_rmse], diagnostics[:sim_safe_cosine]),
            @sprintf("safe samples = %d / %d", diagnostics[:safe_count], diagnostics[:total_count]),
        ]
        text_panel!(fig[2, 3], lines; title="Summary")
        save_figure_checked(path, fig)
    end
    return nothing
end

function initialize_langevin_norm_ensemble(dataset::NormalizedDataset, ntraj::Int, rng::AbstractRNG)
    K, C, N = size(dataset.data)
    z = Array{Float32}(undef, K, C, ntraj)
    @inbounds for b in 1:ntraj
        idx = rand(rng, 1:N)
        z[:, :, b] .= dataset.data[:, :, idx]
    end
    project_zero_modes!(z)
    return z
end

function integrate_score_langevin(model, dataset::NormalizedDataset, params::FHDScoreParams,
        device::ExecutionDevice)
    require_condition(params.langevin_dt > 0.0, "langevin.dt must be positive.")
    require_condition(params.langevin_total_time > params.langevin_burnin_time,
        "langevin.total_time must exceed langevin.burnin_time.")
    require_condition(params.langevin_save_dt >= params.langevin_dt,
        "langevin.save_dt must be at least langevin.dt.")
    nsteps = ceil(Int, params.langevin_total_time / params.langevin_dt)
    burnin_steps = floor(Int, params.langevin_burnin_time / params.langevin_dt)
    save_every = max(1, round(Int, params.langevin_save_dt / params.langevin_dt))
    actual_save_dt = save_every * params.langevin_dt
    nsaved = max(1, fld(max(nsteps - burnin_steps, 0), save_every) + 1)
    rng = MersenneTwister(params.seed + 880_000)
    noise_rng = MersenneTwister(params.seed + 880_001)
    z = initialize_langevin_norm_ensemble(dataset, params.langevin_ntraj, rng)
    K, C, B = size(z)
    zdev = move_array(z, device)
    noise = similar(zdev)
    saved_norm = Array{Float32}(undef, nsaved, K, C, B)
    saved_times = Vector{Float64}(undef, nsaved)
    sqrtdt = Float32(sqrt(2.0 * params.langevin_dt))
    dt32 = Float32(params.langevin_dt)
    save_idx = 0
    progress = Progress(100; desc="Integrating score-only Langevin")
    progress_stride = max(1, nsteps ÷ 100)
    for step in 0:nsteps
        if step >= burnin_steps && (step - burnin_steps) % save_every == 0
            save_idx += 1
            saved_times[save_idx] = (step - burnin_steps) * params.langevin_dt
            zhost = to_host(zdev)
            @inbounds for b in 1:B, c in 1:C, i in 1:K
                saved_norm[save_idx, i, c, b] = zhost[i, c, b]
            end
        end
        step == nsteps && break
        score = score_from_projected_model(model, zdev, params.sigma)
        clamp!(score, -params.langevin_score_clip, params.langevin_score_clip)
        fill_projected_noise!(noise, noise_rng)
        @. zdev = zdev + dt32 * score + sqrtdt * noise
        project_zero_modes!(zdev)
        step % progress_stride == 0 && ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    raw = Array{Float32}(undef, nsaved, K, C, B)
    for t in 1:nsaved
        raw[t, :, :, :] .= denormalize_fhd_tensor(saved_norm[t, :, :, :], dataset.stats)
    end
    return saved_times, raw, actual_save_dt
end

function draw_channel_values(states::Array{Float32, 4}, start_idx::Int, channel::Int,
        max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % npost)
        tmp = linear ÷ npost
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        vals[s] = Float64(states[t, i, channel, tr])
    end
    return vals
end

function draw_velocity_values(states::Array{Float32, 4}, start_idx::Int, phys::FHDPhysicalParams,
        max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % npost)
        tmp = linear ÷ npost
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        vals[s] = Float64(states[t, i, 2, tr]) /
            max(Float64(states[t, i, 1, tr]), phys.velocity_density_floor)
    end
    return vals
end

function draw_eta_values(states::Array{Float32, 4}, start_idx::Int, phys::FHDPhysicalParams,
        max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % npost)
        tmp = linear ÷ npost
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        ip = periodic(i + 1, K)
        rho_edge = max(0.5 * (Float64(states[t, i, 1, tr]) +
            Float64(states[t, ip, 1, tr])), 1.0e-14)
        vals[s] = phys.eta0 * (rho_edge / phys.rho0)^phys.zeta
    end
    return vals
end

function histogram_density_local(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Float64, length(edges) - 1)
    lo = first(edges)
    hi = last(edges)
    width = (hi - lo) / length(counts)
    @inbounds for v in values
        if lo <= v <= hi
            idx = clamp(floor(Int, (v - lo) / width) + 1, 1, length(counts))
            counts[idx] += 1.0
        end
    end
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    density = counts ./ max(sum(counts) * width, eps(Float64))
    return centers, density
end

function pdf_rel_l2(obs::Vector{Float64}, model::Vector{Float64})
    return sqrt(sum(abs2, model .- obs) / max(sum(abs2, obs), eps(Float64)))
end

function sampled_covariance(states::Array{Float32, 4}, start_idx::Int, max_samples::Int,
        rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * ntraj
    nsamp = min(max_samples, total)
    D = 2K
    X = Matrix{Float64}(undef, D, nsamp)
    @inbounds for s in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % npost)
        tr = (linear ÷ npost) + 1
        for i in 1:K
            X[i, s] = Float64(states[t, i, 1, tr])
            X[K + i, s] = Float64(states[t, i, 2, tr])
        end
    end
    mu = mean(X; dims=2)
    X .-= mu
    return (X * transpose(X)) ./ max(nsamp - 1, 1), vec(mu)
end

function spatial_power_spectrum_local(states::Array{Float32, 4}, start_idx::Int, channel::Int)
    nt, K, _, ntraj = size(states)
    npost = nt - start_idx + 1
    powers = zeros(Float64, K ÷ 2 + 1)
    count = 0
    @inbounds for tr in 1:ntraj, t in start_idx:nt
        vals = Float64.(states[t, :, channel, tr])
        vals .-= mean(vals)
        for k in 0:(K ÷ 2)
            re = 0.0
            im = 0.0
            for j in 1:K
                angle = -2.0 * pi * k * (j - 1) / K
                re += vals[j] * cos(angle)
                im += vals[j] * sin(angle)
            end
            powers[k + 1] += (re * re + im * im) / K
        end
        count += 1
    end
    return collect(0:(K ÷ 2)), powers ./ max(count, 1)
end

function zero_mode_drift(states::Array{Float32, 4})
    nt, _, _, ntraj = size(states)
    mass0 = [sum(Float64, @view states[1, :, 1, tr]) for tr in 1:ntraj]
    mom0 = [sum(Float64, @view states[1, :, 2, tr]) for tr in 1:ntraj]
    max_mass = 0.0
    max_mom = 0.0
    @inbounds for tr in 1:ntraj, t in 1:nt
        max_mass = max(max_mass, abs(sum(Float64, @view states[t, :, 1, tr]) - mass0[tr]))
        max_mom = max(max_mom, abs(sum(Float64, @view states[t, :, 2, tr]) - mom0[tr]))
    end
    return max_mass, max_mom
end

function save_score_langevin(path::AbstractString, times::Vector{Float64},
        states::Array{Float32, 4}, actual_save_dt::Float64, metrics::Dict)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time", times)
        write(h5, "/states", states)
        write(h5, "/metadata/save_dt", actual_save_dt)
        for (key, val) in metrics
            val isa Number && write(h5, "/metrics/$(String(key))", val)
        end
    end
    @printf("Saved score-Langevin trajectory to %s\n", path)
    return nothing
end

function render_score_langevin_figure(path::AbstractString, obs_states::Array{Float32, 4},
        obs_start::Int, lang_states::Array{Float32, 4}, phys::FHDPhysicalParams,
        metrics::Dict, params::FHDScoreParams)
    rng = MersenneTwister(params.seed + 881_000)
    maxs = params.langevin_max_pdf_samples
    rho_obs = draw_channel_values(obs_states, obs_start, 1, maxs, rng)
    m_obs = draw_channel_values(obs_states, obs_start, 2, maxs, rng)
    u_obs = draw_velocity_values(obs_states, obs_start, phys, maxs, rng)
    eta_obs = draw_eta_values(obs_states, obs_start, phys, maxs, rng)
    rho_l = draw_channel_values(lang_states, 1, 1, maxs, rng)
    m_l = draw_channel_values(lang_states, 1, 2, maxs, rng)
    u_l = draw_velocity_values(lang_states, 1, phys, maxs, rng)
    eta_l = draw_eta_values(lang_states, 1, phys, maxs, rng)
    value_sets = [(rho_obs, rho_l, "rho PDF", "rho"),
        (m_obs, m_l, "m PDF", "m"),
        (u_obs, u_l, "velocity PDF", "u"),
        (eta_obs, eta_l, "edge viscosity PDF", "eta")]
    spectra_rho_obs = spatial_power_spectrum_local(obs_states, obs_start, 1)
    spectra_rho_l = spatial_power_spectrum_local(lang_states, 1, 1)
    spectra_m_obs = spatial_power_spectrum_local(obs_states, obs_start, 2)
    spectra_m_l = spatial_power_spectrum_local(lang_states, 1, 2)
    cov_obs, mean_obs = sampled_covariance(obs_states, obs_start, 120_000, rng)
    cov_l, mean_l = sampled_covariance(lang_states, 1, 120_000, rng)
    cov_err = cov_l .- cov_obs
    clim = max(maximum(abs, cov_err), 1.0e-8)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("PDF mean rel.L2=%.3e, covariance rel.RMSE=%.3e, zero-mode drift=(%.1e, %.1e)",
            metrics[:mean_pdf_rel_l2], metrics[:covariance_rel_rmse],
            metrics[:max_mass_drift], metrics[:max_momentum_drift])
        figure_title!(fig, "FHDChain score-only Langevin validation"; subtitle=subtitle)
        for (idx, (obs, lang, ttl, xl)) in enumerate(value_sets)
            lo = quantile(obs, 0.001)
            hi = quantile(obs, 0.999)
            pad = 0.08 * max(hi - lo, 1.0e-8)
            edges = collect(range(lo - pad, hi + pad; length=141))
            centers, pobs = histogram_density_local(obs, edges)
            _, pl = histogram_density_local(lang, edges)
            ax = Axis(fig[1, idx]; title=ttl, xlabel=xl, ylabel="density")
            lines!(ax, centers, pobs; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
            lines!(ax, centers, pl; color=STYLE_PRIMARY, linestyle=:dash,
                linewidth=curve_linewidth(), label="score Langevin")
            idx == 1 && axislegend(ax; position=:rt)
        end
        ax5 = Axis(fig[2, 1]; title="rho spatial spectrum", xlabel="mode", ylabel="power")
        lines!(ax5, spectra_rho_obs[1], spectra_rho_obs[2]; color=STYLE_REFERENCE, linewidth=curve_linewidth())
        lines!(ax5, spectra_rho_l[1], spectra_rho_l[2]; color=STYLE_PRIMARY, linestyle=:dash,
            linewidth=curve_linewidth())
        ax6 = Axis(fig[2, 2]; title="m spatial spectrum", xlabel="mode", ylabel="power")
        lines!(ax6, spectra_m_obs[1], spectra_m_obs[2]; color=STYLE_REFERENCE, linewidth=curve_linewidth())
        lines!(ax6, spectra_m_l[1], spectra_m_l[2]; color=STYLE_PRIMARY, linestyle=:dash,
            linewidth=curve_linewidth())
        ax7 = Axis(fig[2, 3]; title="covariance error", xlabel="column", ylabel="row")
        hm = heatmap!(ax7, cov_err; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
        Colorbar(fig[2, 3, Right()], hm; label="model - data")
        ax8 = Axis(fig[2, 4]; title="mean state", xlabel="component", ylabel="mean")
        xs = 1:length(mean_obs)
        scatter!(ax8, xs, mean_obs; color=STYLE_REFERENCE, markersize=8, label="data")
        scatter!(ax8, xs, mean_l; color=STYLE_PRIMARY, markersize=8, label="score Langevin")
        axislegend(ax8; position=:rt)
        lines = String[
            @sprintf("dt = %.5g, save_dt = %.5g", params.langevin_dt, params.langevin_save_dt),
            @sprintf("ntraj = %d, saved snapshots = %d", params.langevin_ntraj, size(lang_states, 1)),
            @sprintf("rho PDF rel.L2 = %.6e", metrics[:rho_pdf_rel_l2]),
            @sprintf("m PDF rel.L2 = %.6e", metrics[:m_pdf_rel_l2]),
            @sprintf("u PDF rel.L2 = %.6e", metrics[:u_pdf_rel_l2]),
            @sprintf("eta PDF rel.L2 = %.6e", metrics[:eta_pdf_rel_l2]),
            @sprintf("covariance rel.RMSE = %.6e", metrics[:covariance_rel_rmse]),
            "Analytic score is not used in this validation.",
        ]
        text_panel!(fig[3, 1:4], lines; title="Diagnostics")
        save_figure_checked(path, fig)
    end
    return nothing
end

function score_langevin_metrics(obs_states::Array{Float32, 4}, obs_start::Int,
        lang_states::Array{Float32, 4}, phys::FHDPhysicalParams, params::FHDScoreParams)
    rng = MersenneTwister(params.seed + 882_000)
    maxs = params.langevin_max_pdf_samples
    obs_draws = (
        draw_channel_values(obs_states, obs_start, 1, maxs, rng),
        draw_channel_values(obs_states, obs_start, 2, maxs, rng),
        draw_velocity_values(obs_states, obs_start, phys, maxs, rng),
        draw_eta_values(obs_states, obs_start, phys, maxs, rng),
    )
    lang_draws = (
        draw_channel_values(lang_states, 1, 1, maxs, rng),
        draw_channel_values(lang_states, 1, 2, maxs, rng),
        draw_velocity_values(lang_states, 1, phys, maxs, rng),
        draw_eta_values(lang_states, 1, phys, maxs, rng),
    )
    rels = Float64[]
    for k in 1:4
        lo = quantile(obs_draws[k], 0.001)
        hi = quantile(obs_draws[k], 0.999)
        pad = 0.08 * max(hi - lo, 1.0e-8)
        edges = collect(range(lo - pad, hi + pad; length=141))
        _, po = histogram_density_local(obs_draws[k], edges)
        _, pl = histogram_density_local(lang_draws[k], edges)
        push!(rels, pdf_rel_l2(po, pl))
    end
    cov_obs, _ = sampled_covariance(obs_states, obs_start, 120_000, rng)
    cov_l, _ = sampled_covariance(lang_states, 1, 120_000, rng)
    mass_drift, mom_drift = zero_mode_drift(lang_states)
    return Dict{Symbol, Float64}(
        :rho_pdf_rel_l2 => rels[1],
        :m_pdf_rel_l2 => rels[2],
        :u_pdf_rel_l2 => rels[3],
        :eta_pdf_rel_l2 => rels[4],
        :mean_pdf_rel_l2 => mean(rels),
        :covariance_rel_rmse => sqrt(sum(abs2, cov_l .- cov_obs) / max(sum(abs2, cov_obs), eps(Float64))),
        :max_mass_drift => mass_drift,
        :max_momentum_drift => mom_drift,
    )
end

function run_score_langevin_validation(model, dataset::NormalizedDataset,
        input_hdf5::AbstractString, params::FHDScoreParams, phys::FHDPhysicalParams,
        device::ExecutionDevice, base_dir::AbstractString)
    times, lang_states, actual_save_dt = integrate_score_langevin(model, dataset, params, device)
    _, obs_states = load_fhd_states(input_hdf5)
    obs_start = burnin_start_index(size(obs_states, 1), params.burnin_fraction)
    metrics = score_langevin_metrics(obs_states, obs_start, lang_states, phys, params)
    h5_path = resolve_path(base_dir, params.output_langevin_hdf5)
    png_path = resolve_path(base_dir, params.output_langevin_png)
    save_score_langevin(h5_path, times, lang_states, actual_save_dt, metrics)
    render_score_langevin_figure(png_path, obs_states, obs_start, lang_states, phys, metrics, params)
    @printf("Score-Langevin validation: mean PDF rel.L2=%.6e, covariance rel.RMSE=%.6e\n",
        metrics[:mean_pdf_rel_l2], metrics[:covariance_rel_rmse])
    return metrics
end

function save_score_model(path::AbstractString, model, model_cfg, dataset::NormalizedDataset,
        params::FHDScoreParams, phys::FHDPhysicalParams, history, diagnostics, stein)
    host_model = cpu(model)
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    trainer_cfg = trainer_config(params)
    metadata = Dict(
        :system => "fhd_chain",
        :score_type => "stationary_dsm_zero_mode_projected",
        :state_layout => "K x 2 x batch, channels=(rho,m)",
        :sigma => params.sigma,
        :normalization => String(params.model_normalization),
        :analytic_score_used_for_training => false,
        :required_gpu_name => params.required_gpu_name,
        :stein_relative_error => get(diagnostics, :stein_relative_error, NaN),
        :sim_safe_rel_rmse => get(diagnostics, :sim_safe_rel_rmse, NaN),
        :sim_safe_cosine => get(diagnostics, :sim_safe_cosine, NaN),
    )
    BSON.@save path host_model model_cfg trainer_cfg stats metadata history diagnostics stein phys
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    output_bson = resolve_path(base_dir, params.output_bson)
    output_png = resolve_path(base_dir, params.output_png)
    ensure_parent_dir(output_bson)
    ensure_parent_dir(output_png)
    require_condition(isfile(input_hdf5), "Input HDF5 not found: $(input_hdf5)")

    device = detect_fhd_device(params.device_name, params.required_gpu_name)
    activate_and_describe_device!(device, params.device_name, params.required_gpu_name)

    rng = MersenneTwister(params.seed)
    phys = load_fhd_physical_params(input_hdf5)

    if params.train
        dataset, _, _, _ = load_fhd_dataset(input_hdf5, params.burnin_fraction, params.max_samples, rng)
        @printf("Loaded FHD score dataset: samples=%d, K=%d, channels=%d\n",
            length(dataset), size(dataset.data, 1), size(dataset.data, 2))
        model, model_cfg, history = train_score_model(dataset, params, device)
        checkpoint = string(output_bson, ".checkpoint")
        basic_diag = Dict{Symbol, Any}(:checkpoint => true)
        save_score_model(checkpoint, model, model_cfg, dataset, params, phys, history, basic_diag,
            zeros(Float64, 2 * size(dataset.data, 1), 2 * size(dataset.data, 1)))
        if params.evaluate
            diagnostics, stein = score_diagnostics(model, dataset, phys, params, device)
            scatter_exact, scatter_pred = diagnostic_scatter_data(model, dataset, phys, params, device)
            save_score_model(output_bson, model, model_cfg, dataset, params, phys, history, diagnostics, stein)
            rm(checkpoint; force=true)
            create_score_figure(output_png, params, dataset, history, diagnostics, stein, scatter_exact, scatter_pred)
            @printf("Done. DSM loss=%.6e, Stein rel=%.6e, sim safe score rel=%.6e, cosine=%.6f\n",
                history[:train_loss][end], diagnostics[:stein_relative_error],
                diagnostics[:sim_safe_rel_rmse], diagnostics[:sim_safe_cosine])
        else
            save_score_model(output_bson, model, model_cfg, dataset, params, phys, history, basic_diag,
                zeros(Float64, 2 * size(dataset.data, 1), 2 * size(dataset.data, 1)))
            rm(checkpoint; force=true)
            @printf("Done. DSM loss=%.6e\n", history[:train_loss][end])
        end
    else
        require_condition(isfile(output_bson), "run.train=false requires an existing score checkpoint: $(output_bson)")
        model, stats, checkpoint_sigma, history, diagnostics, _ = load_score_model_for_validation(output_bson, device)
        require_condition(abs(checkpoint_sigma - params.sigma) < 1.0f-7,
            "Checkpoint sigma $(checkpoint_sigma) does not match configured sigma $(params.sigma).")
        times, states = load_fhd_states(input_hdf5)
        start_idx = burnin_start_index(length(times), params.burnin_fraction)
        raw = sample_state_tensor(states, start_idx, params.max_samples, rng)
        dataset = NormalizedDataset(apply_fhd_stats(raw, stats), stats)
        @printf("Loaded existing score checkpoint and validation dataset: samples=%d, K=%d, channels=%d\n",
            length(dataset), size(dataset.data, 1), size(dataset.data, 2))
        if params.evaluate
            diagnostics, stein = score_diagnostics(model, dataset, phys, params, device)
            scatter_exact, scatter_pred = diagnostic_scatter_data(model, dataset, phys, params, device)
            create_score_figure(output_png, params, dataset, history, diagnostics, stein, scatter_exact, scatter_pred)
            @printf("Checkpoint diagnostics. Stein rel=%.6e, sim safe score rel=%.6e, cosine=%.6f\n",
                diagnostics[:stein_relative_error], diagnostics[:sim_safe_rel_rmse],
                diagnostics[:sim_safe_cosine])
        end
    end
    if params.langevin_validate
        run_score_langevin_validation(model, dataset, input_hdf5, params, phys, device, base_dir)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
