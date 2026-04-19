#!/usr/bin/env julia

# Agent summary:
# - Trains the stationary score network s(x) from the simulated observations.
# - Saves the trained score model and its diagnostic figure for reuse by the mobility-fitting scripts.

import Pkg

function ensure_packages(packages::Vector{String})
    missing = String[]
    for pkg in packages
        if Base.find_package(pkg) === nothing
            push!(missing, pkg)
        end
    end
    if !isempty(missing)
        @info "Installing missing Julia packages" missing
        Pkg.add(missing)
    end
    return nothing
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ensure_packages(["Flux", "BSON", "HDF5", "GLMakie", "CUDA", "ProgressMeter", "KernelDensity"])

using BSON
using CUDA
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics
using TOML

include(joinpath(@__DIR__, "src", "figure_style.jl"))

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "score.toml")

struct ScoreTrainingParams
    input_hdf5::String
    burnin_fraction::Float64
    data_every::Int
    max_samples::Int
    sigma::Float64
    widths::Vector{Int}
    epochs::Int
    batch_size::Int
    learning_rate::Float64
    seed::Int
    sample_dt::Float64
    sample_steps::Int
    sample_burnin_steps::Int
    sample_save_stride::Int
    sample_chains::Int
    pdf_bins::Int
    output_bson::String
    output_png::String
end

struct DeviceConfig
    use_gpu::Bool
    name::String
end

struct PDFDiagnostics
    kl_x::Float64
    kl_y::Float64
    kl_xy::Float64
    mean_kl::Float64
    accuracy::Float64
    data_points::Int
    rom_points::Int
    bins_1d::Int
    bins_2d::Tuple{Int, Int}
end

struct ObservedPDFReference
    samples::Matrix{Float32}
    xcenters::Vector{Float64}
    xdensity::Vector{Float64}
    ycenters::Vector{Float64}
    ydensity::Vector{Float64}
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    density2d::Matrix{Float64}
    x_range::Tuple{Float64, Float64}
    y_range::Tuple{Float64, Float64}
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data_cfg = raw["data"]
    training_cfg = raw["training"]
    sampling_cfg = raw["sampling"]
    figure_cfg = raw["figure"]
    output_cfg = raw["output"]

    params = ScoreTrainingParams(
        String(data_cfg["input_hdf5"]),
        Float64(data_cfg["burnin_fraction"]),
        Int(data_cfg["data_every"]),
        Int(data_cfg["max_samples"]),
        Float64(training_cfg["sigma"]),
        Int.(training_cfg["widths"]),
        Int(training_cfg["epochs"]),
        Int(training_cfg["batch_size"]),
        Float64(training_cfg["learning_rate"]),
        Int(training_cfg["seed"]),
        Float64(sampling_cfg["dt"]),
        Int(sampling_cfg["steps"]),
        Int(sampling_cfg["burnin_steps"]),
        Int(sampling_cfg["save_stride"]),
        Int(sampling_cfg["chains"]),
        Int(figure_cfg["pdf_bins"]),
        String(output_cfg["model_bson"]),
        String(output_cfg["figure_png"]),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.data_every >= 1, "data_every must be >= 1.")
    require_condition(params.max_samples >= 10_000, "max_samples should be at least 10000.")
    require_condition(isapprox(params.sigma, 0.05; atol=1e-12), "This script is configured for sigma = 0.05.")
    require_condition(!isempty(params.widths), "widths must contain at least one hidden layer width.")
    require_condition(all(width -> width >= 8, params.widths), "Each entry in widths must be at least 8.")
    require_condition(params.epochs >= 1, "epochs must be >= 1.")
    require_condition(params.batch_size >= 16, "batch_size must be >= 16.")
    require_condition(params.learning_rate > 0.0, "learning_rate must be positive.")
    require_condition(params.sample_dt > 0.0, "sample_dt must be positive.")
    require_condition(params.sample_steps > params.sample_burnin_steps, "sample_steps must exceed sample_burnin_steps.")
    require_condition(params.sample_save_stride >= 1, "sample_save_stride must be >= 1.")
    require_condition(params.sample_chains >= 1, "sample_chains must be >= 1.")
    require_condition(params.pdf_bins >= 20, "pdf_bins must be >= 20.")
    return params
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? path : normpath(joinpath(base_dir, path))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function detect_device()
    if CUDA.functional()
        CUDA.allowscalar(false)
        device_name = try
            unsafe_string(CUDA.name(CUDA.device()))
        catch
            "CUDA GPU"
        end
        return DeviceConfig(true, device_name)
    end
    return DeviceConfig(false, "CPU")
end

to_device(x, device::DeviceConfig) = device.use_gpu ? cu(x) : x
to_host(x) = x isa CUDA.AbstractGPUArray ? Array(x) : x

function build_model(widths::Vector{Int})
    layers = Any[]
    in_dim = 2
    for width in widths
        push!(layers, Dense(in_dim, width, tanh))
        in_dim = width
    end
    push!(layers, Dense(in_dim, 2))
    return Chain(layers...)
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function load_state_tensor(path::AbstractString)
    times = h5read(path, "/trajectories/time")
    states = h5read(path, "/trajectories/states")
    require_condition(ndims(states) == 3, "Expected /trajectories/states to be a rank-3 tensor.")
    require_condition(size(states, 2) == 2, "Expected the state dimension to have size 2.")
    require_condition(size(states, 1) == length(times),
        "The saved trajectory tensor does not match the saved time axis. This indicates a legacy sim.h5 written by the old low-level HDF5 path in sim.jl. Regenerate the simulation output with the fixed sim.jl before running score.jl.")
    return states
end

function load_samples(path::AbstractString, burnin_fraction::Float64, data_every::Int, max_samples::Int, rng::AbstractRNG)
    states = load_state_tensor(path)
    nt, ndim, ntraj = size(states)
    require_condition(ndim == 2, "Expected a 2D state in /trajectories/states.")
    start_idx = burnin_start_index(nt, burnin_fraction)
    total_available = (nt - start_idx + 1) * ntraj
    nsamples = cld(total_available, data_every)
    raw = Matrix{Float32}(undef, 2, nsamples)

    cursor = 1
    flat_idx = 1
    @inbounds for traj_idx in 1:ntraj
        for time_idx in start_idx:nt
            if (flat_idx - 1) % data_every == 0
                raw[1, cursor] = Float32(states[time_idx, 1, traj_idx])
                raw[2, cursor] = Float32(states[time_idx, 2, traj_idx])
                cursor += 1
            end
            flat_idx += 1
        end
    end
    raw = raw[:, 1:(cursor - 1)]

    if size(raw, 2) > max_samples
        keep = randperm(rng, size(raw, 2))[1:max_samples]
        raw = raw[:, keep]
    end
    return raw
end

function histogram_range_from_centers(centers::Vector{Float64})
    if length(centers) <= 1
        halfwidth = 0.5
    else
        halfwidth = 0.5 * (centers[2] - centers[1])
    end
    return (centers[1] - halfwidth, centers[end] + halfwidth)
end

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function load_observed_pdf_reference(path::AbstractString, burnin_fraction::Float64)
    states = load_state_tensor(path)
    xcenters = h5read(path, "/statistics/pdf/x_centers")
    xdensity = h5read(path, "/statistics/pdf/x_density")
    ycenters = h5read(path, "/statistics/pdf/y_centers")
    ydensity = h5read(path, "/statistics/pdf/y_density")
    xgrid = h5read(path, "/statistics/pdf/xy_x_grid")
    ygrid = h5read(path, "/statistics/pdf/xy_y_grid")
    density2d = h5read(path, "/statistics/pdf/xy_density")

    nt, ndim, ntraj = size(states)
    require_condition(ndim == 2, "Expected a 2D state in /trajectories/states.")
    start_idx = burnin_start_index(nt, burnin_fraction)
    nsamples = (nt - start_idx + 1) * ntraj
    samples = Matrix{Float32}(undef, 2, nsamples)

    cursor = 1
    @inbounds for traj_idx in 1:ntraj
        for time_idx in start_idx:nt
            samples[1, cursor] = Float32(states[time_idx, 1, traj_idx])
            samples[2, cursor] = Float32(states[time_idx, 2, traj_idx])
            cursor += 1
        end
    end

    x_range = histogram_range_from_centers(xcenters)
    y_range = histogram_range_from_centers(ycenters)

    return ObservedPDFReference(samples, xcenters, xdensity, ycenters, ydensity, xgrid, ygrid, density2d, x_range, y_range)
end

function dsm_loss_with_noise(model, batch::Matrix{Float32}, sigma::Float32, noise::Matrix{Float32})
    noisy = batch .+ noise
    target = -noise ./ (sigma * sigma)
    pred = model(noisy)
    dim = Float32(size(batch, 1))
    return (sigma * sigma / dim) * mean(sum(abs2, pred .- target; dims=1))
end

function dsm_loss_with_noise(model, batch, sigma::Float32, noise)
    noisy = batch .+ noise
    target = -noise ./ (sigma * sigma)
    pred = model(noisy)
    dim = Float32(size(batch, 1))
    return (sigma * sigma / dim) * mean(sum(abs2, pred .- target; dims=1))
end

function parameter_norm(model)
    total = 0.0
    for p in Flux.trainables(model)
        total += sum(abs2, p)
    end
    return sqrt(total)
end

function mean_score_norm(model, data)
    scores = model(data)
    norms = sqrt.(vec(sum(abs2, scores; dims=1)))
    return Float64(mean(to_host(norms)))
end

function train_score_model(model, train_data::Matrix{Float32}, params::ScoreTrainingParams, device::DeviceConfig)
    sigma = Float32(params.sigma)
    rng = MersenneTwister(params.seed)
    device_model = to_device(model, device)
    train_device = to_device(train_data, device)
    opt = Flux.setup(Flux.Adam(params.learning_rate), device_model)

    history = Dict(
        :train_loss => Float64[],
        :score_norm => Float64[],
        :param_norm => Float64[],
    )

    epoch_progress = Progress(params.epochs; desc="Training ", dt=0.5)

    for epoch in 1:params.epochs
        perm = randperm(rng, size(train_data, 2))
        epoch_losses = Float64[]

        for batch_start in 1:params.batch_size:size(train_data, 2)
            batch_stop = min(batch_start + params.batch_size - 1, size(train_data, 2))
            batch_idx = perm[batch_start:batch_stop]
            batch = train_device[:, batch_idx]
            noise = device.use_gpu ? sigma .* CUDA.randn(Float32, size(batch)) : sigma .* randn(rng, Float32, size(batch))

            loss_value, grads = Flux.withgradient(device_model) do current_model
                dsm_loss_with_noise(current_model, batch, sigma, noise)
            end
            Flux.update!(opt, device_model, grads[1])
            push!(epoch_losses, Float64(to_host(loss_value)))
        end

        train_loss = mean(epoch_losses)
        score_norm = mean_score_norm(device_model, train_device)
        pnorm = parameter_norm(device_model)

        push!(history[:train_loss], train_loss)
        push!(history[:score_norm], score_norm)
        push!(history[:param_norm], pnorm)

        next!(epoch_progress; showvalues=[
            (:loss, @sprintf("%.3e", train_loss)),
            (:score_norm, @sprintf("%.3e", score_norm)),
            (:param_norm, @sprintf("%.3e", pnorm)),
        ])
    end

    finish!(epoch_progress)

    return device_model, history
end

function stein_matrix(model, data::Matrix{Float32}, sigma::Float32, rng::AbstractRNG)
    noisy_data = data .+ sigma .* randn(rng, Float32, size(data))
    scores = model(noisy_data)
    return -Float64.(scores * noisy_data') ./ size(noisy_data, 2)
end

function stein_matrix(model, data, sigma::Float32, rng::AbstractRNG, device::DeviceConfig)
    noise = device.use_gpu ? sigma .* CUDA.randn(Float32, size(data)) : sigma .* randn(rng, Float32, size(data))
    noisy_data = data .+ noise
    scores = model(noisy_data)
    return -Float64.(to_host(scores * noisy_data')) ./ size(noisy_data, 2)
end

function compute_histogram_1d(data::AbstractVector{<:Real}, bins::Int; range_override=nothing)
    boundary = range_override === nothing ? kde_range(data) : range_override
    kde_result = kde(Float64.(data); npoints=bins, boundary=boundary)
    return collect(kde_result.x), collect(kde_result.density), boundary
end

function compute_histogram_2d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, bins::Int; x_range=nothing, y_range=nothing)
    x_boundary = x_range === nothing ? kde_range(x) : x_range
    y_boundary = y_range === nothing ? kde_range(y) : y_range
    kde_result = kde((Float64.(x), Float64.(y)); npoints=(bins, bins), boundary=(x_boundary, y_boundary))
    return collect(kde_result.x), collect(kde_result.y), Array(kde_result.density), x_boundary, y_boundary
end

function integrate_score_sde(model, init_states::Matrix{Float32}, dt::Float32, steps::Int, burnin_steps::Int, save_stride::Int, seed::Int, device::DeviceConfig)
    rng = MersenneTwister(seed)
    states = to_device(copy(init_states), device)
    nchains = size(states, 2)
    nsaved = (steps - burnin_steps) ÷ save_stride
    samples = Matrix{Float32}(undef, 2, nsaved * nchains)
    noise_scale = sqrt(2.0f0 * dt)
    cursor = 1
    integration_progress = Progress(steps; desc="Langevin ", dt=0.5)

    for step in 1:steps
        drift = model(states)
        if device.use_gpu
            states .+= dt .* drift .+ noise_scale .* CUDA.randn(Float32, size(states))
        else
            states .+= dt .* drift .+ noise_scale .* randn(rng, Float32, size(states))
        end
        if step > burnin_steps && (step - burnin_steps) % save_stride == 0
            samples[:, cursor:(cursor + nchains - 1)] .= to_host(states)
            cursor += nchains
        end
        next!(integration_progress)
    end

    finish!(integration_progress)

    return samples
end

function kl_divergence_from_density_1d(p_density::Vector{Float64}, q_density::Vector{Float64}, width::Float64)
    eps = 1e-12
    p = p_density .* width
    q = q_density .* width
    p .+= eps
    q .+= eps
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function kl_divergence_from_density_2d(p_density::Matrix{Float64}, q_density::Matrix{Float64}, xwidth::Float64, ywidth::Float64)
    eps = 1e-12
    p = vec(p_density .* (xwidth * ywidth))
    q = vec(q_density .* (xwidth * ywidth))
    p .+= eps
    q .+= eps
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function compute_pdf_diagnostics(observed_pdf::ObservedPDFReference, gen_samples::Matrix{Float32})
    gen_x = Float64.(vec(gen_samples[1, :]))
    gen_y = Float64.(vec(gen_samples[2, :]))

    bins = length(observed_pdf.xcenters)
    _, xdens_gen, _ = compute_histogram_1d(gen_x, bins; range_override=observed_pdf.x_range)
    _, ydens_gen, _ = compute_histogram_1d(gen_y, bins; range_override=observed_pdf.y_range)
    _, _, dens_gen, _, _ = compute_histogram_2d(gen_x, gen_y, bins; x_range=observed_pdf.x_range, y_range=observed_pdf.y_range)

    xwidth = length(observed_pdf.xcenters) > 1 ? (observed_pdf.xcenters[2] - observed_pdf.xcenters[1]) : 1.0
    ywidth = length(observed_pdf.ycenters) > 1 ? (observed_pdf.ycenters[2] - observed_pdf.ycenters[1]) : 1.0
    xwidth2 = length(observed_pdf.xgrid) > 1 ? (observed_pdf.xgrid[2] - observed_pdf.xgrid[1]) : 1.0
    ywidth2 = length(observed_pdf.ygrid) > 1 ? (observed_pdf.ygrid[2] - observed_pdf.ygrid[1]) : 1.0

    kl_x = kl_divergence_from_density_1d(observed_pdf.xdensity, xdens_gen, xwidth)
    kl_y = kl_divergence_from_density_1d(observed_pdf.ydensity, ydens_gen, ywidth)
    kl_xy = kl_divergence_from_density_2d(observed_pdf.density2d, dens_gen, xwidth2, ywidth2)
    mean_kl = (kl_x + kl_y + kl_xy) / 3.0
    accuracy = exp(-mean_kl)

    return PDFDiagnostics(
        kl_x,
        kl_y,
        kl_xy,
        mean_kl,
        accuracy,
        size(observed_pdf.samples, 2),
        length(gen_x),
        bins,
        (bins, bins),
    )
end

function summary_lines(params::ScoreTrainingParams, observed_samples::Matrix{Float32}, gen_samples::Matrix{Float32}, stein_mat::Matrix{Float64}, history, pdf_diag::PDFDiagnostics)
    data_mean = vec(mean(observed_samples; dims=2))
    gen_mean = vec(mean(gen_samples; dims=2))
    data_var = vec(var(observed_samples; dims=2))
    gen_var = vec(var(gen_samples; dims=2))
    stein_error = norm(stein_mat - Matrix{Float64}(I, 2, 2))

    return [
        @sprintf("sigma = %.3f", params.sigma),
        @sprintf("widths = %s", string(params.widths)),
        @sprintf("epochs = %d", params.epochs),
        @sprintf("batch_size = %d", params.batch_size),
        @sprintf("data_every = %d", params.data_every),
        @sprintf("train_loss(final) = %.3e", history[:train_loss][end]),
        @sprintf("stein_error = %.3e", stein_error),
        @sprintf("KL_x = %.3e", pdf_diag.kl_x),
        @sprintf("KL_y = %.3e", pdf_diag.kl_y),
        @sprintf("KL_xy = %.3e", pdf_diag.kl_xy),
        @sprintf("pdf_accuracy = %.6f", pdf_diag.accuracy),
        @sprintf("pdf_data_pts = %d", pdf_diag.data_points),
        @sprintf("pdf_rom_pts = %d", pdf_diag.rom_points),
        @sprintf("data mean = [%.3f, %.3f]", data_mean[1], data_mean[2]),
        @sprintf("gen mean  = [%.3f, %.3f]", gen_mean[1], gen_mean[2]),
        @sprintf("data var  = [%.3f, %.3f]", data_var[1], data_var[2]),
        @sprintf("gen var   = [%.3f, %.3f]", gen_var[1], gen_var[2]),
    ]
end

function create_diagnostics_figure(params::ScoreTrainingParams, history, stein_mat::Matrix{Float64}, observed_pdf::ObservedPDFReference, gen_samples::Matrix{Float32}, output_path::AbstractString)
    epochs = collect(1:length(history[:train_loss]))

    gen_x = Float64.(vec(gen_samples[1, :]))
    gen_y = Float64.(vec(gen_samples[2, :]))

    _, xdens_gen, _ = compute_histogram_1d(gen_x, length(observed_pdf.xcenters); range_override=observed_pdf.x_range)
    _, ydens_gen, _ = compute_histogram_1d(gen_y, length(observed_pdf.ycenters); range_override=observed_pdf.y_range)
    _, _, dens_gen, _, _ = compute_histogram_2d(gen_x, gen_y, length(observed_pdf.xgrid); x_range=observed_pdf.x_range, y_range=observed_pdf.y_range)
    pdf_diag = compute_pdf_diagnostics(observed_pdf, gen_samples)

    fig = Figure(; size=(1800, 1250))

    ax1 = Axis(fig[1, 1]; xlabel="Epoch", ylabel="Loss", yscale=log10,
        title="Training DSM Loss")
    lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, label="train")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[1, 2]; xlabel="Epoch", ylabel="Norm", title="Score Norm")
    lines!(ax2, epochs, history[:score_norm]; color=STYLE_ACCENT, label="score norm")
    axislegend(ax2; position=:rt)

    ax3 = Axis(fig[1, 3]; xlabel="Epoch", ylabel="Norm", title="Parameter Norm")
    lines!(ax3, epochs, history[:param_norm]; color=STYLE_HIGHLIGHT, label="param norm")
    axislegend(ax3; position=:rt)

    clim = max(maximum(abs.(stein_mat)), 1e-6)
    ax4 = Axis(fig[2, 1]; xlabel="j", ylabel="i",
        title="Stein matrix  -E[s_i z_j],  z = x + σ·ξ",
        xticks=(1:2, ["x", "y"]), yticks=(1:2, ["x", "y"]),
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    hm4 = heatmap!(ax4, 1:2, 1:2, stein_mat;
        colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
    Colorbar(fig[2, 1, Right()], hm4)
    for i in 1:2, j in 1:2
        text!(ax4, j, i; text=(@sprintf("%.3f", stein_mat[i, j])),
            align=(:center, :center), color=STYLE_REFERENCE, fontsize=14)
    end

    ax5 = Axis(fig[2, 2]; xlabel="x", ylabel="density",
        title="Marginal density p(x)")
    lines!(ax5, observed_pdf.xcenters, observed_pdf.xdensity;
        color=STYLE_REFERENCE, label="data")
    lines!(ax5, observed_pdf.xcenters, xdens_gen;
        color=STYLE_PRIMARY, linestyle=:dash, label="score SDE")
    axislegend(ax5; position=:rt)

    ax6 = Axis(fig[2, 3]; xlabel="y", ylabel="density",
        title="Marginal density p(y)")
    lines!(ax6, observed_pdf.ycenters, observed_pdf.ydensity;
        color=STYLE_REFERENCE, label="data")
    lines!(ax6, observed_pdf.ycenters, ydens_gen;
        color=STYLE_SECONDARY, linestyle=:dash, label="score SDE")
    axislegend(ax6; position=:rt)

    ax7 = Axis(fig[3, 1]; xlabel="x", ylabel="y",
        title="Data joint density  p(x,y)",
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    hm7 = heatmap!(ax7, observed_pdf.xgrid, observed_pdf.ygrid, observed_pdf.density2d;
        colormap=STYLE_SEQUENTIAL)
    Colorbar(fig[3, 1, Right()], hm7)

    ax8 = Axis(fig[3, 2]; xlabel="x", ylabel="y",
        title="Score-SDE joint density  p(x,y)",
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    hm8 = heatmap!(ax8, observed_pdf.xgrid, observed_pdf.ygrid, dens_gen;
        colormap=STYLE_SEQUENTIAL)
    Colorbar(fig[3, 2, Right()], hm8)

    text_panel!(fig[3, 3],
        summary_lines(params, observed_pdf.samples, gen_samples, stein_mat, history, pdf_diag);
        title="Diagnostic Summary", fontsize=14, titlefontsize=18)

    save_figure(output_path, fig)
    return nothing
end

function save_model(path::AbstractString, model, history, stein_mat::Matrix{Float64}, params::ScoreTrainingParams)
    metadata = Dict(
        :sigma => params.sigma,
        :widths => params.widths,
        :epochs => params.epochs,
        :batch_size => params.batch_size,
        :learning_rate => params.learning_rate,
    )
    host_model = cpu(model)
    BSON.@save path host_model history stein_mat metadata
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

    rng = MersenneTwister(params.seed)
    device = detect_device()
    @printf("Training device: %s\n", device.name)
    @printf("Loading samples from %s\n", input_hdf5)
    training_samples = load_samples(input_hdf5, params.burnin_fraction, params.data_every, params.max_samples, rng)
    observed_pdf = load_observed_pdf_reference(input_hdf5, params.burnin_fraction)
    @printf("Training samples: %d\n", size(training_samples, 2))
    @printf("Observed PDF reference samples: %d\n", size(observed_pdf.samples, 2))

    model = build_model(params.widths)
    model, history = train_score_model(model, training_samples, params, device)

    observed_device = to_device(training_samples, device)
    stein_rng = MersenneTwister(params.seed + 1)
    stein_mat = stein_matrix(model, observed_device, Float32(params.sigma), stein_rng, device)

    init_idx = rand(rng, 1:size(training_samples, 2), params.sample_chains)
    init_states = training_samples[:, init_idx]
    gen_samples = integrate_score_sde(model, init_states, Float32(params.sample_dt), params.sample_steps,
        params.sample_burnin_steps, params.sample_save_stride, params.seed + 10, device)

    @printf("Saving model to %s\n", output_bson)
    save_model(output_bson, model, history, stein_mat, params)

    @printf("Saving diagnostics figure to %s\n", output_png)
    create_diagnostics_figure(params, history, stein_mat, observed_pdf, gen_samples, output_png)

    @printf("Done. Final train DSM loss = %.6e\n", history[:train_loss][end])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
