#!/usr/bin/env julia

# Agent summary:
# - Trains the lagged joint score model needed to estimate conditional scores s_{t|0}(x_t | x_0).
# - Saves the trained conditional-score model and its diagnostics for the mobility-fitting scripts.

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

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "joint_score.toml")
const JOINT_DIM = 4
const MODEL_INPUT_DIM = 5

struct JointScoreTrainingParams
    input_hdf5::String
    burnin_fraction::Float64
    tau_min::Float64
    max_eval_pairs::Int
    sigma::Float64
    widths::Vector{Int}
    epochs::Int
    batches_per_epoch::Int
    batch_size::Int
    learning_rate::Float64
    seed::Int
    sample_dt::Float64
    sample_steps::Int
    sample_burnin_steps::Int
    sample_save_stride::Int
    sample_chains::Int
    eval_tau_count::Int
    pdf_bins::Int
    output_bson::String
    output_png::String
end

struct DeviceConfig
    use_gpu::Bool
    name::String
end

struct PairSampler
    states::Array{Float64, 3}
    times::Vector{Float64}
    start_idx::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    lag_tnorm::Vector{Float32}
    tau_min::Float64
    tau_max::Float64
end

struct PairDensity2D
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    density::Matrix{Float64}
    xrange::Tuple{Float64, Float64}
    yrange::Tuple{Float64, Float64}
end

struct JointPDFDiagnostics
    tau::Float64
    tnorm::Float64
    kl_xpair::Float64
    kl_ypair::Float64
    mean_kl::Float64
    accuracy::Float64
    stein_error::Float64
    observed_points::Int
    gen_points::Int
end

struct JointEvalRecord
    tau::Float64
    tnorm::Float32
    observed_pairs::Matrix{Float32}
    generated_pairs::Matrix{Float32}
    observed_xpair::PairDensity2D
    generated_xpair::PairDensity2D
    observed_ypair::PairDensity2D
    generated_ypair::PairDensity2D
    stein_mat::Matrix{Float64}
    diag::JointPDFDiagnostics
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

    params = JointScoreTrainingParams(
        String(data_cfg["input_hdf5"]),
        Float64(data_cfg["burnin_fraction"]),
        Float64(data_cfg["tau_min"]),
        Int(data_cfg["max_eval_pairs"]),
        Float64(training_cfg["sigma"]),
        Int.(training_cfg["widths"]),
        Int(training_cfg["epochs"]),
        Int(training_cfg["batches_per_epoch"]),
        Int(training_cfg["batch_size"]),
        Float64(training_cfg["learning_rate"]),
        Int(training_cfg["seed"]),
        Float64(sampling_cfg["dt"]),
        Int(sampling_cfg["steps"]),
        Int(sampling_cfg["burnin_steps"]),
        Int(sampling_cfg["save_stride"]),
        Int(sampling_cfg["chains"]),
        Int(figure_cfg["eval_tau_count"]),
        Int(figure_cfg["pdf_bins"]),
        String(output_cfg["model_bson"]),
        String(output_cfg["figure_png"]),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.tau_min > 0.0, "tau_min must be positive.")
    require_condition(params.max_eval_pairs >= 1_000, "max_eval_pairs must be at least 1000.")
    require_condition(params.sigma > 0.0, "sigma must be positive.")
    require_condition(!isempty(params.widths), "widths must contain at least one hidden layer width.")
    require_condition(all(width -> width >= 8, params.widths), "Each width must be at least 8.")
    require_condition(params.epochs >= 1, "epochs must be >= 1.")
    require_condition(params.batches_per_epoch >= 1, "batches_per_epoch must be >= 1.")
    require_condition(params.batch_size >= 16, "batch_size must be >= 16.")
    require_condition(params.learning_rate > 0.0, "learning_rate must be positive.")
    require_condition(params.sample_dt > 0.0, "sample_dt must be positive.")
    require_condition(params.sample_steps > params.sample_burnin_steps, "sample_steps must exceed sample_burnin_steps.")
    require_condition(params.sample_save_stride >= 1, "sample_save_stride must be >= 1.")
    require_condition(params.sample_chains >= 1, "sample_chains must be >= 1.")
    require_condition(params.eval_tau_count >= 1, "eval_tau_count must be >= 1.")
    require_condition(params.pdf_bins >= 24, "pdf_bins must be >= 24.")
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
    in_dim = MODEL_INPUT_DIM
    for width in widths
        push!(layers, Dense(in_dim, width, tanh))
        in_dim = width
    end
    push!(layers, Dense(in_dim, JOINT_DIM))
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
    require_condition(size(states, 1) == length(times), "The saved trajectory tensor does not match the saved time axis.")
    return times, states
end

function build_pair_sampler(path::AbstractString, burnin_fraction::Float64, tau_min::Float64)
    times, states = load_state_tensor(path)
    t_decorrelation = read(h5open(path, "r")["/statistics/correlations/t_decorrelation"])
    save_dt = length(times) > 1 ? (times[2] - times[1]) : 0.0
    require_condition(save_dt > 0.0, "A positive saved time step is required.")
    start_idx = burnin_start_index(length(times), burnin_fraction)
    tau_max = min(t_decorrelation, times[end] - times[start_idx])
    require_condition(tau_max >= tau_min, "tau_min exceeds the available lag range up to tD.")

    min_lag = max(1, ceil(Int, tau_min / save_dt - 1e-9))
    max_lag = min(length(times) - start_idx, floor(Int, tau_max / save_dt + 1e-9))
    require_condition(max_lag >= min_lag, "No lag steps are available in [tau_min, tD].")

    lag_steps = collect(min_lag:max_lag)
    lag_times = lag_steps .* save_dt
    denom = max(tau_max - tau_min, eps())
    lag_tnorm = Float32.((lag_times .- tau_min) ./ denom)
    return PairSampler(states, times, start_idx, lag_steps, lag_times, lag_tnorm, tau_min, tau_max)
end

function random_pair_matrix(sampler::PairSampler, lag::Int, tnorm::Float32, npairs::Int, rng::AbstractRNG)
    nt, _, ntraj = size(sampler.states)
    z = Matrix{Float32}(undef, JOINT_DIM, npairs)
    upper = nt - lag
    require_condition(upper >= sampler.start_idx, "Lag exceeds the available post-burn-in time window.")

    @inbounds for n in 1:npairs
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:upper)
        z[1, n] = Float32(sampler.states[time_idx, 1, traj_idx])
        z[2, n] = Float32(sampler.states[time_idx, 2, traj_idx])
        z[3, n] = Float32(sampler.states[time_idx + lag, 1, traj_idx])
        z[4, n] = Float32(sampler.states[time_idx + lag, 2, traj_idx])
    end

    trow = fill(tnorm, 1, npairs)
    return z, trow
end

function sample_training_batch!(z::Matrix{Float32}, trow::Matrix{Float32}, sampler::PairSampler, rng::AbstractRNG)
    nt, _, ntraj = size(sampler.states)
    nlags = length(sampler.lag_steps)

    @inbounds for n in 1:size(z, 2)
        lag_pos = rand(rng, 1:nlags)
        lag = sampler.lag_steps[lag_pos]
        upper = nt - lag
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:upper)

        z[1, n] = Float32(sampler.states[time_idx, 1, traj_idx])
        z[2, n] = Float32(sampler.states[time_idx, 2, traj_idx])
        z[3, n] = Float32(sampler.states[time_idx + lag, 1, traj_idx])
        z[4, n] = Float32(sampler.states[time_idx + lag, 2, traj_idx])
        trow[1, n] = sampler.lag_tnorm[lag_pos]
    end
    return nothing
end

function input_from_z_and_t(z, trow)
    return vcat(z, trow)
end

function dsm_loss_with_noise(model, z_batch::Matrix{Float32}, trow::Matrix{Float32}, sigma::Float32, noise::Matrix{Float32})
    noisy_z = z_batch .+ noise
    pred = model(input_from_z_and_t(noisy_z, trow))
    target = -noise ./ (sigma * sigma)
    dim = Float32(size(z_batch, 1))
    return (sigma * sigma / dim) * mean(sum(abs2, pred .- target; dims=1))
end

function dsm_loss_with_noise(model, z_batch, trow, sigma::Float32, noise)
    noisy_z = z_batch .+ noise
    pred = model(input_from_z_and_t(noisy_z, trow))
    target = -noise ./ (sigma * sigma)
    dim = Float32(size(noisy_z, 1))
    return (sigma * sigma / dim) * mean(sum(abs2, pred .- target; dims=1))
end

function parameter_norm(model)
    total = 0.0
    for p in Flux.trainables(model)
        total += sum(abs2, p)
    end
    return sqrt(total)
end

function mean_score_norm(model, z::Matrix{Float32}, trow::Matrix{Float32}, device::DeviceConfig)
    scores = model(input_from_z_and_t(to_device(z, device), to_device(trow, device)))
    norms = sqrt.(vec(sum(abs2, scores; dims=1)))
    return Float64(mean(to_host(norms)))
end

function train_joint_score_model(model, sampler::PairSampler, params::JointScoreTrainingParams, device::DeviceConfig)
    sigma = Float32(params.sigma)
    rng = MersenneTwister(params.seed)
    device_model = to_device(model, device)
    opt = Flux.setup(Flux.Adam(params.learning_rate), device_model)

    batch_z = Matrix{Float32}(undef, JOINT_DIM, params.batch_size)
    batch_t = Matrix{Float32}(undef, 1, params.batch_size)

    history = Dict(
        :train_loss => Float64[],
        :score_norm => Float64[],
        :param_norm => Float64[],
    )

    epoch_progress = Progress(params.epochs; desc="Training ", dt=0.5)

    for epoch in 1:params.epochs
        epoch_losses = Float64[]

        for _ in 1:params.batches_per_epoch
            sample_training_batch!(batch_z, batch_t, sampler, rng)
            z_device = to_device(batch_z, device)
            t_device = to_device(batch_t, device)
            noise = device.use_gpu ? sigma .* CUDA.randn(Float32, size(z_device)) : sigma .* randn(rng, Float32, size(batch_z))

            loss_value, grads = Flux.withgradient(device_model) do current_model
                dsm_loss_with_noise(current_model, z_device, t_device, sigma, noise)
            end
            Flux.update!(opt, device_model, grads[1])
            push!(epoch_losses, Float64(to_host(loss_value)))
        end

        sample_training_batch!(batch_z, batch_t, sampler, rng)
        train_loss = mean(epoch_losses)
        score_norm = mean_score_norm(device_model, batch_z, batch_t, device)
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

function stein_matrix(model, z::Matrix{Float32}, trow::Matrix{Float32}, sigma::Float32, rng::AbstractRNG, device::DeviceConfig)
    z_device = to_device(z, device)
    t_device = to_device(trow, device)
    noise = device.use_gpu ? sigma .* CUDA.randn(Float32, size(z_device)) : sigma .* randn(rng, Float32, size(z))
    noisy_z = z_device .+ noise
    scores = model(input_from_z_and_t(noisy_z, t_device))
    return -Float64.(to_host(scores * noisy_z')) ./ size(z, 2)
end

function integrate_joint_score_sde(model, init_z::Matrix{Float32}, tnorm::Float32, dt::Float32, steps::Int,
        burnin_steps::Int, save_stride::Int, seed::Int, device::DeviceConfig)
    rng = MersenneTwister(seed)
    states = to_device(copy(init_z), device)
    nchains = size(states, 2)
    nsaved = (steps - burnin_steps) ÷ save_stride
    samples = Matrix{Float32}(undef, JOINT_DIM, nsaved * nchains)
    noise_scale = sqrt(2.0f0 * dt)
    trow = to_device(fill(tnorm, 1, nchains), device)
    cursor = 1
    integration_progress = Progress(steps; desc="Langevin ", dt=0.5)

    for step in 1:steps
        drift = model(input_from_z_and_t(states, trow))
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

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function compute_histogram_2d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, bins::Int; x_range=nothing, y_range=nothing)
    x_boundary = x_range === nothing ? kde_range(x) : x_range
    y_boundary = y_range === nothing ? kde_range(y) : y_range
    kde_result = kde((Float64.(x), Float64.(y)); npoints=(bins, bins), boundary=(x_boundary, y_boundary))
    return collect(kde_result.x), collect(kde_result.y), Array(kde_result.density), x_boundary, y_boundary
end

function compute_histogram_1d(x::AbstractVector{<:Real}, bins::Int; range_override=nothing)
    boundary = range_override === nothing ? kde_range(x) : range_override
    kde_result = kde(Float64.(x); npoints=bins, boundary=boundary)
    return collect(kde_result.x), collect(kde_result.density), boundary
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

function pair_density(samples::Matrix{Float32}, coord_a::Int, coord_b::Int, bins::Int; x_range=nothing, y_range=nothing)
    x = Float64.(vec(samples[coord_a, :]))
    y = Float64.(vec(samples[coord_b, :]))
    xgrid, ygrid, density, xrange, yrange = compute_histogram_2d(x, y, bins; x_range=x_range, y_range=y_range)
    return PairDensity2D(xgrid, ygrid, density, xrange, yrange)
end

function compute_pair_diagnostics(observed_xpair::PairDensity2D, observed_ypair::PairDensity2D,
        generated_xpair::PairDensity2D, generated_ypair::PairDensity2D)
    xwidth = length(observed_xpair.xgrid) > 1 ? (observed_xpair.xgrid[2] - observed_xpair.xgrid[1]) : 1.0
    ywidth = length(observed_xpair.ygrid) > 1 ? (observed_xpair.ygrid[2] - observed_xpair.ygrid[1]) : 1.0
    xwidth_y = length(observed_ypair.xgrid) > 1 ? (observed_ypair.xgrid[2] - observed_ypair.xgrid[1]) : 1.0
    ywidth_y = length(observed_ypair.ygrid) > 1 ? (observed_ypair.ygrid[2] - observed_ypair.ygrid[1]) : 1.0

    kl_xpair = kl_divergence_from_density_2d(observed_xpair.density, generated_xpair.density, xwidth, ywidth)
    kl_ypair = kl_divergence_from_density_2d(observed_ypair.density, generated_ypair.density, xwidth_y, ywidth_y)
    mean_kl = 0.5 * (kl_xpair + kl_ypair)
    accuracy = exp(-mean_kl)
    return kl_xpair, kl_ypair, mean_kl, accuracy
end

function choose_eval_lags(sampler::PairSampler, neval::Int)
    total = length(sampler.lag_steps)
    nsel = min(neval, total)
    idxs = if nsel == 1
        [1]
    else
        unique(round.(Int, range(1, total, length=nsel)))
    end
    return sampler.lag_steps[idxs], sampler.lag_times[idxs], sampler.lag_tnorm[idxs]
end

function maybe_subsample_columns(data::Matrix{Float32}, max_cols::Int, rng::AbstractRNG)
    if size(data, 2) <= max_cols
        return data
    end
    keep = randperm(rng, size(data, 2))[1:max_cols]
    return data[:, keep]
end

function evaluate_tau(model, sampler::PairSampler, tau::Float64, lag::Int, tnorm::Float32,
        params::JointScoreTrainingParams, device::DeviceConfig, rng::AbstractRNG, seed_offset::Int)
    observed_pairs, observed_trow = random_pair_matrix(sampler, lag, tnorm, params.max_eval_pairs, rng)
    observed_xpair = pair_density(observed_pairs, 1, 3, params.pdf_bins)
    observed_ypair = pair_density(observed_pairs, 2, 4, params.pdf_bins)

    init_idx = rand(rng, 1:size(observed_pairs, 2), params.sample_chains)
    init_states = observed_pairs[:, init_idx]
    gen_all = integrate_joint_score_sde(model, init_states, tnorm, Float32(params.sample_dt), params.sample_steps,
        params.sample_burnin_steps, params.sample_save_stride, params.seed + seed_offset, device)
    gen_pairs = maybe_subsample_columns(gen_all, params.max_eval_pairs, rng)

    generated_xpair = pair_density(gen_pairs, 1, 3, params.pdf_bins;
        x_range=observed_xpair.xrange, y_range=observed_xpair.yrange)
    generated_ypair = pair_density(gen_pairs, 2, 4, params.pdf_bins;
        x_range=observed_ypair.xrange, y_range=observed_ypair.yrange)

    stein_rng = MersenneTwister(params.seed + 10_000 + seed_offset)
    stein_mat = stein_matrix(model, observed_pairs, observed_trow, Float32(params.sigma), stein_rng, device)
    stein_error = norm(stein_mat - Matrix{Float64}(I, JOINT_DIM, JOINT_DIM))
    kl_xpair, kl_ypair, mean_kl, accuracy = compute_pair_diagnostics(observed_xpair, observed_ypair, generated_xpair, generated_ypair)

    diag = JointPDFDiagnostics(
        tau,
        Float64(tnorm),
        kl_xpair,
        kl_ypair,
        mean_kl,
        accuracy,
        stein_error,
        size(observed_pairs, 2),
        size(gen_pairs, 2),
    )

    return JointEvalRecord(tau, tnorm, observed_pairs, gen_pairs, observed_xpair, generated_xpair, observed_ypair, generated_ypair, stein_mat, diag)
end

function summary_lines(params::JointScoreTrainingParams, sampler::PairSampler, history, eval_records::Vector{JointEvalRecord})
    mean_accuracy = mean(record.diag.accuracy for record in eval_records)
    mean_kl = mean(record.diag.mean_kl for record in eval_records)
    mean_stein = mean(record.diag.stein_error for record in eval_records)

    return [
        @sprintf("tau range = [%.2f, %.2f]", sampler.tau_min, sampler.tau_max),
        @sprintf("lags used = %d", length(sampler.lag_steps)),
        @sprintf("sigma = %.3f", params.sigma),
        @sprintf("widths = %s", string(params.widths)),
        @sprintf("epochs = %d", params.epochs),
        @sprintf("batches/epoch = %d", params.batches_per_epoch),
        @sprintf("batch_size = %d", params.batch_size),
        @sprintf("sample_dt = %.4f", params.sample_dt),
        @sprintf("sample_steps = %d", params.sample_steps),
        @sprintf("max_eval_pairs = %d", params.max_eval_pairs),
        @sprintf("train_loss(final) = %.3e", history[:train_loss][end]),
        @sprintf("mean KL = %.3e", mean_kl),
        @sprintf("mean accuracy = %.6f", mean_accuracy),
        @sprintf("mean stein err = %.3e", mean_stein),
    ]
end

function metrics_panel!(parent, eval_records::Vector{JointEvalRecord})
    taus = [record.tau for record in eval_records]
    accuracies = [record.diag.accuracy for record in eval_records]
    stein_errors = [record.diag.stein_error for record in eval_records]

    ax = Axis(parent; xlabel="τ", ylabel="score",
        title="Joint PDF accuracy across lags")
    ylims!(ax, 0.0, 1.05)
    lines!(ax, taus, accuracies; color=STYLE_PRIMARY, label="accuracy")
    scatter!(ax, taus, accuracies; color=STYLE_PRIMARY, marker=:circle)
    lines!(ax, taus, exp.(-stein_errors); color=STYLE_HIGHLIGHT, label="exp(-Stein err)")
    scatter!(ax, taus, exp.(-stein_errors); color=STYLE_HIGHLIGHT, marker=:diamond)
    axislegend(ax; position=:rb)
    return ax
end

function stein_panel!(parent, stein_mat::Matrix{Float64}, tau::Float64, diag::JointPDFDiagnostics)
    labels = ["x₀", "y₀", "xₜ", "yₜ"]
    clim = max(maximum(abs.(stein_mat)), 1e-6)
    gl = GridLayout(parent)
    ax = Axis(gl[1, 1]; xlabel="j", ylabel="i",
        title=@sprintf("Stein matrix  τ=%.2f\nKL=%.2e   acc=%.3f", tau, diag.mean_kl, diag.accuracy),
        xticks=(1:JOINT_DIM, labels), yticks=(1:JOINT_DIM, labels),
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    hm = heatmap!(ax, 1:JOINT_DIM, 1:JOINT_DIM, stein_mat;
        colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
    Colorbar(gl[1, 2], hm)
    for i in 1:JOINT_DIM, j in 1:JOINT_DIM
        text!(ax, j, i; text=(@sprintf("%.2f", stein_mat[i, j])),
            align=(:center, :center), color=STYLE_REFERENCE, fontsize=12)
    end
    return gl
end

function univariate_pair_panel!(parent, observed_pairs::Matrix{Float32}, generated_pairs::Matrix{Float32},
        coord_a::Int, coord_b::Int, bins::Int, tau::Float64, label_a::String, label_b::String)
    obs_a = Float64.(vec(observed_pairs[coord_a, :]))
    obs_b = Float64.(vec(observed_pairs[coord_b, :]))
    gen_a = Float64.(vec(generated_pairs[coord_a, :]))
    gen_b = Float64.(vec(generated_pairs[coord_b, :]))

    range_a = kde_range(vcat(obs_a, gen_a))
    range_b = kde_range(vcat(obs_b, gen_b))
    centers_a, dens_obs_a, _ = compute_histogram_1d(obs_a, bins; range_override=range_a)
    _, dens_gen_a, _ = compute_histogram_1d(gen_a, bins; range_override=range_a)
    centers_b, dens_obs_b, _ = compute_histogram_1d(obs_b, bins; range_override=range_b)
    _, dens_gen_b, _ = compute_histogram_1d(gen_b, bins; range_override=range_b)

    ax = Axis(parent; xlabel="value", ylabel="density",
        title=@sprintf("Marginal densities  τ=%.2f", tau))
    lines!(ax, centers_a, dens_obs_a; color=STYLE_PRIMARY, label="$(label_a) obs")
    lines!(ax, centers_a, dens_gen_a; color=STYLE_PRIMARY, linestyle=:dash, label="$(label_a) gen")
    lines!(ax, centers_b, dens_obs_b; color=STYLE_ACCENT, label="$(label_b) obs")
    lines!(ax, centers_b, dens_gen_b; color=STYLE_ACCENT, linestyle=:dash, label="$(label_b) gen")
    axislegend(ax; position=:rt, nbanks=2)
    return ax
end

function contour_levels_from_densities(density_a::Matrix{Float64}, density_b::Matrix{Float64})
    vmax = max(maximum(density_a), maximum(density_b))
    return collect(range(0.15 * vmax, 0.9 * vmax, length=6))
end

function contour_pair_panel!(parent, observed::PairDensity2D, generated::PairDensity2D, tau::Float64,
        xlabel::String, ylabel::String, title_label::String)
    levels = contour_levels_from_densities(observed.density, generated.density)
    ax = Axis(parent; xlabel=xlabel, ylabel=ylabel,
        title=@sprintf("%s  τ=%.2f", title_label, tau),
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    contour!(ax, observed.xgrid, observed.ygrid, observed.density;
        levels=levels, color=STYLE_REFERENCE, linewidth=2.0, linestyle=:solid)
    contour!(ax, generated.xgrid, generated.ygrid, generated.density;
        levels=levels, color=STYLE_HIGHLIGHT, linewidth=2.0, linestyle=:dash)
    elements = [LineElement(color=STYLE_REFERENCE, linewidth=2.4),
                LineElement(color=STYLE_HIGHLIGHT, linewidth=2.4, linestyle=:dash)]
    axislegend(ax, elements, ["observed", "generated"]; position=:rt)
    return ax
end

function create_diagnostics_figure(params::JointScoreTrainingParams, sampler::PairSampler, history,
        eval_records::Vector{JointEvalRecord}, output_path::AbstractString)
    epochs = collect(1:length(history[:train_loss]))
    nrows = 1 + length(eval_records)
    panel_h = 460
    panel_w = 540
    fig = Figure(; size=(panel_w * 5, panel_h * nrows))

    ax_loss = Axis(fig[1, 1]; xlabel="Epoch", ylabel="Loss", yscale=log10,
        title="Training DSM Loss")
    lines!(ax_loss, epochs, history[:train_loss]; color=STYLE_PRIMARY, label="train")
    axislegend(ax_loss; position=:rt)

    ax_score = Axis(fig[1, 2]; xlabel="Epoch", ylabel="Norm", title="Score Norm")
    lines!(ax_score, epochs, history[:score_norm]; color=STYLE_ACCENT, label="score norm")
    axislegend(ax_score; position=:rt)

    ax_param = Axis(fig[1, 3]; xlabel="Epoch", ylabel="Norm", title="Parameter Norm")
    lines!(ax_param, epochs, history[:param_norm]; color=STYLE_HIGHLIGHT, label="param norm")
    axislegend(ax_param; position=:rt)

    metrics_panel!(fig[1, 4], eval_records)
    text_panel!(fig[1, 5], summary_lines(params, sampler, history, eval_records);
        title="Diagnostic Summary", fontsize=14, titlefontsize=18)

    for (idx, record) in enumerate(eval_records)
        row = 1 + idx
        univariate_pair_panel!(fig[row, 1], record.observed_pairs, record.generated_pairs,
            1, 3, params.pdf_bins, record.tau, "x₀", "xₜ")
        univariate_pair_panel!(fig[row, 2], record.observed_pairs, record.generated_pairs,
            2, 4, params.pdf_bins, record.tau, "y₀", "yₜ")
        contour_pair_panel!(fig[row, 3], record.observed_xpair, record.generated_xpair,
            record.tau, "x₀", "xₜ", @sprintf("(x₀, xₜ)  KL=%.2e", record.diag.kl_xpair))
        contour_pair_panel!(fig[row, 4], record.observed_ypair, record.generated_ypair,
            record.tau, "y₀", "yₜ", @sprintf("(y₀, yₜ)  KL=%.2e", record.diag.kl_ypair))
        stein_panel!(fig[row, 5], record.stein_mat, record.tau, record.diag)
    end

    save_figure(output_path, fig)
    return nothing
end

function save_model(path::AbstractString, model, history, params::JointScoreTrainingParams, sampler::PairSampler, eval_records::Vector{JointEvalRecord})
    metadata = Dict(
        :sigma => params.sigma,
        :widths => params.widths,
        :epochs => params.epochs,
        :batches_per_epoch => params.batches_per_epoch,
        :batch_size => params.batch_size,
        :learning_rate => params.learning_rate,
        :tau_min => sampler.tau_min,
        :tau_max => sampler.tau_max,
        :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times,
        :eval_taus => [record.tau for record in eval_records],
    )
    diagnostics = Dict(
        :tau => [record.diag.tau for record in eval_records],
        :kl_xpair => [record.diag.kl_xpair for record in eval_records],
        :kl_ypair => [record.diag.kl_ypair for record in eval_records],
        :mean_kl => [record.diag.mean_kl for record in eval_records],
        :accuracy => [record.diag.accuracy for record in eval_records],
        :stein_error => [record.diag.stein_error for record in eval_records],
    )
    host_model = cpu(model)
    BSON.@save path host_model history metadata diagnostics
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

    sampler = build_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min)
    device = detect_device()

    @printf("Training device: %s\n", device.name)
    @printf("Lag window: [%.3f, %.3f] with %d discrete lags\n", sampler.tau_min, sampler.tau_max, length(sampler.lag_steps))

    model = build_model(params.widths)
    model, history = train_joint_score_model(model, sampler, params, device)

    eval_lags, eval_taus, eval_tnorm = choose_eval_lags(sampler, params.eval_tau_count)
    eval_rng = MersenneTwister(params.seed + 50_000)
    eval_records = JointEvalRecord[]
    for (idx, (lag, tau, tnorm)) in enumerate(zip(eval_lags, eval_taus, eval_tnorm))
        @printf("Evaluating tau = %.3f (lag step %d, normalized t = %.3f)\n", tau, lag, tnorm)
        push!(eval_records, evaluate_tau(model, sampler, tau, lag, tnorm, params, device, eval_rng, idx))
    end

    @printf("Saving model to %s\n", output_bson)
    save_model(output_bson, model, history, params, sampler, eval_records)

    @printf("Saving diagnostics figure to %s\n", output_png)
    create_diagnostics_figure(params, sampler, history, eval_records, output_png)

    mean_kl = mean(record.diag.mean_kl for record in eval_records)
    @printf("Done. Mean evaluation KL = %.6e\n", mean_kl)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
