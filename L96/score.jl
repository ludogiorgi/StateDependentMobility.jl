#!/usr/bin/env julia

import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")

function ensure_packages(packages::Vector{String})
    project_deps = Pkg.project().dependencies
    missing = String[]
    for pkg in packages
        if !haskey(project_deps, pkg)
            push!(missing, pkg)
        end
    end
    if !isempty(missing)
        @info "Installing missing Julia packages" missing
        Pkg.add(missing)
    end
    return nothing
end

ensure_packages(["BSON", "CUDA", "cuDNN", "FFTW", "Flux", "Functors", "HDF5", "KernelDensity", "NNlib", "ProgressMeter"])

using BSON
using CUDA
using cuDNN
using FFTW
using Flux
using Functors
using HDF5
using KernelDensity
using LinearAlgebra
using NNlib
using Printf
using ProgressMeter
using Random
using Statistics
using TOML

include(joinpath(SCOREUNET_SRC, "Device.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "PeriodicConv.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "Blocks.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "UNet1D.jl"))
include(joinpath(SCOREUNET_SRC, "data", "DataPipeline.jl"))
include(joinpath(SCOREUNET_SRC, "training", "Trainer.jl"))
include(joinpath(SCOREUNET_SRC, "EnsembleIntegrator.jl"))

const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
isfile(STYLE_FILE) || error("Shared figure style file not found: $(STYLE_FILE)")
include(STYLE_FILE)
GLMakie.activate!()

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "score.toml")

function activation_from_string(name::AbstractString)
    lname = lowercase(name)
    lname == "swish" && return Flux.swish
    lname == "gelu" && return Flux.gelu
    lname == "relu" && return Flux.relu
    lname == "tanh" && return tanh
    lname == "identity" && return identity
    lname == "softplus" && return Flux.softplus
    error("Unsupported activation: $name")
end

Base.@kwdef mutable struct LangevinConfig
    dt::Float64 = 1e-3
    sample_dt::Float64 = 1e-2
    nsteps::Int = 40_000
    resolution::Int = 20
    n_ensembles::Int = 256
    burn_in::Int = 4_000
    nbins::Int = 120
    sigma::Float32 = 0.1f0
    seed::Int = 21
    progress::Bool = false
end

struct ScoreWrapper{M}
    model::M
    sigma::Float32
    L::Int
    C::Int
    dim::Int
end

Functors.@functor ScoreWrapper (model,)

function (sw::ScoreWrapper)(x::AbstractMatrix)
    batch = size(x, 2)
    input_type = eltype(x)
    if input_type <: Float32
        reshaped = reshape(x, sw.L, sw.C, batch)
        scores = score_from_model(sw.model, reshaped, sw.sigma)
        return reshape(scores, sw.dim, batch)
    end
    x_f32 = Float32.(x)
    reshaped = reshape(x_f32, sw.L, sw.C, batch)
    scores = score_from_model(sw.model, reshaped, sw.sigma)
    return input_type.(reshape(scores, sw.dim, batch))
end

struct PairPdfResult
    offset::Int
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct L96ObservedReference
    times::Vector{Float64}
    post_states::Array{Float32, 3}
    flat_post_states::Matrix{Float32}
    univariate_centers::Vector{Float64}
    univariate_density::Vector{Float64}
    pair_pdfs::Vector{PairPdfResult}
    decorrelation_time::Float64
    mean_value::Float64
    std_value::Float64
    spectrum_modes::Vector{Int}
    spectrum::Vector{Float64}
end

struct L96ScoreParams
    input_hdf5::String
    burnin_fraction::Float64
    data_every::Int
    max_samples::Int
    shared_normalization::Bool
    model_config::ScoreUNetConfig
    model_init_seed::Int
    trainer_config::ScoreTrainerConfig
    sampling_config::LangevinConfig
    pair_offsets::Vector{Int}
    pdf_bins::Int
    max_pdf_samples::Int
    stein_samples::Int
    spectrum_samples::Int
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    device_name::String
    verbose::Bool
end

struct L96Diagnostics
    history::Dict{Symbol, Vector{Float64}}
    stein_matrix::Matrix{Float64}
    stein_relative_error::Float64
    generated_univariate_density::Vector{Float64}
    generated_pair_pdfs::Vector{PairPdfResult}
    univariate_kl::Float64
    pair_kls::Vector{Float64}
    mean_kl::Float64
    pdf_accuracy::Float64
    spectrum_modes::Vector{Int}
    generated_spectrum::Vector{Float64}
    spectrum_relative_error::Float64
    finite_generated_snapshots::Int
    total_generated_snapshots::Int
    generated_states::Array{Float32, 3}
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? path : normpath(joinpath(base_dir, path))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function max_compatible_unet_levels(length_dim::Int)
    length_dim >= 2 || return 1
    levels = 0
    current = length_dim
    while current > 1 && iseven(current)
        levels += 1
        current ÷= 2
    end
    return max(levels, 1)
end

function adjust_model_config_for_length(cfg::ScoreUNetConfig, length_dim::Int)
    max_levels = max_compatible_unet_levels(length_dim)
    if length(cfg.channel_multipliers) <= max_levels
        return cfg
    end
    @warn "Reducing U-Net depth for exact periodic upsampling on the L96 lattice" requested=length(cfg.channel_multipliers) compatible=max_levels length_dim
    return ScoreUNetConfig(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_multipliers=cfg.channel_multipliers[1:max_levels],
        kernel_size=cfg.kernel_size,
        periodic=cfg.periodic,
        activation=cfg.activation,
        final_activation=cfg.final_activation,
    )
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function to_host(x)
    return x isa AbstractArray && !(x isa Array) ? Array(x) : x
end

function detect_device(name::AbstractString)
    normalized = uppercase(strip(name))
    normalized == "AUTO" && return CUDA.functional() ? select_device("GPU:0") : CPUDevice()
    normalized == "GPU" && return select_device("GPU:0")
    return select_device(name)
end

function describe_device(device::ExecutionDevice)
    if device isa GPUDevice
        return "GPU:" * join(device.ids, ",")
    end
    return "CPU"
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data_cfg = raw["data"]
    model_cfg = raw["model"]
    training_cfg = raw["training"]
    sampling_cfg = raw["sampling"]
    figure_cfg = raw["figure"]
    output_cfg = raw["output"]
    run_cfg = haskey(raw, "run") ? raw["run"] : Dict{String, Any}()

    model_config = ScoreUNetConfig(
        in_channels=Int(get(model_cfg, "in_channels", 1)),
        base_channels=Int(get(model_cfg, "base_channels", 32)),
        channel_multipliers=Int.(get(model_cfg, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(model_cfg, "kernel_size", 5)),
        periodic=Bool(get(model_cfg, "periodic", true)),
        activation=activation_from_string(get(model_cfg, "activation", "swish")),
        final_activation=activation_from_string(get(model_cfg, "final_activation", "identity")),
    )

    max_steps = get(training_cfg, "max_steps_per_epoch", nothing)
    max_steps = max_steps === nothing ? nothing : Int(max_steps)
    trainer_config = ScoreTrainerConfig(
        batch_size=Int(get(training_cfg, "batch_size", 256)),
        epochs=Int(get(training_cfg, "epochs", 80)),
        lr=Float64(get(training_cfg, "learning_rate", get(training_cfg, "lr", 3e-4))),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        shuffle=Bool(get(training_cfg, "shuffle", true)),
        seed=Int(get(training_cfg, "seed", 20260427)),
        progress=Bool(get(training_cfg, "progress", true)),
        max_steps_per_epoch=max_steps,
        accumulation_steps=Int(get(training_cfg, "accumulation_steps", 1)),
        use_lr_schedule=Bool(get(training_cfg, "use_lr_schedule", true)),
        warmup_steps=Int(get(training_cfg, "warmup_steps", 100)),
        min_lr_factor=Float64(get(training_cfg, "min_lr_factor", 0.05)),
        epoch_subset_size=Int(get(training_cfg, "epoch_subset_size", 0)),
    )

    save_stride = Int(get(sampling_cfg, "save_stride", 20))
    sample_dt = Float64(get(sampling_cfg, "sample_dt", Float64(get(sampling_cfg, "dt", 0.001)) * save_stride))
    sampling_config = LangevinConfig(
        dt=Float64(get(sampling_cfg, "dt", 0.001)),
        sample_dt=sample_dt,
        nsteps=Int(get(sampling_cfg, "steps", 40_000)),
        resolution=save_stride,
        n_ensembles=Int(get(sampling_cfg, "chains", 256)),
        burn_in=Int(get(sampling_cfg, "burnin_steps", 4_000)),
        nbins=Int(get(figure_cfg, "pdf_bins", 120)),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        seed=Int(get(sampling_cfg, "seed", Int(get(training_cfg, "seed", 20260427)) + 101)),
        progress=Bool(get(sampling_cfg, "progress", true)),
    )

    params = L96ScoreParams(
        String(data_cfg["input_hdf5"]),
        Float64(data_cfg["burnin_fraction"]),
        Int(get(data_cfg, "data_every", 1)),
        Int(get(data_cfg, "max_samples", 0)),
        Bool(get(data_cfg, "shared_normalization", true)),
        model_config,
        Int(get(model_cfg, "init_seed", 314159)),
        trainer_config,
        sampling_config,
        Int.(get(figure_cfg, "pair_offsets", [1, 2, 5, 10])),
        Int(get(figure_cfg, "pdf_bins", 120)),
        Int(get(figure_cfg, "max_pdf_samples", 400_000)),
        Int(get(figure_cfg, "stein_samples", 20_000)),
        Int(get(figure_cfg, "spectrum_samples", 200_000)),
        Int(get(figure_cfg, "width", 3200)),
        Int(get(figure_cfg, "height", 2600)),
        String(output_cfg["model_bson"]),
        String(output_cfg["figure_png"]),
        String(get(run_cfg, "device", "AUTO")),
        Bool(get(run_cfg, "verbose", true)),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.data_every >= 1, "data_every must be >= 1.")
    require_condition(params.max_samples >= 0, "max_samples must be nonnegative.")
    require_condition(params.model_config.in_channels == 1, "The L96 score network expects in_channels = 1.")
    require_condition(params.model_config.periodic, "L96 score training should use periodic convolutions.")
    require_condition(!isempty(params.pair_offsets), "pair_offsets must not be empty.")
    require_condition(all(offset -> offset >= 1, params.pair_offsets), "pair_offsets must be positive.")
    require_condition(params.figure_width >= 1800 && params.figure_height >= 1400, "Figure dimensions are too small.")
    return params
end

function parameter_norm(model)
    total = 0.0
    for p in Flux.trainables(model)
        total += sum(abs2, to_host(p))
    end
    return sqrt(total)
end

function mean_score_norm(model, batch, sigma::Float32)
    scores = score_from_model(model, batch, sigma)
    flat_scores = reshape(scores, :, size(scores, 3))
    norms = sqrt.(sum(abs2, flat_scores; dims=1))
    return Float64(mean(to_host(norms)))
end

function shared_data_stats(samples::Array{Float32, 3})
    mean_value64 = mean(Float64, samples)
    mean_value = Float32(mean_value64)
    n = length(samples)
    variance = mean(x -> abs2(Float64(x) - mean_value64), samples) * n / max(n - 1, 1)
    std_value = Float32(sqrt(variance))
    std_value = max(std_value, sqrt(eps(Float32)))
    L = size(samples, 1)
    return DataStats(fill(mean_value, 1, L), fill(std_value, 1, L))
end

function sitewise_data_stats(samples::Array{Float32, 3})
    L = size(samples, 1)
    means = Array{Float32}(undef, 1, L)
    stds = Array{Float32}(undef, 1, L)
    for idx in 1:L
        slice = @view samples[idx, 1, :]
        mean_value64 = mean(Float64, slice)
        means[1, idx] = Float32(mean_value64)
        n = length(slice)
        variance = mean(x -> abs2(Float64(x) - mean_value64), slice) * n / max(n - 1, 1)
        stds[1, idx] = max(Float32(sqrt(variance)), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function apply_stats(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return (samples .- mean_tensor) ./ std_tensor
end

function denormalize_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return samples .* std_tensor .+ mean_tensor
end

function flatten_l96_samples(states::Array{Float32, 3}, start_idx::Int, data_every::Int)
    nt, K, ntraj = size(states)
    nsamples = length(start_idx:data_every:nt) * ntraj
    tensor = Array{Float32}(undef, K, 1, nsamples)
    cursor = 1
    @inbounds for traj_idx in 1:ntraj
        for time_idx in start_idx:data_every:nt
            tensor[:, 1, cursor] .= states[time_idx, :, traj_idx]
            cursor += 1
        end
    end
    return tensor
end

function subset_tensor(tensor::Array{Float32, 3}, max_samples::Int, rng::AbstractRNG)
    max_samples <= 0 && return tensor
    total = size(tensor, 3)
    total <= max_samples && return tensor
    keep = randperm(rng, total)[1:max_samples]
    return tensor[:, :, keep]
end

function load_training_dataset(path::AbstractString, burnin_fraction::Float64, data_every::Int,
        max_samples::Int, shared_normalization::Bool, rng::AbstractRNG)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    require_condition(ndims(states) == 3, "Expected /trajectories/states to be rank 3.")
    nt, K, ntraj = size(states)
    require_condition(K >= 4, "Expected an L96 state dimension >= 4.")
    require_condition(nt == length(times), "Time axis length does not match state tensor.")

    start_idx = burnin_start_index(nt, burnin_fraction)
    raw_tensor = flatten_l96_samples(states, start_idx, data_every)
    raw_tensor = subset_tensor(raw_tensor, max_samples, rng)
    stats = shared_normalization ? shared_data_stats(raw_tensor) : sitewise_data_stats(raw_tensor)
    normalized = apply_stats(raw_tensor, stats)
    return NormalizedDataset(normalized, stats), times, states, start_idx
end

function select_sample_columns(matrix::AbstractMatrix{<:Real}, max_samples::Int, rng::AbstractRNG)
    total = size(matrix, 2)
    max_samples <= 0 && return matrix
    total <= max_samples && return matrix
    keep = randperm(rng, total)[1:max_samples]
    return matrix[:, keep]
end

function keep_finite_sample_columns(matrix::AbstractMatrix{<:Real})
    finite_mask = BitVector(undef, size(matrix, 2))
    @inbounds for col_idx in axes(matrix, 2)
        finite_mask[col_idx] = all(isfinite, @view matrix[:, col_idx])
    end
    kept = count(identity, finite_mask)
    kept > 0 || error("Score-SDE sampling produced no finite snapshots for diagnostics.")
    total = length(finite_mask)
    if kept < total
        @warn "Dropping non-finite score-SDE snapshots before diagnostics" kept total dropped=(total - kept)
    end
    return matrix[:, finite_mask], kept, total
end

function power_spectrum(samples::AbstractMatrix{<:Real}; max_samples::Int=0, rng::AbstractRNG=MersenneTwister(0))
    data = select_sample_columns(samples, max_samples, rng)
    K = size(data, 1)
    nmodes = fld(K, 2) + 1
    accum = zeros(Float64, nmodes)
    scratch = Vector{ComplexF64}(undef, nmodes)
    for col_idx in 1:size(data, 2)
        scratch .= rfft(Float64.(data[:, col_idx]))
        accum .+= abs2.(scratch) ./ K
    end
    accum ./= max(size(data, 2), 1)
    return collect(0:(nmodes - 1)), accum
end

function read_pair_pdf(path::AbstractString, offset::Int)
    base = @sprintf("/statistics/pdf/bivariate/offset_%d", offset)
    x_grid = Float64.(h5read(path, string(base, "/x_grid")))
    y_grid = Float64.(h5read(path, string(base, "/y_grid")))
    density = Float64.(h5read(path, string(base, "/density")))
    return PairPdfResult(offset, x_grid, y_grid, density)
end

function load_observed_reference(path::AbstractString, burnin_fraction::Float64, pair_offsets::Vector{Int},
        spectrum_samples::Int, seed::Int)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    start_idx = burnin_start_index(length(times), burnin_fraction)
    post_states = states[start_idx:end, :, :]
    flat_post_states = reshape(permutedims(post_states, (2, 1, 3)), size(states, 2), :)
    mean_value = Float64(mean(flat_post_states))
    std_value = Float64(std(flat_post_states))
    spectrum_modes, spectrum = power_spectrum(flat_post_states; max_samples=spectrum_samples,
        rng=MersenneTwister(seed + 17))

    pair_pdfs = PairPdfResult[]
    for offset in pair_offsets
        push!(pair_pdfs, read_pair_pdf(path, offset))
    end

    return L96ObservedReference(
        times,
        post_states,
        flat_post_states,
        Float64.(h5read(path, "/statistics/pdf/univariate_centers")),
        Float64.(h5read(path, "/statistics/pdf/univariate_density")),
        pair_pdfs,
        Float64(h5read(path, "/statistics/correlations/t_decorrelation")),
        mean_value,
        std_value,
        spectrum_modes,
        spectrum,
    )
end

function draw_univariate_samples(samples::AbstractMatrix{<:Real}, max_samples::Int, rng::AbstractRNG)
    total = length(samples)
    max_samples <= 0 || max_samples >= total && return Float64.(vec(samples))
    values = Vector{Float64}(undef, max_samples)
    K, nsnaps = size(samples)
    for sample_idx in 1:max_samples
        linear = rand(rng, 0:(total - 1))
        mode_idx = (linear % K) + 1
        snap_idx = (linear ÷ K) + 1
        values[sample_idx] = samples[mode_idx, snap_idx]
    end
    return values
end

function draw_pair_samples(samples::AbstractMatrix{<:Real}, offset::Int, max_samples::Int, rng::AbstractRNG)
    K, nsnaps = size(samples)
    total = K * nsnaps
    if max_samples <= 0 || max_samples >= total
        x_values = Vector{Float64}(undef, total)
        y_values = Vector{Float64}(undef, total)
        cursor = 1
        @inbounds for snap_idx in 1:nsnaps
            for mode_idx in 1:K
                paired_idx = mod1(mode_idx + offset, K)
                x_values[cursor] = samples[mode_idx, snap_idx]
                y_values[cursor] = samples[paired_idx, snap_idx]
                cursor += 1
            end
        end
        return x_values, y_values
    end

    x_values = Vector{Float64}(undef, max_samples)
    y_values = Vector{Float64}(undef, max_samples)
    for sample_idx in 1:max_samples
        linear = rand(rng, 0:(total - 1))
        mode_idx = (linear % K) + 1
        snap_idx = (linear ÷ K) + 1
        paired_idx = mod1(mode_idx + offset, K)
        x_values[sample_idx] = samples[mode_idx, snap_idx]
        y_values[sample_idx] = samples[paired_idx, snap_idx]
    end
    return x_values, y_values
end

function kde_on_grid_1d(values::AbstractVector{<:Real}, grid::Vector{Float64})
    result = kde(Float64.(values); npoints=length(grid), boundary=(grid[1], grid[end]))
    return Float64.(result.density)
end

function kde_on_grid_2d(x_values::AbstractVector{<:Real}, y_values::AbstractVector{<:Real},
        x_grid::Vector{Float64}, y_grid::Vector{Float64})
    result = kde((Float64.(x_values), Float64.(y_values));
        npoints=(length(x_grid), length(y_grid)),
        boundary=((x_grid[1], x_grid[end]), (y_grid[1], y_grid[end])))
    return Float64.(result.density)
end

function grid_spacing(grid::Vector{Float64})
    return length(grid) > 1 ? (grid[2] - grid[1]) : 1.0
end

function kl_divergence_from_density_1d(p_density::Vector{Float64}, q_density::Vector{Float64}, width::Float64)
    eps_value = 1e-12
    p = p_density .* width
    q = q_density .* width
    p .+= eps_value
    q .+= eps_value
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function kl_divergence_from_density_2d(p_density::Matrix{Float64}, q_density::Matrix{Float64}, x_width::Float64, y_width::Float64)
    eps_value = 1e-12
    p = vec(p_density .* (x_width * y_width))
    q = vec(q_density .* (x_width * y_width))
    p .+= eps_value
    q .+= eps_value
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function stein_matrix(model, dataset::NormalizedDataset, sigma::Float32, nsamples::Int, rng::AbstractRNG,
        device::ExecutionDevice)
    max_keep = min(nsamples, length(dataset))
    keep = max_keep < length(dataset) ? randperm(rng, length(dataset))[1:max_keep] : collect(1:length(dataset))
    clean_batch = dataset.data[:, :, keep]
    noisy_batch = clean_batch .+ sigma .* randn(rng, Float32, size(clean_batch))
    device_batch = move_array(noisy_batch, device)
    scores = score_from_model(model, device_batch, sigma)
    dim = size(clean_batch, 1) * size(clean_batch, 2)
    score_flat = reshape(to_host(scores), dim, size(clean_batch, 3))
    noisy_flat = reshape(noisy_batch, dim, size(clean_batch, 3))
    return -(score_flat * noisy_flat') ./ size(clean_batch, 3)
end

function sample_score_sde(model, dataset::NormalizedDataset, cfg::LangevinConfig, device::ExecutionDevice)
    L = size(dataset.data, 1)
    C = size(dataset.data, 2)
    dim = L * C
    nens = max(cfg.n_ensembles, 1)
    rng = MersenneTwister(cfg.seed)
    x0 = Matrix{Float32}(undef, dim, nens)
    for ens_idx in 1:nens
        sample_idx = rand(rng, 1:length(dataset))
        x0[:, ens_idx] .= reshape(dataset.data[:, :, sample_idx], dim)
    end

    wrapper = ScoreWrapper(model, cfg.sigma, L, C, dim)
    phi = Matrix{Float32}(I, dim, dim)
    sigma_mat = Matrix{Float32}(I, dim, dim)
    traj = EnsembleIntegrator.evolve_sde_snapshots(wrapper, x0, phi, sigma_mat;
        dt=cfg.dt,
        n_steps=cfg.nsteps,
        burn_in=cfg.burn_in,
        resolution=cfg.resolution,
        device=is_gpu(device) ? "gpu" : "cpu",
        boundary=nothing,
        progress=cfg.progress,
        progress_desc="L96 score Langevin")
    return reshape(traj, L, C, :)
end

function train_model(dataset::NormalizedDataset, params::L96ScoreParams, device::ExecutionDevice)
    Random.seed!(params.model_init_seed)
    model_cfg = adjust_model_config_for_length(params.model_config, size(dataset.data, 1))
    model = build_unet(model_cfg)
    Random.seed!()
    model = move_model(model, device)

    monitor_count = min(length(dataset), 4096)
    monitor_batch = dataset.data[:, :, 1:monitor_count]
    monitor_device = move_array(monitor_batch, device)
    score_norm_history = Float64[]
    param_norm_history = Float64[]
    epoch_times = Float64[]

    training_history = train!(model, dataset, params.trainer_config;
        device=device,
        epoch_callback=(epoch, current_model, epoch_time) -> begin
            push!(score_norm_history, mean_score_norm(current_model, monitor_device, params.trainer_config.sigma))
            push!(param_norm_history, parameter_norm(current_model))
            push!(epoch_times, epoch_time)
        end)

    history = Dict(
        :train_loss => Float64.(training_history.epoch_losses),
        :score_norm => score_norm_history,
        :param_norm => param_norm_history,
        :epoch_time => epoch_times,
    )
    return model, history
end

function compute_generated_pair_pdfs(samples::AbstractMatrix{<:Real}, observed_pairs::Vector{PairPdfResult},
        max_pdf_samples::Int, seed::Int)
    results = PairPdfResult[]
    for pair in observed_pairs
        rng = MersenneTwister(seed + 1000 + pair.offset)
        x_values, y_values = draw_pair_samples(samples, pair.offset, max_pdf_samples, rng)
        density = kde_on_grid_2d(x_values, y_values, pair.x_grid, pair.y_grid)
        push!(results, PairPdfResult(pair.offset, pair.x_grid, pair.y_grid, density))
    end
    return results
end

function compute_diagnostics(model, dataset::NormalizedDataset, observed::L96ObservedReference,
        history::Dict{Symbol, Vector{Float64}}, params::L96ScoreParams, device::ExecutionDevice)
    Flux.testmode!(model)

    stein_rng = MersenneTwister(params.trainer_config.seed + 1)
    stein_mat = stein_matrix(model, dataset, params.trainer_config.sigma, params.stein_samples, stein_rng, device)
    ident = Matrix{Float64}(I, size(stein_mat, 1), size(stein_mat, 2))
    stein_relative_error = norm(stein_mat - ident) / norm(ident)

    generated_norm_states = sample_score_sde(model, dataset, params.sampling_config, device)
    generated_states = denormalize_tensor(generated_norm_states, dataset.stats)
    generated_matrix = reshape(generated_states, size(generated_states, 1), :)
    generated_matrix, finite_generated_snapshots, total_generated_snapshots = keep_finite_sample_columns(generated_matrix)
    generated_states = reshape(Float32.(generated_matrix), size(dataset.data, 1), size(dataset.data, 2), :)

    uni_rng = MersenneTwister(params.trainer_config.seed + 2)
    uni_values = draw_univariate_samples(generated_matrix, params.max_pdf_samples, uni_rng)
    generated_univariate_density = kde_on_grid_1d(uni_values, observed.univariate_centers)
    univariate_kl = kl_divergence_from_density_1d(
        observed.univariate_density,
        generated_univariate_density,
        grid_spacing(observed.univariate_centers),
    )

    generated_pair_pdfs = compute_generated_pair_pdfs(generated_matrix, observed.pair_pdfs,
        params.max_pdf_samples, params.trainer_config.seed + 3)
    pair_kls = Float64[]
    for (pair_obs, pair_gen) in zip(observed.pair_pdfs, generated_pair_pdfs)
        push!(pair_kls, kl_divergence_from_density_2d(
            pair_obs.density,
            pair_gen.density,
            grid_spacing(pair_obs.x_grid),
            grid_spacing(pair_obs.y_grid),
        ))
    end

    all_kls = [univariate_kl; pair_kls]
    mean_kl = mean(all_kls)
    pdf_accuracy = exp(-mean_kl)

    spectrum_modes, generated_spectrum = power_spectrum(generated_matrix;
        max_samples=params.spectrum_samples,
        rng=MersenneTwister(params.trainer_config.seed + 4))
    spectrum_relative_error = norm(generated_spectrum - observed.spectrum) / max(norm(observed.spectrum), eps(Float64))

    return L96Diagnostics(
        history,
        stein_mat,
        stein_relative_error,
        generated_univariate_density,
        generated_pair_pdfs,
        univariate_kl,
        pair_kls,
        mean_kl,
        pdf_accuracy,
        spectrum_modes,
        generated_spectrum,
        spectrum_relative_error,
        finite_generated_snapshots,
        total_generated_snapshots,
        generated_states,
    )
end

function summary_lines(params::L96ScoreParams, dataset::NormalizedDataset, observed::L96ObservedReference,
        diagnostics::L96Diagnostics)
    observed_matrix = observed.flat_post_states
    generated_matrix = reshape(diagnostics.generated_states, size(diagnostics.generated_states, 1), :)
    data_mean = Float64(mean(observed_matrix))
    data_std = Float64(std(observed_matrix))
    gen_mean = Float64(mean(generated_matrix))
    gen_std = Float64(std(generated_matrix))
    lines = String[
        @sprintf("K = %d", size(dataset.data, 1)),
        @sprintf("train samples = %d", length(dataset)),
        @sprintf("gen samples = %d", size(generated_matrix, 2)),
        @sprintf("finite gen snaps = %d / %d", diagnostics.finite_generated_snapshots, diagnostics.total_generated_snapshots),
        @sprintf("sigma = %.3f", params.trainer_config.sigma),
        @sprintf("epochs = %d", params.trainer_config.epochs),
        @sprintf("batch_size = %d", params.trainer_config.batch_size),
        @sprintf("lr = %.2e", params.trainer_config.lr),
        @sprintf("final DSM loss = %.3e", diagnostics.history[:train_loss][end]),
        @sprintf("smoothed Stein rel. err. = %.3e", diagnostics.stein_relative_error),
        @sprintf("KL 1-point = %.3e", diagnostics.univariate_kl),
        @sprintf("mean pair KL = %.3e", mean(diagnostics.pair_kls)),
        @sprintf("mean KL = %.3e", diagnostics.mean_kl),
        @sprintf("pdf accuracy = %.6f", diagnostics.pdf_accuracy),
        @sprintf("spectrum rel. err. = %.3e", diagnostics.spectrum_relative_error),
        @sprintf("t_dec(data) = %.3f", observed.decorrelation_time),
        @sprintf("data mean/std = %.3f / %.3f", data_mean, data_std),
        @sprintf("gen mean/std = %.3f / %.3f", gen_mean, gen_std),
        params.shared_normalization ? "normalization = shared scalar" : "normalization = sitewise",
    ]
    return lines
end

function pair_kl_lines(observed_pairs::Vector{PairPdfResult}, pair_kls::Vector{Float64})
    return [@sprintf("r=%d  KL = %.3e", pair.offset, kl) for (pair, kl) in zip(observed_pairs, pair_kls)]
end

function plot_pair_panel!(slot, pair::PairPdfResult; title::AbstractString, colorrange::Tuple{Float64, Float64})
    ax = Axis(slot; title=title, xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pair.offset), aspect=DataAspect())
    heatmap!(ax, pair.x_grid, pair.y_grid, pair.density; colormap=STYLE_SEQUENTIAL_BLUE, colorrange=colorrange)
    return ax
end

function create_diagnostics_figure(output_path::AbstractString, params::L96ScoreParams,
        dataset::NormalizedDataset, observed::L96ObservedReference, diagnostics::L96Diagnostics)
    pair_count = min(length(observed.pair_pdfs), 4)
    observed_pairs = observed.pair_pdfs[1:pair_count]
    generated_pairs = diagnostics.generated_pair_pdfs[1:pair_count]
    pair_density_max = 0.0
    for pair in observed_pairs
        pair_density_max = max(pair_density_max, maximum(pair.density))
    end
    for pair in generated_pairs
        pair_density_max = max(pair_density_max, maximum(pair.density))
    end
    pair_density_max = max(pair_density_max, 1e-9)

    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf(
            "K=%d   train=%d   gen=%d   sigma=%.3f   t_dec(data)=%.3f   mean KL=%.3e",
            size(dataset.data, 1),
            length(dataset),
            size(diagnostics.generated_states, 3),
            params.trainer_config.sigma,
            observed.decorrelation_time,
            diagnostics.mean_kl,
        )
        figure_title!(fig, "L96 stationary score training diagnostics"; subtitle=subtitle)

        epochs = collect(1:length(diagnostics.history[:train_loss]))

        ax_loss = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax_loss, epochs, diagnostics.history[:train_loss]; color=STYLE_PRIMARY)

        ax_score = Axis(fig[1, 2]; title="Mean score norm", xlabel="epoch", ylabel="norm")
        lines!(ax_score, epochs, diagnostics.history[:score_norm]; color=STYLE_ACCENT)

        ax_param = Axis(fig[1, 3]; title="Parameter norm", xlabel="epoch", ylabel="norm")
        lines!(ax_param, epochs, diagnostics.history[:param_norm]; color=STYLE_HIGHLIGHT)

        ax_spec = Axis(fig[1, 4]; title="Mean power spectrum", xlabel="Fourier mode", ylabel="power")
        lines!(ax_spec, observed.spectrum_modes, observed.spectrum; color=STYLE_REFERENCE, label="data")
        lines!(ax_spec, diagnostics.spectrum_modes, diagnostics.generated_spectrum; color=STYLE_SECONDARY, linestyle=:dash, label="score SDE")
        axislegend(ax_spec; position=:rt)

        stein_clim = max(maximum(abs.(diagnostics.stein_matrix .- Matrix{Float64}(I, size(diagnostics.stein_matrix, 1), size(diagnostics.stein_matrix, 2)))), 1e-6)
        ax_stein = Axis(fig[2, 1]; title="Smoothed Stein matrix minus identity", xlabel="j", ylabel="i")
        hm_stein = heatmap!(ax_stein, 1:size(diagnostics.stein_matrix, 2), 1:size(diagnostics.stein_matrix, 1), diagnostics.stein_matrix .- Matrix{Float64}(I, size(diagnostics.stein_matrix, 1), size(diagnostics.stein_matrix, 2));
            colormap=STYLE_DIVERGING_SOFT, colorrange=(-stein_clim, stein_clim))
        Colorbar(fig[2, 1, Right()], hm_stein; label="Stein[p_sigma] - I")

        ax_pdf = Axis(fig[2, 2]; title="Translation-averaged 1-point PDF", xlabel="x_i", ylabel="density")
        lines!(ax_pdf, observed.univariate_centers, observed.univariate_density; color=STYLE_REFERENCE, label="data")
        lines!(ax_pdf, observed.univariate_centers, diagnostics.generated_univariate_density; color=STYLE_PRIMARY, linestyle=:dash, label="score SDE")
        axislegend(ax_pdf; position=:rt)

        plot_pair_panel!(fig[2, 3], observed_pairs[1];
            title=@sprintf("Data pair PDF, r=%d", observed_pairs[1].offset),
            colorrange=(0.0, pair_density_max))
        plot_pair_panel!(fig[2, 4], generated_pairs[1];
            title=@sprintf("Score-SDE pair PDF, r=%d", generated_pairs[1].offset),
            colorrange=(0.0, pair_density_max))

        if pair_count >= 2
            plot_pair_panel!(fig[3, 1], observed_pairs[2];
                title=@sprintf("Data pair PDF, r=%d", observed_pairs[2].offset),
                colorrange=(0.0, pair_density_max))
            plot_pair_panel!(fig[3, 2], generated_pairs[2];
                title=@sprintf("Score-SDE pair PDF, r=%d", generated_pairs[2].offset),
                colorrange=(0.0, pair_density_max))
        end
        if pair_count >= 3
            plot_pair_panel!(fig[3, 3], observed_pairs[3];
                title=@sprintf("Data pair PDF, r=%d", observed_pairs[3].offset),
                colorrange=(0.0, pair_density_max))
            plot_pair_panel!(fig[3, 4], generated_pairs[3];
                title=@sprintf("Score-SDE pair PDF, r=%d", generated_pairs[3].offset),
                colorrange=(0.0, pair_density_max))
        end
        if pair_count >= 4
            plot_pair_panel!(fig[4, 1], observed_pairs[4];
                title=@sprintf("Data pair PDF, r=%d", observed_pairs[4].offset),
                colorrange=(0.0, pair_density_max))
            plot_pair_panel!(fig[4, 2], generated_pairs[4];
                title=@sprintf("Score-SDE pair PDF, r=%d", generated_pairs[4].offset),
                colorrange=(0.0, pair_density_max))
        end

        ax_pair_kl = Axis(fig[4, 3]; title="Pair relative entropy", xlabel="offset", ylabel="KL")
        barplot!(ax_pair_kl, 1:pair_count, diagnostics.pair_kls[1:pair_count]; color=STYLE_SECONDARY)
        ax_pair_kl.xticks = (1:pair_count, [string(pair.offset) for pair in observed_pairs])

        text_panel!(fig[4, 4], vcat(summary_lines(params, dataset, observed, diagnostics), "", pair_kl_lines(observed_pairs, diagnostics.pair_kls[1:pair_count]));
            title="Diagnostic summary")

        save_figure(output_path, fig)
    end
    return nothing
end

function save_model(path::AbstractString, model, dataset::NormalizedDataset, params::L96ScoreParams,
        diagnostics::L96Diagnostics)
    host_model = cpu(model)
    metadata = Dict(
        :burnin_fraction => params.burnin_fraction,
        :data_every => params.data_every,
        :max_samples => params.max_samples,
        :pair_offsets => params.pair_offsets,
        :shared_normalization => params.shared_normalization,
        :stein_relative_error => diagnostics.stein_relative_error,
        :univariate_kl => diagnostics.univariate_kl,
        :pair_kls => diagnostics.pair_kls,
        :mean_kl => diagnostics.mean_kl,
        :pdf_accuracy => diagnostics.pdf_accuracy,
        :spectrum_relative_error => diagnostics.spectrum_relative_error,
        :finite_generated_snapshots => diagnostics.finite_generated_snapshots,
        :total_generated_snapshots => diagnostics.total_generated_snapshots,
    )
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    history = diagnostics.history
    stein_matrix = diagnostics.stein_matrix
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history stein_matrix
    return nothing
end

function save_training_checkpoint(path::AbstractString, model, dataset::NormalizedDataset,
        params::L96ScoreParams, history::Dict{Symbol, Vector{Float64}})
    host_model = cpu(model)
    metadata = Dict(
        :burnin_fraction => params.burnin_fraction,
        :data_every => params.data_every,
        :max_samples => params.max_samples,
        :pair_offsets => params.pair_offsets,
        :shared_normalization => params.shared_normalization,
        :checkpoint => true,
    )
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history
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

    require_condition(isfile(input_hdf5), "Input HDF5 file not found: $(input_hdf5)")
    device = detect_device(params.device_name)
    activate_device!(device)

    @printf("Training device request: %s\n", params.device_name)
    @printf("Resolved execution device: %s\n", describe_device(device))
    rng = MersenneTwister(params.trainer_config.seed)
    dataset, _, _, _ = load_training_dataset(input_hdf5, params.burnin_fraction, params.data_every,
        params.max_samples, params.shared_normalization, rng)
    observed = load_observed_reference(input_hdf5, params.burnin_fraction, params.pair_offsets,
        params.spectrum_samples, params.trainer_config.seed)

    @printf("Training samples: %d\n", length(dataset))
    @printf("Observed post-burnin snapshots: %d\n", size(observed.flat_post_states, 2))
    @printf("Observed decorrelation time: %.6f\n", observed.decorrelation_time)

    model, history = train_model(dataset, params, device)
    checkpoint_path = string(output_bson, ".checkpoint")
    @printf("Saving training checkpoint to %s\n", checkpoint_path)
    save_training_checkpoint(checkpoint_path, model, dataset, params, history)

    diagnostics = compute_diagnostics(model, dataset, observed, history, params, device)

    @printf("Saving model to %s\n", output_bson)
    save_model(output_bson, model, dataset, params, diagnostics)
    rm(checkpoint_path; force=true)

    @printf("Saving diagnostics figure to %s\n", output_png)
    create_diagnostics_figure(output_png, params, dataset, observed, diagnostics)

    @printf("Done. Final DSM loss = %.6e, mean KL = %.6e, pdf accuracy = %.6f\n",
        diagnostics.history[:train_loss][end], diagnostics.mean_kl, diagnostics.pdf_accuracy)
    return diagnostics
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
