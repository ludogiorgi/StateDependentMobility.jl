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

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
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
const STATE_CHANNELS = 2

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
    x32 = input_type <: Float32 ? x : Float32.(x)
    scores = score_from_model(sw.model, reshape(x32, sw.L, sw.C, batch), sw.sigma)
    out = reshape(scores, sw.dim, batch)
    return input_type <: Float32 ? out : input_type.(out)
end

struct PairPdfResult
    label::String
    offset::Int
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct ObservedReference
    times::Vector{Float64}
    post_states::Array{Float32, 4}
    sample_tensor::Array{Float32, 3}
    q_centers::Vector{Float64}
    q_density::Vector{Float64}
    p_centers::Vector{Float64}
    p_density::Vector{Float64}
    amp_centers::Vector{Float64}
    amp_density::Vector{Float64}
    pair_pdfs::Vector{PairPdfResult}
    decorrelation_time::Float64
    spectrum_modes::Vector{Int}
    spectrum_q::Vector{Float64}
    spectrum_p::Vector{Float64}
end

struct ScoreParams
    input_hdf5::String
    burnin_fraction::Float64
    data_every::Int
    max_samples::Int
    shared_normalization::Bool
    model_config::ScoreUNetConfig
    model_normalization::Symbol
    model_init_seed::Int
    warm_start_bson::String
    trainer_config::ScoreTrainerConfig
    sampling_config::LangevinConfig
    pair_offsets::Vector{Int}
    pdf_bins::Int
    max_pdf_samples::Int
    stein_samples::Int
    exact_score_samples::Int
    spectrum_samples::Int
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    device_name::String
    run_evaluation::Bool
    verbose::Bool
end

struct ScoreDiagnostics
    history::Dict{Symbol, Vector{Float64}}
    stein_matrix::Matrix{Float64}
    stein_relative_error::Float64
    exact_score_relative_rmse::Float64
    exact_score_cosine::Float64
    generated_q_density::Vector{Float64}
    generated_p_density::Vector{Float64}
    generated_amp_density::Vector{Float64}
    generated_pair_pdfs::Vector{PairPdfResult}
    q_kl::Float64
    p_kl::Float64
    amp_kl::Float64
    pair_kls::Vector{Float64}
    mean_kl::Float64
    pdf_accuracy::Float64
    spectrum_modes::Vector{Int}
    generated_spectrum_q::Vector{Float64}
    generated_spectrum_p::Vector{Float64}
    spectrum_relative_error::Float64
    finite_generated_snapshots::Int
    total_generated_snapshots::Int
    generated_states::Array{Float32, 3}
end

function activation_from_string(name::AbstractString)
    lname = lowercase(name)
    lname == "swish" && return Flux.swish
    lname == "gelu" && return Flux.gelu
    lname == "relu" && return Flux.relu
    lname == "tanh" && return tanh
    lname == "identity" && return identity
    error("Unsupported activation: $name")
end

require_condition(condition::Bool, message::String) = condition || error(message)
resolve_path(base_dir::AbstractString, path::AbstractString) = isabspath(path) ? path : normpath(joinpath(base_dir, path))
ensure_parent_dir(path::AbstractString) = (mkpath(dirname(path)); nothing)
burnin_start_index(nsaved::Int, burnin_fraction::Float64) = clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
to_host(x) = x isa AbstractArray && !(x isa Array) ? Array(x) : x

function detect_device(name::AbstractString)
    normalized = uppercase(strip(name))
    normalized == "AUTO" && return CUDA.functional() ? select_device("GPU:0") : CPUDevice()
    normalized == "GPU" && return select_device("GPU:0")
    return select_device(name)
end

describe_device(device::ExecutionDevice) = device isa GPUDevice ? "GPU:" * join(device.ids, ",") : "CPU"

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
    length(cfg.channel_multipliers) <= max_levels && return cfg
    @warn "Reducing U-Net depth for exact periodic upsampling" requested=length(cfg.channel_multipliers) compatible=max_levels length_dim
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

function normalization_from_string(name)
    s = lowercase(String(name))
    s == "batchnorm" && return :batchnorm
    s == "none" && return :none
    s == "groupnorm" && return :groupnorm
    error("Unsupported model.normalization=$(name); allowed values are batchnorm, none, groupnorm.")
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
        in_channels=Int(get(model_cfg, "in_channels", STATE_CHANNELS)),
        base_channels=Int(get(model_cfg, "base_channels", 64)),
        channel_multipliers=Int.(get(model_cfg, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(model_cfg, "kernel_size", 5)),
        periodic=Bool(get(model_cfg, "periodic", true)),
        activation=activation_from_string(get(model_cfg, "activation", "swish")),
        final_activation=activation_from_string(get(model_cfg, "final_activation", "identity")),
    )

    max_steps = get(training_cfg, "max_steps_per_epoch", nothing)
    trainer_config = ScoreTrainerConfig(
        batch_size=Int(get(training_cfg, "batch_size", 4096)),
        epochs=Int(get(training_cfg, "epochs", 80)),
        lr=Float64(get(training_cfg, "learning_rate", get(training_cfg, "lr", 3e-4))),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        shuffle=Bool(get(training_cfg, "shuffle", true)),
        seed=Int(get(training_cfg, "seed", 20260503)),
        progress=Bool(get(training_cfg, "progress", true)),
        max_steps_per_epoch=max_steps === nothing ? nothing : Int(max_steps),
        accumulation_steps=Int(get(training_cfg, "accumulation_steps", 1)),
        use_lr_schedule=Bool(get(training_cfg, "use_lr_schedule", true)),
        warmup_steps=Int(get(training_cfg, "warmup_steps", 100)),
        min_lr_factor=Float64(get(training_cfg, "min_lr_factor", 0.05)),
        epoch_subset_size=Int(get(training_cfg, "epoch_subset_size", 0)),
    )

    save_stride = Int(get(sampling_cfg, "save_stride", 20))
    sampling_config = LangevinConfig(
        dt=Float64(get(sampling_cfg, "dt", 0.001)),
        sample_dt=Float64(get(sampling_cfg, "sample_dt", Float64(get(sampling_cfg, "dt", 0.001)) * save_stride)),
        nsteps=Int(get(sampling_cfg, "steps", 40_000)),
        resolution=save_stride,
        n_ensembles=Int(get(sampling_cfg, "chains", 256)),
        burn_in=Int(get(sampling_cfg, "burnin_steps", 4_000)),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        seed=Int(get(sampling_cfg, "seed", Int(get(training_cfg, "seed", 20260503)) + 101)),
        progress=Bool(get(sampling_cfg, "progress", true)),
    )

    params = ScoreParams(
        String(data_cfg["input_hdf5"]),
        Float64(data_cfg["burnin_fraction"]),
        Int(get(data_cfg, "data_every", 1)),
        Int(get(data_cfg, "max_samples", 0)),
        Bool(get(data_cfg, "shared_normalization", true)),
        model_config,
        normalization_from_string(get(model_cfg, "normalization", "batchnorm")),
        Int(get(model_cfg, "init_seed", 314159)),
        String(get(model_cfg, "warm_start_bson", "")),
        trainer_config,
        sampling_config,
        Int.(get(figure_cfg, "pair_offsets", [0, 1, 4, 8])),
        Int(get(figure_cfg, "pdf_bins", 120)),
        Int(get(figure_cfg, "max_pdf_samples", 400_000)),
        Int(get(figure_cfg, "stein_samples", 20_000)),
        Int(get(figure_cfg, "exact_score_samples", 20_000)),
        Int(get(figure_cfg, "spectrum_samples", 200_000)),
        Int(get(figure_cfg, "width", 3200)),
        Int(get(figure_cfg, "height", 2600)),
        String(output_cfg["model_bson"]),
        String(output_cfg["figure_png"]),
        String(get(run_cfg, "device", "AUTO")),
        Bool(get(run_cfg, "evaluate", true)),
        Bool(get(run_cfg, "verbose", true)),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.data_every >= 1, "data_every must be >= 1.")
    require_condition(params.max_samples >= 0, "max_samples must be nonnegative.")
    require_condition(params.model_config.in_channels == STATE_CHANNELS, "Complex-amplitude score network expects in_channels = 2.")
    require_condition(params.model_config.periodic, "Complex-amplitude score training should use periodic convolutions.")
    require_condition(!isempty(params.pair_offsets), "pair_offsets must not be empty.")
    require_condition(all(offset -> offset >= 0, params.pair_offsets), "pair_offsets must be nonnegative.")
    return params
end

function load_potential_params(path::AbstractString)
    return h5open(path, "r") do file
        ChainPotentialParams(
            alpha=Float64(read(file["/metadata/alpha"])),
            beta=Float64(read(file["/metadata/beta"])),
            kappa=Float64(read(file["/metadata/kappa"])),
        )
    end
end

function channel_shared_data_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    means = Array{Float32}(undef, C, K)
    stds = Array{Float32}(undef, C, K)
    for c in 1:C
        vals = @view samples[:, c, :]
        mu64 = mean(Float64, vals)
        n = length(vals)
        var64 = mean(x -> abs2(Float64(x) - mu64), vals) * n / max(n - 1, 1)
        means[c, :] .= Float32(mu64)
        stds[c, :] .= max(Float32(sqrt(var64)), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function sitewise_data_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    means = Array{Float32}(undef, C, K)
    stds = Array{Float32}(undef, C, K)
    for c in 1:C, i in 1:K
        vals = @view samples[i, c, :]
        mu64 = mean(Float64, vals)
        n = length(vals)
        var64 = mean(x -> abs2(Float64(x) - mu64), vals) * n / max(n - 1, 1)
        means[c, i] = Float32(mu64)
        stds[c, i] = max(Float32(sqrt(var64)), sqrt(eps(Float32)))
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

function flatten_chain_samples(states::Array{Float32, 4}, start_idx::Int, data_every::Int)
    nt, K, C, ntraj = size(states)
    nsamples = length(start_idx:data_every:nt) * ntraj
    tensor = Array{Float32}(undef, K, C, nsamples)
    cursor = 1
    @inbounds for traj_idx in 1:ntraj
        for time_idx in start_idx:data_every:nt
            tensor[:, :, cursor] .= states[time_idx, :, :, traj_idx]
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
    require_condition(ndims(states) == 4, "Expected /trajectories/states to be rank 4: time x site x channel x trajectory.")
    nt, K, C, _ = size(states)
    require_condition(C == STATE_CHANNELS, "Expected q,p channels in /trajectories/states.")
    require_condition(nt == length(times), "Time axis length does not match state tensor.")
    start_idx = burnin_start_index(nt, burnin_fraction)
    raw_tensor = flatten_chain_samples(states, start_idx, data_every)
    raw_tensor = subset_tensor(raw_tensor, max_samples, rng)
    stats = shared_normalization ? channel_shared_data_stats(raw_tensor) : sitewise_data_stats(raw_tensor)
    normalized = apply_stats(raw_tensor, stats)
    return NormalizedDataset(normalized, stats), times, states, start_idx
end

function select_samples(tensor::Array{Float32, 3}, max_samples::Int, rng::AbstractRNG)
    max_samples <= 0 || size(tensor, 3) <= max_samples && return tensor
    keep = randperm(rng, size(tensor, 3))[1:max_samples]
    return tensor[:, :, keep]
end

function power_spectra(samples::Array{Float32, 3}; max_samples::Int=0, rng::AbstractRNG=MersenneTwister(0))
    data = select_samples(samples, max_samples, rng)
    K, _, ns = size(data)
    nmodes = fld(K, 2) + 1
    spec_q = zeros(Float64, nmodes)
    spec_p = zeros(Float64, nmodes)
    for s in 1:ns
        qhat = rfft(Float64.(@view data[:, 1, s]))
        phat = rfft(Float64.(@view data[:, 2, s]))
        spec_q .+= abs2.(qhat) ./ K
        spec_p .+= abs2.(phat) ./ K
    end
    spec_q ./= max(ns, 1)
    spec_p ./= max(ns, 1)
    return collect(0:(nmodes - 1)), spec_q, spec_p
end

function read_pair_pdf(path::AbstractString, idx::Int)
    base = @sprintf("/statistics/pdf/bivariate/pair_%02d", idx)
    return PairPdfResult(
        String(h5read(path, string(base, "/label"))),
        idx,
        Float64.(h5read(path, string(base, "/x_grid"))),
        Float64.(h5read(path, string(base, "/y_grid"))),
        Float64.(h5read(path, string(base, "/density"))),
    )
end

function load_observed_reference(path::AbstractString, burnin_fraction::Float64, spectrum_samples::Int, seed::Int)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    start_idx = burnin_start_index(length(times), burnin_fraction)
    post_states = states[start_idx:end, :, :, :]
    sample_tensor = flatten_chain_samples(post_states, 1, 1)
    modes, spec_q, spec_p = power_spectra(sample_tensor; max_samples=spectrum_samples, rng=MersenneTwister(seed + 17))
    labels = h5read(path, "/statistics/pdf/bivariate_labels")
    pair_pdfs = [read_pair_pdf(path, idx) for idx in 1:length(labels)]
    return ObservedReference(
        times,
        post_states,
        sample_tensor,
        Float64.(h5read(path, "/statistics/pdf/q_centers")),
        Float64.(h5read(path, "/statistics/pdf/q_density")),
        Float64.(h5read(path, "/statistics/pdf/p_centers")),
        Float64.(h5read(path, "/statistics/pdf/p_density")),
        Float64.(h5read(path, "/statistics/pdf/amplitude_centers")),
        Float64.(h5read(path, "/statistics/pdf/amplitude_density")),
        pair_pdfs,
        Float64(h5read(path, "/statistics/correlations/t_decorrelation")),
        modes,
        spec_q,
        spec_p,
    )
end

function draw_channel_values(samples::Array{Float32, 3}, channel::Int, max_samples::Int, rng::AbstractRNG)
    K, _, ns = size(samples)
    total = K * ns
    n = max_samples <= 0 ? total : min(max_samples, total)
    values = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        linear = rand(rng, 0:(total - 1))
        i = (linear % K) + 1
        s = (linear ÷ K) + 1
        values[j] = samples[i, channel, s]
    end
    return values
end

function draw_amplitudes(samples::Array{Float32, 3}, max_samples::Int, rng::AbstractRNG)
    K, _, ns = size(samples)
    total = K * ns
    n = max_samples <= 0 ? total : min(max_samples, total)
    values = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        linear = rand(rng, 0:(total - 1))
        i = (linear % K) + 1
        s = (linear ÷ K) + 1
        q = samples[i, 1, s]
        p = samples[i, 2, s]
        values[j] = sqrt(q * q + p * p)
    end
    return values
end

function draw_pair_samples(samples::Array{Float32, 3}, label::AbstractString, offset::Int,
        max_samples::Int, rng::AbstractRNG)
    K, _, ns = size(samples)
    total = K * ns
    n = max_samples <= 0 ? total : min(max_samples, total)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        linear = rand(rng, 0:(total - 1))
        i = (linear % K) + 1
        s = (linear ÷ K) + 1
        if startswith(label, "q_i vs q_")
            jj = mod1(i + offset, K)
            x[j] = samples[i, 1, s]
            y[j] = samples[jj, 1, s]
        elseif startswith(label, "p_i vs p_")
            jj = mod1(i + offset, K)
            x[j] = samples[i, 2, s]
            y[j] = samples[jj, 2, s]
        else
            jj = mod1(i + offset, K)
            x[j] = samples[i, 1, s]
            y[j] = samples[jj, 2, s]
        end
    end
    return x, y
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

grid_spacing(grid::Vector{Float64}) = length(grid) > 1 ? (grid[2] - grid[1]) : 1.0

function kl_divergence_from_density_1d(p_density::Vector{Float64}, q_density::Vector{Float64}, width::Float64)
    eps_value = 1e-12
    p = p_density .* width .+ eps_value
    q = q_density .* width .+ eps_value
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function kl_divergence_from_density_2d(p_density::Matrix{Float64}, q_density::Matrix{Float64}, x_width::Float64, y_width::Float64)
    eps_value = 1e-12
    p = vec(p_density .* (x_width * y_width)) .+ eps_value
    q = vec(q_density .* (x_width * y_width)) .+ eps_value
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function raw_score_batch(samples::Array{Float32, 3}, potential::ChainPotentialParams)
    K, _, B = size(samples)
    scores = Array{Float32}(undef, K, STATE_CHANNELS, B)
    @inbounds for b in 1:B, i in 1:K
        im1 = mod1(i - 1, K)
        ip1 = mod1(i + 1, K)
        q = Float64(samples[i, 1, b])
        p = Float64(samples[i, 2, b])
        r2 = q * q + p * p
        gq = potential.alpha * q + potential.beta * r2 * q +
             potential.kappa * (2.0 * q - Float64(samples[im1, 1, b]) - Float64(samples[ip1, 1, b]))
        gp = potential.alpha * p + potential.beta * r2 * p +
             potential.kappa * (2.0 * p - Float64(samples[im1, 2, b]) - Float64(samples[ip1, 2, b]))
        scores[i, 1, b] = Float32(-gq)
        scores[i, 2, b] = Float32(-gp)
    end
    return scores
end

function standardized_exact_score(normalized_samples::Array{Float32, 3}, stats::DataStats, potential::ChainPotentialParams)
    raw_samples = denormalize_tensor(normalized_samples, stats)
    raw_scores = raw_score_batch(raw_samples, potential)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(raw_scores, 1), size(raw_scores, 2), 1)
    return raw_scores .* std_tensor
end

function exact_score_diagnostics(model, dataset::NormalizedDataset, potential::ChainPotentialParams,
        sigma::Float32, nsamples::Int, rng::AbstractRNG, device::ExecutionDevice)
    max_keep = min(nsamples, length(dataset))
    keep = max_keep < length(dataset) ? randperm(rng, length(dataset))[1:max_keep] : collect(1:length(dataset))
    clean = dataset.data[:, :, keep]
    exact = standardized_exact_score(clean, dataset.stats, potential)
    pred = to_host(score_from_model(model, move_array(copy(clean), device), sigma))
    err = pred .- exact
    rel_rmse = sqrt(sum(abs2, err) / max(sum(abs2, exact), eps(Float64)))
    cosine = sum(pred .* exact) / max(sqrt(sum(abs2, pred) * sum(abs2, exact)), eps(Float64))
    return Float64(rel_rmse), Float64(cosine)
end

function stein_matrix(model, dataset::NormalizedDataset, sigma::Float32, nsamples::Int, rng::AbstractRNG,
        device::ExecutionDevice)
    max_keep = min(nsamples, length(dataset))
    keep = max_keep < length(dataset) ? randperm(rng, length(dataset))[1:max_keep] : collect(1:length(dataset))
    clean_batch = dataset.data[:, :, keep]
    noisy_batch = clean_batch .+ sigma .* randn(rng, Float32, size(clean_batch))
    scores = score_from_model(model, move_array(noisy_batch, device), sigma)
    dim = size(clean_batch, 1) * size(clean_batch, 2)
    score_flat = reshape(to_host(scores), dim, size(clean_batch, 3))
    noisy_flat = reshape(noisy_batch, dim, size(clean_batch, 3))
    return -(score_flat * noisy_flat') ./ size(clean_batch, 3)
end

function sample_score_sde(model, dataset::NormalizedDataset, cfg::LangevinConfig, device::ExecutionDevice)
    K = size(dataset.data, 1)
    C = size(dataset.data, 2)
    dim = K * C
    rng = MersenneTwister(cfg.seed)
    x0 = Matrix{Float32}(undef, dim, cfg.n_ensembles)
    for ens_idx in 1:cfg.n_ensembles
        sample_idx = rand(rng, 1:length(dataset))
        x0[:, ens_idx] .= reshape(dataset.data[:, :, sample_idx], dim)
    end
    wrapper = ScoreWrapper(model, cfg.sigma, K, C, dim)
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
        progress_desc="Complex-chain score Langevin")
    return reshape(traj, K, C, :)
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

function train_model(dataset::NormalizedDataset, params::ScoreParams, device::ExecutionDevice, base_dir::AbstractString)
    model_cfg = adjust_model_config_for_length(params.model_config, size(dataset.data, 1))
    if isempty(params.warm_start_bson)
        Random.seed!(params.model_init_seed)
        model = build_unet(model_cfg; normalization=params.model_normalization)
        Random.seed!()
    else
        warm_start_path = resolve_path(base_dir, params.warm_start_bson)
        isfile(warm_start_path) || error("Warm-start score checkpoint not found: $(warm_start_path)")
        blob = BSON.load(warm_start_path)
        model = blob[:host_model]
        @printf("Warm-starting score model from %s\n", warm_start_path)
    end
    model = move_model(model, device)
    monitor_count = min(length(dataset), 4096)
    monitor_device = move_array(dataset.data[:, :, 1:monitor_count], device)
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

function keep_finite_snapshots(samples::Array{Float32, 3})
    finite_mask = BitVector(undef, size(samples, 3))
    @inbounds for s in axes(samples, 3)
        finite_mask[s] = all(isfinite, @view samples[:, :, s])
    end
    kept = count(identity, finite_mask)
    kept > 0 || error("Score-SDE sampling produced no finite snapshots for diagnostics.")
    total = length(finite_mask)
    kept < total && @warn "Dropping non-finite generated snapshots" kept total dropped=(total - kept)
    return samples[:, :, finite_mask], kept, total
end

function compute_generated_pair_pdfs(samples::Array{Float32, 3}, observed_pairs::Vector{PairPdfResult},
        max_pdf_samples::Int, seed::Int)
    results = PairPdfResult[]
    for (idx, pair) in enumerate(observed_pairs)
        rng = MersenneTwister(seed + 1000 + idx)
        x_values, y_values = draw_pair_samples(samples, pair.label, pair.offset, max_pdf_samples, rng)
        density = kde_on_grid_2d(x_values, y_values, pair.x_grid, pair.y_grid)
        push!(results, PairPdfResult(pair.label, pair.offset, pair.x_grid, pair.y_grid, density))
    end
    return results
end

function compute_diagnostics(model, dataset::NormalizedDataset, observed::ObservedReference,
        potential::ChainPotentialParams, history::Dict{Symbol, Vector{Float64}},
        params::ScoreParams, device::ExecutionDevice)
    Flux.testmode!(model)
    stein_mat = stein_matrix(model, dataset, params.trainer_config.sigma, params.stein_samples,
        MersenneTwister(params.trainer_config.seed + 1), device)
    ident = Matrix{Float64}(I, size(stein_mat, 1), size(stein_mat, 2))
    stein_relative_error = norm(stein_mat - ident) / norm(ident)
    exact_rel, exact_cos = exact_score_diagnostics(model, dataset, potential, params.trainer_config.sigma,
        params.exact_score_samples, MersenneTwister(params.trainer_config.seed + 2), device)

    generated_norm = sample_score_sde(model, dataset, params.sampling_config, device)
    generated_states = denormalize_tensor(generated_norm, dataset.stats)
    generated_states, finite_generated_snapshots, total_generated_snapshots = keep_finite_snapshots(generated_states)

    q_vals = draw_channel_values(generated_states, 1, params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 3))
    p_vals = draw_channel_values(generated_states, 2, params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 4))
    a_vals = draw_amplitudes(generated_states, params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 5))
    generated_q_density = kde_on_grid_1d(q_vals, observed.q_centers)
    generated_p_density = kde_on_grid_1d(p_vals, observed.p_centers)
    generated_amp_density = kde_on_grid_1d(a_vals, observed.amp_centers)

    q_kl = kl_divergence_from_density_1d(observed.q_density, generated_q_density, grid_spacing(observed.q_centers))
    p_kl = kl_divergence_from_density_1d(observed.p_density, generated_p_density, grid_spacing(observed.p_centers))
    amp_kl = kl_divergence_from_density_1d(observed.amp_density, generated_amp_density, grid_spacing(observed.amp_centers))

    shown_pairs = observed.pair_pdfs[1:min(length(observed.pair_pdfs), length(params.pair_offsets))]
    generated_pair_pdfs = compute_generated_pair_pdfs(generated_states, shown_pairs, params.max_pdf_samples,
        params.trainer_config.seed + 6)
    pair_kls = [kl_divergence_from_density_2d(obs.density, gen.density, grid_spacing(obs.x_grid), grid_spacing(obs.y_grid))
                for (obs, gen) in zip(shown_pairs, generated_pair_pdfs)]

    all_kls = [q_kl, p_kl, amp_kl, pair_kls...]
    mean_kl = mean(all_kls)
    pdf_accuracy = exp(-mean_kl)
    modes, gen_spec_q, gen_spec_p = power_spectra(generated_states; max_samples=params.spectrum_samples,
        rng=MersenneTwister(params.trainer_config.seed + 7))
    spec_err = sqrt(sum(abs2, gen_spec_q - observed.spectrum_q) + sum(abs2, gen_spec_p - observed.spectrum_p)) /
               max(sqrt(sum(abs2, observed.spectrum_q) + sum(abs2, observed.spectrum_p)), eps(Float64))

    return ScoreDiagnostics(history, stein_mat, stein_relative_error, exact_rel, exact_cos,
        generated_q_density, generated_p_density, generated_amp_density, generated_pair_pdfs,
        q_kl, p_kl, amp_kl, pair_kls, mean_kl, pdf_accuracy, modes, gen_spec_q, gen_spec_p,
        spec_err, finite_generated_snapshots, total_generated_snapshots, generated_states)
end

function plot_pair_heatmap!(slot, pair::PairPdfResult; title::AbstractString, colorrange)
    ax = Axis(slot; title=title, xlabel=split(pair.label, " vs ")[1], ylabel=split(pair.label, " vs ")[end],
        aspect=DataAspect())
    heatmap!(ax, pair.x_grid, pair.y_grid, pair.density; colormap=STYLE_SEQUENTIAL_BLUE, colorrange=colorrange)
    return ax
end

function summary_lines(params::ScoreParams, dataset::NormalizedDataset, observed::ObservedReference,
        diagnostics::ScoreDiagnostics)
    data_mean_q = Float64(mean(@view observed.sample_tensor[:, 1, :]))
    data_mean_p = Float64(mean(@view observed.sample_tensor[:, 2, :]))
    gen_mean_q = Float64(mean(@view diagnostics.generated_states[:, 1, :]))
    gen_mean_p = Float64(mean(@view diagnostics.generated_states[:, 2, :]))
    return String[
        @sprintf("K = %d, channels = q,p", size(dataset.data, 1)),
        @sprintf("train samples = %d", length(dataset)),
        @sprintf("gen samples = %d", size(diagnostics.generated_states, 3)),
        @sprintf("finite gen snaps = %d / %d", diagnostics.finite_generated_snapshots, diagnostics.total_generated_snapshots),
        @sprintf("sigma = %.3f", params.trainer_config.sigma),
        @sprintf("epochs = %d", params.trainer_config.epochs),
        @sprintf("batch_size = %d", params.trainer_config.batch_size),
        @sprintf("final DSM loss = %.3e", diagnostics.history[:train_loss][end]),
        @sprintf("Stein rel. err. = %.3e", diagnostics.stein_relative_error),
        @sprintf("exact score rel. RMSE = %.3e", diagnostics.exact_score_relative_rmse),
        @sprintf("exact score cosine = %.6f", diagnostics.exact_score_cosine),
        @sprintf("KL q/p/amp = %.2e / %.2e / %.2e", diagnostics.q_kl, diagnostics.p_kl, diagnostics.amp_kl),
        @sprintf("mean pair KL = %.3e", mean(diagnostics.pair_kls)),
        @sprintf("mean KL = %.3e", diagnostics.mean_kl),
        @sprintf("pdf accuracy = %.6f", diagnostics.pdf_accuracy),
        @sprintf("spectrum rel. err. = %.3e", diagnostics.spectrum_relative_error),
        @sprintf("data mean q,p = %.3e, %.3e", data_mean_q, data_mean_p),
        @sprintf("gen mean q,p = %.3e, %.3e", gen_mean_q, gen_mean_p),
        params.shared_normalization ? "normalization = channel-shared" : "normalization = sitewise",
    ]
end

function create_diagnostics_figure(output_path::AbstractString, params::ScoreParams,
        dataset::NormalizedDataset, observed::ObservedReference, diagnostics::ScoreDiagnostics)
    pair_count = min(length(observed.pair_pdfs), length(diagnostics.generated_pair_pdfs), 4)
    observed_pairs = observed.pair_pdfs[1:pair_count]
    generated_pairs = diagnostics.generated_pair_pdfs[1:pair_count]
    pair_density_max = maximum([maximum(pair.density) for pair in vcat(observed_pairs, generated_pairs)])
    pair_density_max = max(pair_density_max, 1e-9)

    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("K=%d  train=%d  sigma=%.3f  exact RMSE=%.2e  mean KL=%.2e",
            size(dataset.data, 1), length(dataset), params.trainer_config.sigma,
            diagnostics.exact_score_relative_rmse, diagnostics.mean_kl)
        figure_title!(fig, "Complex-amplitude stationary score diagnostics"; subtitle=subtitle)

        epochs = collect(1:length(diagnostics.history[:train_loss]))
        ax_loss = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax_loss, epochs, diagnostics.history[:train_loss]; color=STYLE_PRIMARY)
        ax_score = Axis(fig[1, 2]; title="Mean score norm", xlabel="epoch", ylabel="norm")
        lines!(ax_score, epochs, diagnostics.history[:score_norm]; color=STYLE_ACCENT)
        ax_spec = Axis(fig[1, 3]; title="Power spectra", xlabel="Fourier mode", ylabel="power")
        lines!(ax_spec, observed.spectrum_modes, observed.spectrum_q; color=STYLE_REFERENCE, label="q data")
        lines!(ax_spec, diagnostics.spectrum_modes, diagnostics.generated_spectrum_q; color=STYLE_PRIMARY, linestyle=:dash, label="q gen")
        lines!(ax_spec, observed.spectrum_modes, observed.spectrum_p; color=STYLE_HIGHLIGHT, label="p data")
        lines!(ax_spec, diagnostics.spectrum_modes, diagnostics.generated_spectrum_p; color=STYLE_SECONDARY, linestyle=:dash, label="p gen")
        axislegend(ax_spec; position=:rt, nbanks=2)
        text_panel!(fig[1, 4], summary_lines(params, dataset, observed, diagnostics); title="Summary")

        ident = Matrix{Float64}(I, size(diagnostics.stein_matrix, 1), size(diagnostics.stein_matrix, 2))
        residual = diagnostics.stein_matrix - ident
        clim = max(maximum(abs.(residual)), 1e-6)
        ax_stein = Axis(fig[2, 1]; title="Smoothed Stein matrix minus identity", xlabel="j", ylabel="i")
        hm = heatmap!(ax_stein, 1:size(residual, 2), 1:size(residual, 1), residual;
            colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
        Colorbar(fig[2, 1, Right()], hm; label="Stein[p_sigma] - I")

        ax_q = Axis(fig[2, 2]; title="q marginal", xlabel="q", ylabel="density")
        lines!(ax_q, observed.q_centers, observed.q_density; color=STYLE_REFERENCE, label="data")
        lines!(ax_q, observed.q_centers, diagnostics.generated_q_density; color=STYLE_PRIMARY, linestyle=:dash, label="score SDE")
        axislegend(ax_q; position=:rt)
        ax_p = Axis(fig[2, 3]; title="p marginal", xlabel="p", ylabel="density")
        lines!(ax_p, observed.p_centers, observed.p_density; color=STYLE_REFERENCE, label="data")
        lines!(ax_p, observed.p_centers, diagnostics.generated_p_density; color=STYLE_PRIMARY, linestyle=:dash, label="score SDE")
        axislegend(ax_p; position=:rt)
        ax_a = Axis(fig[2, 4]; title="amplitude marginal", xlabel="|a|", ylabel="density")
        lines!(ax_a, observed.amp_centers, observed.amp_density; color=STYLE_REFERENCE, label="data")
        lines!(ax_a, observed.amp_centers, diagnostics.generated_amp_density; color=STYLE_PRIMARY, linestyle=:dash, label="score SDE")
        axislegend(ax_a; position=:rt)

        for idx in 1:pair_count
            row = idx <= 2 ? 3 : 4
            col = isodd(idx) ? 1 : 3
            plot_pair_heatmap!(fig[row, col], observed_pairs[idx];
                title=string("Data ", observed_pairs[idx].label), colorrange=(0.0, pair_density_max))
            plot_pair_heatmap!(fig[row, col + 1], generated_pairs[idx];
                title=@sprintf("Score SDE %s  KL=%.2e", generated_pairs[idx].label, diagnostics.pair_kls[idx]),
                colorrange=(0.0, pair_density_max))
        end
        save_figure(output_path, fig)
    end
    return nothing
end

function save_model(path::AbstractString, model, dataset::NormalizedDataset, params::ScoreParams,
        potential::ChainPotentialParams, diagnostics::ScoreDiagnostics)
    host_model = cpu(model)
    metadata = Dict(
        :system => "complex_amplitude_chain",
        :state_layout => "K x 2 x batch, channels=(q,p)",
        :burnin_fraction => params.burnin_fraction,
        :data_every => params.data_every,
        :max_samples => params.max_samples,
        :shared_normalization => params.shared_normalization,
        :model_normalization => String(params.model_normalization),
        :warm_start_bson => params.warm_start_bson,
        :stein_relative_error => diagnostics.stein_relative_error,
        :exact_score_relative_rmse => diagnostics.exact_score_relative_rmse,
        :exact_score_cosine => diagnostics.exact_score_cosine,
        :mean_kl => diagnostics.mean_kl,
        :pdf_accuracy => diagnostics.pdf_accuracy,
        :spectrum_relative_error => diagnostics.spectrum_relative_error,
    )
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    history = diagnostics.history
    stein_matrix = diagnostics.stein_matrix
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history stein_matrix potential
    return nothing
end

function save_training_checkpoint(path::AbstractString, model, dataset::NormalizedDataset,
        params::ScoreParams, history::Dict{Symbol, Vector{Float64}})
    host_model = cpu(model)
    metadata = Dict(:checkpoint => true, :system => "complex_amplitude_chain")
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history
    return nothing
end

function save_basic_model(path::AbstractString, model, dataset::NormalizedDataset,
        params::ScoreParams, potential::ChainPotentialParams, history::Dict{Symbol, Vector{Float64}})
    host_model = cpu(model)
    metadata = Dict(
        :system => "complex_amplitude_chain",
        :state_layout => "K x 2 x batch, channels=(q,p)",
        :burnin_fraction => params.burnin_fraction,
        :data_every => params.data_every,
        :max_samples => params.max_samples,
        :shared_normalization => params.shared_normalization,
        :model_normalization => String(params.model_normalization),
        :warm_start_bson => params.warm_start_bson,
        :diagnostics_skipped => true,
    )
    stats = Dict(:mean => dataset.stats.mean, :std => dataset.stats.std)
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history potential
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
    observed = load_observed_reference(input_hdf5, params.burnin_fraction, params.spectrum_samples,
        params.trainer_config.seed)
    potential = load_potential_params(input_hdf5)
    @printf("Training samples: %d, K=%d, channels=%d\n", length(dataset), size(dataset.data, 1), size(dataset.data, 2))
    @printf("Observed decorrelation time: %.6f\n", observed.decorrelation_time)

    model, history = train_model(dataset, params, device, base_dir)
    checkpoint_path = string(output_bson, ".checkpoint")
    @printf("Saving training checkpoint to %s\n", checkpoint_path)
    save_training_checkpoint(checkpoint_path, model, dataset, params, history)

    if params.run_evaluation
        diagnostics = compute_diagnostics(model, dataset, observed, potential, history, params, device)
        @printf("Saving model to %s\n", output_bson)
        save_model(output_bson, model, dataset, params, potential, diagnostics)
        rm(checkpoint_path; force=true)
        @printf("Saving diagnostics figure to %s\n", output_png)
        create_diagnostics_figure(output_png, params, dataset, observed, diagnostics)
        @printf("Done. Final DSM loss = %.6e, exact score RMSE = %.6e, exact cosine = %.6f, mean KL = %.6e, pdf accuracy = %.6f\n",
            diagnostics.history[:train_loss][end], diagnostics.exact_score_relative_rmse,
            diagnostics.exact_score_cosine, diagnostics.mean_kl, diagnostics.pdf_accuracy)
        return diagnostics
    else
        @printf("Skipping score Langevin/PDF diagnostics because run.evaluate=false.\n")
        @printf("Saving model to %s\n", output_bson)
        save_basic_model(output_bson, model, dataset, params, potential, history)
        rm(checkpoint_path; force=true)
        @printf("Done. Final DSM loss = %.6e\n", history[:train_loss][end])
        return nothing
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
