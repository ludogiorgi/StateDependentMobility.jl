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

const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
isfile(STYLE_FILE) || error("Shared figure style file not found: $(STYLE_FILE)")
const FIGURE_STYLE_LOADED = Ref(false)

function ensure_figure_support_loaded!()
    if !FIGURE_STYLE_LOADED[]
        include(STYLE_FILE)
        Base.invokelatest(() -> GLMakie.activate!())
        FIGURE_STYLE_LOADED[] = true
    end
    return nothing
end

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "joint_score.toml")
const JOINT_STATE_CHANNELS = 4
const JOINT_INPUT_CHANNELS = 5
const DEFAULT_TIME_FEATURES = "scalar"
const DEFAULT_TIME_FOURIER_FREQUENCIES = 0

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

Base.@kwdef mutable struct JointLangevinConfig
    dt::Float64 = 1e-3
    sample_dt::Float64 = 1e-2
    nsteps::Int = 20_000
    resolution::Int = 20
    n_ensembles::Int = 256
    burn_in::Int = 4_000
    sigma::Float32 = 0.1f0
    seed::Int = 21
    progress::Bool = false
end

struct JointPairSampler
    times::Vector{Float64}
    states::Array{Float32, 4}
    start_idx::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    lag_tnorm::Vector{Float32}
    tau_min::Float64
    tau_max::Float64
    decorrelation_time::Float64
end

struct PairPdfResult
    offset::Int
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct L96JointLagDiagnostics
    tau::Float64
    tnorm::Float64
    x0_kl::Float64
    xt_kl::Float64
    pair_kls::Vector{Float64}
    mean_pair_kl::Float64
    mean_kl::Float64
    accuracy::Float64
    stein_relative_error::Float64
    finite_generated_snapshots::Int
    total_generated_snapshots::Int
end

struct L96JointEvalRecord
    tau::Float64
    lag::Int
    tnorm::Float32
    x0_centers::Vector{Float64}
    observed_x0_density::Vector{Float64}
    generated_x0_density::Vector{Float64}
    xt_centers::Vector{Float64}
    observed_xt_density::Vector{Float64}
    generated_xt_density::Vector{Float64}
    observed_pair_pdfs::Vector{PairPdfResult}
    generated_pair_pdfs::Vector{PairPdfResult}
    stein_matrix::Matrix{Float64}
    diag::L96JointLagDiagnostics
end

struct L96JointScoreParams
    input_hdf5::String
    burnin_fraction::Float64
    tau_min::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    max_eval_pairs::Int
    shared_normalization::Bool
    model_config::ScoreUNetConfig
    model_normalization::Symbol
    time_features::String
    time_fourier_frequencies::Int
    include_delta_input::Bool
    model_init_seed::Int
    trainer_config::ScoreTrainerConfig
    batches_per_epoch::Int
    stein_weight::Float64
    mean_score_weight::Float64
    sampling_config::JointLangevinConfig
    eval_tau_count::Int
    pair_offsets::Vector{Int}
    pdf_bins::Int
    max_pdf_samples::Int
    stein_samples::Int
    figure_width::Int
    figure_height::Int
    output_bson::String
    output_png::String
    device_name::String
    run_evaluation::Bool
    verbose::Bool
end

struct JointScoreUNet{M}
    backbone::M
end

Functors.@functor JointScoreUNet (backbone,)

function (model::JointScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:JOINT_STATE_CHANNELS, :]
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

function normalization_from_string(name)
    s = lowercase(String(name))
    s == "batchnorm" && return :batchnorm
    s == "none" && return :none
    s == "groupnorm" && return :groupnorm
    error("Unsupported model.normalization=$(name); allowed values are batchnorm, none, groupnorm.")
end

function normalize_time_features(name)
    s = lowercase(String(name))
    s in ("scalar", "fourier") || error("Unsupported model.time_features=$(name); allowed values are scalar, fourier.")
    return s
end

time_feature_count(features::AbstractString, nfreq::Int) =
    features == "fourier" ? 1 + 2 * nfreq : 1

joint_input_channels(features::AbstractString, nfreq::Int; include_delta_input::Bool=false) =
    JOINT_STATE_CHANNELS + (include_delta_input ? 2 : 0) + time_feature_count(features, nfreq)

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data_cfg = raw["data"]
    model_cfg = raw["model"]
    training_cfg = raw["training"]
    sampling_cfg = raw["sampling"]
    figure_cfg = raw["figure"]
    output_cfg = raw["output"]
    run_cfg = haskey(raw, "run") ? raw["run"] : Dict{String, Any}()

    time_features = normalize_time_features(get(model_cfg, "time_features", DEFAULT_TIME_FEATURES))
    time_fourier_frequencies = Int(get(model_cfg, "time_fourier_frequencies", DEFAULT_TIME_FOURIER_FREQUENCIES))
    include_delta_input = Bool(get(model_cfg, "include_delta_input", false))
    require_condition(time_fourier_frequencies >= 0, "time_fourier_frequencies must be nonnegative.")
    time_features == "fourier" && require_condition(time_fourier_frequencies >= 1,
        "time_fourier_frequencies must be at least 1 when time_features=fourier.")
    required_in_channels = joint_input_channels(time_features, time_fourier_frequencies;
        include_delta_input=include_delta_input)
    if haskey(model_cfg, "in_channels") && Int(model_cfg["in_channels"]) != required_in_channels
        @warn "Ignoring model.in_channels; it is determined by state plus time features" requested=model_cfg["in_channels"] required=required_in_channels
    end

    model_config = ScoreUNetConfig(
        in_channels=required_in_channels,
        base_channels=Int(get(model_cfg, "base_channels", 32)),
        channel_multipliers=Int.(get(model_cfg, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(model_cfg, "kernel_size", 5)),
        periodic=Bool(get(model_cfg, "periodic", true)),
        activation=activation_from_string(get(model_cfg, "activation", "swish")),
        final_activation=activation_from_string(get(model_cfg, "final_activation", "identity")),
    )

    trainer_config = ScoreTrainerConfig(
        batch_size=Int(get(training_cfg, "batch_size", 2048)),
        epochs=Int(get(training_cfg, "epochs", 40)),
        lr=Float64(get(training_cfg, "learning_rate", get(training_cfg, "lr", 3e-4))),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        shuffle=true,
        seed=Int(get(training_cfg, "seed", 20260427)),
        progress=Bool(get(training_cfg, "progress", true)),
        max_steps_per_epoch=nothing,
        accumulation_steps=Int(get(training_cfg, "accumulation_steps", 1)),
        use_lr_schedule=Bool(get(training_cfg, "use_lr_schedule", true)),
        warmup_steps=Int(get(training_cfg, "warmup_steps", 200)),
        min_lr_factor=Float64(get(training_cfg, "min_lr_factor", 0.1)),
        epoch_subset_size=0,
    )

    save_stride = Int(get(sampling_cfg, "save_stride", 20))
    sample_dt = Float64(get(sampling_cfg, "sample_dt", Float64(get(sampling_cfg, "dt", 0.001)) * save_stride))
    sampling_config = JointLangevinConfig(
        dt=Float64(get(sampling_cfg, "dt", 0.001)),
        sample_dt=sample_dt,
        nsteps=Int(get(sampling_cfg, "steps", 20_000)),
        resolution=save_stride,
        n_ensembles=Int(get(sampling_cfg, "chains", 256)),
        burn_in=Int(get(sampling_cfg, "burnin_steps", 4_000)),
        sigma=Float32(get(training_cfg, "sigma", 0.1)),
        seed=Int(get(sampling_cfg, "seed", Int(get(training_cfg, "seed", 20260427)) + 201)),
        progress=Bool(get(sampling_cfg, "progress", true)),
    )

    params = L96JointScoreParams(
        String(data_cfg["input_hdf5"]),
        Float64(data_cfg["burnin_fraction"]),
        Float64(data_cfg["tau_min"]),
        Float64(get(data_cfg, "tau_max_decorrelation_multiples", 3.0)),
        Int(get(data_cfg, "lag_stride", 1)),
        Int(get(data_cfg, "max_eval_pairs", 120_000)),
        Bool(get(data_cfg, "shared_normalization", true)),
        model_config,
        normalization_from_string(get(model_cfg, "normalization", "batchnorm")),
        time_features,
        time_fourier_frequencies,
        include_delta_input,
        Int(get(model_cfg, "init_seed", 314159)),
        trainer_config,
        Int(get(training_cfg, "batches_per_epoch", 512)),
        Float64(get(training_cfg, "stein_weight", 0.0)),
        Float64(get(training_cfg, "mean_score_weight", 0.0)),
        sampling_config,
        Int(get(figure_cfg, "eval_tau_count", 4)),
        Int.(get(figure_cfg, "pair_offsets", [0, 1, 5])),
        Int(get(figure_cfg, "pdf_bins", 100)),
        Int(get(figure_cfg, "max_pdf_samples", 250_000)),
        Int(get(figure_cfg, "stein_samples", 6_000)),
        Int(get(figure_cfg, "width", 3400)),
        Int(get(figure_cfg, "height", 3200)),
        String(output_cfg["model_bson"]),
        String(output_cfg["figure_png"]),
        String(get(run_cfg, "device", "AUTO")),
        Bool(get(run_cfg, "evaluate", true)),
        Bool(get(run_cfg, "verbose", true)),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.tau_min > 0.0, "tau_min must be positive.")
    require_condition(params.tau_max_decorrelation_multiples > 0.0, "tau_max_decorrelation_multiples must be positive.")
    require_condition(params.lag_stride >= 1, "lag_stride must be >= 1.")
    require_condition(params.max_eval_pairs >= 1_000, "max_eval_pairs must be at least 1000.")
    require_condition(params.model_config.in_channels == joint_input_channels(params.time_features,
            params.time_fourier_frequencies; include_delta_input=params.include_delta_input),
        "The complex-amplitude joint-score U-Net input channels must match state plus time features.")
    require_condition(params.model_config.periodic, "Complex-amplitude joint score training should use periodic convolutions.")
    require_condition(params.batches_per_epoch >= 1, "batches_per_epoch must be >= 1.")
    require_condition(params.eval_tau_count >= 1, "eval_tau_count must be >= 1.")
    require_condition(!isempty(params.pair_offsets), "pair_offsets must not be empty.")
    require_condition(all(offset -> offset >= 0, params.pair_offsets), "pair_offsets must be nonnegative.")
    require_condition(params.figure_width >= 2200 && params.figure_height >= 1800, "Figure dimensions are too small.")
    return params
end

function score_from_joint_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function parameter_norm(model)
    total = 0.0
    for p in Flux.trainables(model)
        total += sum(abs2, to_host(p))
    end
    return sqrt(total)
end

function mean_joint_score_norm(model, batch, sigma::Float32)
    scores = score_from_joint_model(model, batch, sigma)
    flat_scores = reshape(scores, :, size(scores, 3))
    norms = sqrt.(sum(abs2, flat_scores; dims=1))
    return Float64(mean(to_host(norms)))
end

function channel_shared_data_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    means = Array{Float32}(undef, C, K)
    stds = Array{Float32}(undef, C, K)
    for c in 1:C
        vals = @view samples[:, c, :]
        mean_value64 = mean(Float64, vals)
        n = length(vals)
        variance = mean(x -> abs2(Float64(x) - mean_value64), vals) * n / max(n - 1, 1)
        means[c, :] .= Float32(mean_value64)
        stds[c, :] .= max(Float32(sqrt(variance)), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function sitewise_data_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    means = Array{Float32}(undef, C, K)
    stds = Array{Float32}(undef, C, K)
    for c in 1:C, idx in 1:K
        slice = @view samples[idx, c, :]
        mean_value64 = mean(Float64, slice)
        means[c, idx] = Float32(mean_value64)
        n = length(slice)
        variance = mean(x -> abs2(Float64(x) - mean_value64), slice) * n / max(n - 1, 1)
        stds[c, idx] = max(Float32(sqrt(variance)), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function apply_pair_stats(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return (samples .- mean_tensor) ./ std_tensor
end

function denormalize_pair_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return samples .* std_tensor .+ mean_tensor
end

function build_pair_sampler(path::AbstractString, burnin_fraction::Float64, tau_min::Float64,
        tau_max_decorrelation_multiples::Float64, lag_stride::Int)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    require_condition(ndims(states) == 4, "Expected /trajectories/states to be rank 4: time x site x channel x trajectory.")
    require_condition(size(states, 3) == 2, "Expected q,p channels in /trajectories/states.")
    require_condition(size(states, 1) == length(times), "Time axis length does not match state tensor.")

    start_idx = burnin_start_index(length(times), burnin_fraction)
    save_dt = length(times) > 1 ? (times[2] - times[1]) : 0.0
    require_condition(save_dt > 0.0, "A positive saved time step is required.")

    t_decorrelation = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    available_tau_max = times[end] - times[start_idx]
    tau_max = min(tau_max_decorrelation_multiples * t_decorrelation, available_tau_max)
    require_condition(tau_max >= tau_min, "tau_min exceeds the available lag range up to tau_max.")

    min_lag = max(1, ceil(Int, tau_min / save_dt - 1e-9))
    max_lag = min(length(times) - start_idx, floor(Int, tau_max / save_dt + 1e-9))
    require_condition(max_lag >= min_lag, "No lag steps are available in [tau_min, tau_max].")

    lag_steps = collect(min_lag:lag_stride:max_lag)
    lag_times = lag_steps .* save_dt
    denom = max(tau_max - tau_min, eps())
    lag_tnorm = Float32.((lag_times .- tau_min) ./ denom)
    return JointPairSampler(times, states, start_idx, lag_steps, lag_times, lag_tnorm, tau_min, tau_max, t_decorrelation)
end

function compute_state_stats(sampler::JointPairSampler, shared_normalization::Bool)
    nt, K, _, ntraj = size(sampler.states)
    means2 = Array{Float32}(undef, 2, K)
    stds2 = Array{Float32}(undef, 2, K)
    if shared_normalization
        for c in 1:2
            total = 0.0
            count = 0
            @inbounds for traj in 1:ntraj, t in sampler.start_idx:nt, i in 1:K
                total += sampler.states[t, i, c, traj]
                count += 1
            end
            mu = total / count
            ss = 0.0
            @inbounds for traj in 1:ntraj, t in sampler.start_idx:nt, i in 1:K
                delta = Float64(sampler.states[t, i, c, traj]) - mu
                ss += delta * delta
            end
            means2[c, :] .= Float32(mu)
            stds2[c, :] .= max(Float32(sqrt(ss / max(count - 1, 1))), sqrt(eps(Float32)))
        end
    else
        for c in 1:2, i in 1:K
            total = 0.0
            count = 0
            @inbounds for traj in 1:ntraj, t in sampler.start_idx:nt
                total += sampler.states[t, i, c, traj]
                count += 1
            end
            mu = total / count
            ss = 0.0
            @inbounds for traj in 1:ntraj, t in sampler.start_idx:nt
                delta = Float64(sampler.states[t, i, c, traj]) - mu
                ss += delta * delta
            end
            means2[c, i] = Float32(mu)
            stds2[c, i] = max(Float32(sqrt(ss / max(count - 1, 1))), sqrt(eps(Float32)))
        end
    end
    means = Array{Float32}(undef, JOINT_STATE_CHANNELS, K)
    stds = Array{Float32}(undef, JOINT_STATE_CHANNELS, K)
    means[1, :] .= means2[1, :]
    means[2, :] .= means2[2, :]
    means[3, :] .= means2[1, :]
    means[4, :] .= means2[2, :]
    stds[1, :] .= stds2[1, :]
    stds[2, :] .= stds2[2, :]
    stds[3, :] .= stds2[1, :]
    stds[4, :] .= stds2[2, :]
    return DataStats(means, stds)
end

function sample_pair_batch!(state_buffer::Array{Float32, 3}, tnorm_buffer::Vector{Float32},
        sampler::JointPairSampler, rngs::Vector{<:AbstractRNG})
    nt, K, _, ntraj = size(sampler.states)
    nlags = length(sampler.lag_steps)
    batch_size = size(state_buffer, 3)

    Threads.@threads for sample_idx in 1:batch_size
        rng = rngs[Threads.threadid()]
        lag_pos = rand(rng, 1:nlags)
        lag = sampler.lag_steps[lag_pos]
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:(nt - lag))

        @inbounds for mode_idx in 1:K
            state_buffer[mode_idx, 1, sample_idx] = sampler.states[time_idx, mode_idx, 1, traj_idx]
            state_buffer[mode_idx, 2, sample_idx] = sampler.states[time_idx, mode_idx, 2, traj_idx]
            state_buffer[mode_idx, 3, sample_idx] = sampler.states[time_idx + lag, mode_idx, 1, traj_idx]
            state_buffer[mode_idx, 4, sample_idx] = sampler.states[time_idx + lag, mode_idx, 2, traj_idx]
        end
        tnorm_buffer[sample_idx] = sampler.lag_tnorm[lag_pos]
    end
    return nothing
end

function random_pair_tensor(sampler::JointPairSampler, lag::Int, tnorm::Float32, npairs::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(sampler.states)
    pair_tensor = Array{Float32}(undef, K, JOINT_STATE_CHANNELS, npairs)
    tnorm_buffer = fill(tnorm, npairs)
    upper = nt - lag
    require_condition(upper >= sampler.start_idx, "Lag exceeds the available post-burn-in window.")

    @inbounds for sample_idx in 1:npairs
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:upper)
        for mode_idx in 1:K
            pair_tensor[mode_idx, 1, sample_idx] = sampler.states[time_idx, mode_idx, 1, traj_idx]
            pair_tensor[mode_idx, 2, sample_idx] = sampler.states[time_idx, mode_idx, 2, traj_idx]
            pair_tensor[mode_idx, 3, sample_idx] = sampler.states[time_idx + lag, mode_idx, 1, traj_idx]
            pair_tensor[mode_idx, 4, sample_idx] = sampler.states[time_idx + lag, mode_idx, 2, traj_idx]
        end
    end

    return pair_tensor, tnorm_buffer
end

function encode_time_features!(input_buffer::Array{Float32, 3}, tval::Float32,
        mode_idx::Int, sample_idx::Int, first_channel::Int,
        time_features::AbstractString, nfreq::Int)
    input_buffer[mode_idx, first_channel, sample_idx] = tval
    if time_features == "fourier"
        channel = first_channel + 1
        t64 = Float64(tval)
        for freq in 1:nfreq
            angle = Float32(2.0 * pi * freq * t64)
            input_buffer[mode_idx, channel, sample_idx] = sin(angle)
            input_buffer[mode_idx, channel + 1, sample_idx] = cos(angle)
            channel += 2
        end
    end
    return nothing
end

function encode_joint_input!(input_buffer::AbstractArray{Float32, 3}, normalized_pair::AbstractArray{Float32, 3},
        tnorm_buffer::AbstractVector{Float32};
        time_features::AbstractString=DEFAULT_TIME_FEATURES,
        time_fourier_frequencies::Int=DEFAULT_TIME_FOURIER_FREQUENCIES,
        include_delta_input::Bool=false)
    K, _, batch = size(normalized_pair)
    expected_channels = joint_input_channels(time_features, time_fourier_frequencies;
        include_delta_input=include_delta_input)
    require_condition(size(input_buffer, 2) == expected_channels,
        @sprintf("Input buffer has %d channels but %s time features require %d.",
            size(input_buffer, 2), time_features, expected_channels))
    @inbounds for sample_idx in 1:batch
        tval = tnorm_buffer[sample_idx]
        for mode_idx in 1:K
            input_buffer[mode_idx, 1, sample_idx] = normalized_pair[mode_idx, 1, sample_idx]
            input_buffer[mode_idx, 2, sample_idx] = normalized_pair[mode_idx, 2, sample_idx]
            input_buffer[mode_idx, 3, sample_idx] = normalized_pair[mode_idx, 3, sample_idx]
            input_buffer[mode_idx, 4, sample_idx] = normalized_pair[mode_idx, 4, sample_idx]
            time_start = JOINT_STATE_CHANNELS + 1
            if include_delta_input
                input_buffer[mode_idx, time_start, sample_idx] =
                    normalized_pair[mode_idx, 3, sample_idx] - normalized_pair[mode_idx, 1, sample_idx]
                input_buffer[mode_idx, time_start + 1, sample_idx] =
                    normalized_pair[mode_idx, 4, sample_idx] - normalized_pair[mode_idx, 2, sample_idx]
                time_start += 2
            end
            encode_time_features!(input_buffer, tval, mode_idx, sample_idx, time_start,
                time_features, time_fourier_frequencies)
        end
    end
    return nothing
end

function refresh_delta_input_channels!(input_buffer::AbstractArray{Float32, 3})
    @views begin
        input_buffer[:, JOINT_STATE_CHANNELS + 1, :] .= input_buffer[:, 3, :] .- input_buffer[:, 1, :]
        input_buffer[:, JOINT_STATE_CHANNELS + 2, :] .= input_buffer[:, 4, :] .- input_buffer[:, 2, :]
    end
    return nothing
end

function choose_eval_lags(sampler::JointPairSampler, neval::Int)
    total = length(sampler.lag_steps)
    nsel = min(neval, total)
    idxs = if nsel == 1
        [1]
    else
        unique(round.(Int, range(1, total, length=nsel)))
    end
    return sampler.lag_steps[idxs], sampler.lag_times[idxs], sampler.lag_tnorm[idxs]
end

function train_model(sampler::JointPairSampler, stats::DataStats, params::L96JointScoreParams,
        device::ExecutionDevice)
    Random.seed!(params.model_init_seed)
    model_cfg = adjust_model_config_for_length(params.model_config, size(sampler.states, 2))
    model = JointScoreUNet(build_unet(model_cfg; normalization=params.model_normalization))
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
    noise_cpu = Array{Float32}(undef, K, JOINT_STATE_CHANNELS, B)
    stein_dim = K * 2
    stein_eye_device = move_array(Matrix{Float32}(I, stein_dim, stein_dim), device)

    noisy_device = device isa GPUDevice ? CUDA.CuArray{Float32}(undef, K, params.model_config.in_channels, B) :
        Array{Float32}(undef, K, params.model_config.in_channels, B)
    noise_device = device isa GPUDevice ? CUDA.CuArray{Float32}(undef, K, JOINT_STATE_CHANNELS, B) : noise_cpu

    history = Dict(
        :train_loss => Float64[],
        :score_norm => Float64[],
        :param_norm => Float64[],
        :epoch_time => Float64[],
    )

    progress = params.trainer_config.progress ? Progress(params.trainer_config.epochs; desc="Training joint score network") : nothing

    for epoch in 1:params.trainer_config.epochs
        epoch_t0 = time_ns()
        epoch_losses = Float64[]
        accumulated_grads = nothing
        accum_count = 0

        for _ in 1:params.batches_per_epoch
            global_step += 1
            if lr_scheduler !== nothing
                Flux.adjust!(opt_state, lr_scheduler(global_step))
            end

            sample_pair_batch!(state_cpu, tnorm_cpu, sampler, thread_rngs)
            norm_cpu .= apply_pair_stats(state_cpu, stats)
            encode_joint_input!(input_cpu, norm_cpu, tnorm_cpu;
                time_features=params.time_features,
                time_fourier_frequencies=params.time_fourier_frequencies,
                include_delta_input=params.include_delta_input)

            if device isa GPUDevice
                CUDA.@allowscalar copyto!(noisy_device, input_cpu)
                fill_gpu_noise!(noise_device)
                @views noisy_device[:, 1:JOINT_STATE_CHANNELS, :] .+= sigma .* noise_device
            else
                noisy_device .= input_cpu
                Threads.@threads for sample_idx in 1:B
                    rng = thread_rngs[Threads.threadid()]
                    @inbounds for idx in 1:(K * JOINT_STATE_CHANNELS)
                        noise_cpu[idx + (sample_idx - 1) * K * JOINT_STATE_CHANNELS] = randn(rng, Float32)
                    end
                end
                @views noisy_device[:, 1:JOINT_STATE_CHANNELS, :] .+= sigma .* noise_cpu
            end
            params.include_delta_input && refresh_delta_input_channels!(noisy_device)

            loss_value, grads = Flux.withgradient(model) do current_model
                pred = current_model(noisy_device)
                loss = Flux.Losses.mse(pred, noise_device)
                if params.stein_weight > 0.0
                    score_x0 = (-pred[:, 1:2, :] ./ sigma) .* 1.0f0
                    noisy_x0 = copy(noisy_device[:, 1:2, :])
                    score_flat = reshape(score_x0, stein_dim, B)
                    x_flat = reshape(noisy_x0, stein_dim, B)
                    stein_mat = (score_flat * transpose(x_flat)) ./ Float32(B)
                    loss += Float32(params.stein_weight) * Flux.Losses.mse(stein_mat, -stein_eye_device)
                end
                if params.mean_score_weight > 0.0
                    score_x0 = (-pred[:, 1:2, :] ./ sigma) .* 1.0f0
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
        push!(history[:score_norm], mean_joint_score_norm(model, monitor_batch, sigma))
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
    Flux.testmode!(model)
    return model, history
end

function maybe_subsample_columns(samples::Array{Float32, 3}, max_cols::Int, rng::AbstractRNG)
    if size(samples, 3) <= max_cols
        return samples
    end
    keep = randperm(rng, size(samples, 3))[1:max_cols]
    return samples[:, :, keep]
end

function keep_finite_pair_snapshots(samples::Array{Float32, 3})
    finite_mask = BitVector(undef, size(samples, 3))
    @inbounds for snap_idx in axes(samples, 3)
        finite_mask[snap_idx] = all(isfinite, @view samples[:, :, snap_idx])
    end
    kept = count(identity, finite_mask)
    kept > 0 || error("Joint score Langevin sampling produced no finite snapshots for diagnostics.")
    total = length(finite_mask)
    if kept < total
        @warn "Dropping non-finite joint-score snapshots before diagnostics" kept total dropped=(total - kept)
    end
    return samples[:, :, finite_mask], kept, total
end

function integrate_joint_score_sde(model, init_states::Array{Float32, 3}, tnorm::Float32,
        cfg::JointLangevinConfig, device::ExecutionDevice;
        time_features::AbstractString=DEFAULT_TIME_FEATURES,
        time_fourier_frequencies::Int=DEFAULT_TIME_FOURIER_FREQUENCIES)
    Flux.testmode!(model)
    K, _, nchains = size(init_states)
    state_dev = move_array(init_states, device)
    input_channels = joint_input_channels(time_features, time_fourier_frequencies)
    input_cpu = zeros(Float32, K, input_channels, nchains)
    input_dev = move_array(input_cpu, device)
    noise_dev = device isa GPUDevice ? CUDA.CuArray{Float32}(undef, K, JOINT_STATE_CHANNELS, nchains) : Array{Float32}(undef, K, JOINT_STATE_CHANNELS, nchains)
    rngs = seed_thread_rngs(cfg.seed)

    total_saved = fld(max(cfg.nsteps - cfg.burn_in, 0), max(cfg.resolution, 1))
    total_saved > 0 || error("Burn-in removes all samples; increase nsteps or reduce burn_in/resolution.")
    samples = Array{Float32}(undef, K, JOINT_STATE_CHANNELS, total_saved * nchains)
    cursor = 1
    sqrt_2dt = sqrt(2.0f0 * Float32(cfg.dt))
    dt32 = Float32(cfg.dt)

    progress = cfg.progress ? Progress(cfg.nsteps; desc="Complex-chain joint score Langevin", dt=0.5) : nothing

    for step in 1:cfg.nsteps
        input_cpu .= 0
        @views input_cpu[:, 1:JOINT_STATE_CHANNELS, :] .= to_host(state_dev)
        encode_joint_input!(input_cpu, input_cpu[:, 1:JOINT_STATE_CHANNELS, :], fill(tnorm, nchains);
            time_features=time_features, time_fourier_frequencies=time_fourier_frequencies)
        input_dev .= move_array(input_cpu, device)
        scores = score_from_joint_model(model, input_dev, cfg.sigma)

        if device isa GPUDevice
            fill_gpu_noise!(noise_dev)
        else
            Threads.@threads for chain_idx in 1:nchains
                rng = rngs[Threads.threadid()]
                @inbounds for idx in 1:(K * JOINT_STATE_CHANNELS)
                    noise_dev[idx + (chain_idx - 1) * K * JOINT_STATE_CHANNELS] = randn(rng, Float32)
                end
            end
        end

        state_dev .+= dt32 .* scores .+ sqrt_2dt .* noise_dev

        if step > cfg.burn_in && (step - cfg.burn_in) % cfg.resolution == 0
            state_host = Array(state_dev)
            @inbounds samples[:, :, cursor:(cursor + nchains - 1)] .= state_host
            cursor += nchains
        end
        progress !== nothing && ProgressMeter.next!(progress)
    end

    progress !== nothing && ProgressMeter.finish!(progress)
    return samples
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

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
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

function draw_channel_values(samples::Array{Float32, 3}, channel::Int, max_samples::Int, rng::AbstractRNG)
    K, _, nsnaps = size(samples)
    total = K * nsnaps
    if max_samples <= 0 || max_samples >= total
        return Float64.(vec(@view samples[:, channel, :]))
    end

    values = Vector{Float64}(undef, max_samples)
    for sample_idx in 1:max_samples
        linear = rand(rng, 0:(total - 1))
        mode_idx = (linear % K) + 1
        snap_idx = (linear ÷ K) + 1
        values[sample_idx] = samples[mode_idx, channel, snap_idx]
    end
    return values
end

function draw_offset_pair_samples(samples::Array{Float32, 3}, offset::Int, max_samples::Int, rng::AbstractRNG)
    K, _, nsnaps = size(samples)
    total = K * nsnaps
    if max_samples <= 0 || max_samples >= total
        x_values = Vector{Float64}(undef, total)
        y_values = Vector{Float64}(undef, total)
        cursor = 1
        @inbounds for snap_idx in 1:nsnaps
            for mode_idx in 1:K
                paired_idx = mod1(mode_idx + offset, K)
                x_values[cursor] = samples[mode_idx, 1, snap_idx]
                y_values[cursor] = samples[paired_idx, 3, snap_idx]
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
        x_values[sample_idx] = samples[mode_idx, 1, snap_idx]
        y_values[sample_idx] = samples[paired_idx, 3, snap_idx]
    end
    return x_values, y_values
end

function pair_density_from_samples(samples::Array{Float32, 3}, offset::Int, bins::Int, max_samples::Int,
        rng::AbstractRNG; x_grid::Union{Nothing, Vector{Float64}}=nothing,
        y_grid::Union{Nothing, Vector{Float64}}=nothing)
    x_values, y_values = draw_offset_pair_samples(samples, offset, max_samples, rng)
    if x_grid === nothing || y_grid === nothing
        x_range = kde_range(x_values)
        y_range = kde_range(y_values)
        density_result = kde((x_values, y_values); npoints=(bins, bins), boundary=(x_range, y_range))
        return PairPdfResult(offset, collect(density_result.x), collect(density_result.y), Array(density_result.density))
    end
    density = kde_on_grid_2d(x_values, y_values, x_grid, y_grid)
    return PairPdfResult(offset, x_grid, y_grid, density)
end

function marginal_density_from_samples(samples::Array{Float32, 3}, channel::Int, bins::Int, max_samples::Int,
        rng::AbstractRNG; grid::Union{Nothing, Vector{Float64}}=nothing)
    values = draw_channel_values(samples, channel, max_samples, rng)
    if grid === nothing
        boundary = kde_range(values)
        density_result = kde(values; npoints=bins, boundary=boundary)
        return collect(density_result.x), Array(density_result.density)
    end
    return grid, kde_on_grid_1d(values, grid)
end

function stein_matrix(model, normalized_pairs::Array{Float32, 3}, tnorm::Float32, sigma::Float32,
        nsamples::Int, rng::AbstractRNG, device::ExecutionDevice;
        time_features::AbstractString=DEFAULT_TIME_FEATURES,
        time_fourier_frequencies::Int=DEFAULT_TIME_FOURIER_FREQUENCIES)
    max_keep = min(nsamples, size(normalized_pairs, 3))
    keep = max_keep < size(normalized_pairs, 3) ? randperm(rng, size(normalized_pairs, 3))[1:max_keep] : collect(1:size(normalized_pairs, 3))
    clean_batch = normalized_pairs[:, :, keep]
    noisy_input = Array{Float32}(undef, size(clean_batch, 1),
        joint_input_channels(time_features, time_fourier_frequencies), size(clean_batch, 3))
    encode_joint_input!(noisy_input, clean_batch, fill(tnorm, size(clean_batch, 3));
        time_features=time_features, time_fourier_frequencies=time_fourier_frequencies)

    noisy_state = noisy_input[:, 1:JOINT_STATE_CHANNELS, :]
    noisy_state .+= sigma .* randn(rng, Float32, size(noisy_state))

    device_batch = move_array(noisy_input, device)
    scores = score_from_joint_model(model, device_batch, sigma)
    dim = size(clean_batch, 1) * size(clean_batch, 2)
    score_flat = reshape(to_host(scores), dim, size(clean_batch, 3))
    noisy_flat = reshape(noisy_state, dim, size(clean_batch, 3))
    return -(score_flat * noisy_flat') ./ size(clean_batch, 3)
end

function evaluate_tau(model, sampler::JointPairSampler, stats::DataStats, lag::Int, tau::Float64, tnorm::Float32,
        params::L96JointScoreParams, device::ExecutionDevice, rng::AbstractRNG, seed_offset::Int)
    observed_raw, _ = random_pair_tensor(sampler, lag, tnorm, params.max_eval_pairs, rng)
    observed_norm = apply_pair_stats(observed_raw, stats)

    x0_centers, observed_x0_density = marginal_density_from_samples(observed_raw, 1, params.pdf_bins,
        params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 10_000 + seed_offset))
    xt_centers, observed_xt_density = marginal_density_from_samples(observed_raw, 3, params.pdf_bins,
        params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 11_000 + seed_offset))

    observed_pair_pdfs = PairPdfResult[]
    for offset in params.pair_offsets
        push!(observed_pair_pdfs, pair_density_from_samples(observed_raw, offset, params.pdf_bins,
            params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 12_000 + seed_offset + 17 * offset)))
    end

    init_idx = rand(rng, 1:size(observed_norm, 3), params.sampling_config.n_ensembles)
    init_states = observed_norm[:, :, init_idx]
    generated_norm = integrate_joint_score_sde(model, init_states, tnorm, params.sampling_config, device;
        time_features=params.time_features,
        time_fourier_frequencies=params.time_fourier_frequencies)
    generated_norm, finite_generated_snapshots, total_generated_snapshots = keep_finite_pair_snapshots(generated_norm)
    generated_raw = denormalize_pair_tensor(generated_norm, stats)
    generated_raw = maybe_subsample_columns(generated_raw, params.max_eval_pairs, rng)

    _, generated_x0_density = marginal_density_from_samples(generated_raw, 1, params.pdf_bins,
        params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 13_000 + seed_offset); grid=x0_centers)
    _, generated_xt_density = marginal_density_from_samples(generated_raw, 3, params.pdf_bins,
        params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 14_000 + seed_offset); grid=xt_centers)

    generated_pair_pdfs = PairPdfResult[]
    pair_kls = Float64[]
    for observed_pair in observed_pair_pdfs
        generated_pair = pair_density_from_samples(generated_raw, observed_pair.offset, params.pdf_bins,
            params.max_pdf_samples, MersenneTwister(params.trainer_config.seed + 15_000 + seed_offset + 19 * observed_pair.offset);
            x_grid=observed_pair.x_grid, y_grid=observed_pair.y_grid)
        push!(generated_pair_pdfs, generated_pair)
        push!(pair_kls, kl_divergence_from_density_2d(
            observed_pair.density,
            generated_pair.density,
            grid_spacing(observed_pair.x_grid),
            grid_spacing(observed_pair.y_grid),
        ))
    end

    x0_kl = kl_divergence_from_density_1d(observed_x0_density, generated_x0_density, grid_spacing(x0_centers))
    xt_kl = kl_divergence_from_density_1d(observed_xt_density, generated_xt_density, grid_spacing(xt_centers))
    mean_pair_kl = mean(pair_kls)
    mean_kl = mean(vcat([x0_kl, xt_kl], pair_kls))
    accuracy = exp(-mean_kl)

    stein_rng = MersenneTwister(params.trainer_config.seed + 16_000 + seed_offset)
    stein_mat = stein_matrix(model, observed_norm, tnorm, params.trainer_config.sigma, params.stein_samples,
        stein_rng, device; time_features=params.time_features,
        time_fourier_frequencies=params.time_fourier_frequencies)
    dim = size(stein_mat, 1)
    ident = Matrix{Float64}(I, dim, dim)
    stein_relative_error = norm(stein_mat - ident) / norm(ident)

    diag = L96JointLagDiagnostics(
        tau,
        Float64(tnorm),
        x0_kl,
        xt_kl,
        pair_kls,
        mean_pair_kl,
        mean_kl,
        accuracy,
        stein_relative_error,
        finite_generated_snapshots,
        total_generated_snapshots,
    )

    return L96JointEvalRecord(
        tau,
        lag,
        tnorm,
        x0_centers,
        observed_x0_density,
        generated_x0_density,
        xt_centers,
        observed_xt_density,
        generated_xt_density,
        observed_pair_pdfs,
        generated_pair_pdfs,
        stein_mat,
        diag,
    )
end

function summary_lines(params::L96JointScoreParams, sampler::JointPairSampler, history,
        eval_records::Vector{L96JointEvalRecord})
    mean_accuracy = mean(record.diag.accuracy for record in eval_records)
    mean_kl = mean(record.diag.mean_kl for record in eval_records)
    mean_pair_kl = mean(record.diag.mean_pair_kl for record in eval_records)
    mean_stein = mean(record.diag.stein_relative_error for record in eval_records)

    return [
        @sprintf("tau range = [%.3f, %.3f]", sampler.tau_min, sampler.tau_max),
        @sprintf("tau_max / t_dec = %.2f", sampler.tau_max / sampler.decorrelation_time),
        @sprintf("lag count = %d", length(sampler.lag_steps)),
        @sprintf("sigma = %.3f", params.trainer_config.sigma),
        @sprintf("epochs = %d", params.trainer_config.epochs),
        @sprintf("batches/epoch = %d", params.batches_per_epoch),
        @sprintf("batch_size = %d", params.trainer_config.batch_size),
        @sprintf("lr = %.2e", params.trainer_config.lr),
        @sprintf("normalization = %s", String(params.model_normalization)),
        @sprintf("time features = %s (%d Fourier freqs)",
            params.time_features, params.time_fourier_frequencies),
        @sprintf("include_delta_input = %s", params.include_delta_input),
        @sprintf("sample_dt = %.4f", params.sampling_config.sample_dt),
        @sprintf("chains = %d", params.sampling_config.n_ensembles),
        @sprintf("final DSM loss = %.3e", history[:train_loss][end]),
        @sprintf("mean KL = %.3e", mean_kl),
        @sprintf("mean pair KL = %.3e", mean_pair_kl),
        @sprintf("mean pdf accuracy = %.6f", mean_accuracy),
        @sprintf("mean smoothed Stein rel. err. = %.3e", mean_stein),
    ]
end

function metrics_panel!(parent, eval_records::Vector{L96JointEvalRecord})
    taus = [record.tau for record in eval_records]
    accuracies = [record.diag.accuracy for record in eval_records]
    stein_scores = exp.(-[record.diag.stein_relative_error for record in eval_records])

    ax = Axis(parent; xlabel="tau", ylabel="score", title="Lag-wise diagnostics")
    ylims!(ax, 0.0, 1.05)
    lines!(ax, taus, accuracies; color=STYLE_PRIMARY, label="pdf accuracy")
    scatter!(ax, taus, accuracies; color=STYLE_PRIMARY, marker=:circle)
    lines!(ax, taus, stein_scores; color=STYLE_HIGHLIGHT, label="exp(-Stein rel. err.)")
    scatter!(ax, taus, stein_scores; color=STYLE_HIGHLIGHT, marker=:diamond)
    axislegend(ax; position=:rb)
    return ax
end

function marginal_panel!(parent, record::L96JointEvalRecord)
    ax = Axis(parent; xlabel="value", ylabel="density",
        title=@sprintf("q marginal densities  tau=%.2f", record.tau))
    lines!(ax, record.x0_centers, record.observed_x0_density; color=STYLE_PRIMARY, label="q0 data")
    lines!(ax, record.x0_centers, record.generated_x0_density; color=STYLE_PRIMARY, linestyle=:dash, label="q0 gen")
    lines!(ax, record.xt_centers, record.observed_xt_density; color=STYLE_ACCENT, label="q_tau data")
    lines!(ax, record.xt_centers, record.generated_xt_density; color=STYLE_ACCENT, linestyle=:dash, label="q_tau gen")
    axislegend(ax; position=:rt, nbanks=2)
    return ax
end

function contour_levels_from_densities(density_a::Matrix{Float64}, density_b::Matrix{Float64})
    vmax = max(maximum(density_a), maximum(density_b))
    return collect(range(0.15 * vmax, 0.9 * vmax, length=6))
end

function contour_pair_panel!(parent, observed::PairPdfResult, generated::PairPdfResult, tau::Float64, kl::Float64)
    levels = contour_levels_from_densities(observed.density, generated.density)
    ax = Axis(parent; xlabel="q0_i", ylabel=@sprintf("q_tau,i+%d", observed.offset),
        title=@sprintf("Offset %d  tau=%.2f  KL=%.2e", observed.offset, tau, kl),
        aspect=DataAspect(),
        xgridvisible=false, ygridvisible=false)
    contour!(ax, observed.x_grid, observed.y_grid, observed.density;
        levels=levels, color=STYLE_REFERENCE, linewidth=2.0, linestyle=:solid)
    contour!(ax, generated.x_grid, generated.y_grid, generated.density;
        levels=levels, color=STYLE_HIGHLIGHT, linewidth=2.0, linestyle=:dash)
    elements = [
        LineElement(color=STYLE_REFERENCE, linewidth=2.4),
        LineElement(color=STYLE_HIGHLIGHT, linewidth=2.4, linestyle=:dash),
    ]
    axislegend(ax, elements, ["observed", "generated"]; position=:rt)
    return ax
end

function stein_panel!(parent, stein_mat::Matrix{Float64}, tau::Float64, diag::L96JointLagDiagnostics)
    dim = size(stein_mat, 1)
    ident = Matrix{Float64}(I, dim, dim)
    residual = stein_mat - ident
    clim = max(maximum(abs.(residual)), 1e-6)
    ax = Axis(parent[1, 1]; xlabel="j", ylabel="i",
        title=@sprintf("Smoothed Stein[p_sigma]-I  tau=%.2f\nrel.err.=%.2e", tau, diag.stein_relative_error))
    hm = heatmap!(ax, 1:dim, 1:dim, residual; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
    Colorbar(parent[1, 2], hm; label="Stein[p_sigma] - I")
    return ax
end

function create_diagnostics_figure(output_path::AbstractString, params::L96JointScoreParams,
        sampler::JointPairSampler, history, eval_records::Vector{L96JointEvalRecord})
    ensure_figure_support_loaded!()
    render = function (_)
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf(
            "tau in [%.2f, %.2f], eval lags=%d, sigma=%.3f, mean KL=%.3e",
            sampler.tau_min,
            sampler.tau_max,
            length(eval_records),
            params.trainer_config.sigma,
            mean(record.diag.mean_kl for record in eval_records),
        )
        figure_title!(fig, "Complex-amplitude lag-conditioned joint score diagnostics"; subtitle=subtitle)

        epochs = collect(1:length(history[:train_loss]))
        ax_loss = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax_loss, epochs, history[:train_loss]; color=STYLE_PRIMARY)

        ax_score = Axis(fig[1, 2]; title="Mean score norm", xlabel="epoch", ylabel="norm")
        lines!(ax_score, epochs, history[:score_norm]; color=STYLE_ACCENT)

        ax_param = Axis(fig[1, 3]; title="Parameter norm", xlabel="epoch", ylabel="norm")
        lines!(ax_param, epochs, history[:param_norm]; color=STYLE_HIGHLIGHT)

        metrics_panel!(fig[1, 4], eval_records)
        text_panel!(fig[1, 5], summary_lines(params, sampler, history, eval_records);
            title="Diagnostic summary")

        for (idx, record) in enumerate(eval_records)
            row = idx + 1
            marginal_panel!(fig[row, 1], record)
            shown_pairs = min(length(record.observed_pair_pdfs), 3)
            for pair_idx in 1:shown_pairs
                contour_pair_panel!(fig[row, pair_idx + 1], record.observed_pair_pdfs[pair_idx],
                    record.generated_pair_pdfs[pair_idx], record.tau, record.diag.pair_kls[pair_idx])
            end
            if shown_pairs < 3
                for col in (shown_pairs + 2):4
                    text_panel!(fig[row, col], ["No pair offset configured for this panel."]; title="Unused")
                end
            end
            sg = GridLayout(fig[row, 5])
            stein_panel!(sg, record.stein_matrix, record.tau, record.diag)
        end

        save_figure(output_path, fig)
    end
    Base.invokelatest(with_scaled_figure_style, render, params.figure_width, params.figure_height)
    return nothing
end

function save_model(path::AbstractString, model, stats::DataStats, params::L96JointScoreParams,
        sampler::JointPairSampler, history, eval_records::Vector{L96JointEvalRecord})
    host_model = cpu(model)
    metadata = Dict(
        :tau_min => sampler.tau_min,
        :tau_max => sampler.tau_max,
        :decorrelation_time => sampler.decorrelation_time,
        :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times,
        :eval_taus => [record.tau for record in eval_records],
        :pair_offsets => params.pair_offsets,
        :shared_normalization => params.shared_normalization,
        :model_normalization => String(params.model_normalization),
        :time_features => params.time_features,
        :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
    )
    diagnostics = Dict(
        :tau => [record.diag.tau for record in eval_records],
        :x0_kl => [record.diag.x0_kl for record in eval_records],
        :xt_kl => [record.diag.xt_kl for record in eval_records],
        :mean_pair_kl => [record.diag.mean_pair_kl for record in eval_records],
        :mean_kl => [record.diag.mean_kl for record in eval_records],
        :accuracy => [record.diag.accuracy for record in eval_records],
        :stein_relative_error => [record.diag.stein_relative_error for record in eval_records],
        :finite_generated_snapshots => [record.diag.finite_generated_snapshots for record in eval_records],
        :total_generated_snapshots => [record.diag.total_generated_snapshots for record in eval_records],
    )
    model_cfg = params.model_config
    trainer_cfg = params.trainer_config
    sampling_cfg = params.sampling_config
    BSON.@save path host_model model_cfg trainer_cfg sampling_cfg stats metadata history diagnostics
    return nothing
end

function save_training_checkpoint(path::AbstractString, model, stats::DataStats, params::L96JointScoreParams,
        sampler::JointPairSampler, history)
    host_model = cpu(model)
    metadata = Dict(
        :tau_min => sampler.tau_min,
        :tau_max => sampler.tau_max,
        :decorrelation_time => sampler.decorrelation_time,
        :lag_steps => sampler.lag_steps,
        :lag_times => sampler.lag_times,
        :checkpoint => true,
        :model_normalization => String(params.model_normalization),
        :time_features => params.time_features,
        :time_fourier_frequencies => params.time_fourier_frequencies,
        :include_delta_input => params.include_delta_input,
    )
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
    sampler = build_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    stats = compute_state_stats(sampler, params.shared_normalization)

    device = detect_device(params.device_name)
    activate_device!(device)
    @printf("Training device request: %s\n", params.device_name)
    @printf("Resolved execution device: %s\n", describe_device(device))
    @printf("Lag window: [%.3f, %.3f] with %d discrete lags\n", sampler.tau_min, sampler.tau_max, length(sampler.lag_steps))
    @printf("Observed decorrelation time: %.6f\n", sampler.decorrelation_time)

    model, history = train_model(sampler, stats, params, device)

    checkpoint_path = string(output_bson, ".checkpoint")
    @printf("Saving training checkpoint to %s\n", checkpoint_path)
    save_training_checkpoint(checkpoint_path, model, stats, params, sampler, history)

    eval_records = L96JointEvalRecord[]
    if params.run_evaluation
        eval_lags, eval_taus, eval_tnorm = choose_eval_lags(sampler, params.eval_tau_count)
        eval_rng = MersenneTwister(params.trainer_config.seed + 50_000)
        for (idx, (lag, tau, tnorm)) in enumerate(zip(eval_lags, eval_taus, eval_tnorm))
            @printf("Evaluating tau = %.3f (lag step %d, normalized t = %.3f)\n", tau, lag, tnorm)
            push!(eval_records, evaluate_tau(model, sampler, stats, lag, tau, tnorm, params, device, eval_rng, idx))
        end
    else
        @printf("Skipping joint Langevin/PDF evaluation because run.evaluate=false.\n")
    end

    @printf("Saving model to %s\n", output_bson)
    save_model(output_bson, model, stats, params, sampler, history, eval_records)
    rm(checkpoint_path; force=true)

    if params.run_evaluation
        @printf("Saving diagnostics figure to %s\n", output_png)
        create_diagnostics_figure(output_png, params, sampler, history, eval_records)

        mean_kl = mean(record.diag.mean_kl for record in eval_records)
        mean_accuracy = mean(record.diag.accuracy for record in eval_records)
        @printf("Done. Mean evaluation KL = %.6e, mean pdf accuracy = %.6f\n", mean_kl, mean_accuracy)
    else
        @printf("Done. Saved trained model without Langevin/PDF evaluation.\n")
    end
    return eval_records
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
