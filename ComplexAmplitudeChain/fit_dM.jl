#!/usr/bin/env julia

import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")
const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "fit_dM.toml")
const STATE_CHANNELS = 2
const JOINT_STATE_CHANNELS = 4
const JOINT_INPUT_CHANNELS = 5
const DEFAULT_TIME_FEATURES = "scalar"
const DEFAULT_TIME_FOURIER_FREQUENCIES = 0

function ensure_packages(packages::Vector{String})
    project_deps = Pkg.project().dependencies
    missing = String[]
    for pkg in packages
        haskey(project_deps, pkg) || push!(missing, pkg)
    end
    isempty(missing) || Pkg.add(missing)
    return nothing
end

ensure_packages(["BSON", "CUDA", "cuDNN", "Flux", "Functors", "GLMakie", "HDF5", "LaTeXStrings", "NNlib", "TOML"])

using BSON
using CUDA
using cuDNN
using Flux
using Functors
using HDF5
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics
using TOML

const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
isfile(STYLE_FILE) || error("Shared figure style file not found: $(STYLE_FILE)")
include(STYLE_FILE)
GLMakie.activate!()

include(joinpath(SCOREUNET_SRC, "Device.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "PeriodicConv.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "Blocks.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "UNet1D.jl"))
include(joinpath(SCOREUNET_SRC, "data", "DataPipeline.jl"))
include(joinpath(SCOREUNET_SRC, "training", "Trainer.jl"))

# These small type definitions are needed so BSON can deserialize the saved
# score and joint-score checkpoints without including the training scripts.
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

Base.@kwdef mutable struct JointLangevinConfig
    dt::Float64 = 1e-3
    sample_dt::Float64 = 2e-2
    nsteps::Int = 20_000
    resolution::Int = 20
    n_ensembles::Int = 256
    burn_in::Int = 4_000
    sigma::Float32 = 0.1f0
    seed::Int = 21
    progress::Bool = false
end

struct JointScoreUNet{M}
    backbone::M
end

Functors.@functor JointScoreUNet (backbone,)

function (model::JointScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:JOINT_STATE_CHANNELS, :]
end

struct ReverseConditionalScoreUNet{M}
    backbone::M
end

Functors.@functor ReverseConditionalScoreUNet (backbone,)

function (model::ReverseConditionalScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:STATE_CHANNELS, :]
end

struct FitDMParams
    input_hdf5::String
    score_bson::String
    joint_score_bson::String
    reverse_cond_score_bson::String
    burnin_fraction::Float64
    tau_min::Float64
    tau_max::Float64
    lag_stride::Int
    max_fit_lags::Int
    pairs_per_lag_correlation::Int
    pairs_per_lag_phi::Int
    pairs_per_lag_operator::Int
    batch_size::Int
    score_batch_size::Int
    joint_batch_size::Int
    phi_fit_max_lag::Int
    phi_fit_degree::Int
    phi_projection::String
    phi_offsite_warn_ratio::Float64
    conditional_score_source::String
    cdot_local_window::Int
    cdot_poly_degree::Int
    ridge::Float64
    include_coordinates::Bool
    include_cubic_amplitude_coordinates::Bool
    include_amplitude_power_coordinates::Bool
    include_local_amplitude::Bool
    include_laplacian_coordinates::Bool
    include_neighbor_weighted_coordinates::Bool
    neighbor_offsets::Vector{Int}
    amplitude_coordinate_powers::Vector{Int}
    max_selected_observables::Int
    selection_min_signal_fraction::Float64
    selection_min_snr::Float64
    selection_max_roughness::Float64
    selection_max_generator_rel::Float64
    selection_force_coordinates::Bool
    artifact_bson::String
    metrics_txt::String
    figure_png::String
    run_mode::String
    device_name::String
    seed::Int
    verbose::Bool
end

struct PairSampler
    times::Vector{Float64}
    states::Array{Float32, 4}   # time, site, channel(q,p), trajectory
    start_idx::Int
    save_dt::Float64
    K::Int
    D::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    decorrelation_time::Float64
end

struct LoadedModels
    score_model
    joint_model
    score_sigma::Float32
    joint_sigma::Float32
    score_mean::Matrix{Float32} # K x 2
    score_std::Matrix{Float32}
    joint_mean::Matrix{Float32} # K x 4
    joint_std::Matrix{Float32}
    tau_min::Float64
    tau_max::Float64
    joint_input_channels::Int
    joint_time_features::String
    joint_time_fourier_frequencies::Int
    joint_include_delta_input::Bool
end

struct LoadedReverseConditionalModel
    model
    sigma::Float32
    mean::Matrix{Float32}
    std::Matrix{Float32}
    tau_min::Float64
    tau_max::Float64
    input_channels::Int
    time_features::String
    time_fourier_frequencies::Int
    include_delta_input::Bool
    score_type::String
end

struct ObservableLibrary
    names::Vector{String}
    kind::Vector{Symbol}
    component::Vector{Int}
    offset::Vector{Int}
    mean::Vector{Float64}
    mean_r2::Float64
end

struct MobilityFitResult
    coefficients::Vector{Float64}
    coefficient_names::Vector{String}
    normal_matrix::Matrix{Float64}
    rhs::Vector{Float64}
    condition_number::Float64
    residual_rmse::Float64
    target_rms::Float64
    relative_rmse::Float64
end

struct MobilityNNHistory
    epochs::Vector{Int}
    losses::Vector{Float64}
    weights::Vector{Vector{Float64}}
end

struct MobilityMLPHistory
    epochs::Vector{Int}
    losses::Vector{Float64}
    validation_rmse::Vector{Float64}
end

struct MobilityMLPHistoryDetailed
    epochs::Vector{Int}
    losses::Vector{Float64}
    validation_rmse::Vector{Float64}
    mean_abs_delta::Vector{Float64}
    rms_delta::Vector{Float64}
end

struct MobilityDirectTrainingCache
    features::Array{Float32, 4}     # feature(q,p), site, pair, lag
    scond::Array{Float32, 4}        # site, component, pair, lag
    observables::Array{Float32, 4}  # site, observable, pair, lag
    mean_features::Matrix{Float32}  # feature, samples
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

resolve_path(base_dir::AbstractString, path::AbstractString) = isabspath(path) ? path : normpath(joinpath(base_dir, path))
ensure_parent_dir(path::AbstractString) = (mkpath(dirname(path)); nothing)
periodic(i::Int, K::Int) = mod1(i, K)
to_host(x) = x isa AbstractArray && !(x isa Array) ? Array(x) : x

function detect_device(name::AbstractString)
    normalized = uppercase(strip(name))
    normalized == "AUTO" && return CUDA.functional() ? select_device("GPU:0") : CPUDevice()
    normalized == "GPU" && return select_device("GPU:0")
    return select_device(name)
end

describe_device(device::ExecutionDevice) = device isa GPUDevice ? "GPU:" * join(device.ids, ",") : "CPU"

function dict_get(d, key::Symbol)
    haskey(d, key) && return d[key]
    skey = String(key)
    haskey(d, skey) && return d[skey]
    error("Key $(key) not found.")
end

function dict_haskey(d, key::Symbol)
    return haskey(d, key) || haskey(d, String(key))
end

function dict_get_default(d, key::Symbol, default)
    haskey(d, key) && return d[key]
    skey = String(key)
    haskey(d, skey) && return d[skey]
    return default
end

function normalize_time_features(name)
    s = lowercase(String(name))
    s in ("scalar", "fourier") || error("Unsupported time_features=$(name); allowed values are scalar, fourier.")
    return s
end

time_feature_count(features::AbstractString, nfreq::Int) =
    features == "fourier" ? 1 + 2 * nfreq : 1

joint_input_channels(features::AbstractString, nfreq::Int; include_delta_input::Bool=false) =
    JOINT_STATE_CHANNELS + (include_delta_input ? 2 : 0) + time_feature_count(features, nfreq)

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    eval = raw["evaluation"]
    obs = raw["observables"]
    out = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    params = FitDMParams(
        String(data["input_hdf5"]),
        String(data["score_bson"]),
        String(data["joint_score_bson"]),
        String(get(data, "reverse_cond_score_bson", "")),
        Float64(data["burnin_fraction"]),
        Float64(eval["tau_min"]),
        Float64(eval["tau_max"]),
        Int(eval["lag_stride"]),
        Int(eval["max_fit_lags"]),
        Int(eval["pairs_per_lag_correlation"]),
        Int(get(eval, "pairs_per_lag_phi", eval["pairs_per_lag_correlation"])),
        Int(eval["pairs_per_lag_operator"]),
        Int(eval["batch_size"]),
        Int(eval["score_batch_size"]),
        Int(eval["joint_batch_size"]),
        Int(eval["phi_fit_max_lag"]),
        Int(eval["phi_fit_degree"]),
        String(get(eval, "phi_projection", "local_complex")),
        Float64(get(eval, "phi_offsite_warn_ratio", 0.05)),
        String(get(eval, "conditional_score_source", "reverse")),
        Int(eval["cdot_local_window"]),
        Int(eval["cdot_poly_degree"]),
        Float64(eval["ridge"]),
        Bool(get(obs, "include_coordinates", true)),
        Bool(get(obs, "include_cubic_amplitude_coordinates", true)),
        Bool(get(obs, "include_amplitude_power_coordinates", true)),
        Bool(get(obs, "include_local_amplitude", true)),
        Bool(get(obs, "include_laplacian_coordinates", true)),
        Bool(get(obs, "include_neighbor_weighted_coordinates", true)),
        Int.(get(obs, "neighbor_offsets", [1, 2, 4])),
        Int.(get(obs, "amplitude_coordinate_powers", [2, 3])),
        Int(get(obs, "max_selected_observables", 10)),
        Float64(get(obs, "selection_min_signal_fraction", 0.06)),
        Float64(get(obs, "selection_min_snr", 2.5)),
        Float64(get(obs, "selection_max_roughness", 2.0)),
        Float64(get(obs, "selection_max_generator_rel", 0.18)),
        Bool(get(obs, "selection_force_coordinates", true)),
        String(out["artifact_bson"]),
        String(out["metrics_txt"]),
        String(get(out, "figure_png", "outputs/fit_dM_cdot_trueM.png")),
        String(get(run, "mode", "full")),
        String(get(run, "device", "AUTO")),
        Int(get(run, "seed", 20260507)),
        Bool(get(run, "verbose", true)),
    )
    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must lie in [0, 1).")
    require_condition(params.tau_min > 0.0 && params.tau_max >= params.tau_min, "Invalid tau window.")
    require_condition(params.lag_stride >= 1, "lag_stride must be positive.")
    require_condition(params.max_fit_lags >= 1, "max_fit_lags must be positive.")
    require_condition(params.pairs_per_lag_correlation >= 1024, "pairs_per_lag_correlation is too small.")
    require_condition(params.pairs_per_lag_phi >= 1024, "pairs_per_lag_phi is too small.")
    require_condition(params.pairs_per_lag_operator >= 1024, "pairs_per_lag_operator is too small.")
    require_condition(params.batch_size >= 128, "batch_size is too small.")
    require_condition(params.score_batch_size >= 128, "score_batch_size is too small.")
    require_condition(params.joint_batch_size >= 128, "joint_batch_size is too small.")
    require_condition(params.phi_fit_max_lag >= 1, "phi_fit_max_lag must be positive.")
    require_condition(params.phi_fit_degree >= 1, "phi_fit_degree must be at least one.")
    require_condition(params.phi_projection in ("full_profile", "local_2x2", "local_complex"),
        "phi_projection must be one of full_profile, local_2x2, local_complex.")
    require_condition(params.phi_offsite_warn_ratio >= 0.0, "phi_offsite_warn_ratio must be nonnegative.")
    require_condition(params.conditional_score_source in ("reverse", "joint"),
        "conditional_score_source must be either reverse or joint.")
    require_condition(params.cdot_local_window >= 3, "cdot_local_window must be at least three.")
    require_condition(params.cdot_poly_degree >= 1, "cdot_poly_degree must be at least one.")
    require_condition(params.ridge >= 0.0, "ridge must be nonnegative.")
    require_condition(!isempty(params.neighbor_offsets), "neighbor_offsets must not be empty.")
    require_condition(all(p -> p >= 1, params.amplitude_coordinate_powers), "amplitude_coordinate_powers must be positive.")
    require_condition(params.max_selected_observables >= 2, "max_selected_observables must be at least two.")
    require_condition(params.selection_min_signal_fraction >= 0.0, "selection_min_signal_fraction must be nonnegative.")
    require_condition(params.selection_min_snr >= 0.0, "selection_min_snr must be nonnegative.")
    require_condition(params.selection_max_roughness > 0.0, "selection_max_roughness must be positive.")
    require_condition(params.selection_max_generator_rel > 0.0, "selection_max_generator_rel must be positive.")
    require_condition(params.run_mode in ("full", "cdot_diagnostic", "cdot_plot_only",
            "mobility_only", "mobility_fit_only", "phi_constant_only"),
        "run.mode must be \"full\", \"cdot_diagnostic\", \"cdot_plot_only\", \"mobility_only\", \"mobility_fit_only\", or \"phi_constant_only\".")
    return params
end

function load_state_tensor(path::AbstractString, burnin_fraction::Float64, tau_min::Float64,
        tau_max::Float64, lag_stride::Int, max_fit_lags::Int)
    times = Float64.(h5read(path, "/trajectories/time"))
    states_raw = h5read(path, "/trajectories/states")
    require_condition(ndims(states_raw) == 4, "Expected /trajectories/states to have shape time x site x channel x trajectory.")
    states = size(states_raw, 1) == length(times) ? Float32.(states_raw) :
        error("Could not infer time axis for /trajectories/states with shape $(size(states_raw)).")
    nt, K, C, ntraj = size(states)
    require_condition(C == STATE_CHANNELS, "Expected q/p channel count of 2.")
    save_dt = length(times) > 1 ? times[2] - times[1] : 0.0
    require_condition(save_dt > 0.0, "Need positive saved time step.")
    start_idx = clamp(1 + floor(Int, burnin_fraction * (nt - 1)), 1, nt)
    tdec = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    min_lag = max(1, ceil(Int, tau_min / save_dt - 1e-9))
    max_lag = min(nt - start_idx, floor(Int, tau_max / save_dt + 1e-9))
    require_condition(max_lag >= min_lag, "No positive lags available in requested window.")
    lag_steps = collect(min_lag:lag_stride:max_lag)
    if length(lag_steps) > max_fit_lags
        lag_steps = lag_steps[1:max_fit_lags]
    end
    lag_times = lag_steps .* save_dt
    @printf("Loaded states: time=%d, K=%d, channels=%d, trajectories=%d\n", nt, K, C, ntraj)
    @printf("Post-burnin starts at index %d; save_dt=%.6f; fit lag window [%.3f, %.3f] with %d lags\n",
        start_idx, save_dt, first(lag_times), last(lag_times), length(lag_times))
    return PairSampler(times, states, start_idx, save_dt, K, 2K, lag_steps, lag_times, tdec)
end

function stats_matrix(raw_stats, field::Symbol, K::Int, C::Int)
    raw = raw_stats isa DataStats ? getfield(raw_stats, field) : dict_get(raw_stats, field)
    arr = Float32.(to_host(raw))
    if ndims(arr) == 2
        size(arr) == (K, C) && return arr
        size(arr) == (C, K) && return permutedims(arr, (2, 1))
    end
    vec_arr = vec(arr)
    if length(vec_arr) == 1
        return fill(vec_arr[1], K, C)
    elseif length(vec_arr) == C
        return repeat(reshape(vec_arr, 1, C), K, 1)
    elseif length(vec_arr) == K * C
        return reshape(vec_arr, K, C)
    else
        error("Normalization stats for $(field) have incompatible length $(length(vec_arr)); expected 1, $(C), or $(K*C).")
    end
end

function load_models(score_path::AbstractString, joint_path::AbstractString, device::ExecutionDevice, K::Int)
    score_blob = BSON.load(score_path)
    joint_blob = BSON.load(joint_path)
    score_model = move_model(dict_get(score_blob, :host_model), device)
    joint_model = move_model(dict_get(joint_blob, :host_model), device)
    Flux.testmode!(score_model)
    Flux.testmode!(joint_model)
    score_trainer = dict_get(score_blob, :trainer_cfg)
    joint_trainer = dict_get(joint_blob, :trainer_cfg)
    score_stats = dict_get(score_blob, :stats)
    joint_stats = dict_get(joint_blob, :stats)
    joint_meta = dict_get(joint_blob, :metadata)
    joint_time_features = normalize_time_features(dict_get_default(joint_meta, :time_features, DEFAULT_TIME_FEATURES))
    joint_time_fourier_frequencies = Int(dict_get_default(joint_meta, :time_fourier_frequencies, DEFAULT_TIME_FOURIER_FREQUENCIES))
    joint_include_delta_input = Bool(dict_get_default(joint_meta, :include_delta_input, false))
    joint_in_channels = joint_input_channels(joint_time_features, joint_time_fourier_frequencies;
        include_delta_input=joint_include_delta_input)
    return LoadedModels(
        score_model,
        joint_model,
        Float32(score_trainer.sigma),
        Float32(joint_trainer.sigma),
        stats_matrix(score_stats, :mean, K, 2),
        stats_matrix(score_stats, :std, K, 2),
        stats_matrix(joint_stats, :mean, K, 4),
        stats_matrix(joint_stats, :std, K, 4),
        Float64(dict_get(joint_meta, :tau_min)),
        Float64(dict_get(joint_meta, :tau_max)),
        joint_in_channels,
        joint_time_features,
        joint_time_fourier_frequencies,
        joint_include_delta_input,
    )
end

function load_reverse_conditional_model(path::AbstractString, device::ExecutionDevice, K::Int)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    trainer = dict_get(blob, :trainer_cfg)
    stats = dict_get(blob, :stats)
    meta = dict_get(blob, :metadata)
    time_features = normalize_time_features(dict_get_default(meta, :time_features, DEFAULT_TIME_FEATURES))
    time_fourier_frequencies = Int(dict_get_default(meta, :time_fourier_frequencies, DEFAULT_TIME_FOURIER_FREQUENCIES))
    include_delta_input = Bool(dict_get_default(meta, :include_delta_input, false))
    in_channels = joint_input_channels(time_features, time_fourier_frequencies;
        include_delta_input=include_delta_input)
    score_type = String(dict_get_default(meta, :score_type, "reverse_conditional_x0_given_xt"))
    return LoadedReverseConditionalModel(
        model,
        Float32(trainer.sigma),
        stats_matrix(stats, :mean, K, 4),
        stats_matrix(stats, :std, K, 4),
        Float64(dict_get(meta, :tau_min)),
        Float64(dict_get(meta, :tau_max)),
        in_channels,
        time_features,
        time_fourier_frequencies,
        include_delta_input,
        score_type,
    )
end

function flat_index(site::Int, component::Int, K::Int)
    return (component - 1) * K + site
end

function site_component(dim_idx::Int, K::Int)
    component = dim_idx <= K ? 1 : 2
    site = component == 1 ? dim_idx : dim_idx - K
    return site, component
end

function flatten_state!(dest::AbstractMatrix{Float32}, z::AbstractArray{Float32, 3})
    K, C, B = size(z)
    @inbounds for b in 1:B, c in 1:C, i in 1:K
        dest[flat_index(i, c, K), b] = z[i, c, b]
    end
    return dest
end

function sample_pair_batch!(z0::AbstractArray{Float32, 3}, zt::AbstractArray{Float32, 3},
        sampler::PairSampler, lag::Int, rng::AbstractRNG)
    nt, K, C, ntraj = size(sampler.states)
    B = size(z0, 3)
    upper = nt - lag
    require_condition(upper >= sampler.start_idx, "Lag exceeds available post-burnin window.")
    @inbounds for b in 1:B
        traj = rand(rng, 1:ntraj)
        t = rand(rng, sampler.start_idx:upper)
        for c in 1:C, i in 1:K
            z0[i, c, b] = sampler.states[t, i, c, traj]
            zt[i, c, b] = sampler.states[t + lag, i, c, traj]
        end
    end
    return nothing
end

function compute_coordinate_means(states::Array{Float32, 4}, start_idx::Int)
    post = @view states[start_idx:end, :, :, :]
    means = zeros(Float64, 2)
    means[1] = mean(@view post[:, :, 1, :])
    means[2] = mean(@view post[:, :, 2, :])
    return means
end

function build_observable_library(states::Array{Float32, 4}, start_idx::Int, params::FitDMParams)
    nt, K, _, ntraj = size(states)
    r2_sum = 0.0
    count = 0
    for traj in 1:ntraj, t in start_idx:nt, i in 1:K
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        r2_sum += q*q + p*p
        count += 1
    end
    mean_r2 = r2_sum / count
    names = String[]
    kind = Symbol[]
    component = Int[]
    offset = Int[]
    means = Float64[]
    add!(nm, kd, comp, off, μ=0.0) = (push!(names, nm); push!(kind, kd); push!(component, comp); push!(offset, off); push!(means, μ))
    if params.include_coordinates
        add!("q_i", :coord, 1, 0, 0.0)
        add!("p_i", :coord, 2, 0, 0.0)
    end
    if params.include_cubic_amplitude_coordinates
        add!("(r_i^2-<r^2>) q_i", :ampcoord, 1, 0, 0.0)
        add!("(r_i^2-<r^2>) p_i", :ampcoord, 2, 0, 0.0)
    end
    if params.include_amplitude_power_coordinates
        for power in unique(sort(params.amplitude_coordinate_powers))
            power == 1 && continue
            add!(@sprintf("r_i^%d q_i", 2power), :ampcoord_power, 1, power, 0.0)
            add!(@sprintf("r_i^%d p_i", 2power), :ampcoord_power, 2, power, 0.0)
        end
    end
    if params.include_laplacian_coordinates
        add!("lap q_i", :lapcoord, 1, 0, 0.0)
        add!("lap p_i", :lapcoord, 2, 0, 0.0)
    end
    if params.include_local_amplitude
        add!("r_i^2-<r^2>", :amp, 0, 0, 0.0)
    end
    for off in params.neighbor_offsets
        off_mod = mod(off, K)
        off_mod == 0 && continue
        add!(@sprintf("z_i dot z_i+%d", off), :neighbor_dot, 0, off_mod, 0.0)
        add!(@sprintf("z_i cross z_i+%d", off), :neighbor_cross, 0, off_mod, 0.0)
        if params.include_neighbor_weighted_coordinates
            add!(@sprintf("(z_i dot z_i+%d) q_i", off), :neighbor_dot_coord, 1, off_mod, 0.0)
            add!(@sprintf("(z_i dot z_i+%d) p_i", off), :neighbor_dot_coord, 2, off_mod, 0.0)
            add!(@sprintf("(z_i cross z_i+%d) q_i", off), :neighbor_cross_coord, 1, off_mod, 0.0)
            add!(@sprintf("(z_i cross z_i+%d) p_i", off), :neighbor_cross_coord, 2, off_mod, 0.0)
        end
    end

    lib = ObservableLibrary(names, kind, component, offset, means, mean_r2)
    # Center observables empirically from a reproducible subset.  The full
    # post-burnin tensor has millions of site samples, and the expanded
    # candidate library makes a full centering pass unnecessarily expensive.
    sum_vals = zeros(Float64, length(names))
    total = zeros(Int, length(names))
    scratch = Array{Float32}(undef, K, 2, 1)
    vals = Matrix{Float32}(undef, K, length(names))
    center_site_budget = 500_000
    center_states = min((nt - start_idx + 1) * ntraj, max(1, cld(center_site_budget, K)))
    rng = MersenneTwister(params.seed + 5)
    for _ in 1:center_states
        traj = rand(rng, 1:ntraj)
        t = rand(rng, start_idx:nt)
        @inbounds for c in 1:2, i in 1:K
            scratch[i, c, 1] = states[t, i, c, traj]
        end
        observable_values!(vals, @view(scratch[:, :, 1]), lib; center=false)
        @inbounds for a in 1:length(names)
            sum_vals[a] += sum(@view vals[:, a])
            total[a] += K
        end
    end
    centered_means = sum_vals ./ total
    @printf("Observable library (%d site-translated channels): %s\n", length(names), join(names, " | "))
    @printf("Empirical <r^2> = %.8f; centered observables with %d sampled site values\n",
        mean_r2, center_states * K)
    return ObservableLibrary(names, kind, component, offset, centered_means, mean_r2)
end

function observable_values!(out::AbstractMatrix{Float32}, z::AbstractMatrix{Float32},
        lib::ObservableLibrary; center::Bool=true)
    K = size(z, 1)
    @inbounds for a in eachindex(lib.names)
        kd = lib.kind[a]
        comp = lib.component[a]
        off = lib.offset[a]
        μ = center ? Float32(lib.mean[a]) : 0.0f0
        for i in 1:K
            q = z[i, 1]
            p = z[i, 2]
            r2 = q*q + p*p
            val = if kd == :coord
                z[i, comp]
            elseif kd == :ampcoord
                (r2 - Float32(lib.mean_r2)) * z[i, comp]
            elseif kd == :ampcoord_power
                (r2 ^ off) * z[i, comp]
            elseif kd == :lapcoord
                im1 = periodic(i - 1, K)
                ip1 = periodic(i + 1, K)
                2.0f0 * z[i, comp] - z[im1, comp] - z[ip1, comp]
            elseif kd == :amp
                r2
            elseif kd == :neighbor_dot
                j = periodic(i + off, K)
                q * z[j, 1] + p * z[j, 2]
            elseif kd == :neighbor_cross
                j = periodic(i + off, K)
                q * z[j, 2] - p * z[j, 1]
            elseif kd == :neighbor_dot_coord
                j = periodic(i + off, K)
                (q * z[j, 1] + p * z[j, 2]) * z[i, comp]
            elseif kd == :neighbor_cross_coord
                j = periodic(i + off, K)
                (q * z[j, 2] - p * z[j, 1]) * z[i, comp]
            else
                error("Unsupported observable kind $(kd).")
            end
            out[i, a] = val - μ
        end
    end
    return out
end

function normalize_score_input!(dest::AbstractArray{Float32, 3}, z::AbstractArray{Float32, 3},
        mean::Matrix{Float32}, std::Matrix{Float32})
    K, C, B = size(z)
    @inbounds for b in 1:B, c in 1:C, i in 1:K
        dest[i, c, b] = (z[i, c, b] - mean[i, c]) / std[i, c]
    end
    return dest
end

function normalize_joint_input!(dest::AbstractArray{Float32, 3}, z0::AbstractArray{Float32, 3},
        zt::AbstractArray{Float32, 3}, tnorm::Float32, mean::Matrix{Float32}, std::Matrix{Float32};
        time_features::AbstractString=DEFAULT_TIME_FEATURES,
        time_fourier_frequencies::Int=DEFAULT_TIME_FOURIER_FREQUENCIES,
        include_delta_input::Bool=false)
    K, _, B = size(z0)
    expected_channels = joint_input_channels(time_features, time_fourier_frequencies;
        include_delta_input=include_delta_input)
    require_condition(size(dest, 2) >= expected_channels,
        @sprintf("Joint input buffer has %d channels but %s time features require %d.",
            size(dest, 2), time_features, expected_channels))
    @inbounds for b in 1:B, i in 1:K
        dest[i, 1, b] = (z0[i, 1, b] - mean[i, 1]) / std[i, 1]
        dest[i, 2, b] = (z0[i, 2, b] - mean[i, 2]) / std[i, 2]
        dest[i, 3, b] = (zt[i, 1, b] - mean[i, 3]) / std[i, 3]
        dest[i, 4, b] = (zt[i, 2, b] - mean[i, 4]) / std[i, 4]
        ch = JOINT_STATE_CHANNELS + 1
        if include_delta_input
            dest[i, ch, b] = dest[i, 3, b] - dest[i, 1, b]
            dest[i, ch + 1, b] = dest[i, 4, b] - dest[i, 2, b]
            ch += 2
        end
        dest[i, ch, b] = tnorm
        ch += 1
        if time_features == "fourier"
            t64 = Float64(tnorm)
            for freq in 1:time_fourier_frequencies
                angle = Float32(2.0 * pi * freq * t64)
                dest[i, ch, b] = sin(angle)
                dest[i, ch + 1, b] = cos(angle)
                ch += 2
            end
        end
    end
    return dest
end

function evaluate_stationary_score(models::LoadedModels, z0::AbstractArray{Float32, 3},
        batch_size::Int, device::ExecutionDevice)
    K, _, N = size(z0)
    out = Array{Float32}(undef, K, 2, N)
    scratch = Array{Float32}(undef, K, 2, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        bn = stop - start + 1
        normalize_score_input!(@view(scratch[:, :, 1:bn]), @view(z0[:, :, start:stop]), models.score_mean, models.score_std)
        pred = to_host(score_from_model(models.score_model, move_array(@view(scratch[:, :, 1:bn]), device), models.score_sigma))
        @inbounds for b in 1:bn, c in 1:2, i in 1:K
            out[i, c, start + b - 1] = pred[i, c, b] / models.score_std[i, c]
        end
    end
    return out
end

function evaluate_joint_score_x0(models::LoadedModels, z0::AbstractArray{Float32, 3},
        zt::AbstractArray{Float32, 3}, tnorm::Float32, batch_size::Int, device::ExecutionDevice)
    K, _, N = size(z0)
    out = Array{Float32}(undef, K, 2, N)
    scratch = Array{Float32}(undef, K, models.joint_input_channels, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        bn = stop - start + 1
        normalize_joint_input!(@view(scratch[:, :, 1:bn]), @view(z0[:, :, start:stop]),
            @view(zt[:, :, start:stop]), tnorm, models.joint_mean, models.joint_std;
            time_features=models.joint_time_features,
            time_fourier_frequencies=models.joint_time_fourier_frequencies,
            include_delta_input=models.joint_include_delta_input)
        pred = to_host(score_from_model(models.joint_model, move_array(@view(scratch[:, :, 1:bn]), device), models.joint_sigma))
        @inbounds for b in 1:bn, c in 1:2, i in 1:K
            out[i, c, start + b - 1] = pred[i, c, b] / models.joint_std[i, c]
        end
    end
    return out
end

function evaluate_conditional_score_x0(models::LoadedModels, z0::AbstractArray{Float32, 3},
        zt::AbstractArray{Float32, 3}, tau::Float64, params::FitDMParams, device::ExecutionDevice)
    require_condition(models.tau_min <= tau <= models.tau_max + 1e-10,
        @sprintf("tau %.6f lies outside joint-score range [%.6f, %.6f]", tau, models.tau_min, models.tau_max))
    tnorm = Float32((tau - models.tau_min) / max(models.tau_max - models.tau_min, eps(Float64)))
    joint = evaluate_joint_score_x0(models, z0, zt, tnorm, params.joint_batch_size, device)
    stat = evaluate_stationary_score(models, z0, params.score_batch_size, device)
    return joint .- stat
end

function evaluate_reverse_posterior_score_x0(reverse_model::LoadedReverseConditionalModel,
        z0::AbstractArray{Float32, 3}, zt::AbstractArray{Float32, 3}, tau::Float64,
        batch_size::Int, device::ExecutionDevice)
    require_condition(reverse_model.tau_min <= tau <= reverse_model.tau_max + 1e-10,
        @sprintf("tau %.6f lies outside reverse conditional-score range [%.6f, %.6f]",
            tau, reverse_model.tau_min, reverse_model.tau_max))
    K, _, B = size(z0)
    tnorm = Float32((tau - reverse_model.tau_min) /
        max(reverse_model.tau_max - reverse_model.tau_min, eps(Float64)))
    scratch = Array{Float32}(undef, K, reverse_model.input_channels, B)
    out = Array{Float32}(undef, K, STATE_CHANNELS, B)
    for start in 1:batch_size:B
        stop = min(B, start + batch_size - 1)
        bn = stop - start + 1
        normalize_joint_input!(@view(scratch[:, :, 1:bn]), @view(z0[:, :, start:stop]),
            @view(zt[:, :, start:stop]), tnorm, reverse_model.mean, reverse_model.std;
            time_features=reverse_model.time_features,
            time_fourier_frequencies=reverse_model.time_fourier_frequencies,
            include_delta_input=reverse_model.include_delta_input)
        pred = to_host(score_from_model(reverse_model.model,
            move_array(@view(scratch[:, :, 1:bn]), device), reverse_model.sigma))
        @inbounds for b in 1:bn, c in 1:STATE_CHANNELS, i in 1:K
            out[i, c, start + b - 1] = pred[i, c, b] / reverse_model.std[i, c]
        end
    end
    return out
end

function evaluate_reverse_conditional_score_x0(models::LoadedModels,
        reverse_model::LoadedReverseConditionalModel, z0::AbstractArray{Float32, 3},
        zt::AbstractArray{Float32, 3}, tau::Float64, params::FitDMParams,
        device::ExecutionDevice)
    if reverse_model.score_type == "reverse_transition_residual_x0_given_xt"
        return evaluate_reverse_posterior_score_x0(reverse_model, z0, zt, tau,
            params.joint_batch_size, device)
    end
    posterior = evaluate_reverse_posterior_score_x0(reverse_model, z0, zt, tau,
        params.joint_batch_size, device)
    stat = evaluate_stationary_score(models, z0, params.score_batch_size, device)
    return posterior .- stat
end

function evaluate_reverse_conditional_score_x0_exact_stationary(
        reverse_model::LoadedReverseConditionalModel, z0::AbstractArray{Float32, 3},
        zt::AbstractArray{Float32, 3}, tau::Float64, params::FitDMParams,
        device::ExecutionDevice, potential_params::Tuple{Float64, Float64, Float64})
    if reverse_model.score_type == "reverse_transition_residual_x0_given_xt"
        return evaluate_reverse_posterior_score_x0(reverse_model, z0, zt, tau,
            params.joint_batch_size, device)
    end
    posterior = evaluate_reverse_posterior_score_x0(reverse_model, z0, zt, tau,
        params.joint_batch_size, device)
    exact = similar(posterior)
    exact_stationary_score!(exact, z0, potential_params...)
    return posterior .- exact
end

function evaluate_transition_score_x0(models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel},
        z0::AbstractArray{Float32, 3}, zt::AbstractArray{Float32, 3},
        tau::Float64, params::FitDMParams, device::ExecutionDevice;
        source::String=params.conditional_score_source)
    if source == "reverse"
        require_condition(reverse_model !== nothing,
            "conditional_score_source=\"reverse\" requires a reverse_cond_score_bson.")
        return evaluate_reverse_conditional_score_x0(models, reverse_model, z0, zt, tau, params, device)
    elseif source == "joint"
        return evaluate_conditional_score_x0(models, z0, zt, tau, params, device)
    else
        error("Unknown conditional score source: $(source)")
    end
end

function accumulate_observable_coordinate_correlation!(sumC::Array{Float64, 3},
        obs_vals::AbstractArray{Float32, 3}, x0_flat::AbstractMatrix{Float32})
    K, nobs, B = size(obs_vals)
    D = size(x0_flat, 1)
    require_condition(B == size(x0_flat, 2), "Observable and coordinate batches have inconsistent sizes.")
    sum_mat = reshape(sumC, K * nobs, D)
    obs_mat = Float64.(reshape(obs_vals, K * nobs, B))
    x_mat = Float64.(x0_flat)
    mul!(sum_mat, obs_mat, transpose(x_mat), 1.0, 1.0)
    return nothing
end

function estimate_correlations(sampler::PairSampler, lib::ObservableLibrary, params::FitDMParams,
        coord_mean::Vector{Float64})
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    short_lags = collect(0:params.phi_fit_max_lag)
    lags = unique(sort(vcat(short_lags, sampler.lag_steps)))
    taus = lags .* sampler.save_dt
    C = Array{Float64}(undef, length(lags), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    rng = MersenneTwister(params.seed + 10)
    for (lag_idx, lag) in enumerate(lags)
        sumC = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_correlation
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            accumulate_observable_coordinate_correlation!(sumC, @view(obs_vals[:, :, 1:bn]), @view(x0_flat[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        C[lag_idx, :, :, :] .= sumC ./ total
        params.verbose && @printf("Estimated data C(t) at lag %d / %d (tau=%.4f)\n", lag_idx, length(lags), taus[lag_idx])
    end
    return lags, taus, C
end

function sample_centered_difference_batch!(z0::AbstractArray{Float32, 3},
        zminus::AbstractArray{Float32, 3}, zplus::AbstractArray{Float32, 3},
        sampler::PairSampler, lag::Int, rng::AbstractRNG)
    nt, K, C, ntraj = size(sampler.states)
    B = size(z0, 3)
    require_condition(lag >= 1, "Centered-difference Cdot requires positive saved lag.")
    upper = nt - lag - 1
    lower = sampler.start_idx
    require_condition(upper >= lower, "Lag exceeds available post-burnin window for centered differences.")
    @inbounds for b in 1:B
        traj = rand(rng, 1:ntraj)
        t = rand(rng, lower:upper)
        for c in 1:C, i in 1:K
            z0[i, c, b] = sampler.states[t, i, c, traj]
            zminus[i, c, b] = sampler.states[t + lag - 1, i, c, traj]
            zplus[i, c, b] = sampler.states[t + lag + 1, i, c, traj]
        end
    end
    return nothing
end

function accumulate_observable_increment_cdot!(sumCdot::Array{Float64, 3},
        obs_plus::AbstractArray{Float32, 3}, obs_minus::AbstractArray{Float32, 3},
        x0_flat::AbstractMatrix{Float32}, inv_two_dt::Float64)
    K, nobs, B = size(obs_plus)
    D = size(x0_flat, 1)
    require_condition(size(obs_minus) == size(obs_plus), "Observable increment arrays have inconsistent sizes.")
    require_condition(B == size(x0_flat, 2), "Observable increment and coordinate batches have inconsistent sizes.")
    sum_mat = reshape(sumCdot, K * nobs, D)
    diff_mat = Float64.(reshape(obs_plus, K * nobs, B) .- reshape(obs_minus, K * nobs, B))
    diff_mat .*= inv_two_dt
    x_mat = Float64.(x0_flat)
    mul!(sum_mat, diff_mat, transpose(x_mat), 1.0, 1.0)
    return nothing
end

function estimate_data_cdot_from_centered_observable_increments(sampler::PairSampler,
        lib::ObservableLibrary, params::FitDMParams, coord_mean::Vector{Float64})
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot = Array{Float64}(undef, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zminus = similar(z0)
    zplus = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    obs_minus = Array{Float32}(undef, K, nobs, params.batch_size)
    obs_plus = similar(obs_minus)
    inv_two_dt = 1.0 / (2.0 * sampler.save_dt)
    rng = MersenneTwister(params.seed + 70)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        sums = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_correlation
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_centered_difference_batch!(@view(z0[:, :, 1:bn]), @view(zminus[:, :, 1:bn]),
                @view(zplus[:, :, 1:bn]), sampler, lag, rng)
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            for b in 1:bn
                observable_values!(@view(obs_minus[:, :, b]), @view(zminus[:, :, b]), lib)
                observable_values!(@view(obs_plus[:, :, b]), @view(zplus[:, :, b]), lib)
            end
            accumulate_observable_increment_cdot!(sums, @view(obs_plus[:, :, 1:bn]), @view(obs_minus[:, :, 1:bn]),
                @view(x0_flat[:, 1:bn]), inv_two_dt)
            total += bn
            remaining -= bn
        end
        Cdot[lag_idx, :, :, :] .= sums ./ total
        params.verbose && @printf("Estimated data Cdot by centered observable increments for tau %.4f (%d / %d)\n",
            sampler.lag_times[lag_idx], lag_idx, length(sampler.lag_steps))
    end
    return Cdot
end

function drift_values!(b::AbstractArray{Float32, 3}, z::AbstractArray{Float32, 3},
        alpha::Float64, beta::Float64, kappa::Float64,
        d0::Float64, d1::Float64, omega0::Float64, omega1::Float64)
    K, _, B = size(z)
    @inbounds for sample in 1:B, i in 1:K
        im1 = periodic(i - 1, K)
        ip1 = periodic(i + 1, K)
        q = Float64(z[i, 1, sample])
        p = Float64(z[i, 2, sample])
        r2 = q*q + p*p
        d = d0 + d1 * r2
        omega = omega0 + omega1 * r2
        gq = alpha * q + beta * r2 * q + kappa * (2.0 * q - Float64(z[im1, 1, sample]) - Float64(z[ip1, 1, sample]))
        gp = alpha * p + beta * r2 * p + kappa * (2.0 * p - Float64(z[im1, 2, sample]) - Float64(z[ip1, 2, sample]))
        b[i, 1, sample] = Float32(-d * gq + omega * gp + 2.0 * d1 * q - 2.0 * omega1 * p)
        b[i, 2, sample] = Float32(-d * gp - omega * gq + 2.0 * d1 * p + 2.0 * omega1 * q)
    end
    return b
end

function generator_observable_values!(out::AbstractMatrix{Float32}, z::AbstractMatrix{Float32},
        drift::AbstractMatrix{Float32}, lib::ObservableLibrary,
        d0::Float64, d1::Float64)
    K = size(z, 1)
    @inbounds for a in eachindex(lib.names)
        kd = lib.kind[a]
        comp = lib.component[a]
        off = lib.offset[a]
        for i in 1:K
            q = Float64(z[i, 1])
            p = Float64(z[i, 2])
            bq = Float64(drift[i, 1])
            bp = Float64(drift[i, 2])
            r2 = q*q + p*p
            d = d0 + d1 * r2
            val = if kd == :coord
                comp == 1 ? bq : bp
            elseif kd == :ampcoord
                if comp == 1
                    (3.0 * q*q + p*p - lib.mean_r2) * bq + 2.0 * p * q * bp + 8.0 * d * q
                else
                    2.0 * q * p * bq + (q*q + 3.0 * p*p - lib.mean_r2) * bp + 8.0 * d * p
                end
            elseif kd == :ampcoord_power
                power = off
                rpow = r2^power
                rpow_m1 = power == 0 ? 0.0 : r2^(power - 1)
                if comp == 1
                    (rpow + 2.0 * power * q*q * rpow_m1) * bq +
                        (2.0 * power * p * q * rpow_m1) * bp +
                        4.0 * power * (power + 1.0) * d * q * rpow_m1
                else
                    (2.0 * power * q * p * rpow_m1) * bq +
                        (rpow + 2.0 * power * p*p * rpow_m1) * bp +
                        4.0 * power * (power + 1.0) * d * p * rpow_m1
                end
            elseif kd == :lapcoord
                im1 = periodic(i - 1, K)
                ip1 = periodic(i + 1, K)
                2.0 * Float64(drift[i, comp]) - Float64(drift[im1, comp]) - Float64(drift[ip1, comp])
            elseif kd == :amp
                2.0 * q * bq + 2.0 * p * bp + 4.0 * d
            elseif kd == :neighbor_dot
                j = periodic(i + off, K)
                bq * Float64(z[j, 1]) + bp * Float64(z[j, 2]) +
                    Float64(drift[j, 1]) * q + Float64(drift[j, 2]) * p
            elseif kd == :neighbor_cross
                j = periodic(i + off, K)
                bq * Float64(z[j, 2]) + q * Float64(drift[j, 2]) -
                    bp * Float64(z[j, 1]) - p * Float64(drift[j, 1])
            elseif kd == :neighbor_dot_coord
                j = periodic(i + off, K)
                qj = Float64(z[j, 1])
                pj = Float64(z[j, 2])
                dot_ij = q * qj + p * pj
                ldot = bq * qj + bp * pj + Float64(drift[j, 1]) * q + Float64(drift[j, 2]) * p
                zic = comp == 1 ? q : p
                bic = Float64(drift[i, comp])
                diffusion = 2.0 * d * (comp == 1 ? qj : pj)
                ldot * zic + dot_ij * bic + diffusion
            elseif kd == :neighbor_cross_coord
                j = periodic(i + off, K)
                qj = Float64(z[j, 1])
                pj = Float64(z[j, 2])
                cross_ij = q * pj - p * qj
                lcross = bq * pj + q * Float64(drift[j, 2]) -
                    bp * qj - p * Float64(drift[j, 1])
                zic = comp == 1 ? q : p
                bic = Float64(drift[i, comp])
                diffusion = 2.0 * d * (comp == 1 ? pj : -qj)
                lcross * zic + cross_ij * bic + diffusion
            else
                error("Unsupported observable kind $(kd).")
            end
            out[i, a] = Float32(val)
        end
    end
    return out
end

function estimate_generator_cdot_from_true_model(sampler::PairSampler, lib::ObservableLibrary,
        params::FitDMParams, coord_mean::Vector{Float64}, input_hdf5::AbstractString)
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot = Array{Float64}(undef, length(sampler.lag_steps), K, nobs, D)
    alpha, beta, kappa = load_potential_params(input_hdf5)
    d0, d1, omega0, omega1 = load_true_mobility_params(input_hdf5)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    drift = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    gen_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    rng = MersenneTwister(params.seed + 75)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        sums = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_correlation
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            drift_values!(@view(drift[:, :, 1:bn]), @view(zt[:, :, 1:bn]), alpha, beta, kappa, d0, d1, omega0, omega1)
            for b in 1:bn
                generator_observable_values!(@view(gen_vals[:, :, b]), @view(zt[:, :, b]),
                    @view(drift[:, :, b]), lib, d0, d1)
            end
            accumulate_observable_coordinate_correlation!(sums, @view(gen_vals[:, :, 1:bn]), @view(x0_flat[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot[lag_idx, :, :, :] .= sums ./ total
        params.verbose && @printf("Estimated ex-post generator Cdot for tau %.4f (%d / %d)\n",
            sampler.lag_times[lag_idx], lag_idx, length(sampler.lag_steps))
    end
    return Cdot
end

function polynomial_derivative_at(x::Vector{Float64}, y::Vector{Float64}, x0::Float64, degree::Int)
    deg = min(degree, length(x) - 1)
    X = hcat([x .^ k for k in 0:deg]...)
    coeff = X \ y
    deriv = 0.0
    for k in 1:deg
        deriv += k * coeff[k + 1] * x0^(k - 1)
    end
    return deriv
end

function local_polynomial_derivatives(taus::Vector{Float64}, C::Array{Float64, 4},
        window::Int, degree::Int)
    nt, K, nobs, D = size(C)
    half = max(1, window ÷ 2)
    Cdot = similar(C)
    for t in 1:nt
        lo = max(1, t - half)
        hi = min(nt, t + half)
        if hi - lo + 1 < degree + 1
            lo = max(1, min(lo, t - degree))
            hi = min(nt, max(hi, lo + degree))
        end
        xs = taus[lo:hi]
        for i in 1:K, a in 1:nobs, n in 1:D
            ys = vec(C[lo:hi, i, a, n])
            Cdot[t, i, a, n] = polynomial_derivative_at(xs, ys, taus[t], degree)
        end
    end
    return Cdot
end

function estimate_phi_from_coordinate_cdot0(taus::Vector{Float64}, C::Array{Float64, 4},
        lib::ObservableLibrary, params::FitDMParams, K::Int)
    q_idx = findfirst(i -> lib.kind[i] == :coord && lib.component[i] == 1, eachindex(lib.names))
    p_idx = findfirst(i -> lib.kind[i] == :coord && lib.component[i] == 2, eachindex(lib.names))
    require_condition(q_idx !== nothing && p_idx !== nothing, "Coordinate observables are required to estimate Phi.")
    L = min(params.phi_fit_max_lag, length(taus) - 1)
    idxs = collect(1:(L + 1))
    D = 2K
    Cdot0 = zeros(Float64, D, D)
    for i in 1:K
        rowq = flat_index(i, 1, K)
        rowp = flat_index(i, 2, K)
        for n in 1:D
            Cdot0[rowq, n] = polynomial_derivative_at(taus[idxs], vec(C[idxs, i, q_idx, n]), 0.0, params.phi_fit_degree)
            Cdot0[rowp, n] = polynomial_derivative_at(taus[idxs], vec(C[idxs, i, p_idx, n]), 0.0, params.phi_fit_degree)
        end
    end
    Phi = -Cdot0
    return Cdot0, Phi
end

function matrix_from_block_profile(profile::Array{Float64, 3})
    K = size(profile, 1)
    D = 2K
    M = zeros(Float64, D, D)
    @inbounds for i in 1:K, j in 1:K, cm in 1:2, cn in 1:2
        r = mod(j - i, K) + 1
        M[flat_index(i, cm, K), flat_index(j, cn, K)] = profile[r, cm, cn]
    end
    return M
end

function project_phi_from_profile(Phi_full::Matrix{Float64}, K::Int, projection::AbstractString)
    profile = block_profile(Phi_full, K)
    projected_profile = zeros(Float64, size(profile))
    if projection == "full_profile"
        projected_profile .= profile
    elseif projection == "local_2x2"
        projected_profile[1, :, :] .= profile[1, :, :]
    elseif projection == "local_complex"
        b = @view profile[1, :, :]
        d = 0.5 * (b[1, 1] + b[2, 2])
        omega = 0.5 * (b[2, 1] - b[1, 2])
        projected_profile[1, 1, 1] = d
        projected_profile[1, 1, 2] = -omega
        projected_profile[1, 2, 1] = omega
        projected_profile[1, 2, 2] = d
    else
        error("Unsupported phi projection: $(projection)")
    end
    Phi_projected = matrix_from_block_profile(projected_profile)
    offsite_ratio = norm(profile[2:end, :, :]) / max(norm(profile), eps(Float64))
    projection_change = norm(Phi_projected - Phi_full) / max(norm(Phi_full), eps(Float64))
    eigs = eigen(Symmetric(sympart(Phi_projected))).values
    return Phi_projected, projected_profile, offsite_ratio, projection_change,
        minimum(eigs), maximum(eigs)
end

function estimate_phi_from_short_lag_coordinate_profiles(sampler::PairSampler, params::FitDMParams,
        coord_mean::Vector{Float64}; max_lag::Int=params.phi_fit_max_lag,
        poly_degree::Int=params.phi_fit_degree, pairs_per_lag::Int=params.pairs_per_lag_phi)
    K = sampler.K
    L = min(max_lag, length(sampler.times) - sampler.start_idx)
    lags = collect(0:L)
    taus = lags .* sampler.save_dt
    profiles = zeros(Float64, length(lags), K, 2, 2)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    rng = MersenneTwister(params.seed + 30 + 17L + 1009poly_degree)
    for (lag_pos, lag) in enumerate(lags)
        sums = zeros(Float64, K, 2, 2)
        remaining = pairs_per_lag
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            @inbounds for b in 1:bn, i in 1:K
                qt = Float64(zt[i, 1, b]) - coord_mean[1]
                pt = Float64(zt[i, 2, b]) - coord_mean[2]
                for r0 in 0:(K - 1)
                    j = periodic(i + r0, K)
                    q0 = Float64(z0[j, 1, b]) - coord_mean[1]
                    p0 = Float64(z0[j, 2, b]) - coord_mean[2]
                    rp = r0 + 1
                    sums[rp, 1, 1] += qt * q0
                    sums[rp, 1, 2] += qt * p0
                    sums[rp, 2, 1] += pt * q0
                    sums[rp, 2, 2] += pt * p0
                end
            end
            total += bn * K
            remaining -= bn
        end
        profiles[lag_pos, :, :, :] .= sums ./ total
        params.verbose && @printf("Estimated short-lag coordinate profile %d / %d (tau=%.4f)\n",
            lag_pos, length(lags), taus[lag_pos])
    end
    cdot_profile = zeros(Float64, K, 2, 2)
    for r in 1:K, cm in 1:2, cn in 1:2
        cdot_profile[r, cm, cn] = polynomial_derivative_at(taus, vec(profiles[:, r, cm, cn]), 0.0, poly_degree)
    end
    phi_profile = -cdot_profile
    Phi = matrix_from_block_profile(phi_profile)
    Cdot0 = -Phi
    return taus, profiles, Cdot0, Phi
end

function estimate_stein_corrected_projected_phi(sampler::PairSampler, models::LoadedModels,
        params::FitDMParams, coord_mean::Vector{Float64}, device::ExecutionDevice)
    phi_taus, phi_profiles, Cdot0_raw, Phi_raw =
        estimate_phi_from_short_lag_coordinate_profiles(sampler, params, coord_mean)
    stein_matrix = estimate_stationary_stein_matrix(sampler, models, params, coord_mean, device)
    stein_projected = matrix_from_block_profile(block_profile(stein_matrix, sampler.K))

    # The learned DSM score only approximately satisfies -<s_theta(x) x'> = I.
    # Right-correct the short-lag derivative by this data-only Stein matrix before
    # applying the configured spatial/local projection to Phi.
    Phi_full = Phi_raw / stein_projected
    Cdot0 = -Phi_full
    Phi, phi_projected_profile, phi_offsite_ratio, phi_projection_change,
        phi_projected_min_eig, phi_projected_max_eig =
        project_phi_from_profile(Phi_full, sampler.K, params.phi_projection)
    return (
        phi_taus=phi_taus,
        phi_profiles=phi_profiles,
        Cdot0=Cdot0,
        Cdot0_raw=Cdot0_raw,
        Phi_raw=Phi_raw,
        Phi_full=Phi_full,
        Phi=Phi,
        stein_matrix=stein_matrix,
        stein_projected=stein_projected,
        phi_projected_profile=phi_projected_profile,
        phi_offsite_ratio=phi_offsite_ratio,
        phi_projection_change=phi_projection_change,
        phi_projected_min_eig=phi_projected_min_eig,
        phi_projected_max_eig=phi_projected_max_eig,
    )
end

function phi_action!(dest::AbstractMatrix{Float32}, Phi::AbstractMatrix{Float64},
        scond::AbstractArray{Float32, 3})
    K, _, B = size(scond)
    D = 2K
    sflat = Matrix{Float64}(undef, D, B)
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        sflat[flat_index(i, c, K), b] = scond[i, c, b]
    end
    action = transpose(Phi) * sflat
    @inbounds for b in 1:B, n in 1:D
        dest[n, b] = Float32(action[n, b])
    end
    return dest
end

function basis_actions!(d_action::AbstractMatrix{Float32}, omega_action::AbstractMatrix{Float32},
        z0::AbstractArray{Float32, 3}, scond::AbstractArray{Float32, 3}, mean_r2::Float64)
    K, _, B = size(z0)
    @inbounds for b in 1:B, i in 1:K
        q = z0[i, 1, b]
        p = z0[i, 2, b]
        f = q*q + p*p - Float32(mean_r2)
        sq = scond[i, 1, b]
        sp = scond[i, 2, b]
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        d_action[qrow, b] = f * sq
        d_action[prow, b] = f * sp
        omega_action[qrow, b] = f * sp
        omega_action[prow, b] = -f * sq
    end
    return nothing
end

function local_mobility_nn_features(states::Array{Float32, 4}, start_idx::Int, mean_r2::Float64,
        max_sites::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * K * ntraj
    nsamp = min(max_sites, total)
    f = Matrix{Float32}(undef, 1, nsamp)
    for n in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        i = mod(linear, K) + 1
        tmp = fld(linear, K)
        t = start_idx + mod(tmp, nt - start_idx + 1)
        traj = fld(tmp, nt - start_idx + 1) + 1
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        f[1, n] = Float32(q*q + p*p - mean_r2)
    end
    return f
end

function local_mobility_full_features(states::Array{Float32, 4}, start_idx::Int,
        max_sites::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * K * ntraj
    nsamp = min(max_sites, total)
    f = Matrix{Float32}(undef, 2, nsamp)
    for n in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        i = mod(linear, K) + 1
        tmp = fld(linear, K)
        t = start_idx + mod(tmp, nt - start_idx + 1)
        traj = fld(tmp, nt - start_idx + 1) + 1
        f[1, n] = states[t, i, 1, traj]
        f[2, n] = states[t, i, 2, traj]
    end
    return f
end

function build_local_mobility_nn()
    return Chain(
        Dense(2 => 128, tanh),
        Dense(128 => 128, tanh),
        Dense(128 => 4),
    )
end

function model_weight_decay(model)
    total = 0.0f0
    count = 0
    for param in Flux.trainables(model)
        total += sum(abs2, param)
        count += length(param)
    end
    return count == 0 ? 0.0f0 : total / count
end

function local_mobility_nn_weights(model)::Vector{Float64}
    if length(model.layers) == 1 && hasproperty(model[1], :weight)
        return vec(Float64.(Array(model[1].weight)))
    end
    return Float64[]
end

function train_local_mobility_nn(coeff::Vector{Float64}, states::Array{Float32, 4}, start_idx::Int,
        mean_r2::Float64; seed::Int, epochs::Int=300, batch_size::Int=8192,
        learning_rate::Float64=2.0e-2, max_sites::Int=262144)
    rng = MersenneTwister(seed)
    model = build_local_mobility_nn()
    model[1].weight .= 0.0f0
    features = local_mobility_nn_features(states, start_idx, mean_r2, max_sites, rng)
    targets = Matrix{Float32}(undef, 2, size(features, 2))
    @. targets[1, :] = Float32(coeff[1]) * features[1, :]
    @. targets[2, :] = Float32(coeff[2]) * features[1, :]
    opt_state = Flux.setup(Flux.Adam(learning_rate), model)
    history = MobilityNNHistory(Int[], Float64[], Vector{Float64}[])
    n = size(features, 2)
    for epoch in 1:epochs
        perm = randperm(rng, n)
        epoch_loss = 0.0
        nbatches = 0
        for start in 1:batch_size:n
            stop = min(start + batch_size - 1, n)
            idx = perm[start:stop]
            x = features[:, idx]
            y = targets[:, idx]
            loss, grads = Flux.withgradient(model) do current_model
                Flux.Losses.mse(current_model(x), y)
            end
            opt_state, model = Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float64(loss)
            nbatches += 1
        end
        if epoch == 1 || epoch % 25 == 0 || epoch == epochs
            push!(history.epochs, epoch)
            push!(history.losses, epoch_loss / nbatches)
            push!(history.weights, local_mobility_nn_weights(model))
            @printf("Local mobility NN epoch %d: loss %.6e weights [%.6e, %.6e]\n",
                epoch, history.losses[end], history.weights[end][1], history.weights[end][2])
        end
    end
    return model, history
end

function local_nn_delta!(delta::AbstractMatrix{Float32}, model, z0::AbstractArray{Float32, 3}, mean_r2::Float64)
    K, _, B = size(z0)
    features = Matrix{Float32}(undef, 2, K * B)
    @inbounds for b in 1:B, i in 1:K
        features[1, i + (b - 1) * K] = z0[i, 1, b]
        features[2, i + (b - 1) * K] = z0[i, 2, b]
    end
    pred = model(features)
    @inbounds for b in 1:B, i in 1:K
        col = i + (b - 1) * K
        # rows store local full 2x2 block: m11,m12,m21,m22
        delta[4 * (i - 1) + 1, b] = pred[1, col]
        delta[4 * (i - 1) + 2, b] = pred[2, col]
        delta[4 * (i - 1) + 3, b] = pred[3, col]
        delta[4 * (i - 1) + 4, b] = pred[4, col]
    end
    return delta
end

function nn_local_transpose_action!(dest::AbstractMatrix{Float32}, model,
        z0::AbstractArray{Float32, 3}, scond::AbstractArray{Float32, 3}, mean_r2::Float64)
    K, _, B = size(z0)
    delta = Matrix{Float32}(undef, 4K, B)
    local_nn_delta!(delta, model, z0, mean_r2)
    @inbounds for b in 1:B, i in 1:K
        m11 = delta[4 * (i - 1) + 1, b]
        m12 = delta[4 * (i - 1) + 2, b]
        m21 = delta[4 * (i - 1) + 3, b]
        m22 = delta[4 * (i - 1) + 4, b]
        sq = scond[i, 1, b]
        sp = scond[i, 2, b]
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        dest[qrow, b] = m11 * sq + m21 * sp
        dest[prow, b] = m12 * sq + m22 * sp
    end
    return dest
end

function nn_local_row_action_and_divergence!(dest::AbstractMatrix{Float64}, model,
        z::AbstractArray{Float64, 3}, score::AbstractArray{Float32, 3}, mean_r2::Float64)
    K, _, B = size(z)
    features = Matrix{Float32}(undef, 2, K * B)
    eps_f = 1.0f-3
    @inbounds for b in 1:B, i in 1:K
        features[1, i + (b - 1) * K] = Float32(z[i, 1, b])
        features[2, i + (b - 1) * K] = Float32(z[i, 2, b])
    end
    pred = model(features)
    fqp = copy(features); fqm = copy(features); fpp = copy(features); fpm = copy(features)
    fqp[1, :] .+= eps_f
    fqm[1, :] .-= eps_f
    fpp[2, :] .+= eps_f
    fpm[2, :] .-= eps_f
    dq = (model(fqp) .- model(fqm)) ./ (2.0f0 * eps_f)
    dp = (model(fpp) .- model(fpm)) ./ (2.0f0 * eps_f)
    @inbounds for b in 1:B, i in 1:K
        col = i + (b - 1) * K
        m11 = Float64(pred[1, col])
        m12 = Float64(pred[2, col])
        m21 = Float64(pred[3, col])
        m22 = Float64(pred[4, col])
        sq = Float64(score[i, 1, b])
        sp = Float64(score[i, 2, b])
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        divq = Float64(dq[1, col] + dp[2, col])
        divp = Float64(dq[3, col] + dp[4, col])
        dest[qrow, b] += m11 * sq + m12 * sp + divq
        dest[prow, b] += m21 * sq + m22 * sp + divp
    end
    return dest
end

function feature_power_moments(states::Array{Float32, 4}, start_idx::Int, mean_r2::Float64,
        nfeatures::Int; max_sites::Int=1_000_000, seed::Int=71_337)
    rng = MersenneTwister(seed)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * K * ntraj
    nsamp = min(max_sites, total)
    moments = zeros(Float64, nfeatures)
    for _ in 1:nsamp
        linear = rand(rng, 0:(total - 1))
        i = mod(linear, K) + 1
        tmp = fld(linear, K)
        t = start_idx + mod(tmp, nt - start_idx + 1)
        traj = fld(tmp, nt - start_idx + 1) + 1
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        f = q*q + p*p - mean_r2
        fp = f
        for k in 1:nfeatures
            moments[k] += fp
            fp *= f
        end
    end
    moments ./= nsamp
    return moments
end

function mobility_feature_value(f::Float64, power::Int, moments::Vector{Float64})
    return f^power - moments[power]
end

function basis_feature_actions!(d_action::AbstractMatrix{Float32}, omega_action::AbstractMatrix{Float32},
        z0::AbstractArray{Float32, 3}, scond::AbstractArray{Float32, 3},
        mean_r2::Float64, moments::Vector{Float64}, power::Int)
    K, _, B = size(z0)
    @inbounds for b in 1:B, i in 1:K
        q = Float64(z0[i, 1, b])
        p = Float64(z0[i, 2, b])
        f = Float32(mobility_feature_value(q*q + p*p - mean_r2, power, moments))
        sq = scond[i, 1, b]
        sp = scond[i, 2, b]
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        d_action[qrow, b] = f * sq
        d_action[prow, b] = f * sp
        omega_action[qrow, b] = f * sp
        omega_action[prow, b] = -f * sq
    end
    return nothing
end

function profile_from_action_sums!(profiles::Array{Float64, 5}, basis_idx::Int,
        sums::Array{Float64, 3}, total::Int, K::Int)
    _, nobs, D = size(sums)
    @inbounds for a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)
        acc = 0.0
        for i in 1:K
            j = periodic(i + r0, K)
            n = flat_index(j, comp, K)
            acc += sums[i, a, n]
        end
        profiles[a, comp, r0 + 1, basis_idx, 1] = -acc / (total * K)
    end
    return nothing
end

function estimate_mobility_basis_A_profiles(sampler::PairSampler, models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel},
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice,
        moments::Vector{Float64})
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    nfeatures = length(moments)
    nbasis = 2 * nfeatures
    basis_profiles = zeros(Float64, length(sampler.lag_steps), nobs, 2, K, nbasis)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    d_act = Matrix{Float32}(undef, D, params.batch_size)
    omega_act = similar(d_act)
    rng = MersenneTwister(params.seed + 610_000)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[lag_idx]
        basis_sums = [zeros(Float64, K, nobs, D) for _ in 1:nbasis]
        remaining = params.pairs_per_lag_operator
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = evaluate_transition_score_x0(models, reverse_model,
                @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), tau, params, device)
            for power in 1:nfeatures
                basis_feature_actions!(@view(d_act[:, 1:bn]), @view(omega_act[:, 1:bn]),
                    @view(z0[:, :, 1:bn]), scond, lib.mean_r2, moments, power)
                accumulate_action_sums!(basis_sums[power], @view(obs_vals[:, :, 1:bn]), @view(d_act[:, 1:bn]))
                accumulate_action_sums!(basis_sums[nfeatures + power], @view(obs_vals[:, :, 1:bn]), @view(omega_act[:, 1:bn]))
            end
            total += bn
            remaining -= bn
        end
        tmp = zeros(Float64, nobs, 2, K, nbasis, 1)
        for bidx in 1:nbasis
            profile_from_action_sums!(tmp, bidx, basis_sums[bidx], total, K)
        end
        basis_profiles[lag_idx, :, :, :, :] .= tmp[:, :, :, :, 1]
        params.verbose && @printf("Estimated polynomial mobility A basis profiles for tau %.4f (%d / %d)\n",
            tau, lag_idx, length(sampler.lag_steps))
    end
    return basis_profiles
end

function fit_mobility_basis_coefficients(A_target::Array{Float64, 4}, basis_profiles::Array{Float64, 5};
        ridge::Float64=1.0e-6)
    nt, nobs, _, _, nbasis = size(basis_profiles)
    X = reshape(basis_profiles, nt * nobs * 2 * size(basis_profiles, 4), nbasis)
    y = vec(A_target)
    mask = isfinite.(y)
    for j in 1:nbasis
        mask .&= isfinite.(X[:, j])
    end
    Xf = X[mask, :]
    yf = y[mask]
    scale_arr = similar(A_target)
    for a in 1:nobs
        vals = vec(@view A_target[:, a, :, :])
        finite_vals = vals[isfinite.(vals)]
        rmsv = isempty(finite_vals) ? 1.0 : sqrt(mean(finite_vals .^ 2))
        scale_arr[:, a, :, :] .= 1.0 / max(rmsv, 1.0e-8)
    end
    scales = vec(scale_arr)[mask]
    Xw = Xf .* scales
    yw = yf .* scales
    coeff = (transpose(Xw) * Xw + ridge * I) \ (transpose(Xw) * yw)
    pred = reshape(reshape(basis_profiles, :, nbasis) * coeff, size(A_target))
    rel, corrv = profile_agreement_metrics(A_target, pred)
    return collect(coeff), pred, rel, corrv
end

function polynomial_delta_targets(features::Matrix{Float32}, coeff::Vector{Float64}, moments::Vector{Float64})
    nfeatures = length(moments)
    out = Matrix{Float32}(undef, 2, size(features, 2))
    @inbounds for col in axes(features, 2)
        f = Float64(features[1, col])
        dd = 0.0
        dw = 0.0
        for power in 1:nfeatures
            val = mobility_feature_value(f, power, moments)
            dd += coeff[power] * val
            dw += coeff[nfeatures + power] * val
        end
        out[1, col] = Float32(dd)
        out[2, col] = Float32(dw)
    end
    return out
end

function train_local_mobility_mlp_from_polynomial(coeff::Vector{Float64}, moments::Vector{Float64},
        states::Array{Float32, 4}, start_idx::Int, mean_r2::Float64;
        seed::Int, epochs::Int=1200, batch_size::Int=8192, learning_rate::Float64=2.0e-3,
        max_sites::Int=524288)
    rng = MersenneTwister(seed)
    model = build_local_mobility_nn()
    features = local_mobility_nn_features(states, start_idx, mean_r2, max_sites, rng)
    targets = polynomial_delta_targets(features, coeff, moments)
    ntrain = floor(Int, 0.9 * size(features, 2))
    train_idx = collect(1:ntrain)
    val_idx = collect((ntrain + 1):size(features, 2))
    opt_state = Flux.setup(Flux.Adam(learning_rate), model)
    history = MobilityMLPHistory(Int[], Float64[], Float64[])
    n = length(train_idx)
    for epoch in 1:epochs
        perm = train_idx[randperm(rng, n)]
        epoch_loss = 0.0
        nbatches = 0
        for start in 1:batch_size:n
            stop = min(start + batch_size - 1, n)
            idx = perm[start:stop]
            x = features[:, idx]
            y = targets[:, idx]
            loss, grads = Flux.withgradient(model) do current_model
                Flux.Losses.mse(current_model(x), y)
            end
            opt_state, model = Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float64(loss)
            nbatches += 1
        end
        if epoch == 1 || epoch % 50 == 0 || epoch == epochs
            val_pred = model(features[:, val_idx])
            val_rmse = sqrt(mean((Array(val_pred) .- targets[:, val_idx]) .^ 2))
            push!(history.epochs, epoch)
            push!(history.losses, epoch_loss / max(nbatches, 1))
            push!(history.validation_rmse, Float64(val_rmse))
            @printf("Local mobility MLP epoch %d: loss %.6e val_RMSE %.6e\n",
                epoch, history.losses[end], history.validation_rmse[end])
        end
    end
    return model, history
end

function build_direct_mobility_training_cache(sampler::PairSampler, models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel},
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice;
        pairs_per_lag::Int=4096, mean_samples::Int=131072, seed::Int=params.seed + 700_000)
    K = sampler.K
    nobs = length(lib.names)
    nlags = length(sampler.lag_steps)
    B = pairs_per_lag
    features = Array{Float32}(undef, 2, K, B, nlags)
    scond_cache = Array{Float32}(undef, K, 2, B, nlags)
    obs_cache = Array{Float32}(undef, K, nobs, B, nlags)
    z0 = Array{Float32}(undef, K, 2, B)
    zt = similar(z0)
    rng = MersenneTwister(seed)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[lag_idx]
        sample_pair_batch!(z0, zt, sampler, lag, rng)
        @inbounds for b in 1:B, i in 1:K
            features[1, i, b, lag_idx] = z0[i, 1, b]
            features[2, i, b, lag_idx] = z0[i, 2, b]
        end
        scond_cache[:, :, :, lag_idx] .=
            evaluate_transition_score_x0(models, reverse_model, z0, zt, tau, params, device)
        for b in 1:B
            observable_values!(@view(obs_cache[:, :, b, lag_idx]), @view(zt[:, :, b]), lib)
        end
        params.verbose && @printf("Built direct mobility training cache for tau %.4f (%d / %d)\n",
            tau, lag_idx, nlags)
    end
    mean_features = local_mobility_full_features(sampler.states, sampler.start_idx,
        mean_samples, MersenneTwister(seed + 1))
    return MobilityDirectTrainingCache(features, scond_cache, obs_cache, mean_features)
end

function predict_A_profiles_direct(model, cache::MobilityDirectTrainingCache)
    _, K, B, T = size(cache.features)
    nobs = size(cache.observables, 2)
    delta = reshape(model(reshape(cache.features, 2, :)), 4, K, B, T)
    sq = @view cache.scond[:, 1, :, :]
    sp = @view cache.scond[:, 2, :, :]
    aq = delta[1, :, :, :] .* sq .+ delta[3, :, :, :] .* sp
    ap = delta[2, :, :, :] .* sq .+ delta[4, :, :, :] .* sp
    vals = [
        begin
        action = comp == 1 ? aq : ap
        shifted = action[[periodic(i + r0, K) for i in 1:K], :, t]
        -mean(cache.observables[:, a, :, t] .* shifted)
        end
        for t in 1:T, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)
    ]
    return reshape(vals, T, nobs, 2, K)
end

function profile_index_list(T::Int, nobs::Int, K::Int)
    return [(t, a, comp, r0) for t in 1:T, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)]
end

function predict_A_values_direct(model, cache::MobilityDirectTrainingCache,
        indices::Vector{NTuple{4, Int}})
    _, K, B, T = size(cache.features)
    delta = reshape(model(reshape(cache.features, 2, :)), 4, K, B, T)
    sq = @view cache.scond[:, 1, :, :]
    sp = @view cache.scond[:, 2, :, :]
    aq = delta[1, :, :, :] .* sq .+ delta[3, :, :, :] .* sp
    ap = delta[2, :, :, :] .* sq .+ delta[4, :, :, :] .* sp
    return [
        begin
            t, a, comp, r0 = idx
            action = comp == 1 ? aq : ap
            shifted = action[[periodic(i + r0, K) for i in 1:K], :, t]
            -mean(cache.observables[:, a, :, t] .* shifted)
        end
        for idx in indices
    ]
end

function direct_mobility_mean_penalty(model, mean_features::Matrix{Float32})
    vals = model(mean_features)
    return mean(vals .^ 2)
end

function mobility_delta_summary(model, mean_features::Matrix{Float32})
    vals = Float64.(model(mean_features))
    mean_block = vec(mean(vals; dims=2))
    mean_abs = norm(mean_block) / sqrt(length(mean_block))
    rms_delta = sqrt(mean(vals .^ 2))
    return mean_abs, rms_delta
end

function train_mobility_nn_direct_loss(A_target::Array{Float64, 4}, cache::MobilityDirectTrainingCache;
        seed::Int, epochs::Int=700, learning_rate::Float64=1.0e-3,
        mean_penalty_weight::Float64=1.0e-2, weight_decay::Float64=1.0e-6,
        profile_batch_size::Int=512)
    Random.seed!(seed)
    model = build_local_mobility_nn()
    Random.seed!()
    target = Float32.(A_target)
    _, nobs, _, _ = size(A_target)
    scale = ones(Float32, size(A_target))
    for a in 1:nobs
        vals = vec(@view A_target[:, a, :, :])
        finite_vals = vals[isfinite.(vals)]
        rmsv = isempty(finite_vals) ? 1.0 : sqrt(mean(finite_vals .^ 2))
        scale[:, a, :, :] .= Float32(max(rmsv, 1.0e-6))
    end
    opt_state = Flux.setup(Flux.Adam(learning_rate), model)
    history = MobilityMLPHistoryDetailed(Int[], Float64[], Float64[], Float64[], Float64[])
    best_model = deepcopy(model)
    best_rmse = Inf
    rng = MersenneTwister(seed + 1)
    all_indices = profile_index_list(size(A_target, 1), size(A_target, 2), size(A_target, 4))
    for epoch in 1:epochs
        batch_indices = all_indices[rand(rng, 1:length(all_indices), min(profile_batch_size, length(all_indices)))]
        target_batch = Float32[A_target[t, a, comp, r0 + 1] for (t, a, comp, r0) in batch_indices]
        scale_batch = Float32[scale[t, a, comp, r0 + 1] for (t, a, comp, r0) in batch_indices]
        loss, grads = Flux.withgradient(model) do current_model
            pred = predict_A_values_direct(current_model, cache, batch_indices)
            data_loss = mean(((pred .- target_batch) ./ scale_batch) .^ 2)
            mean_penalty = direct_mobility_mean_penalty(current_model, cache.mean_features)
            wd = Float32(weight_decay) * model_weight_decay(current_model)
            data_loss + Float32(mean_penalty_weight) * mean_penalty + wd
        end
        opt_state, model = Flux.update!(opt_state, model, grads[1])
        if epoch == 1 || epoch % 25 == 0 || epoch == epochs
            pred = Array(predict_A_profiles_direct(model, cache))
            rel, _ = profile_agreement_metrics(A_target, Float64.(pred))
            push!(history.epochs, epoch)
            push!(history.losses, Float64(loss))
            push!(history.validation_rmse, rel)
            mean_abs, rms_delta = mobility_delta_summary(model, cache.mean_features)
            push!(history.mean_abs_delta, mean_abs)
            push!(history.rms_delta, rms_delta)
            if rel < best_rmse
                best_rmse = rel
                best_model = deepcopy(model)
            end
            @printf("Direct mobility NN epoch %d: loss %.6e A_rel_RMSE %.6e\n",
                epoch, Float64(loss), rel)
        end
    end
    return best_model, history
end

function build_random_feature_mobility_nn(hidden_width::Int, seed::Int)
    Random.seed!(seed)
    model = Chain(
        Dense(2 => hidden_width, tanh),
        Dense(hidden_width => 4),
    )
    Random.seed!()
    model[2].weight .= 0.0f0
    model[2].bias .= 0.0f0
    return model
end

function hidden_features_with_bias(model, features::AbstractMatrix{Float32})
    hidden = model[1](features)
    return vcat(hidden, ones(Float32, 1, size(hidden, 2)))
end

function design_column(output_component::Int, hidden_index::Int, n_hidden_bias::Int)
    return (output_component - 1) * n_hidden_bias + hidden_index
end

function fit_mobility_nn_random_feature_loss(A_target::Array{Float64, 4},
        cache::MobilityDirectTrainingCache; hidden_width::Int=256,
        seed::Int=17, ridge::Float64=1.0e-7, mean_penalty_weight::Float64=10.0)
    model = build_random_feature_mobility_nn(hidden_width, seed)
    _, K, B, T = size(cache.features)
    nobs = size(cache.observables, 2)
    H = hidden_width + 1
    nrows = T * nobs * 2 * K
    ncols = 4H
    X = zeros(Float64, nrows, ncols)
    y = zeros(Float64, nrows)
    scale_arr = ones(Float64, size(A_target))
    for a in 1:nobs
        vals = vec(@view A_target[:, a, :, :])
        finite_vals = vals[isfinite.(vals)]
        rmsv = isempty(finite_vals) ? 1.0 : sqrt(mean(finite_vals .^ 2))
        scale_arr[:, a, :, :] .= 1.0 / max(rmsv, 1.0e-6)
    end
    hidden = reshape(hidden_features_with_bias(model, reshape(cache.features, 2, :)), H, K, B, T)
    inv_count = 1.0 / (K * B)
    ncols_profile = nobs * K
    nsite_batch = K * B
    @inbounds for t in 1:T
        hidden_t = Float64.(reshape(@view(hidden[:, :, :, t]), H, nsite_batch))
        for comp in 1:2
            v_sq = zeros(Float64, nsite_batch, ncols_profile)
            v_sp = zeros(Float64, nsite_batch, ncols_profile)
            for a in 1:nobs, r0 in 0:(K - 1)
                profile_col = (a - 1) * K + r0 + 1
                row_scale = scale_arr[t, a, comp, r0 + 1]
                design_row = (((t - 1) * nobs + (a - 1)) * 2 + (comp - 1)) * K + r0 + 1
                y[design_row] = A_target[t, a, comp, r0 + 1] * row_scale
                for b in 1:B, j in 1:K
                    i = periodic(j - r0, K)
                    site_row = j + (b - 1) * K
                    coeff = -row_scale * Float64(cache.observables[i, a, b, t]) * inv_count
                    v_sq[site_row, profile_col] = coeff * Float64(cache.scond[j, 1, b, t])
                    v_sp[site_row, profile_col] = coeff * Float64(cache.scond[j, 2, b, t])
                end
            end
            g_sq = hidden_t * v_sq
            g_sp = hidden_t * v_sp
            for a in 1:nobs, r0 in 0:(K - 1)
                profile_col = (a - 1) * K + r0 + 1
                design_row = (((t - 1) * nobs + (a - 1)) * 2 + (comp - 1)) * K + r0 + 1
                if comp == 1
                    X[design_row, 1:H] .= @view g_sq[:, profile_col]
                    X[design_row, (2H + 1):(3H)] .= @view g_sp[:, profile_col]
                else
                    X[design_row, (H + 1):(2H)] .= @view g_sq[:, profile_col]
                    X[design_row, (3H + 1):(4H)] .= @view g_sp[:, profile_col]
                end
            end
        end
    end

    mean_hidden = vec(mean(hidden_features_with_bias(model, cache.mean_features); dims=2))
    if mean_penalty_weight > 0.0
        Xmean = zeros(Float64, 4, ncols)
        weight = sqrt(mean_penalty_weight)
        for out in 1:4, h in 1:H
            Xmean[out, design_column(out, h, H)] = weight * mean_hidden[h]
        end
        X = vcat(X, Xmean)
        y = vcat(y, zeros(Float64, 4))
    end

    theta = (transpose(X) * X + ridge * I) \ (transpose(X) * y)
    W = reshape(theta, H, 4)'
    model[2].weight .= Float32.(W[:, 1:hidden_width])
    model[2].bias .= Float32.(W[:, H])
    pred = Float64.(predict_A_profiles_direct(model, cache))
    final_rel, _ = profile_agreement_metrics(A_target, pred)
    initial_rel = 1.0
    history = MobilityMLPHistory([0, 1], [initial_rel^2, final_rel^2], [initial_rel, final_rel])
    @printf("Random-feature mobility NN ridge solve: hidden=%d, A_rel_RMSE=%.6e\n",
        hidden_width, final_rel)
    return model, history
end

function accumulate_operator_sums!(phi_sum::Array{Float64, 3}, d_sum::Array{Float64, 3},
        omega_sum::Array{Float64, 3}, obs_vals::AbstractArray{Float32, 3},
        phi_act::AbstractMatrix{Float32}, d_act::AbstractMatrix{Float32},
        omega_act::AbstractMatrix{Float32})
    K, nobs, B = size(obs_vals)
    D = size(phi_act, 1)
    @inbounds for i in 1:K, a in 1:nobs, n in 1:D
        pred_phi = 0.0
        pred_d = 0.0
        pred_omega = 0.0
        for b in 1:B
            obs = Float64(obs_vals[i, a, b])
            pred_phi += obs * Float64(phi_act[n, b])
            pred_d += obs * Float64(d_act[n, b])
            pred_omega += obs * Float64(omega_act[n, b])
        end
        phi_sum[i, a, n] += pred_phi
        d_sum[i, a, n] += pred_d
        omega_sum[i, a, n] += pred_omega
    end
    return nothing
end

function accumulate_action_sums!(sumA::Array{Float64, 3}, obs_vals::AbstractArray{Float32, 3},
        action::AbstractMatrix{Float32})
    K, nobs, B = size(obs_vals)
    D = size(action, 1)
    require_condition(B == size(action, 2), "Observable and action batches have inconsistent sizes.")
    sum_mat = reshape(sumA, K * nobs, D)
    obs_mat = Float64.(reshape(obs_vals, K * nobs, B))
    action_mat = Float64.(action)
    mul!(sum_mat, obs_mat, transpose(action_mat), 1.0, 1.0)
    return nothing
end

function accumulate_normal_equations!(G::Matrix{Float64}, h::Vector{Float64}, stats::Vector{Float64},
        phi_sum::Array{Float64, 3}, d_sum::Array{Float64, 3}, omega_sum::Array{Float64, 3},
        target_cdot::AbstractArray{Float64, 3}, total_pairs::Int, fit_mask::Vector{Bool})
    K, nobs, D = size(phi_sum)
    invN = 1.0 / total_pairs
    @inbounds for i in 1:K, a in 1:nobs, n in 1:D
        fit_mask[a] || continue
        pred_phi = -phi_sum[i, a, n] * invN
        x1 = -d_sum[i, a, n] * invN
        x2 = -omega_sum[i, a, n] * invN
        y = target_cdot[i, a, n] - pred_phi
        G[1, 1] += x1 * x1
        G[1, 2] += x1 * x2
        G[2, 2] += x2 * x2
        h[1] += x1 * y
        h[2] += x2 * y
        stats[1] += 1
        stats[2] += target_cdot[i, a, n]^2
        stats[3] += y^2
    end
    return nothing
end

function estimate_mobility_coefficients(sampler::PairSampler, models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel}, lib::ObservableLibrary,
        params::FitDMParams, device::ExecutionDevice, Phi::Matrix{Float64}, correlation_lags::Vector{Int},
        Cdot_data::Array{Float64, 4})
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    G = zeros(Float64, 2, 2)
    h = zeros(Float64, 2)
    stats = zeros(Float64, 4)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    phi_act = Matrix{Float32}(undef, D, params.batch_size)
    d_act = similar(phi_act)
    omega_act = similar(phi_act)
    fit_mask = [kd == :ampcoord for kd in lib.kind]
    require_condition(any(fit_mask), "No amplitude-coordinate observables are available for the local mobility coefficient solve.")
    @printf("Coefficient solve uses observables: %s\n", join(lib.names[fit_mask], " | "))
    rng = MersenneTwister(params.seed + 50)
    for fit_idx in eachindex(sampler.lag_steps)
        lag = sampler.lag_steps[fit_idx]
        tau = sampler.lag_times[fit_idx]
        lag_idx = searchsortedfirst(correlation_lags, lag)
        require_condition(lag_idx <= length(correlation_lags) && correlation_lags[lag_idx] == lag,
            "Internal error: fit lag $(lag) was not present in the correlation grid.")
        phi_sum = zeros(Float64, K, nobs, D)
        d_sum = zeros(Float64, K, nobs, D)
        omega_sum = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_operator
        total_pairs = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = evaluate_transition_score_x0(models, reverse_model,
                @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), tau, params, device)
            phi_action!(@view(phi_act[:, 1:bn]), Phi, scond)
            basis_actions!(@view(d_act[:, 1:bn]), @view(omega_act[:, 1:bn]),
                @view(z0[:, :, 1:bn]), scond, lib.mean_r2)
            accumulate_operator_sums!(phi_sum, d_sum, omega_sum, @view(obs_vals[:, :, 1:bn]),
                @view(phi_act[:, 1:bn]), @view(d_act[:, 1:bn]), @view(omega_act[:, 1:bn]))
            total_pairs += bn
            remaining -= bn
        end
        accumulate_normal_equations!(G, h, stats, phi_sum, d_sum, omega_sum,
            @view(Cdot_data[lag_idx, :, :, :]), total_pairs, fit_mask)
        params.verbose && @printf("Accumulated operator rows for tau %.4f (%d / %d)\n",
            tau, fit_idx, length(sampler.lag_steps))
    end
    G[2, 1] = G[1, 2]
    H = G + params.ridge * I
    coeff = H \ h
    residual2 = 0.0
    # Reconstruct residual norm from normal equations.
    residual2 = stats[3] - 2.0 * dot(coeff, h) + dot(coeff, G * coeff)
    row_count = max(stats[1], 1.0)
    residual_rmse = sqrt(max(residual2, 0.0) / row_count)
    target_rms = sqrt(stats[2] / row_count)
    rel = residual_rmse / max(target_rms, eps(Float64))
    return MobilityFitResult(
        collect(coeff),
        ["d1_zero_mean_r2", "omega1_zero_mean_r2"],
        G,
        h,
        cond(Matrix(H)),
        residual_rmse,
        target_rms,
        rel,
    )
end

function mean_mobility_from_ansatz(Phi::Matrix{Float64}, lib::ObservableLibrary, coeff::Vector{Float64},
        states::Array{Float32, 4}, start_idx::Int)
    K = size(states, 2)
    D = 2K
    meanM = copy(Phi)
    delta_mean = zeros(Float64, D, D)
    nt, _, _, ntraj = size(states)
    count = 0
    for traj in 1:ntraj, t in start_idx:nt, i in 1:K
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        f = q*q + p*p - lib.mean_r2
        qidx = flat_index(i, 1, K)
        pidx = flat_index(i, 2, K)
        delta_mean[qidx, qidx] += coeff[1] * f
        delta_mean[pidx, pidx] += coeff[1] * f
        delta_mean[qidx, pidx] += -coeff[2] * f
        delta_mean[pidx, qidx] += coeff[2] * f
        count += 1
    end
    # Each matrix entry receives one contribution per site sample.
    delta_mean ./= max(count ÷ K, 1)
    meanM .+= delta_mean
    return meanM, delta_mean
end

function block_profile(M::AbstractMatrix{<:Real}, K::Int)
    prof = zeros(Float64, K, 2, 2)
    counts = zeros(Int, K)
    for i in 1:K, j in 1:K
        r = mod(j - i, K) + 1
        prof[r, 1, 1] += M[flat_index(i, 1, K), flat_index(j, 1, K)]
        prof[r, 1, 2] += M[flat_index(i, 1, K), flat_index(j, 2, K)]
        prof[r, 2, 1] += M[flat_index(i, 2, K), flat_index(j, 1, K)]
        prof[r, 2, 2] += M[flat_index(i, 2, K), flat_index(j, 2, K)]
        counts[r] += 1
    end
    for r in 1:K
        prof[r, :, :] ./= counts[r]
    end
    return prof
end

function load_true_mean_mobility(path::AbstractString)
    mean_d = Float64(h5read(path, "/mobility/mean_d"))
    mean_omega = Float64(h5read(path, "/mobility/mean_omega"))
    d1 = Float64(h5read(path, "/metadata/d1"))
    omega1 = Float64(h5read(path, "/metadata/omega1"))
    return mean_d, mean_omega, d1, omega1
end

function true_mean_mobility_matrix(K::Int, mean_d::Float64, mean_omega::Float64)
    M = zeros(Float64, 2K, 2K)
    @inbounds for i in 1:K
        qidx = flat_index(i, 1, K)
        pidx = flat_index(i, 2, K)
        M[qidx, qidx] = mean_d
        M[pidx, pidx] = mean_d
        M[qidx, pidx] = -mean_omega
        M[pidx, qidx] = mean_omega
    end
    return M
end

function estimate_stationary_stein_matrix(sampler::PairSampler, models::LoadedModels, params::FitDMParams,
        coord_mean::Vector{Float64}, device::ExecutionDevice; nsamples::Int=min(params.pairs_per_lag_phi, 200_000))
    K = sampler.K
    D = sampler.D
    z = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z)
    xflat = Matrix{Float32}(undef, D, params.batch_size)
    total = zeros(Float64, D, D)
    rng = MersenneTwister(params.seed + 610_000)
    remaining = nsamples
    count = 0
    while remaining > 0
        bn = min(remaining, params.batch_size)
        sample_pair_batch!(@view(z[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, 0, rng)
        scores = evaluate_stationary_score(models, @view(z[:, :, 1:bn]), params.score_batch_size, device)
        flatten_state!(@view(xflat[:, 1:bn]), @view(z[:, :, 1:bn]))
        @inbounds for b in 1:bn, i in 1:K
            xflat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
            xflat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
        end
        sflat = reshape(scores, D, bn)
        total .+= Matrix{Float64}(sflat) * transpose(Matrix{Float64}(@view(xflat[:, 1:bn])))
        count += bn
        remaining -= bn
    end
    return -total ./ max(count, 1)
end

function estimate_exact_stationary_stein_matrix(sampler::PairSampler, params::FitDMParams,
        coord_mean::Vector{Float64}, potential_params::Tuple{Float64, Float64, Float64};
        nsamples::Int=min(params.pairs_per_lag_phi, 500_000))
    K = sampler.K
    D = sampler.D
    z = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z)
    score = similar(z)
    xflat = Matrix{Float32}(undef, D, params.batch_size)
    total = zeros(Float64, D, D)
    rng = MersenneTwister(params.seed + 615_000)
    remaining = nsamples
    count = 0
    while remaining > 0
        bn = min(remaining, params.batch_size)
        sample_pair_batch!(@view(z[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, 0, rng)
        exact_stationary_score!(@view(score[:, :, 1:bn]), @view(z[:, :, 1:bn]), potential_params...)
        flatten_state!(@view(xflat[:, 1:bn]), @view(z[:, :, 1:bn]))
        @inbounds for b in 1:bn, i in 1:K
            xflat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
            xflat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
        end
        sflat = reshape(@view(score[:, :, 1:bn]), D, bn)
        total .+= Matrix{Float64}(sflat) * transpose(Matrix{Float64}(@view(xflat[:, 1:bn])))
        count += bn
        remaining -= bn
    end
    return -total ./ max(count, 1)
end

function estimate_phi_from_true_generator_coordinate_cdot0(sampler::PairSampler, params::FitDMParams,
        coord_mean::Vector{Float64}, input_hdf5::AbstractString;
        nsamples::Int=min(params.pairs_per_lag_phi, 500_000))
    K = sampler.K
    D = sampler.D
    alpha, beta, kappa = load_potential_params(input_hdf5)
    d0, d1, omega0, omega1 = load_true_mobility_params(input_hdf5)
    z = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z)
    drift = similar(z)
    sums = zeros(Float64, D, D)
    rng = MersenneTwister(params.seed + 620_000)
    remaining = nsamples
    count = 0
    while remaining > 0
        bn = min(remaining, params.batch_size)
        sample_pair_batch!(@view(z[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, 0, rng)
        drift_values!(@view(drift[:, :, 1:bn]), @view(z[:, :, 1:bn]),
            alpha, beta, kappa, d0, d1, omega0, omega1)
        @inbounds for b in 1:bn, i in 1:K
            rowq = flat_index(i, 1, K)
            rowp = flat_index(i, 2, K)
            bq = Float64(drift[i, 1, b])
            bp = Float64(drift[i, 2, b])
            for j in 1:K
                xq = Float64(z[j, 1, b]) - coord_mean[1]
                xp = Float64(z[j, 2, b]) - coord_mean[2]
                sums[rowq, flat_index(j, 1, K)] += bq * xq
                sums[rowq, flat_index(j, 2, K)] += bq * xp
                sums[rowp, flat_index(j, 1, K)] += bp * xq
                sums[rowp, flat_index(j, 2, K)] += bp * xp
            end
        end
        count += bn
        remaining -= bn
    end
    Cdot0 = sums ./ max(count, 1)
    Phi = -Cdot0
    return Cdot0, Phi
end

function phi_agreement_metrics(Phi::AbstractMatrix{<:Real}, Phi_true::AbstractMatrix{<:Real})
    P = Matrix{Float64}(Phi)
    T = Matrix{Float64}(Phi_true)
    S = sympart(P)
    ST = sympart(T)
    A = 0.5 .* (P .- transpose(P))
    AT = 0.5 .* (T .- transpose(T))
    eig = eigen(Symmetric(S)).values
    return Dict{Symbol, Float64}(
        :relative_rmse => norm(P - T) / max(norm(T), eps(Float64)),
        :sym_relative_rmse => norm(S - ST) / max(norm(ST), eps(Float64)),
        :anti_relative_rmse => norm(A - AT) / max(norm(AT), eps(Float64)),
        :offsite_relative_norm => norm(block_profile(P, size(P, 1) ÷ 2)[2:end, :, :]) / max(norm(T), eps(Float64)),
        :min_sym_eig => minimum(eig),
        :max_sym_eig => maximum(eig),
        :trace_sym_ratio => tr(S) / max(tr(ST), eps(Float64)),
    )
end

function print_phi_metrics(label::AbstractString, Phi::AbstractMatrix{<:Real}, Phi_true::AbstractMatrix{<:Real}, K::Int)
    m = phi_agreement_metrics(Phi, Phi_true)
    prof = block_profile(Phi, K)
    @printf("=== Phi diagnostic: %s ===\n", label)
    @printf("r=0 block = [[%.8e, %.8e], [%.8e, %.8e]]\n",
        prof[1, 1, 1], prof[1, 1, 2], prof[1, 2, 1], prof[1, 2, 2])
    @printf("||Phi - <M_true>|| / ||<M_true>||       = %.8e\n", m[:relative_rmse])
    @printf("||sym(Phi)-sym(<M_true>)|| / ||sym||    = %.8e\n", m[:sym_relative_rmse])
    @printf("||anti(Phi)-anti(<M_true>)|| / ||anti|| = %.8e\n", m[:anti_relative_rmse])
    @printf("off-site profile norm / ||<M_true>||    = %.8e\n", m[:offsite_relative_norm])
    @printf("min eig sym(Phi)                        = %.8e\n", m[:min_sym_eig])
    @printf("max eig sym(Phi)                        = %.8e\n", m[:max_sym_eig])
    @printf("trace sym(Phi) / trace sym(<M_true>)    = %.8e\n", m[:trace_sym_ratio])
    return m
end

function render_phi_constant_stats(path::AbstractString, sampler::PairSampler, obs_states::Array{Float32, 4},
        phi_states::Array{Float32, 4}, save_dt::Float64; pdf_bins::Int=120, max_pdf_samples::Int=250_000,
        title_suffix::AbstractString="")
    ensure_parent_dir(path)
    rng = MersenneTwister(812_345)
    q_obs = draw_channel_values_from_states(obs_states, 1, max_pdf_samples, rng)
    p_obs = draw_channel_values_from_states(obs_states, 2, max_pdf_samples, rng)
    a_obs = draw_amplitude_values_from_states(obs_states, max_pdf_samples, rng)
    r2_obs = draw_radius_squared_values_from_states(obs_states, max_pdf_samples, rng)
    ranges = (
        collect(range(quantile(q_obs, 0.001), quantile(q_obs, 0.999); length=pdf_bins + 1)),
        collect(range(quantile(p_obs, 0.001), quantile(p_obs, 0.999); length=pdf_bins + 1)),
        collect(range(max(0.0, quantile(a_obs, 0.001)), quantile(a_obs, 0.999); length=pdf_bins + 1)),
        collect(range(max(0.0, quantile(r2_obs, 0.001)), quantile(r2_obs, 0.999); length=pdf_bins + 1)),
    )
    q_cent, q_obs_d = histogram_density(q_obs, ranges[1])
    p_cent, p_obs_d = histogram_density(p_obs, ranges[2])
    a_cent, a_obs_d = histogram_density(a_obs, ranges[3])
    r2_cent, r2_obs_d = histogram_density(r2_obs, ranges[4])
    _, q_phi_d = histogram_density(draw_channel_values_from_states(phi_states, 1, max_pdf_samples, rng), ranges[1])
    _, p_phi_d = histogram_density(draw_channel_values_from_states(phi_states, 2, max_pdf_samples, rng), ranges[2])
    _, a_phi_d = histogram_density(draw_amplitude_values_from_states(phi_states, max_pdf_samples, rng), ranges[3])
    _, r2_phi_d = histogram_density(draw_radius_squared_values_from_states(phi_states, max_pdf_samples, rng), ranges[4])

    max_lag = min(size(obs_states, 1), size(phi_states, 1)) - 1
    max_lag = min(max_lag, round(Int, 9.0 / save_dt))
    stride = max(1, round(Int, 0.06 / save_dt))
    lag_steps, acf_q_obs = channel_acf(obs_states, 1, max_lag, stride)
    _, acf_q_phi = channel_acf(phi_states, 1, max_lag, stride)
    _, acf_p_obs = channel_acf(obs_states, 2, max_lag, stride)
    _, acf_p_phi = channel_acf(phi_states, 2, max_lag, stride)
    _, acf_r_obs = amplitude_acf(obs_states, max_lag, stride)
    _, acf_r_phi = amplitude_acf(phi_states, max_lag, stride)
    _, cross_obs = channel_cross_same_site(obs_states, max_lag, stride)
    _, cross_phi = channel_cross_same_site(phi_states, max_lag, stride)
    lag_times = lag_steps .* save_dt

    obs_summary = state_summary_vector(obs_states)
    phi_summary = state_summary_vector(phi_states)
    summary_labels = ["mean q", "mean p", "mean r", "std q", "std p", "std r", "mean r²"]
    modes, q_obs_spec = spatial_power_spectrum(obs_states, 1)
    _, q_phi_spec = spatial_power_spectrum(phi_states, 1)
    _, p_obs_spec = spatial_power_spectrum(obs_states, 2)
    _, p_phi_spec = spatial_power_spectrum(phi_states, 2)

    width, height = 3600, 2700
    with_scaled_figure_style(width, height; scale_override=1.0) do _
        fig = Figure(size=(width, height))
        figure_title!(fig, "Complex-amplitude constant-mobility Langevin validation";
            subtitle=title_suffix)
        specs = [
            (q_cent, q_obs_d, q_phi_d, "q PDF", "q", "density", fig[1, 1]),
            (p_cent, p_obs_d, p_phi_d, "p PDF", "p", "density", fig[1, 2]),
            (a_cent, a_obs_d, a_phi_d, "amplitude PDF", "r", "density", fig[1, 3]),
            (r2_cent, r2_obs_d, r2_phi_d, "amplitude-squared PDF", "r²", "density", fig[1, 4]),
            (lag_times, acf_q_obs, acf_q_phi, "q autocorrelation", "τ", "C/C(0)", fig[2, 1]),
            (lag_times, acf_p_obs, acf_p_phi, "p autocorrelation", "τ", "C/C(0)", fig[2, 2]),
            (lag_times, acf_r_obs, acf_r_phi, "r autocorrelation", "τ", "C/C(0)", fig[2, 3]),
            (lag_times, cross_obs, cross_phi, "q(t) p(0) cross-correlation", "τ", "corr", fig[2, 4]),
            (modes, q_obs_spec, q_phi_spec, "q spatial spectrum", "mode", "power", fig[3, 3]),
            (modes, p_obs_spec, p_phi_spec, "p spatial spectrum", "mode", "power", fig[3, 4]),
        ]
        for (x, obs, phi, ttl, xlabel, ylabel, slot) in specs
            ax = Axis(slot; title=ttl, xlabel=xlabel, ylabel=ylabel)
            lines!(ax, x, obs; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
            lines!(ax, x, phi; color=STYLE_SECONDARY, linestyle=:dash,
                linewidth=curve_linewidth(emphasis=0.95), label="M=Φ")
            ttl == "q PDF" && axislegend(ax; position=:rt)
        end
        ax_mean = Axis(fig[3, 1]; title="Mean statistics", xlabel="statistic", ylabel="value",
            xticks=(1:3, summary_labels[1:3]))
        barplot!(ax_mean, (1:3) .- 0.16, obs_summary[1:3]; width=0.28, color=STYLE_REFERENCE, label="data")
        barplot!(ax_mean, (1:3) .+ 0.16, phi_summary[1:3]; width=0.28, color=STYLE_SECONDARY, label="M=Φ")
        ax_mean.xticklabelrotation = pi / 8
        axislegend(ax_mean; position=:lt)
        ax_scale = Axis(fig[3, 2]; title="Scale statistics", xlabel="statistic", ylabel="value",
            xticks=(1:4, summary_labels[4:7]))
        barplot!(ax_scale, (1:4) .- 0.16, obs_summary[4:7]; width=0.28, color=STYLE_REFERENCE, label="data")
        barplot!(ax_scale, (1:4) .+ 0.16, phi_summary[4:7]; width=0.28, color=STYLE_SECONDARY, label="M=Φ")
        ax_scale.xticklabelrotation = pi / 8
        apply_publication_grid!(fig.layout, 3, 4; row_gap=30, col_gap=30)
        save_figure(path, fig)
    end
    return Dict(
        :lag_times => lag_times,
        :acf_q_obs => acf_q_obs, :acf_q_phi => acf_q_phi,
        :acf_p_obs => acf_p_obs, :acf_p_phi => acf_p_phi,
        :acf_r_obs => acf_r_obs, :acf_r_phi => acf_r_phi,
        :cross_obs => cross_obs, :cross_phi => cross_phi,
        :summary_labels => summary_labels, :obs_summary => obs_summary, :phi_summary => phi_summary,
        :modes => modes, :q_obs_spec => q_obs_spec, :q_phi_spec => q_phi_spec,
        :p_obs_spec => p_obs_spec, :p_phi_spec => p_phi_spec,
    )
end

function render_phi_recovery_figure(path::AbstractString, Phi_raw::Matrix{Float64},
        Phi_corr::Matrix{Float64}, Phi_true::Matrix{Float64}, V::Matrix{Float64}, sensitivities)
    ensure_parent_dir(path)
    K = size(Phi_true, 1) ÷ 2
    prof_true = block_profile(Phi_true, K)
    prof_raw = block_profile(Phi_raw, K)
    prof_corr = block_profile(Phi_corr, K)
    r = collect(0:(K - 1))
    eig_true = eigen(Symmetric(sympart(Phi_true))).values
    eig_raw = eigen(Symmetric(sympart(Phi_raw))).values
    eig_corr = eigen(Symmetric(sympart(Phi_corr))).values
    width, height = 3000, 2100
    with_scaled_figure_style(width, height; scale_override=1.0) do _
        fig = Figure(size=(width, height))
        figure_title!(fig, "Complex-amplitude Phi recovery diagnostic";
            subtitle="Phi from short-lag Cdot(0), ex-post compared with stored <M_true>")
        ax1 = Axis(fig[1, 1]; title="true <M>", xlabel="column", ylabel="row")
        hm1 = heatmap!(ax1, Phi_true; colormap=:balance)
        Colorbar(fig[1, 1, Right()], hm1)
        ax2 = Axis(fig[1, 2]; title="data Phi, raw", xlabel="column", ylabel="row")
        hm2 = heatmap!(ax2, Phi_raw; colormap=:balance)
        Colorbar(fig[1, 2, Right()], hm2)
        ax3 = Axis(fig[1, 3]; title="data Phi, Stein-corrected", xlabel="column", ylabel="row")
        hm3 = heatmap!(ax3, Phi_corr; colormap=:balance)
        Colorbar(fig[1, 3, Right()], hm3)
        ax4 = Axis(fig[2, 1]; title="r=0 block profiles", xlabel="site offset", ylabel="entry")
        entries = ((1, 1, "qq"), (1, 2, "qp"), (2, 1, "pq"), (2, 2, "pp"))
        colors = (STYLE_REFERENCE, STYLE_SECONDARY, STYLE_PRIMARY, STYLE_ACCENT)
        for (idx, (cm, cn, lbl)) in enumerate(entries)
            lines!(ax4, r, prof_true[:, cm, cn]; color=colors[idx], linewidth=curve_linewidth(), label="true " * lbl)
            lines!(ax4, r, prof_raw[:, cm, cn]; color=colors[idx], linestyle=:dash,
                linewidth=curve_linewidth(emphasis=0.75), label="raw " * lbl)
            lines!(ax4, r, prof_corr[:, cm, cn]; color=colors[idx], linestyle=:dot,
                linewidth=curve_linewidth(emphasis=0.75), label="corr " * lbl)
        end
        axislegend(ax4; position=:rt, nbanks=2)
        ax5 = Axis(fig[2, 2]; title="short-lag Phi sensitivity", xlabel="fit lag count L", ylabel="rel. error")
        lag_counts = [s[:L] for s in sensitivities]
        raw_errs = [s[:raw_rel] for s in sensitivities]
        corr_errs = [s[:corr_rel] for s in sensitivities]
        circcorr_errs = [s[:circcorr_rel] for s in sensitivities]
        lines!(ax5, lag_counts, raw_errs; color=STYLE_SECONDARY, label="raw")
        scatter!(ax5, lag_counts, raw_errs; color=STYLE_SECONDARY)
        lines!(ax5, lag_counts, corr_errs; color=STYLE_PRIMARY, label="full Stein")
        scatter!(ax5, lag_counts, corr_errs; color=STYLE_PRIMARY)
        lines!(ax5, lag_counts, circcorr_errs; color=STYLE_ACCENT, label="circulant Stein")
        scatter!(ax5, lag_counts, circcorr_errs; color=STYLE_ACCENT)
        axislegend(ax5; position=:rt)
        ax6 = Axis(fig[2, 3]; title="sym(Phi) eigenvalues", xlabel="index", ylabel="eigenvalue")
        idxs = 1:length(eig_true)
        lines!(ax6, idxs, eig_true; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="true")
        lines!(ax6, idxs, eig_raw; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(), label="raw")
        lines!(ax6, idxs, eig_corr; color=STYLE_PRIMARY, linestyle=:dot, linewidth=curve_linewidth(), label="corr")
        axislegend(ax6; position=:lt)
        ax7 = Axis(fig[3, 1]; title="Stein matrix V", xlabel="column", ylabel="row")
        ident = Matrix{Float64}(I, size(V, 1), size(V, 2))
        hm7 = heatmap!(ax7, V .- ident; colormap=:balance)
        Colorbar(fig[3, 1, Right()], hm7; label="V-I")
        ax8 = Axis(fig[3, 2]; title="raw Phi error", xlabel="column", ylabel="row")
        hm8 = heatmap!(ax8, Phi_raw .- Phi_true; colormap=:balance)
        Colorbar(fig[3, 2, Right()], hm8)
        ax9 = Axis(fig[3, 3]; title="corrected Phi error", xlabel="column", ylabel="row")
        hm9 = heatmap!(ax9, Phi_corr .- Phi_true; colormap=:balance)
        Colorbar(fig[3, 3, Right()], hm9)
        apply_publication_grid!(fig.layout, 3, 3; row_gap=34, col_gap=34)
        save_figure(path, fig)
    end
    return nothing
end

function load_true_mobility_params(path::AbstractString)
    return (
        Float64(h5read(path, "/metadata/d0")),
        Float64(h5read(path, "/metadata/d1")),
        Float64(h5read(path, "/metadata/omega0")),
        Float64(h5read(path, "/metadata/omega1")),
    )
end

function load_potential_params(path::AbstractString)
    return (
        Float64(h5read(path, "/metadata/alpha")),
        Float64(h5read(path, "/metadata/beta")),
        Float64(h5read(path, "/metadata/kappa")),
    )
end

function exact_stationary_score!(dest::AbstractArray{Float32, 3}, z::AbstractArray{Float32, 3},
        alpha::Float64, beta::Float64, kappa::Float64)
    K, _, B = size(z)
    @inbounds for b in 1:B, i in 1:K
        im1 = periodic(i - 1, K)
        ip1 = periodic(i + 1, K)
        q = Float64(z[i, 1, b])
        p = Float64(z[i, 2, b])
        r2 = q*q + p*p
        gq = alpha * q + beta * r2 * q + kappa * (2.0 * q - Float64(z[im1, 1, b]) - Float64(z[ip1, 1, b]))
        gp = alpha * p + beta * r2 * p + kappa * (2.0 * p - Float64(z[im1, 2, b]) - Float64(z[ip1, 2, b]))
        dest[i, 1, b] = Float32(-gq)
        dest[i, 2, b] = Float32(-gp)
    end
    return dest
end

function evaluate_conditional_score_x0_exact_stationary(models::LoadedModels,
        z0::AbstractArray{Float32, 3}, zt::AbstractArray{Float32, 3}, tau::Float64,
        params::FitDMParams, device::ExecutionDevice,
        potential_params::Tuple{Float64, Float64, Float64};
        joint_stein_correction::Union{Nothing, Matrix{Float64}}=nothing)
    require_condition(models.tau_min <= tau <= models.tau_max + 1e-10,
        @sprintf("tau %.6f lies outside joint-score range [%.6f, %.6f]", tau, models.tau_min, models.tau_max))
    tnorm = Float32((tau - models.tau_min) / max(models.tau_max - models.tau_min, eps(Float64)))
    joint = evaluate_joint_score_x0(models, z0, zt, tnorm, params.joint_batch_size, device)
    if joint_stein_correction !== nothing
        apply_score_left_correction!(joint, joint_stein_correction)
    end
    stat = similar(joint)
    exact_stationary_score!(stat, z0, potential_params...)
    return joint .- stat
end

function apply_score_left_correction!(score::AbstractArray{Float32, 3}, correction::Matrix{Float64})
    K, _, B = size(score)
    D = 2K
    flat = Matrix{Float64}(undef, D, B)
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        flat[flat_index(i, c, K), b] = score[i, c, b]
    end
    corrected = correction * flat
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        score[i, c, b] = Float32(corrected[flat_index(i, c, K), b])
    end
    return score
end

function estimate_joint_x0_stein_correction(sampler::PairSampler, models::LoadedModels,
        params::FitDMParams, device::ExecutionDevice, lag::Int, tau::Float64,
        coord_mean::Vector{Float64})
    K = sampler.K
    D = sampler.D
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    V = zeros(Float64, D, D)
    rng = MersenneTwister(params.seed + 90_000 + lag)
    remaining = params.pairs_per_lag_operator
    total = 0
    tnorm = Float32((tau - models.tau_min) / max(models.tau_max - models.tau_min, eps(Float64)))
    while remaining > 0
        bn = min(remaining, params.batch_size)
        sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
        flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
        @inbounds for b in 1:bn, i in 1:K
            x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
            x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
        end
        joint = evaluate_joint_score_x0(models, @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]),
            tnorm, params.joint_batch_size, device)
        @inbounds for b in 1:bn, c in 1:2, i in 1:K
            row = flat_index(i, c, K)
            sval = -Float64(joint[i, c, b])
            for n in 1:D
                V[row, n] += sval * Float64(x0_flat[n, b])
            end
        end
        total += bn
        remaining -= bn
    end
    V ./= total
    correction = inv(V + 1.0e-6 * I)
    return correction, V
end

function estimate_joint_x0_stein_corrections(sampler::PairSampler, models::LoadedModels,
        params::FitDMParams, device::ExecutionDevice, coord_mean::Vector{Float64})
    corrections = Matrix{Float64}[]
    matrices = Matrix{Float64}[]
    for (idx, lag) in enumerate(sampler.lag_steps)
        correction, V = estimate_joint_x0_stein_correction(sampler, models, params, device,
            lag, sampler.lag_times[idx], coord_mean)
        push!(corrections, correction)
        push!(matrices, V)
        params.verbose && @printf("Estimated joint-score Stein correction for tau %.4f (%d / %d), ||V-I||/||I||=%.4e\n",
            sampler.lag_times[idx], idx, length(sampler.lag_steps), norm(V - I) / sqrt(size(V, 1)))
    end
    return corrections, matrices
end

function estimate_conditional_score_self_consistency(sampler::PairSampler, models::LoadedModels,
        params::FitDMParams, device::ExecutionDevice, coord_mean::Vector{Float64},
        potential_params::Tuple{Float64, Float64, Float64})
    D = sampler.D
    K = sampler.K
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    exact_stat = similar(z0)
    lags = sampler.lag_times
    diag = Dict(
        :taus => copy(lags),
        :cond_mean_norm_learned => zeros(Float64, length(lags)),
        :cond_x0_norm_learned => zeros(Float64, length(lags)),
        :cond_mean_norm_exactstat => zeros(Float64, length(lags)),
        :cond_x0_norm_exactstat => zeros(Float64, length(lags)),
        :joint_x0_stein_rel => zeros(Float64, length(lags)),
        :stat_x0_stein_rel => zeros(Float64, length(lags)),
    )
    rng = MersenneTwister(params.seed + 95_000)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[lag_idx]
        tnorm = Float32((tau - models.tau_min) / max(models.tau_max - models.tau_min, eps(Float64)))
        mean_learned = zeros(Float64, D)
        mean_exactstat = zeros(Float64, D)
        cond_x0_learned = zeros(Float64, D, D)
        cond_x0_exactstat = zeros(Float64, D, D)
        joint_x0 = zeros(Float64, D, D)
        stat_x0 = zeros(Float64, D, D)
        total = 0
        remaining = params.pairs_per_lag_operator
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            joint = evaluate_joint_score_x0(models, @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]),
                tnorm, params.joint_batch_size, device)
            stat = evaluate_stationary_score(models, @view(z0[:, :, 1:bn]), params.score_batch_size, device)
            exact_stationary_score!(@view(exact_stat[:, :, 1:bn]), @view(z0[:, :, 1:bn]), potential_params...)
            joint_flat = Matrix{Float64}(undef, D, bn)
            stat_flat = similar(joint_flat)
            exact_flat = similar(joint_flat)
            @inbounds for b in 1:bn, c in 1:2, i in 1:K
                idx = flat_index(i, c, K)
                joint_flat[idx, b] = joint[i, c, b]
                stat_flat[idx, b] = stat[i, c, b]
                exact_flat[idx, b] = exact_stat[i, c, b]
            end
            cond_learned = joint_flat .- stat_flat
            cond_exactstat = joint_flat .- exact_flat
            mean_learned .+= vec(sum(cond_learned; dims=2))
            mean_exactstat .+= vec(sum(cond_exactstat; dims=2))
            x0 = Float64.(@view(x0_flat[:, 1:bn]))
            cond_x0_learned .+= cond_learned * x0'
            cond_x0_exactstat .+= cond_exactstat * x0'
            joint_x0 .+= joint_flat * x0'
            stat_x0 .+= stat_flat * x0'
            total += bn
            remaining -= bn
        end
        inv_total = 1.0 / total
        mean_learned .*= inv_total
        mean_exactstat .*= inv_total
        cond_x0_learned .*= inv_total
        cond_x0_exactstat .*= inv_total
        joint_x0 .*= inv_total
        stat_x0 .*= inv_total
        diag[:cond_mean_norm_learned][lag_idx] = norm(mean_learned) / sqrt(D)
        diag[:cond_mean_norm_exactstat][lag_idx] = norm(mean_exactstat) / sqrt(D)
        diag[:cond_x0_norm_learned][lag_idx] = norm(cond_x0_learned) / sqrt(D)
        diag[:cond_x0_norm_exactstat][lag_idx] = norm(cond_x0_exactstat) / sqrt(D)
        diag[:joint_x0_stein_rel][lag_idx] = norm(joint_x0 + I) / sqrt(D)
        diag[:stat_x0_stein_rel][lag_idx] = norm(stat_x0 + I) / sqrt(D)
        params.verbose && @printf("Score self-consistency tau %.4f: ||E[s_cond x0']||/||I|| learned %.3e exactstat %.3e, joint Stein %.3e\n",
            tau, diag[:cond_x0_norm_learned][lag_idx], diag[:cond_x0_norm_exactstat][lag_idx],
            diag[:joint_x0_stein_rel][lag_idx])
    end
    return diag
end

function estimate_reverse_conditional_score_self_consistency(sampler::PairSampler,
        models::LoadedModels, reverse_model::LoadedReverseConditionalModel,
        params::FitDMParams, device::ExecutionDevice, coord_mean::Vector{Float64},
        potential_params::Tuple{Float64, Float64, Float64})
    D = sampler.D
    K = sampler.K
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, params.batch_size)
    exact_stat = similar(z0)
    diag = Dict(
        :taus => copy(sampler.lag_times),
        :cond_mean_norm_reverse => zeros(Float64, length(sampler.lag_steps)),
        :cond_x0_norm_reverse => zeros(Float64, length(sampler.lag_steps)),
        :cond_mean_norm_reverse_exactstat => zeros(Float64, length(sampler.lag_steps)),
        :cond_x0_norm_reverse_exactstat => zeros(Float64, length(sampler.lag_steps)),
        :posterior_x0_stein_rel => zeros(Float64, length(sampler.lag_steps)),
    )
    rng = MersenneTwister(params.seed + 96_000)
    for (lag_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[lag_idx]
        mean_reverse = zeros(Float64, D)
        mean_exactstat = zeros(Float64, D)
        cond_x0_reverse = zeros(Float64, D, D)
        cond_x0_exactstat = zeros(Float64, D, D)
        posterior_x0 = zeros(Float64, D, D)
        total = 0
        remaining = params.pairs_per_lag_operator
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            posterior = evaluate_reverse_posterior_score_x0(reverse_model, @view(z0[:, :, 1:bn]),
                @view(zt[:, :, 1:bn]), tau, params.joint_batch_size, device)
            stat = evaluate_stationary_score(models, @view(z0[:, :, 1:bn]), params.score_batch_size, device)
            exact_stationary_score!(@view(exact_stat[:, :, 1:bn]), @view(z0[:, :, 1:bn]), potential_params...)
            posterior_flat = Matrix{Float64}(undef, D, bn)
            stat_flat = similar(posterior_flat)
            exact_flat = similar(posterior_flat)
            @inbounds for b in 1:bn, c in 1:2, i in 1:K
                idx = flat_index(i, c, K)
                posterior_flat[idx, b] = posterior[i, c, b]
                stat_flat[idx, b] = stat[i, c, b]
                exact_flat[idx, b] = exact_stat[i, c, b]
            end
            if reverse_model.score_type == "reverse_transition_residual_x0_given_xt"
                cond_reverse = posterior_flat
                cond_exactstat = posterior_flat
            else
                cond_reverse = posterior_flat .- stat_flat
                cond_exactstat = posterior_flat .- exact_flat
            end
            mean_reverse .+= vec(sum(cond_reverse; dims=2))
            mean_exactstat .+= vec(sum(cond_exactstat; dims=2))
            x0 = Float64.(@view(x0_flat[:, 1:bn]))
            cond_x0_reverse .+= cond_reverse * x0'
            cond_x0_exactstat .+= cond_exactstat * x0'
            posterior_x0 .+= posterior_flat * x0'
            total += bn
            remaining -= bn
        end
        inv_total = 1.0 / total
        mean_reverse .*= inv_total
        mean_exactstat .*= inv_total
        cond_x0_reverse .*= inv_total
        cond_x0_exactstat .*= inv_total
        posterior_x0 .*= inv_total
        diag[:cond_mean_norm_reverse][lag_idx] = norm(mean_reverse) / sqrt(D)
        diag[:cond_mean_norm_reverse_exactstat][lag_idx] = norm(mean_exactstat) / sqrt(D)
        diag[:cond_x0_norm_reverse][lag_idx] = norm(cond_x0_reverse) / sqrt(D)
        diag[:cond_x0_norm_reverse_exactstat][lag_idx] = norm(cond_x0_exactstat) / sqrt(D)
        diag[:posterior_x0_stein_rel][lag_idx] =
            reverse_model.score_type == "reverse_transition_residual_x0_given_xt" ?
                norm(posterior_x0) / sqrt(D) : norm(posterior_x0 + I) / sqrt(D)
        params.verbose && @printf("Reverse score self-consistency tau %.4f: ||E[s_cond x0']||/||I|| learned %.3e exactstat %.3e, posterior Stein %.3e\n",
            tau, diag[:cond_x0_norm_reverse][lag_idx],
            diag[:cond_x0_norm_reverse_exactstat][lag_idx], diag[:posterior_x0_stein_rel][lag_idx])
    end
    return diag
end

function true_mobility_transpose_action!(dest::AbstractMatrix{Float32},
        z0::AbstractArray{Float32, 3}, scond::AbstractArray{Float32, 3},
        d0::Float64, d1::Float64, omega0::Float64, omega1::Float64)
    K, _, B = size(z0)
    @inbounds for b in 1:B, i in 1:K
        q = Float64(z0[i, 1, b])
        p = Float64(z0[i, 2, b])
        r2 = q*q + p*p
        d = Float32(d0 + d1 * r2)
        omega = Float32(omega0 + omega1 * r2)
        sq = scond[i, 1, b]
        sp = scond[i, 2, b]
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        dest[qrow, b] = d * sq + omega * sp
        dest[prow, b] = -omega * sq + d * sp
    end
    return nothing
end

function true_mobility_row_action!(dest::AbstractMatrix{Float32},
        z0::AbstractArray{Float32, 3}, scond::AbstractArray{Float32, 3},
        d0::Float64, d1::Float64, omega0::Float64, omega1::Float64)
    K, _, B = size(z0)
    @inbounds for b in 1:B, i in 1:K
        q = Float64(z0[i, 1, b])
        p = Float64(z0[i, 2, b])
        r2 = q*q + p*p
        d = Float32(d0 + d1 * r2)
        omega = Float32(omega0 + omega1 * r2)
        sq = scond[i, 1, b]
        sp = scond[i, 2, b]
        qrow = flat_index(i, 1, K)
        prow = flat_index(i, 2, K)
        dest[qrow, b] = d * sq - omega * sp
        dest[prow, b] = omega * sq + d * sp
    end
    return nothing
end

function estimate_true_mobility_cdot_from_conditional_score(sampler::PairSampler, models::LoadedModels,
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice,
        true_mobility_params::Tuple{Float64, Float64, Float64, Float64};
        use_transpose_action::Bool=true,
        potential_params::Union{Nothing, Tuple{Float64, Float64, Float64}}=nothing,
        joint_stein_corrections::Union{Nothing, Vector{Matrix{Float64}}}=nothing)
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot_true = fill(NaN, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    action = Matrix{Float32}(undef, D, params.batch_size)
    rng = MersenneTwister(params.seed + 80)
    d0, d1, omega0, omega1 = true_mobility_params
    for fit_idx in eachindex(sampler.lag_steps)
        lag = sampler.lag_steps[fit_idx]
        tau = sampler.lag_times[fit_idx]
        sums = zeros(Float64, K, nobs, D)
        total = 0
        remaining = params.pairs_per_lag_operator
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = potential_params === nothing ?
                evaluate_conditional_score_x0(models, @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]),
                    tau, params, device) :
                evaluate_conditional_score_x0_exact_stationary(models, @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]),
                    tau, params, device, potential_params;
                    joint_stein_correction=joint_stein_corrections === nothing ? nothing : joint_stein_corrections[fit_idx])
            if use_transpose_action
                true_mobility_transpose_action!(@view(action[:, 1:bn]), @view(z0[:, :, 1:bn]),
                    scond, d0, d1, omega0, omega1)
            else
                true_mobility_row_action!(@view(action[:, 1:bn]), @view(z0[:, :, 1:bn]),
                    scond, d0, d1, omega0, omega1)
            end
            accumulate_action_sums!(sums, @view(obs_vals[:, :, 1:bn]), @view(action[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot_true[fit_idx, :, :, :] .= .-sums ./ total
        params.verbose && @printf("Estimated true-M conditional-score Cdot for tau %.4f (%d / %d)\n",
            tau, fit_idx, length(sampler.lag_steps))
    end
    return Cdot_true
end

function estimate_phi_cdot_from_conditional_score(sampler::PairSampler, models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel},
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice, Phi::Matrix{Float64})
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot_phi = fill(NaN, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    action = Matrix{Float32}(undef, D, params.batch_size)
    rng = MersenneTwister(params.seed + 135_000)
    for (fit_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[fit_idx]
        sums = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_operator
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = evaluate_transition_score_x0(models, reverse_model,
                @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), tau, params, device)
            phi_action!(@view(action[:, 1:bn]), Phi, scond)
            accumulate_action_sums!(sums, @view(obs_vals[:, :, 1:bn]), @view(action[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot_phi[fit_idx, :, :, :] .= .-sums ./ total
        params.verbose && @printf("Estimated Phi conditional-score Cdot for tau %.4f (%d / %d)\n",
            tau, fit_idx, length(sampler.lag_steps))
    end
    return Cdot_phi
end

function estimate_phi_cdot_from_reverse_conditional_score(sampler::PairSampler,
        models::LoadedModels, reverse_model::LoadedReverseConditionalModel,
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice,
        Phi::Matrix{Float64};
        potential_params::Union{Nothing, Tuple{Float64, Float64, Float64}}=nothing)
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot_phi = fill(NaN, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    action = Matrix{Float32}(undef, D, params.batch_size)
    rng = MersenneTwister(params.seed + 136_000)
    for (fit_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[fit_idx]
        sums = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_operator
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = potential_params === nothing ?
                evaluate_reverse_conditional_score_x0(models, reverse_model, @view(z0[:, :, 1:bn]),
                    @view(zt[:, :, 1:bn]), tau, params, device) :
                evaluate_reverse_conditional_score_x0_exact_stationary(reverse_model, @view(z0[:, :, 1:bn]),
                    @view(zt[:, :, 1:bn]), tau, params, device, potential_params)
            phi_action!(@view(action[:, 1:bn]), Phi, scond)
            accumulate_action_sums!(sums, @view(obs_vals[:, :, 1:bn]), @view(action[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot_phi[fit_idx, :, :, :] .= .-sums ./ total
        label = potential_params === nothing ? "DSM-score" : "exact-score"
        params.verbose && @printf("Estimated Phi reverse %s Cdot for tau %.4f (%d / %d)\n",
            label, tau, fit_idx, length(sampler.lag_steps))
    end
    return Cdot_phi
end

function estimate_nn_mobility_cdot_from_conditional_score(sampler::PairSampler, models::LoadedModels,
        reverse_model::Union{Nothing, LoadedReverseConditionalModel},
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice, Phi::Matrix{Float64},
        mobility_model)
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot_nn = fill(NaN, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    phi_act = Matrix{Float32}(undef, D, params.batch_size)
    nn_act = similar(phi_act)
    action = similar(phi_act)
    rng = MersenneTwister(params.seed + 145_000)
    for (fit_idx, lag) in enumerate(sampler.lag_steps)
        tau = sampler.lag_times[fit_idx]
        sums = zeros(Float64, K, nobs, D)
        remaining = params.pairs_per_lag_operator
        total = 0
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = evaluate_transition_score_x0(models, reverse_model,
                @view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), tau, params, device)
            phi_action!(@view(phi_act[:, 1:bn]), Phi, scond)
            nn_local_transpose_action!(@view(nn_act[:, 1:bn]), mobility_model,
                @view(z0[:, :, 1:bn]), scond, lib.mean_r2)
            @views action[:, 1:bn] .= phi_act[:, 1:bn] .+ nn_act[:, 1:bn]
            accumulate_action_sums!(sums, @view(obs_vals[:, :, 1:bn]), @view(action[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot_nn[fit_idx, :, :, :] .= .-sums ./ total
        params.verbose && @printf("Estimated NN-M conditional-score Cdot for tau %.4f (%d / %d)\n",
            tau, fit_idx, length(sampler.lag_steps))
    end
    return Cdot_nn
end

function estimate_true_mobility_cdot_from_reverse_conditional_score(sampler::PairSampler,
        models::LoadedModels, reverse_model::LoadedReverseConditionalModel,
        lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice,
        true_mobility_params::Tuple{Float64, Float64, Float64, Float64};
        use_transpose_action::Bool=true,
        potential_params::Union{Nothing, Tuple{Float64, Float64, Float64}}=nothing)
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot_true = fill(NaN, length(sampler.lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, params.batch_size)
    zt = similar(z0)
    obs_vals = Array{Float32}(undef, K, nobs, params.batch_size)
    action = Matrix{Float32}(undef, D, params.batch_size)
    rng = MersenneTwister(params.seed + 81)
    d0, d1, omega0, omega1 = true_mobility_params
    for fit_idx in eachindex(sampler.lag_steps)
        lag = sampler.lag_steps[fit_idx]
        tau = sampler.lag_times[fit_idx]
        sums = zeros(Float64, K, nobs, D)
        total = 0
        remaining = params.pairs_per_lag_operator
        while remaining > 0
            bn = min(remaining, params.batch_size)
            sample_pair_batch!(@view(z0[:, :, 1:bn]), @view(zt[:, :, 1:bn]), sampler, lag, rng)
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            scond = potential_params === nothing ?
                evaluate_reverse_conditional_score_x0(models, reverse_model, @view(z0[:, :, 1:bn]),
                    @view(zt[:, :, 1:bn]), tau, params, device) :
                evaluate_reverse_conditional_score_x0_exact_stationary(reverse_model, @view(z0[:, :, 1:bn]),
                    @view(zt[:, :, 1:bn]), tau, params, device, potential_params)
            if use_transpose_action
                true_mobility_transpose_action!(@view(action[:, 1:bn]), @view(z0[:, :, 1:bn]),
                    scond, d0, d1, omega0, omega1)
            else
                true_mobility_row_action!(@view(action[:, 1:bn]), @view(z0[:, :, 1:bn]),
                    scond, d0, d1, omega0, omega1)
            end
            accumulate_action_sums!(sums, @view(obs_vals[:, :, 1:bn]), @view(action[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        Cdot_true[fit_idx, :, :, :] .= .-sums ./ total
        params.verbose && @printf("Estimated reverse-conditional true-M Cdot for tau %.4f (%d / %d)\n",
            tau, fit_idx, length(sampler.lag_steps))
    end
    return Cdot_true
end

function translation_profiles(Cdot::AbstractArray{<:Real, 4}, K::Int)
    nt, _, nobs, D = size(Cdot)
    prof = fill(NaN, nt, nobs, 2, K)
    @inbounds for t in 1:nt, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)
        total = 0.0
        count = 0
        for i in 1:K
            j = periodic(i + r0, K)
            n = flat_index(j, comp, K)
            val = Cdot[t, i, a, n]
            if isfinite(val)
                total += Float64(val)
                count += 1
            end
        end
        prof[t, a, comp, r0 + 1] = count > 0 ? total / count : NaN
    end
    return prof
end

function observable_translation_noise(Cdot::AbstractArray{<:Real, 4}, profiles::Array{Float64, 4}, K::Int)
    nt, _, nobs, _ = size(Cdot)
    noise = zeros(Float64, nobs)
    counts = zeros(Int, nobs)
    @inbounds for t in 1:nt, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)
        mean_val = profiles[t, a, comp, r0 + 1]
        isfinite(mean_val) || continue
        for i in 1:K
            j = periodic(i + r0, K)
            n = flat_index(j, comp, K)
            val = Float64(Cdot[t, i, a, n])
            isfinite(val) || continue
            noise[a] += (val - mean_val)^2
            counts[a] += 1
        end
    end
    out = fill(NaN, nobs)
    for a in 1:nobs
        if counts[a] > 0
            # Standard error of the translation-averaged profile, not raw site scatter.
            out[a] = sqrt(noise[a] / counts[a]) / sqrt(K)
        end
    end
    return out
end

function observable_profile_roughness(profiles::Array{Float64, 4})
    nt, nobs, _, _ = size(profiles)
    rough = zeros(Float64, nobs)
    counts = zeros(Int, nobs)
    nt < 3 && return fill(0.0, nobs)
    @inbounds for a in 1:nobs, comp in 1:2, r in axes(profiles, 4), t in 2:(nt - 1)
        v0 = profiles[t - 1, a, comp, r]
        v1 = profiles[t, a, comp, r]
        v2 = profiles[t + 1, a, comp, r]
        if isfinite(v0) && isfinite(v1) && isfinite(v2)
            rough[a] += (v2 - 2.0 * v1 + v0)^2
            counts[a] += 1
        end
    end
    out = fill(NaN, nobs)
    for a in 1:nobs
        out[a] = counts[a] > 0 ? sqrt(rough[a] / counts[a]) : NaN
    end
    return out
end

function subset_observable_library(lib::ObservableLibrary, selected::Vector{Int})
    return ObservableLibrary(
        lib.names[selected],
        lib.kind[selected],
        lib.component[selected],
        lib.offset[selected],
        lib.mean[selected],
        lib.mean_r2,
    )
end

function select_observables_for_mobility(lib::ObservableLibrary, Cdot_data::Array{Float64, 4},
        Cdot_generator::Array{Float64, 4}, params::FitDMParams, K::Int)
    profiles = translation_profiles(Cdot_data, K)
    generator_profiles = translation_profiles(Cdot_generator, K)
    nobs = length(lib.names)
    signal = zeros(Float64, nobs)
    generator_signal = zeros(Float64, nobs)
    generator_rel = fill(Inf, nobs)
    for a in 1:nobs
        vals = vec(@view profiles[:, a, :, :])
        vals = vals[isfinite.(vals)]
        signal[a] = isempty(vals) ? 0.0 : sqrt(mean(vals .^ 2))
        gvals = vec(@view generator_profiles[:, a, :, :])
        gvals = gvals[isfinite.(gvals)]
        generator_signal[a] = isempty(gvals) ? 0.0 : sqrt(mean(gvals .^ 2))
        mask = isfinite.(@view profiles[:, a, :, :]) .& isfinite.(@view generator_profiles[:, a, :, :])
        if any(mask)
            d = profiles[:, a, :, :][mask]
            g = generator_profiles[:, a, :, :][mask]
            generator_rel[a] = sqrt(mean((d .- g) .^ 2)) / max(sqrt(mean(g .^ 2)), eps(Float64))
        end
    end
    noise = observable_translation_noise(Cdot_data, profiles, K)
    rough_abs = observable_profile_roughness(profiles)
    snr = signal ./ max.(noise, eps(Float64))
    rough_rel = rough_abs ./ max.(signal, eps(Float64))
    score = generator_signal .* snr ./ ((1.0 .+ rough_rel) .* (1.0 .+ generator_rel))

    valid_signal_pool = findall(a -> isfinite(generator_rel[a]) &&
        generator_rel[a] <= params.selection_max_generator_rel, 1:nobs)
    signal_reference = isempty(valid_signal_pool) ? maximum(signal) : maximum(signal[valid_signal_pool])
    signal_cut = params.selection_min_signal_fraction * signal_reference
    eligible = findall(a -> signal[a] >= signal_cut &&
        snr[a] >= params.selection_min_snr &&
        rough_rel[a] <= params.selection_max_roughness &&
        generator_rel[a] <= params.selection_max_generator_rel, 1:nobs)
    ranked = sort(eligible; by=a -> score[a], rev=true)

    selected = Int[]
    if params.selection_force_coordinates
        append!(selected, findall(a -> lib.kind[a] == :coord, 1:nobs))
    end
    for a in ranked
        a in selected && continue
        push!(selected, a)
        length(selected) >= params.max_selected_observables && break
    end
    selected = selected[1:min(length(selected), params.max_selected_observables)]
    selection = Dict(
        :selected_indices => selected,
        :signal_rms => signal,
        :translation_noise_se => noise,
        :snr => snr,
        :roughness_relative => rough_rel,
        :generator_signal_rms => generator_signal,
        :generator_relative_rmse => generator_rel,
        :selection_score => score,
        :signal_cut => signal_cut,
    )
    @printf("Selected %d / %d observables for mobility training diagnostics:\n", length(selected), nobs)
    for a in selected
        @printf("  [%02d] %-34s signal=%.4e gen_rel=%.3f snr=%.3f rough=%.3f score=%.4e\n",
            a, lib.names[a], signal[a], generator_rel[a], snr[a], rough_rel[a], score[a])
    end
    return selected, selection
end

function true_mobility_agreement_metrics(data_prof::Array{Float64, 4}, true_prof::Array{Float64, 4})
    mask = isfinite.(data_prof) .& isfinite.(true_prof)
    data_vals = data_prof[mask]
    true_vals = true_prof[mask]
    diff = true_vals .- data_vals
    rmse = sqrt(mean(diff .^ 2))
    data_rms = sqrt(mean(data_vals .^ 2))
    corr = cor(data_vals, true_vals)
    rel = rmse / max(data_rms, eps(Float64))
    sign_flip_rmse = sqrt(mean((-true_vals .- data_vals) .^ 2))
    nt, nobs, _, _ = size(data_prof)
    per_observable_rel = fill(NaN, nobs)
    per_observable_corr = fill(NaN, nobs)
    per_lag_rel = fill(NaN, nt)
    for a in 1:nobs
        obs_mask = mask[:, a, :, :]
        dv = data_prof[:, a, :, :][obs_mask]
        tv = true_prof[:, a, :, :][obs_mask]
        if !isempty(dv)
            per_observable_rel[a] = sqrt(mean((tv .- dv) .^ 2)) / max(sqrt(mean(dv .^ 2)), eps(Float64))
            per_observable_corr[a] = length(dv) > 1 ? cor(dv, tv) : NaN
        end
    end
    for t in 1:nt
        lag_mask = mask[t, :, :, :]
        dv = data_prof[t, :, :, :][lag_mask]
        tv = true_prof[t, :, :, :][lag_mask]
        if !isempty(dv)
            per_lag_rel[t] = sqrt(mean((tv .- dv) .^ 2)) / max(sqrt(mean(dv .^ 2)), eps(Float64))
        end
    end
    return Dict(
        :rmse => rmse,
        :relative_rmse => rel,
        :correlation => corr,
        :data_rms => data_rms,
        :sign_flip_rmse => sign_flip_rmse,
        :npoints => length(data_vals),
        :per_observable_rel => per_observable_rel,
        :per_observable_corr => per_observable_corr,
        :per_lag_rel => per_lag_rel,
    )
end

function profile_kind_metrics(lib::ObservableLibrary, reference_prof::Array{Float64, 4},
        estimate_prof::Array{Float64, 4}, kinds::Set{Symbol})
    selected = findall(a -> lib.kind[a] in kinds, eachindex(lib.names))
    isempty(selected) && return nothing
    return true_mobility_agreement_metrics(reference_prof[:, selected, :, :], estimate_prof[:, selected, :, :])
end

function dominant_profile_channels(reference_prof::Array{Float64, 4})
    _, nobs, ncomp, nr = size(reference_prof)
    channels = Vector{Tuple{Int, Int}}(undef, nobs)
    for a in 1:nobs
        best = (1, 1)
        best_rms = -Inf
        for comp in 1:ncomp, r in 1:nr
            vals = reference_prof[:, a, comp, r]
            finite_vals = vals[isfinite.(vals)]
            isempty(finite_vals) && continue
            rms = sqrt(mean(finite_vals .^ 2))
            if rms > best_rms
                best_rms = rms
                best = (comp, r)
            end
        end
        channels[a] = best
    end
    return channels
end

function render_true_mobility_cdot_figure(path::AbstractString, taus::Vector{Float64},
        lib::ObservableLibrary, data_prof::Array{Float64, 4}, generator_prof::Array{Float64, 4},
        learned_prof::Array{Float64, 4}, exactstat_prof::Array{Float64, 4},
        row_prof::Array{Float64, 4}, metrics, score_diag;
        reverse_prof=nothing, reverse_exactstat_prof=nothing,
        phi_reverse_prof=nothing, phi_reverse_exactstat_prof=nothing,
        reverse_score_diag=nothing)
    ensure_parent_dir(path)
    nobs = length(lib.names)
    cols = 3
    summary_panels = 4
    rows = ceil(Int, (nobs + summary_panels) / cols)
    fig = Figure(size=(4400, max(2200, 690 * rows + 320)), fontsize=21)
    data_metrics = metrics[:data_vs_generator]
    has_reverse = reverse_prof !== nothing && haskey(metrics, :reverse_vs_generator)
    has_reverse_exact = reverse_exactstat_prof !== nothing && haskey(metrics, :reverse_exactstat_vs_generator)
    has_phi_reverse = phi_reverse_prof !== nothing && haskey(metrics, :phi_reverse_vs_generator)
    has_phi_reverse_exact = phi_reverse_exactstat_prof !== nothing &&
        haskey(metrics, :phi_reverse_exactstat_vs_generator)
    reverse_metrics = has_reverse ? metrics[:reverse_vs_generator] : nothing
    reverse_exact_metrics = has_reverse_exact ? metrics[:reverse_exactstat_vs_generator] : nothing
    phi_reverse_metrics = has_phi_reverse ? metrics[:phi_reverse_vs_generator] : nothing
    phi_reverse_exact_metrics = has_phi_reverse_exact ? metrics[:phi_reverse_exactstat_vs_generator] : nothing
    metric_text = @sprintf("data/gen: rel=%.3e corr=%.4f",
        data_metrics[:relative_rmse], data_metrics[:correlation])
    if has_reverse
        metric_text *= @sprintf("    reverse true-M DSM/gen: rel=%.3e corr=%.4f",
            reverse_metrics[:relative_rmse], reverse_metrics[:correlation])
    end
    if has_reverse_exact
        metric_text *= @sprintf("\nreverse true-M exact/gen: rel=%.3e corr=%.4f",
            reverse_exact_metrics[:relative_rmse], reverse_exact_metrics[:correlation])
    end
    if has_phi_reverse
        metric_text *= @sprintf("    reverse Phi(DSM) /gen: rel=%.3e corr=%.4f",
            phi_reverse_metrics[:relative_rmse], phi_reverse_metrics[:correlation])
    end
    if has_phi_reverse_exact
        metric_text *= @sprintf("\nreverse Phi(exact-score) /gen: rel=%.3e corr=%.4f",
            phi_reverse_exact_metrics[:relative_rmse], phi_reverse_exact_metrics[:correlation])
    end
    Label(fig[0, :], "Complex-amplitude chain: generator-validated Cdot diagnostics\n" * metric_text,
        fontsize=28, font=:bold, tellwidth=false)
    channels = dominant_profile_channels(generator_prof)
    for a in 1:nobs
        row = div(a - 1, cols) + 1
        col = mod(a - 1, cols) + 1
        comp, rpos = channels[a]
        comp_name = comp == 1 ? "q" : "p"
        reverse_rel = has_reverse ? reverse_metrics[:per_observable_rel][a] : NaN
        phi_rel = has_phi_reverse ? phi_reverse_metrics[:per_observable_rel][a] : NaN
        ax = Axis(fig[row, col],
            title=@sprintf("%s  %s r=%d  true-M %.2e Phi %.2e",
                lib.names[a], comp_name, rpos - 1, reverse_rel, phi_rel),
            xlabel="tau", ylabel="profile Cdot")
        lines!(ax, taus, data_prof[:, a, comp, rpos]; color=:gray35, linestyle=:dot, linewidth=2.5,
            label="finite diff data")
        lines!(ax, taus, generator_prof[:, a, comp, rpos]; color=:black, linewidth=2.8,
            label="exact generator")
        if has_reverse
            lines!(ax, taus, reverse_prof[:, a, comp, rpos]; color=:seagreen4, linestyle=:dash, linewidth=2.6,
                label="true M + reverse DSM")
        end
        if has_reverse_exact
            lines!(ax, taus, reverse_exactstat_prof[:, a, comp, rpos]; color=:purple4, linestyle=:dashdot, linewidth=2.4,
                label="true M + reverse exact")
        end
        if has_phi_reverse
            lines!(ax, taus, phi_reverse_prof[:, a, comp, rpos]; color=:darkorange3, linestyle=:dash, linewidth=2.6,
                label="Phi DSM + reverse DSM")
        end
        if has_phi_reverse_exact
            lines!(ax, taus, phi_reverse_exactstat_prof[:, a, comp, rpos]; color=:deepskyblue4,
                linestyle=:dashdot, linewidth=2.4, label="Phi exact + reverse exact")
        end
        if a == 1
            axislegend(ax, position=:rt, framevisible=true, labelsize=13)
        end
    end

    panel = nobs + 1
    scatter_row = div(panel - 1, cols) + 1
    scatter_col = mod(panel - 1, cols) + 1
    ax = Axis(fig[scatter_row, scatter_col], title="All profile channels vs generator",
        xlabel="exact generator Cdot", ylabel="estimate Cdot")
    mask_data = isfinite.(generator_prof) .& isfinite.(data_prof)
    mask_reverse = has_reverse ? (isfinite.(generator_prof) .& isfinite.(reverse_prof)) : falses(size(generator_prof))
    mask_reverse_exact = has_reverse_exact ? (isfinite.(generator_prof) .& isfinite.(reverse_exactstat_prof)) :
        falses(size(generator_prof))
    mask_phi_reverse = has_phi_reverse ? (isfinite.(generator_prof) .& isfinite.(phi_reverse_prof)) :
        falses(size(generator_prof))
    mask_phi_reverse_exact = has_phi_reverse_exact ?
        (isfinite.(generator_prof) .& isfinite.(phi_reverse_exactstat_prof)) : falses(size(generator_prof))
    scatter!(ax, vec(generator_prof[mask_data]), vec(data_prof[mask_data]); color=(:gray35, 0.18), markersize=4, label="data")
    if has_reverse
        scatter!(ax, vec(generator_prof[mask_reverse]), vec(reverse_prof[mask_reverse]);
            color=(:seagreen4, 0.22), markersize=4, label="true M DSM")
    end
    if has_reverse_exact
        scatter!(ax, vec(generator_prof[mask_reverse_exact]), vec(reverse_exactstat_prof[mask_reverse_exact]);
            color=(:purple4, 0.20), markersize=4, label="true M exact")
    end
    if has_phi_reverse
        scatter!(ax, vec(generator_prof[mask_phi_reverse]), vec(phi_reverse_prof[mask_phi_reverse]);
            color=(:darkorange3, 0.20), markersize=4, label="Phi DSM")
    end
    if has_phi_reverse_exact
        scatter!(ax, vec(generator_prof[mask_phi_reverse_exact]), vec(phi_reverse_exactstat_prof[mask_phi_reverse_exact]);
            color=(:deepskyblue4, 0.20), markersize=4, label="Phi exact")
    end
    vals = vcat(vec(generator_prof[mask_data]), vec(data_prof[mask_data]))
    has_reverse && (vals = vcat(vals, vec(reverse_prof[mask_reverse])))
    has_reverse_exact && (vals = vcat(vals, vec(reverse_exactstat_prof[mask_reverse_exact])))
    has_phi_reverse && (vals = vcat(vals, vec(phi_reverse_prof[mask_phi_reverse])))
    has_phi_reverse_exact && (vals = vcat(vals, vec(phi_reverse_exactstat_prof[mask_phi_reverse_exact])))
    lo, hi = extrema(vals)
    lines!(ax, [lo, hi], [lo, hi]; color=:red, linewidth=2)
    axislegend(ax, position=:lt, framevisible=true, labelsize=13)

    panel += 1
    ax_err = Axis(fig[div(panel - 1, cols) + 1, mod(panel - 1, cols) + 1],
        title="Lagwise relative error vs generator", xlabel="tau", ylabel="relative RMSE")
    lines!(ax_err, taus, data_metrics[:per_lag_rel]; color=:gray35, linewidth=2.4, label="data")
    if has_reverse
        lines!(ax_err, taus, metrics[:reverse_vs_generator][:per_lag_rel]; color=:seagreen4,
            linewidth=2.4, label="true M DSM")
    end
    if has_reverse_exact
        lines!(ax_err, taus, metrics[:reverse_exactstat_vs_generator][:per_lag_rel]; color=:purple4,
            linewidth=2.4, label="true M exact")
    end
    if has_phi_reverse
        lines!(ax_err, taus, metrics[:phi_reverse_vs_generator][:per_lag_rel]; color=:darkorange3,
            linewidth=2.4, label="Phi DSM")
    end
    if has_phi_reverse_exact
        lines!(ax_err, taus, metrics[:phi_reverse_exactstat_vs_generator][:per_lag_rel]; color=:deepskyblue4,
            linewidth=2.4, label="Phi exact")
    end
    axislegend(ax_err, position=:lt, framevisible=true, labelsize=13)

    panel += 1
    ax_score = Axis(fig[div(panel - 1, cols) + 1, mod(panel - 1, cols) + 1],
        title="Conditional-score self-consistency", xlabel="tau", ylabel="matrix norm / ||I||")
    if reverse_score_diag !== nothing
        lines!(ax_score, reverse_score_diag[:taus], reverse_score_diag[:cond_x0_norm_reverse];
            color=:seagreen4, linewidth=2.4, label="reverse DSM E[s_cond x0']")
        if haskey(reverse_score_diag, :cond_x0_norm_reverse_exactstat)
            lines!(ax_score, reverse_score_diag[:taus], reverse_score_diag[:cond_x0_norm_reverse_exactstat];
                color=:purple4, linewidth=2.4, linestyle=:dashdot, label="reverse exact E[s_cond x0']")
        end
        if haskey(reverse_score_diag, :posterior_x0_stein_rel)
            lines!(ax_score, reverse_score_diag[:taus], reverse_score_diag[:posterior_x0_stein_rel];
                color=:black, linewidth=2.4, linestyle=:dash, label="posterior Stein")
        end
    end
    axislegend(ax_score, position=:rt, framevisible=true, labelsize=13)

    panel += 1
    ax_mean = Axis(fig[div(panel - 1, cols) + 1, mod(panel - 1, cols) + 1],
        title="Conditional-score mean", xlabel="tau", ylabel="||E[s_cond]||/sqrt(D)")
    if reverse_score_diag !== nothing
        lines!(ax_mean, reverse_score_diag[:taus], reverse_score_diag[:cond_mean_norm_reverse];
            color=:seagreen4, linewidth=2.4, label="reverse DSM")
        if haskey(reverse_score_diag, :cond_mean_norm_reverse_exactstat)
            lines!(ax_mean, reverse_score_diag[:taus], reverse_score_diag[:cond_mean_norm_reverse_exactstat];
                color=:purple4, linewidth=2.4, linestyle=:dashdot, label="reverse exact")
        end
    end
    axislegend(ax_mean, position=:rt, framevisible=true, labelsize=13)

    save(path, fig)
    return nothing
end

function profile_agreement_metrics(reference::Array{Float64, 4}, estimate::Array{Float64, 4})
    mask = isfinite.(reference) .& isfinite.(estimate)
    ref = vec(reference[mask])
    est = vec(estimate[mask])
    rel = sqrt(mean((est .- ref) .^ 2)) / max(sqrt(mean(ref .^ 2)), eps(Float64))
    corr = length(ref) > 1 ? cor(ref, est) : NaN
    return rel, corr
end

function render_three_method_profiles(path::AbstractString, title::AbstractString, ylabel::AbstractString,
        taus::Vector{Float64}, lib::ObservableLibrary, data_prof::Array{Float64, 4},
        true_prof::Array{Float64, 4}, nn_prof::Array{Float64, 4};
        true_label::AbstractString="true M + cond score",
        nn_label::AbstractString="NN M + cond score",
        summary_true_label::AbstractString="true-M",
        summary_nn_label::AbstractString="NN-M")
    ensure_parent_dir(path)
    nobs = length(lib.names)
    rows, cols = panel_grid_dims(nobs + 1; max_cols=4)
    width, height = publication_panel_figure_size(rows, cols;
        base_w=2800, base_h=1700, panel_w=920, panel_h=600, max_w=5600, max_h=4300)
    with_scaled_figure_style(width, height; scale_override=1.05) do _
        fig = Figure(size=(width, height))
        nn_rel, nn_corr = profile_agreement_metrics(data_prof, nn_prof)
        true_rel, true_corr = profile_agreement_metrics(data_prof, true_prof)
        subtitle = @sprintf("%s/data rel=%.3e corr=%.4f   |   %s/data rel=%.3e corr=%.4f",
            summary_true_label, true_rel, true_corr, summary_nn_label, nn_rel, nn_corr)
        figure_title!(fig, title; subtitle=subtitle)
        channels = dominant_profile_channels(data_prof)
        for a in 1:nobs
            row, col = centered_panel_rc(a, nobs + 1, cols)
            comp, rpos = channels[a]
            ax = Axis(fig[row, col];
                title=@sprintf("%02d  %s  %s r=%d", a, lib.names[a], comp == 1 ? "q" : "p", rpos - 1),
                xlabel="τ", ylabel=ylabel)
            hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            lines!(ax, taus, data_prof[:, a, comp, rpos]; color=STYLE_REFERENCE,
                linewidth=curve_linewidth(), label="data")
            lines!(ax, taus, true_prof[:, a, comp, rpos]; color=STYLE_SECONDARY,
                linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9), label=true_label)
            lines!(ax, taus, nn_prof[:, a, comp, rpos]; color=STYLE_PRIMARY,
                linestyle=:dashdot, linewidth=curve_linewidth(emphasis=0.9), label=nn_label)
            a == 1 && axislegend(ax; position=:rt)
        end

        panel = nobs + 1
        row, col = centered_panel_rc(panel, nobs + 1, cols)
        ax = Axis(fig[row, col]; title="All profile values", xlabel="data", ylabel="model")
        mask_true = isfinite.(data_prof) .& isfinite.(true_prof)
        mask_nn = isfinite.(data_prof) .& isfinite.(nn_prof)
        scatter!(ax, vec(data_prof[mask_true]), vec(true_prof[mask_true]);
            color=(STYLE_SECONDARY, 0.22), markersize=4, label=summary_true_label)
        scatter!(ax, vec(data_prof[mask_nn]), vec(nn_prof[mask_nn]);
            color=(STYLE_PRIMARY, 0.22), markersize=4, label=summary_nn_label)
        vals = vcat(vec(data_prof[mask_true]), vec(true_prof[mask_true]), vec(nn_prof[mask_nn]))
        lo, hi = extrema(vals)
        lim = max(abs(lo), abs(hi))
        lines!(ax, [-lim, lim], [-lim, lim]; color=STYLE_ZERO, linewidth=guide_linewidth())
        xlims!(ax, -lim, lim)
        ylims!(ax, -lim, lim)
        axislegend(ax; position=:lt)
        apply_publication_grid!(fig.layout, rows, cols; row_gap=28, col_gap=30)
        save_figure(path, fig)
    end
    return nothing
end

function per_observable_profile_rel(reference::Array{Float64, 4}, estimate::Array{Float64, 4})
    nobs = size(reference, 2)
    rel = fill(NaN, nobs)
    corrv = fill(NaN, nobs)
    for a in 1:nobs
        mask = isfinite.(@view reference[:, a, :, :]) .& isfinite.(@view estimate[:, a, :, :])
        rv = reference[:, a, :, :][mask]
        ev = estimate[:, a, :, :][mask]
        if !isempty(rv)
            rel[a] = sqrt(mean((ev .- rv) .^ 2)) / max(sqrt(mean(rv .^ 2)), eps(Float64))
            corrv[a] = length(rv) > 1 ? cor(rv, ev) : NaN
        end
    end
    return rel, corrv
end

history_epochs(history) = history.epochs
history_losses(history) = history.losses
history_validation_rmse(history) = history.validation_rmse
history_mean_abs_delta(history) = hasproperty(history, :mean_abs_delta) ?
    getproperty(history, :mean_abs_delta) : fill(NaN, length(history_epochs(history)))
history_rms_delta(history) = hasproperty(history, :rms_delta) ?
    getproperty(history, :rms_delta) : fill(NaN, length(history_epochs(history)))

function render_mobility_training_diagnostic(path::AbstractString, history,
        taus::Vector{Float64}, lib::ObservableLibrary, A_target::Array{Float64, 4},
        A_basis::Array{Float64, 4}, A_nn::Array{Float64, 4}, coeff::Vector{Float64},
        moments::Vector{Float64}; A_true::Union{Nothing, Array{Float64, 4}}=nothing)
    ensure_parent_dir(path)
    width, height = 3100, 920
    with_scaled_figure_style(width, height; scale_override=1.08) do _
        fig = Figure(size=(width, height); figure_padding=(34, 34, 72, 28))
        figure_title!(fig, "Complex-amplitude mobility NN training diagnostics";
            subtitle="Direct paper-loss training of the state-dependent correction δMθ")
        rowgap!(fig.layout, 26)
        colgap!(fig.layout, 32)

        epochs = history_epochs(history)
        losses = history_losses(history)
        val = history_validation_rmse(history)
        mean_abs = history_mean_abs_delta(history)
        rms_delta = history_rms_delta(history)

        ax_loss = Axis(fig[1, 1]; title="Training objective", xlabel="epoch", ylabel="loss")
        lines!(ax_loss, epochs, losses; color=STYLE_REFERENCE, linewidth=curve_linewidth())

        ax_rmse = Axis(fig[1, 2]; title="Validation A residual RMSE", xlabel="epoch", ylabel="relative RMSE")
        lines!(ax_rmse, epochs, val; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="normalized")
        axislegend(ax_rmse; position=:rt)

        ax_mean = Axis(fig[1, 3]; title="Mean and RMS δM", xlabel="epoch", ylabel="value")
        lines!(ax_mean, epochs, mean_abs; color=STYLE_PRIMARY, linewidth=curve_linewidth(),
            label="|<δM>|")
        lines!(ax_mean, epochs, rms_delta; color=STYLE_SECONDARY, linewidth=curve_linewidth(),
            label="RMS δM")
        axislegend(ax_mean; position=:rt)
        save_figure(path, fig)
    end
    return nothing
end

function output_variant(path::AbstractString, suffix::AbstractString)
    root, ext = splitext(path)
    return root * suffix * (isempty(ext) ? ".png" : ext)
end

function sympart(A::AbstractMatrix{<:Real})
    return 0.5 .* (Matrix{Float64}(A) .+ transpose(Matrix{Float64}(A)))
end

function psd_sqrt(A::AbstractMatrix{<:Real}; floor::Float64=1.0e-9)
    S = sympart(A)
    eig = eigen(Symmetric(S))
    vals = max.(eig.values, floor)
    return eig.vectors * Diagonal(sqrt.(vals)) * transpose(eig.vectors), minimum(eig.values)
end

function sample_initial_ensemble(sampler::PairSampler, ntraj::Int, rng::AbstractRNG)
    nt, K, _, ndata = size(sampler.states)
    z = Array{Float64}(undef, K, 2, ntraj)
    for b in 1:ntraj
        traj = rand(rng, 1:ndata)
        t = rand(rng, sampler.start_idx:nt)
        @inbounds for c in 1:2, i in 1:K
            z[i, c, b] = Float64(sampler.states[t, i, c, traj])
        end
    end
    return z
end

function data_support_bounds(sampler::PairSampler; pad_fraction::Float64=0.25)
    post = @view sampler.states[sampler.start_idx:end, :, :, :]
    bounds = Matrix{Float64}(undef, 2, 2)
    for c in 1:2
        vals = vec(Float64.(@view post[:, :, c, :]))
        lo = quantile(vals, 0.001)
        hi = quantile(vals, 0.999)
        pad = pad_fraction * max(hi - lo, eps(Float64))
        bounds[c, 1] = lo - pad
        bounds[c, 2] = hi + pad
    end
    return bounds
end

function clamp_state_to_support!(z::Array{Float64, 3}, bounds::Matrix{Float64})
    K, _, B = size(z)
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        z[i, c, b] = clamp(z[i, c, b], bounds[c, 1], bounds[c, 2])
    end
    return z
end

function evaluate_stationary_score_forward(models::LoadedModels, z::Array{Float64, 3},
        batch_size::Int, device::ExecutionDevice)
    z32 = Float32.(z)
    return evaluate_stationary_score(models, z32, batch_size, device)
end

function evaluate_exact_stationary_score_forward(z::Array{Float64, 3},
        potential_params::Tuple{Float64, Float64, Float64})
    z32 = Float32.(z)
    out = similar(z32)
    exact_stationary_score!(out, z32, potential_params...)
    return out
end

function flatten_state64!(dest::AbstractMatrix{Float64}, z::AbstractArray{Float64, 3})
    K, C, B = size(z)
    @inbounds for b in 1:B, c in 1:C, i in 1:K
        dest[flat_index(i, c, K), b] = z[i, c, b]
    end
    return dest
end

function unflatten_state64!(dest::AbstractArray{Float64, 3}, flat::AbstractMatrix{Float64})
    K, _, B = size(dest)
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        dest[i, c, b] = flat[flat_index(i, c, K), b]
    end
    return dest
end

function phi_forward_drift!(drift_flat::AbstractMatrix{Float64}, Phi::Matrix{Float64},
        score::AbstractArray{Float32, 3})
    K, _, B = size(score)
    D = 2K
    score_flat = Matrix{Float64}(undef, D, B)
    @inbounds for b in 1:B, c in 1:2, i in 1:K
        score_flat[flat_index(i, c, K), b] = Float64(score[i, c, b])
    end
    mul!(drift_flat, Phi, score_flat)
    return drift_flat
end

function nn_forward_drift!(drift_flat::AbstractMatrix{Float64}, Phi::Matrix{Float64}, mobility_model,
        z::Array{Float64, 3}, score::AbstractArray{Float32, 3}, mean_r2::Float64)
    phi_forward_drift!(drift_flat, Phi, score)
    nn_local_row_action_and_divergence!(drift_flat, mobility_model, z, score, mean_r2)
    return drift_flat
end

function local_diffusion_sqrt_for_sample(Phi_sym::Matrix{Float64}, mobility_model, z::Array{Float64, 3},
        mean_r2::Float64, b::Int)
    K = size(z, 1)
    Dmat = copy(Phi_sym)
    features = Matrix{Float32}(undef, 2, K)
    @inbounds for i in 1:K
        features[1, i] = Float32(z[i, 1, b])
        features[2, i] = Float32(z[i, 2, b])
    end
    pred = mobility_model(features)
    @inbounds for i in 1:K
        m11 = Float64(pred[1, i])
        m12 = Float64(pred[2, i])
        m21 = Float64(pred[3, i])
        m22 = Float64(pred[4, i])
        qidx = flat_index(i, 1, K)
        pidx = flat_index(i, 2, K)
        Dmat[qidx, qidx] += m11
        Dmat[pidx, pidx] += m22
        sym12 = 0.5 * (m12 + m21)
        Dmat[qidx, pidx] += sym12
        Dmat[pidx, qidx] += sym12
    end
    return psd_sqrt(Dmat; floor=1.0e-8)
end

function integrate_forward_langevin(models::LoadedModels, Phi::Matrix{Float64}, mobility_model,
        sampler::PairSampler, lib::ObservableLibrary, params::FitDMParams, device::ExecutionDevice;
        mode::Symbol, dt::Float64=0.002, total_time::Float64=36.0,
        burnin_time::Float64=6.0, save_dt::Float64=sampler.save_dt,
        ntraj::Int=96, seed::Int=params.seed + 500_000,
        score_clip::Float32=50.0f0, hard_clamp_state::Bool=true,
        score_source::Symbol=:learned,
        potential_params::Union{Nothing, Tuple{Float64, Float64, Float64}}=nothing)
    save_every = max(1, round(Int, save_dt / dt))
    actual_save_dt = save_every * dt
    nsteps = ceil(Int, total_time / dt)
    burnin_steps = floor(Int, burnin_time / dt)
    nsaved = max(1, fld(max(nsteps - burnin_steps, 0), save_every) + 1)
    K = sampler.K
    D = sampler.D
    rng = MersenneTwister(seed)
    z = sample_initial_ensemble(sampler, ntraj, rng)
    support_bounds = data_support_bounds(sampler)
    hard_clamp_state && clamp_state_to_support!(z, support_bounds)
    saved = Array{Float32}(undef, nsaved, K, 2, ntraj)
    saved_times = Vector{Float64}(undef, nsaved)
    Phi_sym = sympart(Phi)
    sqrt_phi, min_phi = psd_sqrt(Phi_sym; floor=1.0e-8)
    min_diffusion = min_phi
    drift = Matrix{Float64}(undef, D, ntraj)
    flat = Matrix{Float64}(undef, D, ntraj)
    noise = Matrix{Float64}(undef, D, ntraj)
    save_idx = 0
    progress_stride = max(1, nsteps ÷ 10)
    for step in 0:nsteps
        if step >= burnin_steps && (step - burnin_steps) % save_every == 0
            save_idx += 1
            saved_times[save_idx] = (step - burnin_steps) * dt
            @inbounds for b in 1:ntraj, c in 1:2, i in 1:K
                saved[save_idx, i, c, b] = Float32(z[i, c, b])
            end
            save_idx >= nsaved && step == nsteps && break
        end
        step == nsteps && break
        score = if score_source == :learned
            evaluate_stationary_score_forward(models, z, params.score_batch_size, device)
        elseif score_source == :exact
            potential_params === nothing && error("potential_params are required for score_source=:exact.")
            evaluate_exact_stationary_score_forward(z, potential_params)
        else
            error("Unknown forward score source $(score_source).")
        end
        clamp!(score, -score_clip, score_clip)
        if mode == :phi
            phi_forward_drift!(drift, Phi, score)
        elseif mode == :nn
            nn_forward_drift!(drift, Phi, mobility_model, z, score, lib.mean_r2)
        else
            error("Unknown forward mode $(mode).")
        end
        randn!(rng, noise)
        if mode == :phi
            noise .= sqrt_phi * noise
        else
            for b in 1:ntraj
                Lb, mineig = local_diffusion_sqrt_for_sample(Phi_sym, mobility_model, z, lib.mean_r2, b)
                min_diffusion = min(min_diffusion, mineig)
                noise[:, b] .= Lb * noise[:, b]
            end
        end
        flatten_state64!(flat, z)
        @. flat = flat + dt * drift + sqrt(2.0 * dt) * noise
        unflatten_state64!(z, flat)
        hard_clamp_state && clamp_state_to_support!(z, support_bounds)
        if params.verbose && (step == 1 || step % progress_stride == 0)
            @printf("Forward %s Langevin step %d / %d, finite=%s\n",
                String(mode), step, nsteps, string(all(isfinite, z)))
        end
        all(isfinite, z) || error("Forward $(mode) Langevin integration produced non-finite states at step $(step).")
    end
    if save_idx < nsaved
        saved = saved[1:save_idx, :, :, :]
        saved_times = saved_times[1:save_idx]
    end
    @printf("Forward %s Langevin saved %d snapshots x %d trajectories, save_dt %.6f, min diffusion eig %.3e\n",
        String(mode), length(saved_times), ntraj, actual_save_dt, min_diffusion)
    return saved_times, saved, min_diffusion
end

function draw_channel_values_from_states(states::Array{Float32, 4}, channel::Int, max_samples::Int,
        rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = nt * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        idx = rand(rng, 0:(total - 1))
        i = mod(idx, K) + 1
        tmp = fld(idx, K)
        t = mod(tmp, nt) + 1
        traj = fld(tmp, nt) + 1
        vals[s] = Float64(states[t, i, channel, traj])
    end
    return vals
end

function draw_amplitude_values_from_states(states::Array{Float32, 4}, max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = nt * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        idx = rand(rng, 0:(total - 1))
        i = mod(idx, K) + 1
        tmp = fld(idx, K)
        t = mod(tmp, nt) + 1
        traj = fld(tmp, nt) + 1
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        vals[s] = sqrt(q*q + p*p)
    end
    return vals
end

function draw_radius_squared_values_from_states(states::Array{Float32, 4}, max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = nt * K * ntraj
    nsamp = min(max_samples, total)
    vals = Vector{Float64}(undef, nsamp)
    @inbounds for s in 1:nsamp
        idx = rand(rng, 0:(total - 1))
        i = mod(idx, K) + 1
        tmp = fld(idx, K)
        t = mod(tmp, nt) + 1
        traj = fld(tmp, nt) + 1
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        vals[s] = q*q + p*p
    end
    return vals
end

function histogram_density(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Float64, length(edges) - 1)
    lo = first(edges)
    hi = last(edges)
    nb = length(counts)
    width = (hi - lo) / nb
    @inbounds for v in values
        if lo <= v <= hi
            idx = v == hi ? nb : floor(Int, (v - lo) / width) + 1
            counts[idx] += 1.0
        end
    end
    dens = counts ./ max(sum(counts) * width, eps(Float64))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, dens
end

function channel_acf(states::Array{Float32, 4}, channel::Int, max_lag::Int, stride::Int)
    nt, K, _, ntraj = size(states)
    lags = collect(0:stride:max_lag)
    vals = Float64.(states[:, :, channel, :])
    μ = mean(vals)
    varv = mean((vals .- μ).^2)
    acf = zeros(Float64, length(lags))
    for (idx, lag) in enumerate(lags)
        total = 0.0
        count = 0
        @inbounds for traj in 1:ntraj, t in 1:(nt - lag), i in 1:K
            total += (Float64(states[t + lag, i, channel, traj]) - μ) *
                (Float64(states[t, i, channel, traj]) - μ)
            count += 1
        end
        acf[idx] = total / max(count * varv, eps(Float64))
    end
    return lags, acf
end

function amplitude_acf(states::Array{Float32, 4}, max_lag::Int, stride::Int)
    nt, K, _, ntraj = size(states)
    lags = collect(0:stride:max_lag)
    vals = Array{Float64}(undef, nt, K, ntraj)
    @inbounds for traj in 1:ntraj, t in 1:nt, i in 1:K
        q = Float64(states[t, i, 1, traj])
        p = Float64(states[t, i, 2, traj])
        vals[t, i, traj] = sqrt(q*q + p*p)
    end
    μ = mean(vals)
    varv = mean((vals .- μ).^2)
    acf = zeros(Float64, length(lags))
    for (idx, lag) in enumerate(lags)
        total = 0.0
        count = 0
        @inbounds for traj in 1:ntraj, t in 1:(nt - lag), i in 1:K
            total += (vals[t + lag, i, traj] - μ) * (vals[t, i, traj] - μ)
            count += 1
        end
        acf[idx] = total / max(count * varv, eps(Float64))
    end
    return lags, acf
end

function channel_cross_same_site(states::Array{Float32, 4}, max_lag::Int, stride::Int)
    nt, K, _, ntraj = size(states)
    lags = collect(0:stride:max_lag)
    qvals = Float64.(states[:, :, 1, :])
    pvals = Float64.(states[:, :, 2, :])
    μq = mean(qvals)
    μp = mean(pvals)
    scale = sqrt(mean((qvals .- μq).^2) * mean((pvals .- μp).^2))
    cross = zeros(Float64, length(lags))
    for (idx, lag) in enumerate(lags)
        total = 0.0
        count = 0
        @inbounds for traj in 1:ntraj, t in 1:(nt - lag), i in 1:K
            total += (Float64(states[t + lag, i, 1, traj]) - μq) *
                (Float64(states[t, i, 2, traj]) - μp)
            count += 1
        end
        cross[idx] = total / max(count * scale, eps(Float64))
    end
    return lags, cross
end

function state_summary_vector(states::Array{Float32, 4})
    q = Float64.(states[:, :, 1, :])
    p = Float64.(states[:, :, 2, :])
    r = sqrt.(q .^ 2 .+ p .^ 2)
    return [
        mean(q), mean(p), mean(r),
        std(vec(q)), std(vec(p)), std(vec(r)),
        mean(r .^ 2),
    ]
end

function spatial_power_spectrum(states::Array{Float32, 4}, channel::Int)
    nt, K, _, ntraj = size(states)
    modes = collect(0:(K ÷ 2))
    μ = mean(Float64.(states[:, :, channel, :]))
    power = zeros(Float64, length(modes))
    @inbounds for traj in 1:ntraj, t in 1:nt
        for (midx, k) in enumerate(modes)
            re = 0.0
            im = 0.0
            for i in 1:K
                angle = -2π * k * (i - 1) / K
                val = Float64(states[t, i, channel, traj]) - μ
                re += val * cos(angle)
                im += val * sin(angle)
            end
            power[midx] += (re*re + im*im) / K
        end
    end
    power ./= nt * ntraj
    return modes, power
end

function observed_validation_window(sampler::PairSampler, max_snapshots::Int)
    nt = size(sampler.states, 1)
    stop = min(nt, sampler.start_idx + max_snapshots - 1)
    return sampler.states[sampler.start_idx:stop, :, :, :]
end

function render_forward_validation_stats(path::AbstractString, sampler::PairSampler,
        obs_states::Array{Float32, 4}, phi_states::Array{Float32, 4}, nn_states::Array{Float32, 4},
        save_dt::Float64; pdf_bins::Int=120, max_pdf_samples::Int=250_000)
    ensure_parent_dir(path)
    rng = MersenneTwister(912_345)
    q_obs = draw_channel_values_from_states(obs_states, 1, max_pdf_samples, rng)
    p_obs = draw_channel_values_from_states(obs_states, 2, max_pdf_samples, rng)
    a_obs = draw_amplitude_values_from_states(obs_states, max_pdf_samples, rng)
    r2_obs = draw_radius_squared_values_from_states(obs_states, max_pdf_samples, rng)
    ranges = (
        range(quantile(q_obs, 0.001), quantile(q_obs, 0.999); length=pdf_bins + 1),
        range(quantile(p_obs, 0.001), quantile(p_obs, 0.999); length=pdf_bins + 1),
        range(max(0.0, quantile(a_obs, 0.001)), quantile(a_obs, 0.999); length=pdf_bins + 1),
        range(max(0.0, quantile(r2_obs, 0.001)), quantile(r2_obs, 0.999); length=pdf_bins + 1),
    )
    q_cent, q_obs_d = histogram_density(q_obs, collect(ranges[1]))
    p_cent, p_obs_d = histogram_density(p_obs, collect(ranges[2]))
    a_cent, a_obs_d = histogram_density(a_obs, collect(ranges[3]))
    r2_cent, r2_obs_d = histogram_density(r2_obs, collect(ranges[4]))
    _, q_phi_d = histogram_density(draw_channel_values_from_states(phi_states, 1, max_pdf_samples, rng), collect(ranges[1]))
    _, p_phi_d = histogram_density(draw_channel_values_from_states(phi_states, 2, max_pdf_samples, rng), collect(ranges[2]))
    _, a_phi_d = histogram_density(draw_amplitude_values_from_states(phi_states, max_pdf_samples, rng), collect(ranges[3]))
    _, r2_phi_d = histogram_density(draw_radius_squared_values_from_states(phi_states, max_pdf_samples, rng), collect(ranges[4]))
    _, q_nn_d = histogram_density(draw_channel_values_from_states(nn_states, 1, max_pdf_samples, rng), collect(ranges[1]))
    _, p_nn_d = histogram_density(draw_channel_values_from_states(nn_states, 2, max_pdf_samples, rng), collect(ranges[2]))
    _, a_nn_d = histogram_density(draw_amplitude_values_from_states(nn_states, max_pdf_samples, rng), collect(ranges[3]))
    _, r2_nn_d = histogram_density(draw_radius_squared_values_from_states(nn_states, max_pdf_samples, rng), collect(ranges[4]))

    max_lag = min(size(obs_states, 1), size(phi_states, 1), size(nn_states, 1)) - 1
    max_lag = min(max_lag, round(Int, 9.0 / save_dt))
    stride = max(1, round(Int, 0.06 / save_dt))
    lag_steps, acf_q_obs = channel_acf(obs_states, 1, max_lag, stride)
    _, acf_q_phi = channel_acf(phi_states, 1, max_lag, stride)
    _, acf_q_nn = channel_acf(nn_states, 1, max_lag, stride)
    _, acf_p_obs = channel_acf(obs_states, 2, max_lag, stride)
    _, acf_p_phi = channel_acf(phi_states, 2, max_lag, stride)
    _, acf_p_nn = channel_acf(nn_states, 2, max_lag, stride)
    _, acf_r_obs = amplitude_acf(obs_states, max_lag, stride)
    _, acf_r_phi = amplitude_acf(phi_states, max_lag, stride)
    _, acf_r_nn = amplitude_acf(nn_states, max_lag, stride)
    _, cross_obs = channel_cross_same_site(obs_states, max_lag, stride)
    _, cross_phi = channel_cross_same_site(phi_states, max_lag, stride)
    _, cross_nn = channel_cross_same_site(nn_states, max_lag, stride)
    lag_times = lag_steps .* save_dt

    obs_summary = state_summary_vector(obs_states)
    phi_summary = state_summary_vector(phi_states)
    nn_summary = state_summary_vector(nn_states)
    summary_labels = ["mean q", "mean p", "mean r", "std q", "std p", "std r", "mean r²"]
    modes, q_obs_spec = spatial_power_spectrum(obs_states, 1)
    _, q_phi_spec = spatial_power_spectrum(phi_states, 1)
    _, q_nn_spec = spatial_power_spectrum(nn_states, 1)
    _, p_obs_spec = spatial_power_spectrum(obs_states, 2)
    _, p_phi_spec = spatial_power_spectrum(phi_states, 2)
    _, p_nn_spec = spatial_power_spectrum(nn_states, 2)

    width, height = 4300, 3000
    with_scaled_figure_style(width, height; scale_override=1.02) do _
        fig = Figure(size=(width, height))
        figure_title!(fig, "Complex-amplitude forward Langevin validation";
            subtitle="Observed trajectories vs constant mobility M=Φ and learned state-dependent mobility Mθ")
        specs = [
            (q_cent, q_obs_d, q_phi_d, q_nn_d, "q PDF", "q", "density", fig[1, 1]),
            (p_cent, p_obs_d, p_phi_d, p_nn_d, "p PDF", "p", "density", fig[1, 2]),
            (a_cent, a_obs_d, a_phi_d, a_nn_d, "amplitude PDF", "r", "density", fig[1, 3]),
            (r2_cent, r2_obs_d, r2_phi_d, r2_nn_d, "amplitude-squared PDF", "r²", "density", fig[1, 4]),
            (lag_times, acf_q_obs, acf_q_phi, acf_q_nn, "q autocorrelation", "τ", "C/C(0)", fig[2, 1]),
            (lag_times, acf_p_obs, acf_p_phi, acf_p_nn, "p autocorrelation", "τ", "C/C(0)", fig[2, 2]),
            (lag_times, acf_r_obs, acf_r_phi, acf_r_nn, "r autocorrelation", "τ", "C/C(0)", fig[2, 3]),
            (lag_times, cross_obs, cross_phi, cross_nn, "q(t) p(0) cross-correlation", "τ", "corr", fig[2, 4]),
            (modes, q_obs_spec, q_phi_spec, q_nn_spec, "q spatial spectrum", "mode", "power", fig[3, 3]),
            (modes, p_obs_spec, p_phi_spec, p_nn_spec, "p spatial spectrum", "mode", "power", fig[3, 4]),
        ]
        for (x, obs, phi, nn, title, xlabel, ylabel, slot) in specs
            ax = Axis(slot; title=title, xlabel=xlabel, ylabel=ylabel)
            lines!(ax, x, obs; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
            lines!(ax, x, phi; color=STYLE_SECONDARY, linestyle=:dash,
                linewidth=curve_linewidth(emphasis=0.9), label="M=Φ")
            lines!(ax, x, nn; color=STYLE_PRIMARY, linestyle=:dashdot,
                linewidth=curve_linewidth(emphasis=0.9), label="NN M")
            title == "q PDF" && axislegend(ax; position=:rt)
        end

        ax_mean = Axis(fig[3, 1]; title="Mean statistics", xlabel="statistic", ylabel="value",
            xticks=(1:3, summary_labels[1:3]))
        barplot!(ax_mean, (1:3) .- 0.24, obs_summary[1:3]; width=0.22, color=STYLE_REFERENCE, label="data")
        barplot!(ax_mean, 1:3, phi_summary[1:3]; width=0.22, color=STYLE_SECONDARY, label="M=Φ")
        barplot!(ax_mean, (1:3) .+ 0.24, nn_summary[1:3]; width=0.22, color=STYLE_PRIMARY, label="NN M")
        ax_mean.xticklabelrotation = pi / 8
        axislegend(ax_mean; position=:lt)

        ax_scale = Axis(fig[3, 2]; title="Scale statistics", xlabel="statistic", ylabel="value",
            xticks=(1:4, summary_labels[4:7]))
        barplot!(ax_scale, (1:4) .- 0.24, obs_summary[4:7]; width=0.22, color=STYLE_REFERENCE, label="data")
        barplot!(ax_scale, 1:4, phi_summary[4:7]; width=0.22, color=STYLE_SECONDARY, label="M=Φ")
        barplot!(ax_scale, (1:4) .+ 0.24, nn_summary[4:7]; width=0.22, color=STYLE_PRIMARY, label="NN M")
        ax_scale.xticklabelrotation = pi / 8

        apply_publication_grid!(fig.layout, 3, 4; row_gap=30, col_gap=30)
        save_figure(path, fig)
    end
    return Dict(
        :lag_times => lag_times,
        :acf_q_obs => acf_q_obs, :acf_q_phi => acf_q_phi, :acf_q_nn => acf_q_nn,
        :acf_p_obs => acf_p_obs, :acf_p_phi => acf_p_phi, :acf_p_nn => acf_p_nn,
        :acf_r_obs => acf_r_obs, :acf_r_phi => acf_r_phi, :acf_r_nn => acf_r_nn,
        :cross_obs => cross_obs, :cross_phi => cross_phi, :cross_nn => cross_nn,
        :summary_labels => summary_labels, :obs_summary => obs_summary,
        :phi_summary => phi_summary, :nn_summary => nn_summary,
        :modes => modes,
        :q_obs_spec => q_obs_spec, :q_phi_spec => q_phi_spec, :q_nn_spec => q_nn_spec,
        :p_obs_spec => p_obs_spec, :p_phi_spec => p_phi_spec, :p_nn_spec => p_nn_spec,
    )
end

function estimate_c_profiles_from_states(states::Array{Float32, 4}, save_dt::Float64,
        lag_times::Vector{Float64}, lib::ObservableLibrary, coord_mean::Vector{Float64},
        pairs_per_lag::Int, batch_size::Int, seed::Int)
    nt, K, _, ntraj = size(states)
    D = 2K
    nobs = length(lib.names)
    lag_steps = [clamp(round(Int, tau / save_dt), 0, nt - 1) for tau in lag_times]
    C = Array{Float64}(undef, length(lag_steps), K, nobs, D)
    z0 = Array{Float32}(undef, K, 2, batch_size)
    zt = similar(z0)
    x0_flat = Matrix{Float32}(undef, D, batch_size)
    obs_vals = Array{Float32}(undef, K, nobs, batch_size)
    rng = MersenneTwister(seed)
    for (lag_idx, lag) in enumerate(lag_steps)
        sums = zeros(Float64, K, nobs, D)
        total = 0
        remaining = min(pairs_per_lag, max(1, (nt - lag) * ntraj))
        while remaining > 0
            bn = min(remaining, batch_size)
            upper = nt - lag
            @inbounds for b in 1:bn
                traj = rand(rng, 1:ntraj)
                t = rand(rng, 1:upper)
                for c in 1:2, i in 1:K
                    z0[i, c, b] = states[t, i, c, traj]
                    zt[i, c, b] = states[t + lag, i, c, traj]
                end
            end
            flatten_state!(@view(x0_flat[:, 1:bn]), @view(z0[:, :, 1:bn]))
            @inbounds for b in 1:bn, i in 1:K
                x0_flat[flat_index(i, 1, K), b] -= Float32(coord_mean[1])
                x0_flat[flat_index(i, 2, K), b] -= Float32(coord_mean[2])
            end
            for b in 1:bn
                observable_values!(@view(obs_vals[:, :, b]), @view(zt[:, :, b]), lib)
            end
            accumulate_observable_coordinate_correlation!(sums, @view(obs_vals[:, :, 1:bn]), @view(x0_flat[:, 1:bn]))
            total += bn
            remaining -= bn
        end
        C[lag_idx, :, :, :] .= sums ./ total
    end
    return translation_profiles(C, K)
end

function save_forward_trajectories(path::AbstractString, times::Vector{Float64},
        phi_states::Array{Float32, 4}, nn_states::Array{Float32, 4}, min_phi::Float64, min_nn::Float64)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time", times)
        write(h5, "/phi_states", phi_states)
        write(h5, "/nn_states", nn_states)
        write(h5, "/min_phi_diffusion_eig", min_phi)
        write(h5, "/min_nn_diffusion_eig", min_nn)
    end
    return nothing
end

function write_cdot_diagnostic_metrics(path::AbstractString, sampler::PairSampler,
        Phi::Matrix{Float64}, true_diag::Tuple{Float64, Float64, Float64, Float64},
        diagnostic_metrics, score_diag,
        lib::ObservableLibrary, observable_selection; reverse_score_diag=nothing,
        Phi_raw=nothing, Phi_full=nothing, stein_projected=nothing,
        Phi_exact=nothing)
    ensure_parent_dir(path)
    prof = block_profile(Phi, sampler.K)
    mean_d, mean_omega, true_d1, true_omega1 = true_diag
    open(path, "w") do io
        println(io, "ComplexAmplitudeChain fit_dM Cdot diagnostic metrics")
        println(io, @sprintf("K = %d, D = %d", sampler.K, sampler.D))
        println(io, @sprintf("tau range used = [%.6f, %.6f], lag count = %d",
            first(sampler.lag_times), last(sampler.lag_times), length(sampler.lag_times)))
        println(io, @sprintf("Phi block profile r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
            prof[1, 1, 1], prof[1, 1, 2], prof[1, 2, 1], prof[1, 2, 2]))
        if Phi_raw !== nothing
            raw_prof = block_profile(Phi_raw, sampler.K)
            println(io, @sprintf("Raw short-lag Phi r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
                raw_prof[1, 1, 1], raw_prof[1, 1, 2], raw_prof[1, 2, 1], raw_prof[1, 2, 2]))
        end
        if Phi_full !== nothing
            full_prof = block_profile(Phi_full, sampler.K)
            println(io, @sprintf("Stein-corrected full Phi r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
                full_prof[1, 1, 1], full_prof[1, 1, 2], full_prof[1, 2, 1], full_prof[1, 2, 2]))
        end
        if stein_projected !== nothing
            ident = Matrix{Float64}(I, sampler.D, sampler.D)
            println(io, @sprintf("Projected DSM Stein ||V-I||/||I|| = %.8e",
                norm(stein_projected - ident) / norm(ident)))
        end
        if Phi_exact !== nothing
            exact_prof = block_profile(Phi_exact, sampler.K)
            println(io, @sprintf("Ex-post exact-score corrected Phi r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
                exact_prof[1, 1, 1], exact_prof[1, 1, 2], exact_prof[1, 2, 1], exact_prof[1, 2, 2]))
        end
        println(io, @sprintf("Stored true <d>, <omega> diagnostic = %.8e, %.8e", mean_d, mean_omega))
        println(io, @sprintf("Stored true d1, omega1 diagnostic = %.8e, %.8e", true_d1, true_omega1))
        println(io, "")
        println(io, "Generator-validated Cdot diagnostics on selected observables")
        for (name, key) in (
                ("centered finite-difference data vs exact generator", :data_vs_generator),
                ("learned conditional-score true-M vs exact generator", :learned_vs_generator),
                ("exact-stationary conditional-score true-M vs exact generator", :exactstat_vs_generator),
                ("row-action orientation true-M vs exact generator", :row_vs_generator))
            m = diagnostic_metrics[key]
            println(io, @sprintf("%-58s rel.RMSE = %.8e, corr = %.8e, RMSE = %.8e",
                name, m[:relative_rmse], m[:correlation], m[:rmse]))
        end
        if haskey(diagnostic_metrics, :reverse_vs_generator)
            m = diagnostic_metrics[:reverse_vs_generator]
            println(io, @sprintf("%-58s rel.RMSE = %.8e, corr = %.8e, RMSE = %.8e",
                "reverse conditional-score true-M vs exact generator",
                m[:relative_rmse], m[:correlation], m[:rmse]))
        end
        if haskey(diagnostic_metrics, :reverse_exactstat_vs_generator)
            m = diagnostic_metrics[:reverse_exactstat_vs_generator]
            println(io, @sprintf("%-58s rel.RMSE = %.8e, corr = %.8e, RMSE = %.8e",
                "reverse exact-stationary true-M vs exact generator",
                m[:relative_rmse], m[:correlation], m[:rmse]))
        end
        if haskey(diagnostic_metrics, :phi_reverse_vs_generator)
            m = diagnostic_metrics[:phi_reverse_vs_generator]
            println(io, @sprintf("%-58s rel.RMSE = %.8e, corr = %.8e, RMSE = %.8e",
                "reverse DSM-score Phi vs exact generator",
                m[:relative_rmse], m[:correlation], m[:rmse]))
        end
        if haskey(diagnostic_metrics, :phi_reverse_exactstat_vs_generator)
            m = diagnostic_metrics[:phi_reverse_exactstat_vs_generator]
            println(io, @sprintf("%-58s rel.RMSE = %.8e, corr = %.8e, RMSE = %.8e",
                "reverse exact-score Phi vs exact generator",
                m[:relative_rmse], m[:correlation], m[:rmse]))
        end
        kind_metric = get(diagnostic_metrics, :data_vs_generator_coord_lap, nothing)
        if kind_metric !== nothing
            println(io, @sprintf("coordinate/laplacian data vs generator rel.RMSE = %.8e, corr = %.8e",
                kind_metric[:relative_rmse], kind_metric[:correlation]))
        end
        println(io, "")
        println(io, "Conditional-score self-consistency")
        println(io, @sprintf("max ||E[s_cond]||/sqrt(D), learned = %.8e",
            maximum(score_diag[:cond_mean_norm_learned])))
        println(io, @sprintf("max ||E[s_cond]||/sqrt(D), exact-stat = %.8e",
            maximum(score_diag[:cond_mean_norm_exactstat])))
        println(io, @sprintf("max ||E[s_cond x0']||/||I||, learned = %.8e",
            maximum(score_diag[:cond_x0_norm_learned])))
        println(io, @sprintf("max ||E[s_cond x0']||/||I||, exact-stat = %.8e",
            maximum(score_diag[:cond_x0_norm_exactstat])))
        println(io, @sprintf("max joint x0 Stein rel = %.8e", maximum(score_diag[:joint_x0_stein_rel])))
        println(io, @sprintf("max stationary x0 Stein rel = %.8e", maximum(score_diag[:stat_x0_stein_rel])))
        if reverse_score_diag !== nothing
            println(io, @sprintf("max reverse ||E[s_cond]||/sqrt(D), learned = %.8e",
                maximum(reverse_score_diag[:cond_mean_norm_reverse])))
            println(io, @sprintf("max reverse ||E[s_cond x0']||/||I||, learned = %.8e",
                maximum(reverse_score_diag[:cond_x0_norm_reverse])))
            println(io, @sprintf("max reverse posterior x0 Stein rel = %.8e",
                maximum(reverse_score_diag[:posterior_x0_stein_rel])))
        end
        println(io, "")
        println(io, "Selected observables")
        for a in observable_selection[:selected_indices]
            println(io, @sprintf("[%02d] %-34s signal = %.8e, generator_rel = %.8e, snr = %.8e, roughness = %.8e",
                a, lib.names[a], observable_selection[:signal_rms][a],
                observable_selection[:generator_relative_rmse][a],
                observable_selection[:snr][a], observable_selection[:roughness_relative][a]))
        end
    end
    return nothing
end

function write_metrics(path::AbstractString, sampler::PairSampler, Phi::Matrix{Float64},
        fit::MobilityFitResult, meanM::Matrix{Float64}, delta_mean::Matrix{Float64},
        true_diag::Tuple{Float64, Float64, Float64, Float64}, trueM_metrics=nothing,
        lib::Union{Nothing, ObservableLibrary}=nothing, observable_selection=nothing)
    ensure_parent_dir(path)
    prof = block_profile(Phi, sampler.K)
    mean_prof = block_profile(meanM, sampler.K)
    mean_d, mean_omega, true_d1, true_omega1 = true_diag
    open(path, "w") do io
        println(io, "ComplexAmplitudeChain fit_dM first-stage metrics")
        println(io, @sprintf("K = %d, D = %d", sampler.K, sampler.D))
        println(io, @sprintf("tau range used = [%.6f, %.6f], lag count = %d", first(sampler.lag_times), last(sampler.lag_times), length(sampler.lag_times)))
        println(io, @sprintf("Phi block profile r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
            prof[1, 1, 1], prof[1, 1, 2], prof[1, 2, 1], prof[1, 2, 2]))
        println(io, @sprintf("Stored true <d>, <omega> diagnostic = %.8e, %.8e", mean_d, mean_omega))
        println(io, @sprintf("Stored true d1, omega1 diagnostic = %.8e, %.8e", true_d1, true_omega1))
        println(io, @sprintf("fit d1_hat = %.8e", fit.coefficients[1]))
        println(io, @sprintf("fit omega1_hat = %.8e", fit.coefficients[2]))
        println(io, @sprintf("normal condition number = %.8e", fit.condition_number))
        println(io, @sprintf("operator target RMS = %.8e", fit.target_rms))
        println(io, @sprintf("operator residual RMSE = %.8e", fit.residual_rmse))
        println(io, @sprintf("operator relative RMSE = %.8e", fit.relative_rmse))
        println(io, @sprintf("||<M>-Phi|| / ||Phi|| = %.8e", norm(meanM - Phi) / max(norm(Phi), eps(Float64))))
        println(io, @sprintf("||<delta M>|| / ||Phi|| = %.8e", norm(delta_mean) / max(norm(Phi), eps(Float64))))
        println(io, @sprintf("<M> block profile r=0 = [[%.8e, %.8e], [%.8e, %.8e]]",
            mean_prof[1, 1, 1], mean_prof[1, 1, 2], mean_prof[1, 2, 1], mean_prof[1, 2, 2]))
        if trueM_metrics !== nothing
            println(io, "")
            println(io, "True-M conditional-score identity diagnostics")
            println(io, @sprintf("profile RMSE = %.8e", trueM_metrics[:rmse]))
            println(io, @sprintf("profile relative RMSE = %.8e", trueM_metrics[:relative_rmse]))
            println(io, @sprintf("profile correlation = %.8e", trueM_metrics[:correlation]))
            println(io, @sprintf("sign-flip RMSE = %.8e", trueM_metrics[:sign_flip_rmse]))
            println(io, @sprintf("profile points = %d", trueM_metrics[:npoints]))
        end
        if lib !== nothing && observable_selection !== nothing
            println(io, "")
            println(io, "Selected observables")
            for a in observable_selection[:selected_indices]
                println(io, @sprintf("[%02d] %-34s signal = %.8e, generator_rel = %.8e, snr = %.8e, roughness = %.8e",
                    a, lib.names[a], observable_selection[:signal_rms][a],
                    observable_selection[:generator_relative_rmse][a],
                    observable_selection[:snr][a], observable_selection[:roughness_relative][a]))
            end
        end
    end
    return nothing
end

function save_phi_constant_trajectories(path::AbstractString, times::Vector{Float64},
        states::Array{Float32, 4}, min_eig::Float64, Phi::Matrix{Float64})
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time", times)
        write(h5, "/phi_states", states)
        write(h5, "/min_phi_diffusion_eig", min_eig)
        write(h5, "/Phi", Phi)
    end
    return nothing
end

function run_phi_constant_only(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    reverse_bson = isempty(params.reverse_cond_score_bson) ? "" :
        resolve_path(base_dir, params.reverse_cond_score_bson)
    artifact_bson = resolve_path(base_dir, params.artifact_bson)
    metrics_txt = resolve_path(base_dir, params.metrics_txt)
    figure_png = resolve_path(base_dir, params.figure_png)
    ensure_parent_dir(artifact_bson)
    ensure_parent_dir(metrics_txt)
    ensure_parent_dir(figure_png)
    @printf("fit_dM run mode: %s\n", params.run_mode)

    sampler = load_state_tensor(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max, params.lag_stride, params.max_fit_lags)
    coord_mean = compute_coordinate_means(sampler.states, sampler.start_idx)
    @printf("Coordinate means after burnin: q=%.6e, p=%.6e\n", coord_mean[1], coord_mean[2])
    lib = build_observable_library(sampler.states, sampler.start_idx, params)

    device = detect_device(params.device_name)
    @printf("Analysis device: %s\n", describe_device(device))
    models = load_models(score_bson, joint_bson, device, sampler.K)
    @printf("Loaded stationary score model from %s (sigma=%.4f).\n", score_bson, models.score_sigma)

    phi_est = estimate_stein_corrected_projected_phi(sampler, models, params, coord_mean, device)
    phi_taus = phi_est.phi_taus
    phi_profiles = phi_est.phi_profiles
    Cdot0 = phi_est.Cdot0
    Phi_raw = phi_est.Phi_raw
    V = phi_est.stein_matrix
    V_circ = phi_est.stein_projected
    mean_d, mean_omega, true_d1, true_omega1 = load_true_mean_mobility(input_hdf5)
    Phi_true = true_mean_mobility_matrix(sampler.K, mean_d, mean_omega)
    potential_params = load_potential_params(input_hdf5)
    Cdot0_generator, Phi_generator_raw = estimate_phi_from_true_generator_coordinate_cdot0(
        sampler, params, coord_mean, input_hdf5)
    @printf("Stored <M_true> block = [[%.8e, %.8e], [%.8e, %.8e]]; d1=%.8e omega1=%.8e\n",
        mean_d, -mean_omega, mean_omega, mean_d, true_d1, true_omega1)

    V_exact = estimate_exact_stationary_stein_matrix(sampler, params, coord_mean, potential_params)
    V_exact_circ = matrix_from_block_profile(block_profile(V_exact, sampler.K))
    ident = Matrix{Float64}(I, sampler.D, sampler.D)
    eigV = eigen(Symmetric(sympart(V))).values
    eigVc = eigen(Symmetric(sympart(V_circ))).values
    eigVe = eigen(Symmetric(sympart(V_exact))).values
    eigVec = eigen(Symmetric(sympart(V_exact_circ))).values
    @printf("=== Stationary score Stein diagnostic ===\n")
    @printf("||V - I|| / ||I|| = %.8e\n", norm(V - ident) / norm(ident))
    @printf("min eig sym(V)    = %.8e\n", minimum(eigV))
    @printf("max eig sym(V)    = %.8e\n", maximum(eigV))
    @printf("cond(V)           = %.8e\n", cond(V))
    @printf("||V_circ - I|| / ||I|| = %.8e\n", norm(V_circ - ident) / norm(ident))
    @printf("min eig sym(V_circ)    = %.8e\n", minimum(eigVc))
    @printf("max eig sym(V_circ)    = %.8e\n", maximum(eigVc))
    @printf("cond(V_circ)           = %.8e\n", cond(V_circ))
    @printf("||V_exact - I|| / ||I|| = %.8e\n", norm(V_exact - ident) / norm(ident))
    @printf("min eig sym(V_exact)    = %.8e\n", minimum(eigVe))
    @printf("max eig sym(V_exact)    = %.8e\n", maximum(eigVe))
    @printf("cond(V_exact)           = %.8e\n", cond(V_exact))
    @printf("||V_exact_circ - I|| / ||I|| = %.8e\n", norm(V_exact_circ - ident) / norm(ident))
    @printf("min eig sym(V_exact_circ)    = %.8e\n", minimum(eigVec))
    @printf("max eig sym(V_exact_circ)    = %.8e\n", maximum(eigVec))
    @printf("cond(V_exact_circ)           = %.8e\n", cond(V_exact_circ))

    Phi_corr = Phi_raw / V
    Phi_circcorr = phi_est.Phi_full
    Phi_leftcorr = V \ Phi_raw
    Phi_generator_exactcorr = Phi_generator_raw / V_exact
    Phi_generator_exactcirccorr = Phi_generator_raw / V_exact_circ
    raw_metrics = print_phi_metrics("short-lag raw", Phi_raw, Phi_true, sampler.K)
    corr_metrics = print_phi_metrics("short-lag right Stein-corrected Phi/V", Phi_corr, Phi_true, sampler.K)
    circcorr_metrics = print_phi_metrics("short-lag right circulant-Stein-corrected Phi/V_circ", Phi_circcorr, Phi_true, sampler.K)
    left_metrics = print_phi_metrics("short-lag left Stein-corrected V\\Phi", Phi_leftcorr, Phi_true, sampler.K)
    transpose_metrics = print_phi_metrics("short-lag raw transpose", transpose(Phi_raw), Phi_true, sampler.K)
    generator_metrics = print_phi_metrics("analytic-generator Cdot0 raw", Phi_generator_raw, Phi_true, sampler.K)
    generator_exactcorr_metrics = print_phi_metrics("analytic-generator Cdot0 / exact Stein", Phi_generator_exactcorr, Phi_true, sampler.K)
    generator_exactcirccorr_metrics = print_phi_metrics("analytic-generator Cdot0 / exact circulant Stein", Phi_generator_exactcirccorr, Phi_true, sampler.K)

    sensitivities = Vector{Dict{Symbol, Float64}}()
    candidate_labels = Symbol[:raw_Lmax, :right_stein_Lmax, :right_circstein_Lmax, :left_stein_Lmax,
        :generator_raw, :generator_exact_stein, :generator_exact_circstein]
    candidate_phis = Matrix{Float64}[Phi_raw, Phi_corr, Phi_circcorr, Phi_leftcorr,
        Phi_generator_raw, Phi_generator_exactcorr, Phi_generator_exactcirccorr]
    candidate_metrics = [raw_metrics, corr_metrics, circcorr_metrics, left_metrics,
        generator_metrics, generator_exactcorr_metrics, generator_exactcirccorr_metrics]
    maxL = size(phi_profiles, 1) - 1
    for L in sort(unique([1, 2, 3, 4, 6, 8, maxL]))
        1 <= L <= maxL || continue
        taus_L = phi_taus[1:(L + 1)]
        cdot_profile = zeros(Float64, sampler.K, 2, 2)
        for r in 1:sampler.K, cm in 1:2, cn in 1:2
            degree = min(params.phi_fit_degree, L)
            cdot_profile[r, cm, cn] = polynomial_derivative_at(taus_L,
                vec(phi_profiles[1:(L + 1), r, cm, cn]), 0.0, degree)
        end
        Phi_L = matrix_from_block_profile(-cdot_profile)
        Phi_L_corr = Phi_L / V
        Phi_L_circcorr = Phi_L / V_circ
        raw_L_metrics = phi_agreement_metrics(Phi_L, Phi_true)
        corr_L_metrics = phi_agreement_metrics(Phi_L_corr, Phi_true)
        circcorr_L_metrics = phi_agreement_metrics(Phi_L_circcorr, Phi_true)
        push!(candidate_labels, Symbol("raw_L$(L)"))
        push!(candidate_phis, Phi_L)
        push!(candidate_metrics, raw_L_metrics)
        push!(candidate_labels, Symbol("right_stein_L$(L)"))
        push!(candidate_phis, Phi_L_corr)
        push!(candidate_metrics, corr_L_metrics)
        push!(candidate_labels, Symbol("right_circstein_L$(L)"))
        push!(candidate_phis, Phi_L_circcorr)
        push!(candidate_metrics, circcorr_L_metrics)
        push!(sensitivities, Dict{Symbol, Float64}(
            :L => Float64(L),
            :raw_rel => raw_L_metrics[:relative_rmse],
            :corr_rel => corr_L_metrics[:relative_rmse],
            :circcorr_rel => circcorr_L_metrics[:relative_rmse],
            :raw_sym_rel => raw_L_metrics[:sym_relative_rmse],
            :corr_sym_rel => corr_L_metrics[:sym_relative_rmse],
            :circcorr_sym_rel => circcorr_L_metrics[:sym_relative_rmse],
        ))
    end
    @printf("=== Short-lag Phi sensitivity ===\n")
    for s in sensitivities
        @printf("L=%2d raw rel=%.8e corr rel=%.8e circ-corr rel=%.8e raw sym=%.8e corr sym=%.8e circ-corr sym=%.8e\n",
            Int(s[:L]), s[:raw_rel], s[:corr_rel], s[:circcorr_rel],
            s[:raw_sym_rel], s[:corr_sym_rel], s[:circcorr_sym_rel])
    end

    Phi_best = phi_est.Phi
    phi_projected_profile = phi_est.phi_projected_profile
    phi_offsite_ratio = phi_est.phi_offsite_ratio
    phi_projection_change = phi_est.phi_projection_change
    phi_projected_min_eig = phi_est.phi_projected_min_eig
    phi_projected_max_eig = phi_est.phi_projected_max_eig
    best_label = Symbol("data_right_circstein_" * params.phi_projection)
    best_metrics = phi_agreement_metrics(Phi_best, Phi_true)
    raw_profile = block_profile(Phi_raw, sampler.K)
    @printf("=== Data-only Phi projection ===\n")
    @printf("projection mode                      = %s\n", params.phi_projection)
    @printf("Phi_full r=0 block                   = [[%.8e, %.8e], [%.8e, %.8e]]\n",
        raw_profile[1, 1, 1], raw_profile[1, 1, 2], raw_profile[1, 2, 1], raw_profile[1, 2, 2])
    @printf("Phi_projected r=0 block              = [[%.8e, %.8e], [%.8e, %.8e]]\n",
        phi_projected_profile[1, 1, 1], phi_projected_profile[1, 1, 2],
        phi_projected_profile[1, 2, 1], phi_projected_profile[1, 2, 2])
    @printf("off-site norm ratio                  = %.8e\n", phi_offsite_ratio)
    @printf("projection relative change           = %.8e\n", phi_projection_change)
    @printf("eig sym(Phi_projected) min/max       = %.8e / %.8e\n",
        phi_projected_min_eig, phi_projected_max_eig)
    phi_offsite_ratio > params.phi_offsite_warn_ratio &&
        @warn "Data-only Phi has non-negligible off-site profile before projection" ratio=phi_offsite_ratio threshold=params.phi_offsite_warn_ratio
    @printf("Operational Phi is data-only projection %s; ex-post rel.RMSE vs stored <M_true> = %.8e.\n",
        params.phi_projection, best_metrics[:relative_rmse])

    phi_recovery_png = output_variant(figure_png, "_phi_recovery")
    render_phi_recovery_figure(phi_recovery_png, Phi_raw, Phi_best, Phi_true, V_exact, sensitivities)
    @printf("Saved Phi recovery diagnostic figure to %s\n", phi_recovery_png)

    forward_dt = min(0.001, sampler.save_dt / 20.0)
    forward_total_time = max(48.0, 8.0 * sampler.decorrelation_time)
    forward_burnin_time = max(12.0, 2.0 * sampler.decorrelation_time)
    forward_ntraj = 128
    @printf("Starting constant-M Langevin validation only: dt=%.5g, total=%.3f, burnin=%.3f, ntraj=%d.\n",
        forward_dt, forward_total_time, forward_burnin_time, forward_ntraj)
    phi_times, phi_states, min_phi_eig = integrate_forward_langevin(models, Phi_best,
        nothing, sampler, lib, params, device; mode=:phi, dt=forward_dt,
        total_time=forward_total_time, burnin_time=forward_burnin_time,
        save_dt=sampler.save_dt, ntraj=forward_ntraj, seed=params.seed + 810_000,
        hard_clamp_state=false, score_source=:learned, potential_params=potential_params)
    obs_states = observed_validation_window(sampler, size(phi_states, 1))
    phi_stats_png = output_variant(figure_png, "_constant_phi_stats")
    forward_stats = render_phi_constant_stats(phi_stats_png, sampler, obs_states, phi_states, sampler.save_dt;
        title_suffix=@sprintf("NN stationary score, Phi=%s, min eig sym(Phi)=%.3e", String(best_label), min_phi_eig))
    forward_h5 = replace(output_variant(artifact_bson, "_constant_phi_trajectory"), r"\.bson$" => ".h5")
    save_phi_constant_trajectories(forward_h5, phi_times, phi_states, min_phi_eig, Phi_best)
    @printf("Saved constant-M forward statistics figure to %s\n", phi_stats_png)
    @printf("Saved constant-M forward trajectory to %s\n", forward_h5)

    summary_rel = norm(forward_stats[:phi_summary] - forward_stats[:obs_summary]) /
        max(norm(forward_stats[:obs_summary]), eps(Float64))
    acf_q_rel = norm(forward_stats[:acf_q_phi] - forward_stats[:acf_q_obs]) /
        max(norm(forward_stats[:acf_q_obs]), eps(Float64))
    acf_p_rel = norm(forward_stats[:acf_p_phi] - forward_stats[:acf_p_obs]) /
        max(norm(forward_stats[:acf_p_obs]), eps(Float64))
    @printf("Constant-M learned-score summary rel.RMSE = %.8e\n", summary_rel)
    @printf("Constant-M learned-score q ACF rel.RMSE = %.8e\n", acf_q_rel)
    @printf("Constant-M learned-score p ACF rel.RMSE = %.8e\n", acf_p_rel)

    BSON.@save artifact_bson params phi_taus phi_profiles Cdot0 Cdot0_generator Phi_raw Phi_corr Phi_circcorr Phi_leftcorr Phi_generator_raw Phi_generator_exactcorr Phi_generator_exactcirccorr Phi_true V V_circ V_exact V_exact_circ coord_mean raw_metrics corr_metrics circcorr_metrics left_metrics transpose_metrics generator_metrics generator_exactcorr_metrics generator_exactcirccorr_metrics sensitivities best_label Phi_best phi_projected_profile phi_offsite_ratio phi_projection_change phi_projected_min_eig phi_projected_max_eig forward_stats min_phi_eig
    open(metrics_txt, "w") do io
        println(io, "ComplexAmplitudeChain constant Phi diagnostic")
        println(io, @sprintf("score_bson = %s", score_bson))
        println(io, @sprintf("save_dt = %.8e, phi fit max lag = %d, degree = %d",
            sampler.save_dt, params.phi_fit_max_lag, params.phi_fit_degree))
        println(io, @sprintf("stored <d> = %.8e, <omega> = %.8e", mean_d, mean_omega))
        println(io, @sprintf("Stein ||V-I||/||I|| = %.8e, cond(V) = %.8e",
            norm(V - ident) / norm(ident), cond(V)))
        println(io, @sprintf("Circulant Stein ||V_circ-I||/||I|| = %.8e, cond(V_circ) = %.8e",
            norm(V_circ - ident) / norm(ident), cond(V_circ)))
        println(io, @sprintf("Exact-score Stein ||V_exact-I||/||I|| = %.8e, cond(V_exact) = %.8e",
            norm(V_exact - ident) / norm(ident), cond(V_exact)))
        println(io, @sprintf("Exact-score circulant Stein ||V_exact_circ-I||/||I|| = %.8e, cond(V_exact_circ) = %.8e",
            norm(V_exact_circ - ident) / norm(ident), cond(V_exact_circ)))
        for (name, metrics) in (("raw", raw_metrics), ("right_stein", corr_metrics),
                ("right_circstein", circcorr_metrics), ("left_stein", left_metrics),
                ("raw_transpose", transpose_metrics),
                ("generator_raw", generator_metrics),
                ("generator_exact_stein", generator_exactcorr_metrics),
                ("generator_exact_circstein", generator_exactcirccorr_metrics))
            println(io, @sprintf("%-16s Phi rel = %.8e, sym rel = %.8e, anti rel = %.8e, offsite rel = %.8e",
                name, metrics[:relative_rmse], metrics[:sym_relative_rmse],
                metrics[:anti_relative_rmse], metrics[:offsite_relative_norm]))
        end
        println(io, "Short-lag sensitivity")
        for s in sensitivities
            println(io, @sprintf("L=%2d raw rel=%.8e corr rel=%.8e circ-corr rel=%.8e raw sym=%.8e corr sym=%.8e circ-corr sym=%.8e",
                Int(s[:L]), s[:raw_rel], s[:corr_rel], s[:circcorr_rel],
                s[:raw_sym_rel], s[:corr_sym_rel], s[:circcorr_sym_rel]))
        end
        println(io, @sprintf("selected Phi = %s", String(best_label)))
        println(io, @sprintf("Phi projection mode = %s", params.phi_projection))
        println(io, @sprintf("Phi off-site norm ratio = %.8e", phi_offsite_ratio))
        println(io, @sprintf("Phi projection relative change = %.8e", phi_projection_change))
        println(io, @sprintf("Phi_projected eig sym min/max = %.8e %.8e",
            phi_projected_min_eig, phi_projected_max_eig))
        println(io, @sprintf("forward min eig sym(Phi) = %.8e", min_phi_eig))
        println(io, "forward score source = learned DSM stationary score")
        println(io, @sprintf("forward summary rel.RMSE = %.8e", summary_rel))
        println(io, @sprintf("forward q ACF rel.RMSE = %.8e", acf_q_rel))
        println(io, @sprintf("forward p ACF rel.RMSE = %.8e", acf_p_rel))
        println(io, "Phi recovery figure = " * phi_recovery_png)
        println(io, "Constant Phi stats figure = " * phi_stats_png)
        println(io, "Constant Phi trajectory = " * forward_h5)
    end
    @printf("Saved constant Phi diagnostic artifacts to %s\n", artifact_bson)
    @printf("Saved constant Phi diagnostic metrics to %s\n", metrics_txt)
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    reverse_bson = isempty(params.reverse_cond_score_bson) ? "" :
        resolve_path(base_dir, params.reverse_cond_score_bson)
    artifact_bson = resolve_path(base_dir, params.artifact_bson)
    metrics_txt = resolve_path(base_dir, params.metrics_txt)
    figure_png = resolve_path(base_dir, params.figure_png)
    ensure_parent_dir(artifact_bson)
    ensure_parent_dir(metrics_txt)
    ensure_parent_dir(figure_png)
    @printf("fit_dM run mode: %s\n", params.run_mode)

    sampler = load_state_tensor(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max, params.lag_stride, params.max_fit_lags)
    coord_mean = compute_coordinate_means(sampler.states, sampler.start_idx)
    @printf("Coordinate means after burnin: q=%.6e, p=%.6e\n", coord_mean[1], coord_mean[2])

    device = detect_device(params.device_name)
    @printf("Analysis device: %s\n", describe_device(device))
    models = load_models(score_bson, joint_bson, device, sampler.K)
    @printf("Loaded score models. Joint-score tau range: [%.6f, %.6f]\n", models.tau_min, models.tau_max)
    @printf("Operational conditional-score source: %s\n", params.conditional_score_source)
    reverse_model = nothing
    if !isempty(reverse_bson)
        require_condition(isfile(reverse_bson), "Reverse conditional score BSON not found: $(reverse_bson)")
        reverse_model = load_reverse_conditional_model(reverse_bson, device, sampler.K)
        @printf("Loaded reverse conditional score model. Tau range: [%.6f, %.6f]\n",
            reverse_model.tau_min, reverse_model.tau_max)
    end
    require_condition(params.conditional_score_source != "reverse" || reverse_model !== nothing,
        "conditional_score_source=\"reverse\" requires [data].reverse_cond_score_bson.")

    lib = build_observable_library(sampler.states, sampler.start_idx, params)
    correlation_lags = Int[]
    taus = Float64[]
    C = nothing
    Cdot_data = nothing
    if params.run_mode == "full"
        correlation_lags, taus, C = estimate_correlations(sampler, lib, params, coord_mean)
        Cdot_data = local_polynomial_derivatives(taus, C, params.cdot_local_window, params.cdot_poly_degree)
    end
    Cdot_data_direct = estimate_data_cdot_from_centered_observable_increments(sampler, lib, params, coord_mean)
    phi_est = estimate_stein_corrected_projected_phi(sampler, models, params, coord_mean, device)
    phi_taus = phi_est.phi_taus
    phi_profiles = phi_est.phi_profiles
    Cdot0 = phi_est.Cdot0
    Cdot0_raw = phi_est.Cdot0_raw
    Phi_raw = phi_est.Phi_raw
    Phi_full = phi_est.Phi_full
    Phi = phi_est.Phi
    stein_matrix = phi_est.stein_matrix
    stein_projected = phi_est.stein_projected
    phi_projected_profile = phi_est.phi_projected_profile
    phi_offsite_ratio = phi_est.phi_offsite_ratio
    phi_projection_change = phi_est.phi_projection_change
    phi_projected_min_eig = phi_est.phi_projected_min_eig
    phi_projected_max_eig = phi_est.phi_projected_max_eig
    phi_raw_prof = block_profile(Phi_raw, sampler.K)
    phi_full_prof = block_profile(Phi_full, sampler.K)
    stein_rel = norm(stein_projected - I) / norm(Matrix{Float64}(I, sampler.D, sampler.D))
    @printf("Data-only Phi_raw r=0 block = [[%.6e, %.6e], [%.6e, %.6e]]\n",
        phi_raw_prof[1, 1, 1], phi_raw_prof[1, 1, 2], phi_raw_prof[1, 2, 1], phi_raw_prof[1, 2, 2])
    @printf("DSM Stein projected ||V-I||/||I|| = %.6e\n", stein_rel)
    @printf("Data-only Stein-corrected Phi_full r=0 block = [[%.6e, %.6e], [%.6e, %.6e]]\n",
        phi_full_prof[1, 1, 1], phi_full_prof[1, 1, 2], phi_full_prof[1, 2, 1], phi_full_prof[1, 2, 2])
    @printf("Data-only corrected Phi_projected (%s) r=0 block = [[%.6e, %.6e], [%.6e, %.6e]]\n",
        params.phi_projection, phi_projected_profile[1, 1, 1], phi_projected_profile[1, 1, 2],
        phi_projected_profile[1, 2, 1], phi_projected_profile[1, 2, 2])
    @printf("Phi off-site ratio %.6e; projection change %.6e; eig sym min/max %.6e / %.6e\n",
        phi_offsite_ratio, phi_projection_change, phi_projected_min_eig, phi_projected_max_eig)
    phi_offsite_ratio > params.phi_offsite_warn_ratio &&
        @warn "Data-only Phi has non-negligible off-site profile before projection" ratio=phi_offsite_ratio threshold=params.phi_offsite_warn_ratio

    true_diag = load_true_mean_mobility(input_hdf5)
    mean_d, mean_omega, true_d1, true_omega1 = true_diag
    Phi_true_mean = true_mean_mobility_matrix(sampler.K, mean_d, mean_omega)
    phi_dsm_metrics = phi_agreement_metrics(Phi, Phi_true_mean)
    @printf("Ex-post stored diagnostics only: true <d>=%.6e, <omega>=%.6e, d1=%.6e, omega1=%.6e\n",
        mean_d, mean_omega, true_d1, true_omega1)
    @printf("Operational data-only DSM-corrected Phi rel.RMSE vs stored <M_true> = %.6e\n",
        phi_dsm_metrics[:relative_rmse])

    Cdot_generator = estimate_generator_cdot_from_true_model(sampler, lib, params, coord_mean, input_hdf5)
    selected_observable_indices, observable_selection = select_observables_for_mobility(lib, Cdot_data_direct,
        Cdot_generator, params, sampler.K)
    selected_lib = subset_observable_library(lib, selected_observable_indices)
    Cdot_data_direct_selected = Cdot_data_direct[:, :, selected_observable_indices, :]
    Cdot_generator_selected = Cdot_generator[:, :, selected_observable_indices, :]

    data_profiles = translation_profiles(Cdot_data_direct_selected, sampler.K)
    generator_profiles = translation_profiles(Cdot_generator_selected, sampler.K)
    data_vs_generator = true_mobility_agreement_metrics(generator_profiles, data_profiles)
    data_vs_generator_coord_lap = profile_kind_metrics(selected_lib, generator_profiles, data_profiles,
        Set([:coord, :lapcoord]))
    @printf("Data finite-difference Cdot vs exact generator: rel.RMSE=%.6e, corr=%.6f\n",
        data_vs_generator[:relative_rmse], data_vs_generator[:correlation])
    if data_vs_generator_coord_lap !== nothing
        @printf("Coordinate/laplacian subset data vs generator: rel.RMSE=%.6e, corr=%.6f\n",
            data_vs_generator_coord_lap[:relative_rmse], data_vs_generator_coord_lap[:correlation])
    end

    true_mobility_params = load_true_mobility_params(input_hdf5)
    potential_params = load_potential_params(input_hdf5)
    V_exact_stationary = estimate_exact_stationary_stein_matrix(sampler, params, coord_mean, potential_params)
    V_exact_projected = matrix_from_block_profile(block_profile(V_exact_stationary, sampler.K))
    Phi_exact_full = Phi_raw / V_exact_projected
    Phi_exact, phi_exact_projected_profile, phi_exact_offsite_ratio, phi_exact_projection_change,
        phi_exact_min_eig, phi_exact_max_eig =
        project_phi_from_profile(Phi_exact_full, sampler.K, params.phi_projection)
    phi_exact_metrics = phi_agreement_metrics(Phi_exact, Phi_true_mean)
    phi_exact_prof = block_profile(Phi_exact, sampler.K)
    @printf("Ex-post exact-score corrected Phi_projected r=0 block = [[%.6e, %.6e], [%.6e, %.6e]]; rel.RMSE vs <M_true> %.6e\n",
        phi_exact_prof[1, 1, 1], phi_exact_prof[1, 1, 2],
        phi_exact_prof[1, 2, 1], phi_exact_prof[1, 2, 2],
        phi_exact_metrics[:relative_rmse])
    Cdot_trueM_learned = estimate_true_mobility_cdot_from_conditional_score(sampler, models, selected_lib, params, device,
        true_mobility_params)
    Cdot_trueM_exact_stat = estimate_true_mobility_cdot_from_conditional_score(sampler, models, selected_lib, params, device,
        true_mobility_params; potential_params=potential_params)
    Cdot_trueM_row = estimate_true_mobility_cdot_from_conditional_score(sampler, models, selected_lib, params, device,
        true_mobility_params; use_transpose_action=false)
    score_diag = estimate_conditional_score_self_consistency(sampler, models, params, device, coord_mean, potential_params)
    Cdot_trueM_reverse = nothing
    Cdot_trueM_reverse_exact_stat = nothing
    Cdot_Phi_reverse = nothing
    Cdot_Phi_reverse_exact_stat = nothing
    reverse_score_diag = nothing
    if reverse_model !== nothing
        Cdot_trueM_reverse = estimate_true_mobility_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, true_mobility_params)
        Cdot_trueM_reverse_exact_stat = estimate_true_mobility_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, true_mobility_params; potential_params=potential_params)
        Cdot_Phi_reverse = estimate_phi_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, Phi)
        Cdot_Phi_reverse_exact_stat = estimate_phi_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, Phi_exact; potential_params=potential_params)
        reverse_score_diag = estimate_reverse_conditional_score_self_consistency(sampler, models,
            reverse_model, params, device, coord_mean, potential_params)
    end

    learned_profiles = translation_profiles(Cdot_trueM_learned, sampler.K)
    exactstat_profiles = translation_profiles(Cdot_trueM_exact_stat, sampler.K)
    row_profiles = translation_profiles(Cdot_trueM_row, sampler.K)
    reverse_profiles = Cdot_trueM_reverse === nothing ? nothing : translation_profiles(Cdot_trueM_reverse, sampler.K)
    reverse_exactstat_profiles = Cdot_trueM_reverse_exact_stat === nothing ? nothing :
        translation_profiles(Cdot_trueM_reverse_exact_stat, sampler.K)
    phi_reverse_profiles = Cdot_Phi_reverse === nothing ? nothing : translation_profiles(Cdot_Phi_reverse, sampler.K)
    phi_reverse_exactstat_profiles = Cdot_Phi_reverse_exact_stat === nothing ? nothing :
        translation_profiles(Cdot_Phi_reverse_exact_stat, sampler.K)
    diagnostic_metrics = Dict{Symbol, Any}(
        :data_vs_generator => data_vs_generator,
        :data_vs_generator_coord_lap => data_vs_generator_coord_lap,
        :learned_vs_generator => true_mobility_agreement_metrics(generator_profiles, learned_profiles),
        :exactstat_vs_generator => true_mobility_agreement_metrics(generator_profiles, exactstat_profiles),
        :row_vs_generator => true_mobility_agreement_metrics(generator_profiles, row_profiles),
    )
    if reverse_profiles !== nothing
        diagnostic_metrics[:reverse_vs_generator] = true_mobility_agreement_metrics(generator_profiles, reverse_profiles)
    end
    if reverse_exactstat_profiles !== nothing
        diagnostic_metrics[:reverse_exactstat_vs_generator] =
            true_mobility_agreement_metrics(generator_profiles, reverse_exactstat_profiles)
    end
    if phi_reverse_profiles !== nothing
        diagnostic_metrics[:phi_reverse_vs_generator] =
            true_mobility_agreement_metrics(generator_profiles, phi_reverse_profiles)
    end
    if phi_reverse_exactstat_profiles !== nothing
        diagnostic_metrics[:phi_reverse_exactstat_vs_generator] =
            true_mobility_agreement_metrics(generator_profiles, phi_reverse_exactstat_profiles)
    end
    learned_metrics = diagnostic_metrics[:learned_vs_generator]
    exact_metrics = diagnostic_metrics[:exactstat_vs_generator]
    row_metrics = diagnostic_metrics[:row_vs_generator]
    @printf("True-M learned conditional-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
        learned_metrics[:relative_rmse], learned_metrics[:correlation])
    @printf("True-M exact-stationary conditional-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
        exact_metrics[:relative_rmse], exact_metrics[:correlation])
    @printf("True-M row-action orientation vs generator: rel.RMSE=%.6e, corr=%.6f\n",
        row_metrics[:relative_rmse], row_metrics[:correlation])
    if haskey(diagnostic_metrics, :reverse_vs_generator)
        reverse_metrics = diagnostic_metrics[:reverse_vs_generator]
        @printf("True-M reverse conditional-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
            reverse_metrics[:relative_rmse], reverse_metrics[:correlation])
    end
    if haskey(diagnostic_metrics, :reverse_exactstat_vs_generator)
        reverse_exact_metrics = diagnostic_metrics[:reverse_exactstat_vs_generator]
        @printf("True-M reverse exact-stationary vs generator: rel.RMSE=%.6e, corr=%.6f\n",
            reverse_exact_metrics[:relative_rmse], reverse_exact_metrics[:correlation])
    end
    if haskey(diagnostic_metrics, :phi_reverse_vs_generator)
        phi_reverse_metrics = diagnostic_metrics[:phi_reverse_vs_generator]
        @printf("Phi reverse DSM-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
            phi_reverse_metrics[:relative_rmse], phi_reverse_metrics[:correlation])
    end
    if haskey(diagnostic_metrics, :phi_reverse_exactstat_vs_generator)
        phi_reverse_exact_metrics = diagnostic_metrics[:phi_reverse_exactstat_vs_generator]
        @printf("Phi reverse exact-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
            phi_reverse_exact_metrics[:relative_rmse], phi_reverse_exact_metrics[:correlation])
    end
    @printf("Score self-consistency max ||E[s_cond x0']||/||I||: learned %.6e, exact-stat %.6e\n",
        maximum(score_diag[:cond_x0_norm_learned]), maximum(score_diag[:cond_x0_norm_exactstat]))
    if reverse_score_diag !== nothing
        @printf("Reverse score self-consistency max ||E[s_cond x0']||/||I||: learned %.6e\n",
            maximum(reverse_score_diag[:cond_x0_norm_reverse]))
    end
    render_true_mobility_cdot_figure(figure_png, sampler.lag_times, selected_lib, data_profiles,
        generator_profiles, learned_profiles, exactstat_profiles, row_profiles, diagnostic_metrics, score_diag;
        reverse_prof=reverse_profiles, reverse_exactstat_prof=reverse_exactstat_profiles,
        phi_reverse_prof=phi_reverse_profiles, phi_reverse_exactstat_prof=phi_reverse_exactstat_profiles,
        reverse_score_diag=reverse_score_diag)
    @printf("Saved generator-validated Cdot diagnostic figure to %s\n", figure_png)

    fit = nothing
    meanM = nothing
    delta_mean = nothing
    BSON.@save artifact_bson params correlation_lags taus C Cdot_data Cdot_data_direct Cdot_data_direct_selected Cdot_generator Cdot_generator_selected phi_taus phi_profiles Cdot0 Cdot0_raw Phi_raw Phi_full Phi stein_matrix stein_projected Phi_exact_full Phi_exact V_exact_stationary V_exact_projected phi_projected_profile phi_offsite_ratio phi_projection_change phi_projected_min_eig phi_projected_max_eig phi_exact_projected_profile phi_exact_offsite_ratio phi_exact_projection_change phi_exact_min_eig phi_exact_max_eig phi_dsm_metrics phi_exact_metrics lib selected_lib selected_observable_indices observable_selection fit meanM delta_mean coord_mean true_diag true_mobility_params potential_params Cdot_trueM_learned Cdot_trueM_exact_stat Cdot_trueM_row Cdot_trueM_reverse Cdot_trueM_reverse_exact_stat Cdot_Phi_reverse Cdot_Phi_reverse_exact_stat data_profiles generator_profiles learned_profiles exactstat_profiles row_profiles reverse_profiles reverse_exactstat_profiles phi_reverse_profiles phi_reverse_exactstat_profiles diagnostic_metrics score_diag reverse_score_diag
    write_cdot_diagnostic_metrics(metrics_txt, sampler, Phi, true_diag, diagnostic_metrics, score_diag,
        lib, observable_selection; reverse_score_diag=reverse_score_diag,
        Phi_raw=Phi_raw, Phi_full=Phi_full, stein_projected=stein_projected,
        Phi_exact=Phi_exact)
    @printf("Saved Cdot diagnostic artifacts to %s\n", artifact_bson)
    @printf("Saved diagnostic metrics to %s\n", metrics_txt)
    params.run_mode == "cdot_diagnostic" && return nothing

    fit = estimate_mobility_coefficients(sampler, models, reverse_model, lib, params, device, Phi, correlation_lags, Cdot_data)
    @printf("Two-coefficient diagnostic fit: d1_hat=%.6e, omega1_hat=%.6e, rel.RMSE=%.6e\n",
        fit.coefficients[1], fit.coefficients[2], fit.relative_rmse)

    meanM, delta_mean = mean_mobility_from_ansatz(Phi, lib, fit.coefficients, sampler.states, sampler.start_idx)
    mean_rel = norm(meanM - Phi) / max(norm(Phi), eps(Float64))
    delta_rel = norm(delta_mean) / max(norm(Phi), eps(Float64))
    @printf("<M>-Phi check: ||<M>-Phi||/||Phi|| = %.6e; ||<delta M>||/||Phi|| = %.6e\n", mean_rel, delta_rel)

    Cdot_phi_cond = estimate_phi_cdot_from_conditional_score(sampler, models, reverse_model, selected_lib, params, device, Phi)
    phi_cond_profiles = translation_profiles(Cdot_phi_cond, sampler.K)
    A_data_profiles = data_profiles .- phi_cond_profiles
    A_true_profiles = learned_profiles .- phi_cond_profiles
    direct_cache = build_direct_mobility_training_cache(sampler, models, reverse_model, selected_lib, params, device;
        pairs_per_lag=min(256, params.pairs_per_lag_operator), mean_samples=32_768,
        seed=params.seed + 710_000)
    mobility_model, mobility_history = train_mobility_nn_direct_loss(A_data_profiles, direct_cache;
        seed=params.seed + 720_000, epochs=350, learning_rate=8.0e-4,
        mean_penalty_weight=1.0e-2, weight_decay=1.0e-7, profile_batch_size=512)
    A_train_profiles = Float64.(predict_A_profiles_direct(mobility_model, direct_cache))
    Cdot_nn_cond = estimate_nn_mobility_cdot_from_conditional_score(sampler, models, reverse_model, selected_lib, params,
        device, Phi, mobility_model)
    nn_cond_profiles = translation_profiles(Cdot_nn_cond, sampler.K)
    A_nn_profiles = nn_cond_profiles .- phi_cond_profiles
    cdot_three_png = output_variant(figure_png, "_three_methods")
    A_three_png = output_variant(figure_png, "_A_three_methods")
    training_png = output_variant(figure_png, "_M_training")
    render_three_method_profiles(cdot_three_png,
        "Complex-amplitude chain: Cdot comparison on selected observables",
        "profile Cdot", sampler.lag_times, selected_lib, data_profiles, learned_profiles, nn_cond_profiles)
    render_three_method_profiles(A_three_png,
        "Complex-amplitude chain: A = Cdot - Cdot_Phi comparison",
        "profile A", sampler.lag_times, selected_lib, A_data_profiles, A_true_profiles, A_nn_profiles)
    render_mobility_training_diagnostic(training_png, mobility_history, sampler.lag_times,
        selected_lib, A_data_profiles, A_train_profiles, A_nn_profiles, Float64[], Float64[];
        A_true=A_true_profiles)
    @printf("Saved three-method Cdot figure to %s\n", cdot_three_png)
    @printf("Saved three-method A figure to %s\n", A_three_png)
    @printf("Saved mobility training diagnostic figure to %s\n", training_png)

    if params.run_mode == "mobility_fit_only"
        mobility_bson = output_variant(artifact_bson, "_mobility_nn")
        BSON.@save mobility_bson mobility_model mobility_history fit Phi meanM delta_mean lib selected_lib coord_mean
        BSON.@save artifact_bson params correlation_lags taus C Cdot_data Cdot_data_direct Cdot_data_direct_selected Cdot_generator Cdot_generator_selected phi_taus phi_profiles Cdot0 Phi_full Phi phi_projected_profile phi_offsite_ratio phi_projection_change phi_projected_min_eig phi_projected_max_eig lib selected_lib selected_observable_indices observable_selection fit meanM delta_mean coord_mean true_diag true_mobility_params potential_params Cdot_trueM_learned Cdot_trueM_exact_stat Cdot_trueM_row Cdot_trueM_reverse Cdot_trueM_reverse_exact_stat data_profiles generator_profiles learned_profiles exactstat_profiles row_profiles reverse_profiles reverse_exactstat_profiles diagnostic_metrics score_diag reverse_score_diag direct_cache mobility_history Cdot_phi_cond Cdot_nn_cond phi_cond_profiles nn_cond_profiles A_data_profiles A_true_profiles A_train_profiles A_nn_profiles
        write_metrics(metrics_txt, sampler, Phi, fit, meanM, delta_mean, true_diag,
            diagnostic_metrics[:learned_vs_generator], lib, observable_selection)
        open(metrics_txt, "a") do io
            println(io, "")
            println(io, "Mobility NN direct paper-loss training")
            println(io, "mobility_nn training target = direct paper loss J_deltaM using A[delta M_theta] = Cdot_obs - A[Phi]")
            println(io, @sprintf("mobility_nn direct-cache A rel.RMSE = %.8e, corr = %.8e",
                profile_agreement_metrics(A_data_profiles, A_train_profiles)...))
            println(io, @sprintf("mobility_nn resampled A rel.RMSE = %.8e, corr = %.8e",
                profile_agreement_metrics(A_data_profiles, A_nn_profiles)...))
            println(io, "Cdot three-method figure = " * cdot_three_png)
            println(io, "A three-method figure = " * A_three_png)
            println(io, "Mobility training figure = " * training_png)
        end
        @printf("Saved mobility NN to %s\n", mobility_bson)
        @printf("Saved updated mobility-fit artifacts to %s\n", artifact_bson)
        @printf("Saved metrics to %s\n", metrics_txt)
        return nothing
    end

    forward_stats_png = output_variant(figure_png, "_forward_validation_stats")
    forward_cmn_png = output_variant(figure_png, "_forward_validation_cmn")
    forward_h5 = output_variant(artifact_bson, "_forward_trajectories")
    forward_h5 = replace(forward_h5, r"\.bson$" => ".h5")
    @printf("Starting forward Langevin validation with learned score, M=Phi and NN-M.\n")
    forward_dt = min(0.002, sampler.save_dt / 10.0)
    forward_total_time = max(24.0, last(sampler.lag_times) + 18.0)
    forward_burnin_time = 6.0
    forward_ntraj = 96
    phi_forward_times, phi_forward_states, min_phi_eig = integrate_forward_langevin(models, Phi,
        mobility_model, sampler, selected_lib, params, device; mode=:phi, dt=forward_dt,
        total_time=forward_total_time, burnin_time=forward_burnin_time,
        save_dt=sampler.save_dt, ntraj=forward_ntraj, seed=params.seed + 510_000)
    nn_forward_times, nn_forward_states, min_nn_eig = integrate_forward_langevin(models, Phi,
        mobility_model, sampler, selected_lib, params, device; mode=:nn, dt=forward_dt,
        total_time=forward_total_time, burnin_time=forward_burnin_time,
        save_dt=sampler.save_dt, ntraj=forward_ntraj, seed=params.seed + 520_000)
    obs_validation_states = observed_validation_window(sampler, size(phi_forward_states, 1))
    forward_stats = render_forward_validation_stats(forward_stats_png, sampler, obs_validation_states,
        phi_forward_states, nn_forward_states, sampler.save_dt)
    save_forward_trajectories(forward_h5, phi_forward_times, phi_forward_states, nn_forward_states,
        min_phi_eig, min_nn_eig)
    @printf("Saved forward validation statistics figure to %s\n", forward_stats_png)
    @printf("Saved forward validation trajectories to %s\n", forward_h5)

    lag_indices = [begin
        idx = searchsortedfirst(correlation_lags, lag)
        require_condition(idx <= length(correlation_lags) && correlation_lags[idx] == lag,
            "Internal error: lag $(lag) missing from correlation grid.")
        idx
    end for lag in sampler.lag_steps]
    C_obs_selected_profiles = translation_profiles(C[lag_indices, :, selected_observable_indices, :], sampler.K)
    forward_pairs = min(30_000, params.pairs_per_lag_correlation)
    C_phi_forward_profiles = estimate_c_profiles_from_states(phi_forward_states, sampler.save_dt,
        sampler.lag_times, selected_lib, coord_mean, forward_pairs, params.batch_size, params.seed + 530_000)
    C_nn_forward_profiles = estimate_c_profiles_from_states(nn_forward_states, sampler.save_dt,
        sampler.lag_times, selected_lib, coord_mean, forward_pairs, params.batch_size, params.seed + 540_000)
    render_three_method_profiles(forward_cmn_png,
        "Complex-amplitude chain: forward Langevin C_mn(t)",
        "profile C", sampler.lag_times, selected_lib, C_obs_selected_profiles,
        C_phi_forward_profiles, C_nn_forward_profiles;
        true_label="M=Phi Langevin", nn_label="NN M Langevin",
        summary_true_label="Phi", summary_nn_label="NN-M")
    @printf("Saved forward validation Cmn figure to %s\n", forward_cmn_png)

    mobility_bson = output_variant(artifact_bson, "_mobility_nn")
    BSON.@save mobility_bson mobility_model mobility_history fit Phi meanM delta_mean lib selected_lib coord_mean
    @printf("Saved mobility NN to %s\n", mobility_bson)

    BSON.@save artifact_bson params correlation_lags taus C Cdot_data Cdot_data_direct Cdot_data_direct_selected Cdot_generator Cdot_generator_selected phi_taus phi_profiles Cdot0 Phi_full Phi phi_projected_profile phi_offsite_ratio phi_projection_change phi_projected_min_eig phi_projected_max_eig lib selected_lib selected_observable_indices observable_selection fit meanM delta_mean coord_mean true_diag true_mobility_params potential_params Cdot_trueM_learned Cdot_trueM_exact_stat Cdot_trueM_row Cdot_trueM_reverse Cdot_trueM_reverse_exact_stat data_profiles generator_profiles learned_profiles exactstat_profiles row_profiles reverse_profiles reverse_exactstat_profiles diagnostic_metrics score_diag reverse_score_diag mobility_history Cdot_phi_cond Cdot_nn_cond phi_cond_profiles nn_cond_profiles A_data_profiles A_true_profiles A_train_profiles A_nn_profiles forward_stats C_obs_selected_profiles C_phi_forward_profiles C_nn_forward_profiles min_phi_eig min_nn_eig
    write_metrics(metrics_txt, sampler, Phi, fit, meanM, delta_mean, true_diag,
        diagnostic_metrics[:learned_vs_generator], lib, observable_selection)
    open(metrics_txt, "a") do io
        println(io, "")
        println(io, "Mobility NN and forward validation")
        println(io, "mobility_nn training target = direct paper loss J_deltaM using A[delta M_theta] = Cdot_obs - A[Phi]")
        println(io, @sprintf("mobility_nn direct-training A rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(A_data_profiles, A_train_profiles)...))
        println(io, @sprintf("Cdot NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(data_profiles, nn_cond_profiles)...))
        println(io, @sprintf("A NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(A_data_profiles, A_nn_profiles)...))
        println(io, @sprintf("Forward Cmn Phi/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(C_obs_selected_profiles, C_phi_forward_profiles)...))
        println(io, @sprintf("Forward Cmn NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(C_obs_selected_profiles, C_nn_forward_profiles)...))
        println(io, @sprintf("Forward min diffusion eig Phi = %.8e", min_phi_eig))
        println(io, @sprintf("Forward min diffusion eig NN-M = %.8e", min_nn_eig))
        println(io, "Cdot three-method figure = " * cdot_three_png)
        println(io, "A three-method figure = " * A_three_png)
        println(io, "Forward stats figure = " * forward_stats_png)
        println(io, "Forward Cmn figure = " * forward_cmn_png)
        println(io, "Mobility training figure = " * training_png)
        println(io, "Forward trajectories = " * forward_h5)
    end
    @printf("Saved first-stage artifacts to %s\n", artifact_bson)
    @printf("Saved metrics to %s\n", metrics_txt)
    return nothing
end

function run_mobility_stage_from_artifacts(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    artifact_bson = resolve_path(base_dir, params.artifact_bson)
    metrics_txt = resolve_path(base_dir, params.metrics_txt)
    figure_png = resolve_path(base_dir, params.figure_png)
    @printf("fit_dM run mode: %s\n", params.run_mode)

    sampler = load_state_tensor(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max, params.lag_stride, params.max_fit_lags)
    device = detect_device(params.device_name)
    @printf("Analysis device: %s\n", describe_device(device))
    models = load_models(score_bson, joint_bson, device, sampler.K)
    reverse_model = nothing
    if !isempty(reverse_bson)
        require_condition(isfile(reverse_bson), "Reverse conditional score BSON not found: $(reverse_bson)")
        reverse_model = load_reverse_conditional_model(reverse_bson, device, sampler.K)
    end
    require_condition(params.conditional_score_source != "reverse" || reverse_model !== nothing,
        "conditional_score_source=\"reverse\" requires [data].reverse_cond_score_bson.")
    @printf("Operational conditional-score source: %s\n", params.conditional_score_source)
    blob = BSON.load(artifact_bson)
    correlation_lags = dict_get(blob, :correlation_lags)
    taus = dict_get(blob, :taus)
    C = dict_get(blob, :C)
    Cdot_data = dict_get(blob, :Cdot_data)
    Cdot_data_direct = dict_get(blob, :Cdot_data_direct)
    Cdot_data_direct_selected = dict_get(blob, :Cdot_data_direct_selected)
    Cdot_generator = dict_get(blob, :Cdot_generator)
    Cdot_generator_selected = dict_get(blob, :Cdot_generator_selected)
    phi_taus = dict_get(blob, :phi_taus)
    phi_profiles = dict_get(blob, :phi_profiles)
    Cdot0 = dict_get(blob, :Cdot0)
    Phi = dict_get(blob, :Phi)
    lib = dict_get(blob, :lib)
    selected_lib = dict_get(blob, :selected_lib)
    selected_observable_indices = dict_get(blob, :selected_observable_indices)
    observable_selection = dict_get(blob, :observable_selection)
    coord_mean = dict_get(blob, :coord_mean)
    true_diag = dict_get(blob, :true_diag)
    true_mobility_params = dict_get(blob, :true_mobility_params)
    potential_params = dict_get(blob, :potential_params)
    Cdot_trueM_learned = dict_get(blob, :Cdot_trueM_learned)
    Cdot_trueM_exact_stat = dict_get(blob, :Cdot_trueM_exact_stat)
    Cdot_trueM_row = dict_get(blob, :Cdot_trueM_row)
    Cdot_trueM_reverse = dict_get(blob, :Cdot_trueM_reverse)
    Cdot_trueM_reverse_exact_stat = dict_get(blob, :Cdot_trueM_reverse_exact_stat)
    data_profiles = dict_get(blob, :data_profiles)
    generator_profiles = dict_get(blob, :generator_profiles)
    learned_profiles = dict_get(blob, :learned_profiles)
    exactstat_profiles = dict_get(blob, :exactstat_profiles)
    row_profiles = dict_get(blob, :row_profiles)
    reverse_profiles = dict_get(blob, :reverse_profiles)
    reverse_exactstat_profiles = dict_get(blob, :reverse_exactstat_profiles)
    diagnostic_metrics = dict_get(blob, :diagnostic_metrics)
    score_diag = dict_get(blob, :score_diag)
    reverse_score_diag = dict_get(blob, :reverse_score_diag)
    @printf("Loaded diagnostic artifacts from %s\n", artifact_bson)

    fit = haskey(blob, :fit) && dict_get(blob, :fit) !== nothing ? dict_get(blob, :fit) :
        estimate_mobility_coefficients(sampler, models, reverse_model, lib, params, device, Phi, correlation_lags, Cdot_data)
    @printf("Two-coefficient diagnostic fit: d1_hat=%.6e, omega1_hat=%.6e, rel.RMSE=%.6e\n",
        fit.coefficients[1], fit.coefficients[2], fit.relative_rmse)

    meanM, delta_mean = mean_mobility_from_ansatz(Phi, lib, fit.coefficients, sampler.states, sampler.start_idx)
    @printf("<M>-Phi check: ||<M>-Phi||/||Phi|| = %.6e; ||<delta M>||/||Phi|| = %.6e\n",
        norm(meanM - Phi) / max(norm(Phi), eps(Float64)),
        norm(delta_mean) / max(norm(Phi), eps(Float64)))

    Cdot_phi_cond = dict_haskey(blob, :Cdot_phi_cond) ? dict_get(blob, :Cdot_phi_cond) :
        estimate_phi_cdot_from_conditional_score(sampler, models, reverse_model, selected_lib, params, device, Phi)
    phi_cond_profiles = translation_profiles(Cdot_phi_cond, sampler.K)
    A_data_profiles = data_profiles .- phi_cond_profiles
    A_true_profiles = learned_profiles .- phi_cond_profiles
    direct_cache = dict_haskey(blob, :direct_cache) ? dict_get(blob, :direct_cache) :
        build_direct_mobility_training_cache(sampler, models, reverse_model, selected_lib, params, device;
        pairs_per_lag=min(256, params.pairs_per_lag_operator), mean_samples=32_768,
        seed=params.seed + 710_000)
    mobility_model, mobility_history = train_mobility_nn_direct_loss(A_data_profiles, direct_cache;
        seed=params.seed + 720_000, epochs=350, learning_rate=8.0e-4,
        mean_penalty_weight=1.0e-2, weight_decay=1.0e-7, profile_batch_size=512)
    A_train_profiles = Float64.(predict_A_profiles_direct(mobility_model, direct_cache))
    Cdot_nn_cond = estimate_nn_mobility_cdot_from_conditional_score(sampler, models, reverse_model, selected_lib, params,
        device, Phi, mobility_model)
    nn_cond_profiles = translation_profiles(Cdot_nn_cond, sampler.K)
    A_nn_profiles = nn_cond_profiles .- phi_cond_profiles
    cdot_three_png = output_variant(figure_png, "_three_methods")
    A_three_png = output_variant(figure_png, "_A_three_methods")
    training_png = output_variant(figure_png, "_M_training")
    render_three_method_profiles(cdot_three_png,
        "Complex-amplitude chain: Cdot comparison on selected observables",
        "profile Cdot", sampler.lag_times, selected_lib, data_profiles, learned_profiles, nn_cond_profiles)
    render_three_method_profiles(A_three_png,
        "Complex-amplitude chain: A = Cdot - Cdot_Phi comparison",
        "profile A", sampler.lag_times, selected_lib, A_data_profiles, A_true_profiles, A_nn_profiles)
    render_mobility_training_diagnostic(training_png, mobility_history, sampler.lag_times,
        selected_lib, A_data_profiles, A_train_profiles, A_nn_profiles, Float64[], Float64[];
        A_true=A_true_profiles)
    @printf("Saved three-method Cdot figure to %s\n", cdot_three_png)
    @printf("Saved three-method A figure to %s\n", A_three_png)
    @printf("Saved mobility training diagnostic figure to %s\n", training_png)

    if params.run_mode == "mobility_fit_only"
        mobility_bson = output_variant(artifact_bson, "_mobility_nn")
        BSON.@save mobility_bson mobility_model mobility_history fit Phi meanM delta_mean lib selected_lib coord_mean
        BSON.@save artifact_bson params correlation_lags taus C Cdot_data Cdot_data_direct Cdot_data_direct_selected Cdot_generator Cdot_generator_selected phi_taus phi_profiles Cdot0 Phi lib selected_lib selected_observable_indices observable_selection fit meanM delta_mean coord_mean true_diag true_mobility_params potential_params Cdot_trueM_learned Cdot_trueM_exact_stat Cdot_trueM_row Cdot_trueM_reverse Cdot_trueM_reverse_exact_stat data_profiles generator_profiles learned_profiles exactstat_profiles row_profiles reverse_profiles reverse_exactstat_profiles diagnostic_metrics score_diag reverse_score_diag direct_cache mobility_history Cdot_phi_cond Cdot_nn_cond phi_cond_profiles nn_cond_profiles A_data_profiles A_true_profiles A_train_profiles A_nn_profiles
        write_metrics(metrics_txt, sampler, Phi, fit, meanM, delta_mean, true_diag,
            diagnostic_metrics[:learned_vs_generator], lib, observable_selection)
        open(metrics_txt, "a") do io
            println(io, "")
            println(io, "Mobility NN direct paper-loss training")
            println(io, "mobility_nn training target = direct paper loss J_deltaM using A[delta M_theta] = Cdot_obs - A[Phi]")
            println(io, @sprintf("mobility_nn direct-cache A rel.RMSE = %.8e, corr = %.8e",
                profile_agreement_metrics(A_data_profiles, A_train_profiles)...))
            println(io, @sprintf("mobility_nn resampled A rel.RMSE = %.8e, corr = %.8e",
                profile_agreement_metrics(A_data_profiles, A_nn_profiles)...))
            println(io, "Cdot three-method figure = " * cdot_three_png)
            println(io, "A three-method figure = " * A_three_png)
            println(io, "Mobility training figure = " * training_png)
        end
        @printf("Saved mobility NN to %s\n", mobility_bson)
        @printf("Saved updated mobility-fit artifacts to %s\n", artifact_bson)
        @printf("Saved metrics to %s\n", metrics_txt)
        return nothing
    end

    forward_stats_png = output_variant(figure_png, "_forward_validation_stats")
    forward_cmn_png = output_variant(figure_png, "_forward_validation_cmn")
    forward_h5 = replace(output_variant(artifact_bson, "_forward_trajectories"), r"\.bson$" => ".h5")
    @printf("Starting forward Langevin validation with learned score, M=Phi and NN-M.\n")
    forward_dt = min(0.002, sampler.save_dt / 10.0)
    forward_total_time = max(24.0, last(sampler.lag_times) + 18.0)
    forward_burnin_time = 6.0
    forward_ntraj = 96
    phi_forward_times, phi_forward_states, min_phi_eig = integrate_forward_langevin(models, Phi,
        mobility_model, sampler, selected_lib, params, device; mode=:phi, dt=forward_dt,
        total_time=forward_total_time, burnin_time=forward_burnin_time,
        save_dt=sampler.save_dt, ntraj=forward_ntraj, seed=params.seed + 510_000)
    nn_forward_times, nn_forward_states, min_nn_eig = integrate_forward_langevin(models, Phi,
        mobility_model, sampler, selected_lib, params, device; mode=:nn, dt=forward_dt,
        total_time=forward_total_time, burnin_time=forward_burnin_time,
        save_dt=sampler.save_dt, ntraj=forward_ntraj, seed=params.seed + 520_000)
    obs_validation_states = observed_validation_window(sampler, size(phi_forward_states, 1))
    forward_stats = render_forward_validation_stats(forward_stats_png, sampler, obs_validation_states,
        phi_forward_states, nn_forward_states, sampler.save_dt)
    save_forward_trajectories(forward_h5, phi_forward_times, phi_forward_states, nn_forward_states,
        min_phi_eig, min_nn_eig)
    @printf("Saved forward validation statistics figure to %s\n", forward_stats_png)
    @printf("Saved forward validation trajectories to %s\n", forward_h5)

    lag_indices = [begin
        idx = searchsortedfirst(correlation_lags, lag)
        require_condition(idx <= length(correlation_lags) && correlation_lags[idx] == lag,
            "Internal error: lag $(lag) missing from correlation grid.")
        idx
    end for lag in sampler.lag_steps]
    C_obs_selected_profiles = translation_profiles(C[lag_indices, :, selected_observable_indices, :], sampler.K)
    forward_pairs = min(30_000, params.pairs_per_lag_correlation)
    C_phi_forward_profiles = estimate_c_profiles_from_states(phi_forward_states, sampler.save_dt,
        sampler.lag_times, selected_lib, coord_mean, forward_pairs, params.batch_size, params.seed + 530_000)
    C_nn_forward_profiles = estimate_c_profiles_from_states(nn_forward_states, sampler.save_dt,
        sampler.lag_times, selected_lib, coord_mean, forward_pairs, params.batch_size, params.seed + 540_000)
    render_three_method_profiles(forward_cmn_png,
        "Complex-amplitude chain: forward Langevin C_mn(t)",
        "profile C", sampler.lag_times, selected_lib, C_obs_selected_profiles,
        C_phi_forward_profiles, C_nn_forward_profiles;
        true_label="M=Phi Langevin", nn_label="NN M Langevin",
        summary_true_label="Phi", summary_nn_label="NN-M")
    @printf("Saved forward validation Cmn figure to %s\n", forward_cmn_png)

    mobility_bson = output_variant(artifact_bson, "_mobility_nn")
    BSON.@save mobility_bson mobility_model mobility_history fit Phi meanM delta_mean lib selected_lib coord_mean
    BSON.@save artifact_bson params correlation_lags taus C Cdot_data Cdot_data_direct Cdot_data_direct_selected Cdot_generator Cdot_generator_selected phi_taus phi_profiles Cdot0 Phi lib selected_lib selected_observable_indices observable_selection fit meanM delta_mean coord_mean true_diag true_mobility_params potential_params Cdot_trueM_learned Cdot_trueM_exact_stat Cdot_trueM_row Cdot_trueM_reverse Cdot_trueM_reverse_exact_stat data_profiles generator_profiles learned_profiles exactstat_profiles row_profiles reverse_profiles reverse_exactstat_profiles diagnostic_metrics score_diag reverse_score_diag mobility_history Cdot_phi_cond Cdot_nn_cond phi_cond_profiles nn_cond_profiles A_data_profiles A_true_profiles A_train_profiles A_nn_profiles forward_stats C_obs_selected_profiles C_phi_forward_profiles C_nn_forward_profiles min_phi_eig min_nn_eig
    write_metrics(metrics_txt, sampler, Phi, fit, meanM, delta_mean, true_diag,
        diagnostic_metrics[:learned_vs_generator], lib, observable_selection)
    open(metrics_txt, "a") do io
        println(io, "")
        println(io, "Mobility NN and forward validation")
        println(io, "mobility_nn training target = direct paper loss J_deltaM using A[delta M_theta] = Cdot_obs - A[Phi]")
        println(io, @sprintf("mobility_nn direct-training A rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(A_data_profiles, A_train_profiles)...))
        println(io, @sprintf("Cdot NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(data_profiles, nn_cond_profiles)...))
        println(io, @sprintf("A NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(A_data_profiles, A_nn_profiles)...))
        println(io, @sprintf("Forward Cmn Phi/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(C_obs_selected_profiles, C_phi_forward_profiles)...))
        println(io, @sprintf("Forward Cmn NN-M/data rel.RMSE = %.8e, corr = %.8e",
            profile_agreement_metrics(C_obs_selected_profiles, C_nn_forward_profiles)...))
        println(io, @sprintf("Forward min diffusion eig Phi = %.8e", min_phi_eig))
        println(io, @sprintf("Forward min diffusion eig NN-M = %.8e", min_nn_eig))
        println(io, "Cdot three-method figure = " * cdot_three_png)
        println(io, "A three-method figure = " * A_three_png)
        println(io, "Forward stats figure = " * forward_stats_png)
        println(io, "Forward Cmn figure = " * forward_cmn_png)
        println(io, "Mobility training figure = " * training_png)
        println(io, "Forward trajectories = " * forward_h5)
    end
    @printf("Saved mobility NN to %s\n", mobility_bson)
    @printf("Saved updated artifacts to %s\n", artifact_bson)
    @printf("Saved metrics to %s\n", metrics_txt)
    return nothing
end

function run_cdot_plot_only(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    reverse_bson = isempty(params.reverse_cond_score_bson) ? "" :
        resolve_path(base_dir, params.reverse_cond_score_bson)
    artifact_bson = resolve_path(base_dir, params.artifact_bson)
    figure_png = resolve_path(base_dir, params.figure_png)
    cache_bson = output_variant(artifact_bson, "_cdot_plot_cache")
    @printf("fit_dM run mode: %s\n", params.run_mode)
    @printf("Loading already-computed Cdot diagnostic artifact from %s\n", artifact_bson)

    blob = BSON.load(artifact_bson)
    selected_lib = dict_get(blob, :selected_lib)
    data_profiles = dict_get(blob, :data_profiles)
    generator_profiles = dict_get(blob, :generator_profiles)
    learned_profiles = dict_get(blob, :learned_profiles)
    exactstat_profiles = dict_get(blob, :exactstat_profiles)
    row_profiles = dict_get(blob, :row_profiles)
    reverse_profiles = dict_get(blob, :reverse_profiles)
    reverse_exactstat_profiles = dict_get(blob, :reverse_exactstat_profiles)
    diagnostic_metrics = copy(dict_get(blob, :diagnostic_metrics))
    score_diag = dict_get(blob, :score_diag)
    reverse_score_diag = dict_get(blob, :reverse_score_diag)

    sampler = load_state_tensor(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max, params.lag_stride, params.max_fit_lags)
    coord_mean = dict_haskey(blob, :coord_mean) ? dict_get(blob, :coord_mean) :
        compute_coordinate_means(sampler.states, sampler.start_idx)

    cache_ok = false
    phi_reverse_profiles = nothing
    phi_reverse_exactstat_profiles = nothing
    Phi = nothing
    Phi_exact = nothing
    phi_dsm_metrics = nothing
    phi_exact_metrics = nothing
    if isfile(cache_bson)
        cache = BSON.load(cache_bson)
        cache_ok = dict_haskey(cache, :cache_version) &&
            Int(dict_get(cache, :cache_version)) == 2 &&
            Int(dict_get(cache, :phi_fit_max_lag)) == params.phi_fit_max_lag
        if cache_ok
            @printf("Loading cached Phi Cdot curves from %s\n", cache_bson)
            phi_reverse_profiles = dict_get(cache, :phi_reverse_profiles)
            phi_reverse_exactstat_profiles = dict_get(cache, :phi_reverse_exactstat_profiles)
            Phi = dict_get(cache, :Phi)
            Phi_exact = dict_get(cache, :Phi_exact)
            phi_dsm_metrics = dict_get(cache, :phi_dsm_metrics)
            phi_exact_metrics = dict_get(cache, :phi_exact_metrics)
        end
    end

    if !cache_ok
        @printf("Cached Phi curves are missing or stale; computing only the missing Phi-action curves from saved score BSONs.\n")
        @printf("No score model is retrained in this mode.\n")
        # Score checkpoints are only loaded for inference when the Phi-action
        # curves are absent. No score training is performed in this mode.
        device = detect_device(params.device_name)
        @printf("Inference device: %s\n", describe_device(device))
        models = load_models(score_bson, joint_bson, device, sampler.K)
        require_condition(!isempty(reverse_bson) && isfile(reverse_bson),
            "cdot_plot_only requires [data].reverse_cond_score_bson for missing Phi curves.")
        reverse_model = load_reverse_conditional_model(reverse_bson, device, sampler.K)

        phi_est = estimate_stein_corrected_projected_phi(sampler, models, params, coord_mean, device)
        Phi = phi_est.Phi
        true_diag = dict_haskey(blob, :true_diag) ? dict_get(blob, :true_diag) :
            load_true_mean_mobility(input_hdf5)
        mean_d, mean_omega, _, _ = true_diag
        Phi_true_mean = true_mean_mobility_matrix(sampler.K, mean_d, mean_omega)
        potential_params = dict_haskey(blob, :potential_params) ? dict_get(blob, :potential_params) :
            load_potential_params(input_hdf5)
        V_exact_stationary = estimate_exact_stationary_stein_matrix(sampler, params, coord_mean, potential_params)
        V_exact_projected = matrix_from_block_profile(block_profile(V_exact_stationary, sampler.K))
        Phi_exact_full = phi_est.Phi_raw / V_exact_projected
        Phi_exact, _, _, _, _, _ = project_phi_from_profile(Phi_exact_full, sampler.K, params.phi_projection)
        phi_dsm_metrics = phi_agreement_metrics(Phi, Phi_true_mean)
        phi_exact_metrics = phi_agreement_metrics(Phi_exact, Phi_true_mean)
        @printf("Operational data-only Phi rel.RMSE vs stored <M_true> = %.6e\n",
            phi_dsm_metrics[:relative_rmse])
        @printf("Ex-post exact-score Phi rel.RMSE vs stored <M_true> = %.6e\n",
            phi_exact_metrics[:relative_rmse])

        Cdot_Phi_reverse = estimate_phi_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, Phi)
        Cdot_Phi_reverse_exact_stat = estimate_phi_cdot_from_reverse_conditional_score(sampler, models,
            reverse_model, selected_lib, params, device, Phi_exact; potential_params=potential_params)
        phi_reverse_profiles = translation_profiles(Cdot_Phi_reverse, sampler.K)
        phi_reverse_exactstat_profiles = translation_profiles(Cdot_Phi_reverse_exact_stat, sampler.K)
        cache_version = 2
        phi_fit_max_lag = params.phi_fit_max_lag
        BSON.@save cache_bson cache_version phi_fit_max_lag Phi Phi_exact phi_dsm_metrics phi_exact_metrics Cdot_Phi_reverse Cdot_Phi_reverse_exact_stat phi_reverse_profiles phi_reverse_exactstat_profiles
        @printf("Saved cached Phi Cdot curves to %s\n", cache_bson)
    end

    diagnostic_metrics[:phi_reverse_vs_generator] =
        true_mobility_agreement_metrics(generator_profiles, phi_reverse_profiles)
    diagnostic_metrics[:phi_reverse_exactstat_vs_generator] =
        true_mobility_agreement_metrics(generator_profiles, phi_reverse_exactstat_profiles)
    @printf("Phi reverse DSM-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
        diagnostic_metrics[:phi_reverse_vs_generator][:relative_rmse],
        diagnostic_metrics[:phi_reverse_vs_generator][:correlation])
    @printf("Phi reverse exact-score vs generator: rel.RMSE=%.6e, corr=%.6f\n",
        diagnostic_metrics[:phi_reverse_exactstat_vs_generator][:relative_rmse],
        diagnostic_metrics[:phi_reverse_exactstat_vs_generator][:correlation])

    render_true_mobility_cdot_figure(figure_png, sampler.lag_times, selected_lib, data_profiles,
        generator_profiles, learned_profiles, exactstat_profiles, row_profiles, diagnostic_metrics, score_diag;
        reverse_prof=reverse_profiles, reverse_exactstat_prof=reverse_exactstat_profiles,
        phi_reverse_prof=phi_reverse_profiles, phi_reverse_exactstat_prof=phi_reverse_exactstat_profiles,
        reverse_score_diag=reverse_score_diag)
    @printf("Saved regenerated Cdot diagnostic figure to %s\n", figure_png)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE
    params = load_params(param_file)
    if params.run_mode == "phi_constant_only"
        run_phi_constant_only(param_file)
    elseif params.run_mode == "cdot_plot_only"
        run_cdot_plot_only(param_file)
    elseif params.run_mode in ("mobility_only", "mobility_fit_only")
        run_mobility_stage_from_artifacts(param_file)
    else
        run_pipeline(param_file)
    end
end
