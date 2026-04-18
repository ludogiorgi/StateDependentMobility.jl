#!/usr/bin/env julia

# Agent summary:
# - Trains a full-matrix mobility network M_theta(x) on a configurable subset of
#   normalized A_{m,n}(tau) channels for the 2D affine-noise Langevin benchmark.
# - Automatically creates or reuses runs/run_### directories so each mobility run
#   keeps its config, figures, data products, and optional forward-validation outputs together.
# - Optionally runs the learned-full-M and Phi-only forward Langevin comparison stage
#   after fitting, writing its figures, metrics, diagnostics, and trajectories into the same run directory.

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
ensure_packages(["Flux", "BSON", "HDF5", "Plots", "CUDA", "Dierckx"])

using BSON
using CUDA
using Dierckx
using Flux
using HDF5
using LinearAlgebra
using Plots
using Printf
using Random
using SparseArrays
using Statistics
using TOML

gr()
default(fontfamily="DejaVu Sans", lw=2, dpi=180)

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "fit_dm.toml")
const DEFAULT_TRAIN_OBS_INDICES = (3, 4, 5)
const DEFAULT_TRAIN_COORD_INDICES = (1, 2)
const TRAINING_TARGET_SOURCE = "A_data = Gamma - Cdot_data"
const MAX_OBSERVABLE_TOTAL_DEGREE = 6
const MAX_FIT_PANEL_COLUMNS = 4

struct FitDMParams
    input_hdf5::String
    plain_score_bson::String
    joint_score_bson::String
    burnin_fraction::Float64
    tau_min::Float64
    force_recompute_phi::Bool
    lag_stride::Int
    use_all_observed_pairs::Bool
    cphi_spline_smoothing::Float64
    cphi_smoothing_rel_grid::Vector{Float64}
    cphi_smoothing_rmse_tolerance::Float64
    use_all_sphi_pairs::Bool
    use_all_a_pairs::Bool
    a_pairs_per_tau::Int
    score_batch_size::Int
    joint_batch_size::Int
    train_obs_indices::Vector{Int}
    train_coord_indices::Vector{Int}
    train_pairs::Vector{Tuple{Int, Int}}
    train_pair_weights::Vector{Float64}
    mobility_grid_nx::Int
    mobility_grid_ny::Int
    mobility_ridge::Float64
    grid_pad_fraction::Float64
    mobility_nn_pairs_per_tau::Int
    mobility_nn_tau_batch::Int
    mobility_nn_anchor_points::Int
    mobility_nn_epochs::Int
    mobility_nn_eval_every::Int
    mobility_nn_learning_rate::Float64
    mobility_nn_lag_weight_power::Float64
    mobility_nn_mean_penalty_weight::Float64
    mobility_nn_anchor_rms_penalty_weight::Float64
    mobility_nn_mean_penalty_final_scale::Float64
    mobility_nn_anchor_rms_penalty_final_scale::Float64
    mobility_nn_weight_decay::Float64
    mobility_nn_checkpoint_metric::String
    mobility_nn_validation_pair_seeds::Vector{Int}
    mobility_nn_model::String
    mobility_nn_reliability_mode::String
    mobility_nn_reliability_splits::Int
    mobility_nn_reliability_fraction::Float64
    mobility_nn_reliability_min_weight::Float64
    mobility_nn_reliability_strength::Float64
    mobility_nn_mean_entry_weights::Matrix{Float64}
    mobility_nn_anchor_rms_entry_weights::Matrix{Float64}
    mobility_nn_psd_jitter::Float64
    mobility_nn_scale_floor::Float64
    mobility_nn_widths::Vector{Int}
    figure_width::Int
    figure_height::Int
    output_a_png::String
    output_cphi_png::String
    output_mobility_png::String
    output_training_png::String
    output_mobility_bson::String
    output_metrics_txt::String
    output_phi_bson::String
    output_artifact_bson::String
end

struct ManagedRunPaths
    run_dir::String
    config_copy::String
    metadata_toml::String
    fit_stage_config::String
    forward_stage_config::String
    figure_dir::String
    data_dir::String
    fit_a_png::String
    fit_cphi_png::String
    fit_mobility_png::String
    fit_training_png::String
    fit_model_bson::String
    fit_metrics_txt::String
    fit_phi_bson::String
    fit_artifact_bson::String
    validation_stats_png::String
    validation_cphi_png::String
    validation_metrics_txt::String
    validation_diagnostics_bson::String
    validation_trajectories_hdf5::String
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
end

struct AffineModelMetadata
    omega::Float64
    B0::Matrix{Float64}
    B1::Matrix{Float64}
    B2::Matrix{Float64}
end

struct MobilityField
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    rgrid::Matrix{Float64}
    meta::AffineModelMetadata
end

struct ObservableContext
    mu_x::Float64
    mu_y::Float64
end

struct ObservableSpec
    label::String
    px::Int
    py::Int
end

struct SmoothingSelection
    smoothing::Float64
    rmse::Float64
    roughness::Float64
end

struct MobilityNNCache
    x0::Array{Float32, 3}
    scond::Array{Float32, 3}
    phi::Array{Float32, 3}
    grad::Array{Float32, 4}
    mean_x0::Matrix{Float32}
    taus::Vector{Float64}
end

struct MobilityNNHistory
    epochs::Vector{Int}
    train_loss::Vector{Float64}
    normalized_rmse::Vector{Float64}
    physical_rmse::Vector{Float64}
    mean_abs_delta::Vector{Float64}
    mean_delta11::Vector{Float64}
    mean_delta12::Vector{Float64}
    mean_delta21::Vector{Float64}
    mean_delta22::Vector{Float64}
    weight_l2::Vector{Float64}
end

struct RawAffineResidualMobilityModel
    affine
    residual
end

Flux.@layer RawAffineResidualMobilityModel

(model::RawAffineResidualMobilityModel)(x) = model.affine(x) .+ model.residual(x)

struct DirectEntryAffineMobilityModel
    delta_model
    phi_entries
end

Flux.@layer DirectEntryAffineMobilityModel trainable=(delta_model,)

function (model::DirectEntryAffineMobilityModel)(x)
    phi_entries = Flux.Zygote.dropgrad(reshape(model.phi_entries, :, 1))
    return model.delta_model(x) .+ phi_entries
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function validate_training_indices(indices::Vector{Int}, max_index::Int, name::String)
    require_condition(!isempty(indices), name * " must not be empty.")
    require_condition(all(idx -> 1 <= idx <= max_index, indices),
        name * @sprintf(" entries must lie in [1, %d].", max_index))
    require_condition(length(unique(indices)) == length(indices), name * " must not contain duplicates.")
    return nothing
end

function build_training_pairs(train_obs_indices::Vector{Int}, train_coord_indices::Vector{Int}, raw_pairs)
    if raw_pairs === nothing
        return Tuple{Int, Int}[(obs_idx, coord_idx) for obs_idx in train_obs_indices for coord_idx in train_coord_indices]
    end
    pairs = Tuple{Int, Int}[]
    for entry in raw_pairs
        require_condition(length(entry) == 2, "Each train_pairs entry must have exactly two indices.")
        push!(pairs, (Int(entry[1]), Int(entry[2])))
    end
    return pairs
end

function observable_index_from_label(label::AbstractString)
    idx = findfirst(==(String(label)), observable_labels())
    require_condition(!isnothing(idx),
        "Unknown observable label '" * String(label) * "'. Valid labels are: " * join(observable_labels(), ", "))
    return idx
end

function build_training_pairs(train_obs_indices::Vector{Int}, train_coord_indices::Vector{Int}, raw_pairs, raw_pair_labels)
    require_condition(raw_pairs === nothing || raw_pair_labels === nothing,
        "Specify only one of training_observables.train_pairs or training_observables.train_pairs_labels.")
    if raw_pair_labels !== nothing
        pairs = Tuple{Int, Int}[]
        for entry in raw_pair_labels
            require_condition(length(entry) == 2, "Each train_pairs_labels entry must have exactly two labels.")
            push!(pairs, (observable_index_from_label(entry[1]), observable_index_from_label(entry[2])))
        end
        return pairs
    end
    return build_training_pairs(train_obs_indices, train_coord_indices, raw_pairs)
end

function validate_training_pairs(pairs::Vector{Tuple{Int, Int}}, max_index::Int, name::String)
    require_condition(!isempty(pairs), name * " must not be empty.")
    require_condition(all(pair -> 1 <= pair[1] <= max_index && 1 <= pair[2] <= max_index, pairs),
        name * @sprintf(" entries must lie in [1, %d] x [1, %d].", max_index, max_index))
    require_condition(length(unique(pairs)) == length(pairs), name * " must not contain duplicates.")
    return nothing
end

function validate_training_weights(weights::Vector{Float64}, npairs::Int)
    require_condition(length(weights) == npairs,
        @sprintf("train_pair_weights must have the same length as train_pairs (got %d weights for %d pairs).",
            length(weights), npairs))
    require_condition(all(weight -> weight > 0.0, weights), "train_pair_weights entries must be positive.")
    return nothing
end

function normalized_channel_instability(split_targets::Array{Float64, 3}, target_scale::Vector{Float64})
    split_std = size(split_targets, 3) > 1 ? std(split_targets; dims=3, corrected=true)[:, :, 1] : zeros(Float64, size(split_targets, 1), size(split_targets, 2))
    out = Vector{Float64}(undef, size(split_targets, 1))
    for channel_idx in 1:size(split_targets, 1)
        out[channel_idx] = sqrt(mean((@view(split_std[channel_idx, :]) ./ target_scale[channel_idx]) .^ 2))
    end
    return out
end

function reliability_weights_from_instability(instability::Vector{Float64}, min_weight::Float64, strength::Float64)
    weights = 1.0 ./ (1.0 .+ instability)
    weights .= max.(weights, min_weight)
    if strength < 1.0
        weights .= weights .^ strength
    end
    weights ./= mean(weights)
    return weights
end

function parse_penalty_entry_weights(raw_value, name::String)
    rows = raw_value === nothing ? [[1.0, 1.0], [1.0, 1.0]] : raw_value
    require_condition(length(rows) == 2 && all(length(row) == 2 for row in rows), name * " must be a 2x2 array.")
    weights = Matrix{Float64}(undef, 2, 2)
    for i in 1:2, j in 1:2
        weights[i, j] = Float64(rows[i][j])
    end
    require_condition(all(x -> x >= 0.0, weights), name * " must contain nonnegative entries.")
    require_condition(sum(weights) > 0.0, name * " must contain at least one positive entry.")
    return weights
end

function unique_pair_indices(pairs::Vector{Tuple{Int, Int}})
    return unique(first.(pairs)), unique(last.(pairs))
end

function active_observable_subset(train_pairs::Vector{Tuple{Int, Int}})
    active_indices = sort!(unique(vcat(first.(train_pairs), last.(train_pairs))))
    global_to_local = Dict{Int, Int}(obs_idx => local_idx for (local_idx, obs_idx) in enumerate(active_indices))
    local_pairs = Tuple{Int, Int}[(global_to_local[m], global_to_local[n]) for (m, n) in train_pairs]
    local_labels = observable_labels()[active_indices]
    return active_indices, local_pairs, local_labels
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data_cfg = raw["data"]
    eval_cfg = raw["evaluation"]
    obs_cfg = get(raw, "training_observables", Dict{String, Any}())
    mobility_cfg = raw["mobility"]
    mobility_nn_cfg = raw["mobility_nn"]
    fig_cfg = raw["figure"]
    out_cfg = raw["output"]
    raw_train_obs_indices = Int.(get(obs_cfg, "train_obs_indices", collect(DEFAULT_TRAIN_OBS_INDICES)))
    raw_train_coord_indices = Int.(get(obs_cfg, "train_coord_indices", collect(DEFAULT_TRAIN_COORD_INDICES)))
    raw_train_pairs = get(obs_cfg, "train_pairs", nothing)
    raw_train_pair_labels = get(obs_cfg, "train_pairs_labels", nothing)
    train_pairs = build_training_pairs(raw_train_obs_indices, raw_train_coord_indices, raw_train_pairs, raw_train_pair_labels)
    if raw_train_pair_labels !== nothing
        raw_train_obs_indices, raw_train_coord_indices = unique_pair_indices(train_pairs)
    end
    raw_train_pair_weights = Float64.(get(obs_cfg, "train_pair_weights", ones(length(train_pairs))))
    mean_entry_weights = parse_penalty_entry_weights(get(mobility_nn_cfg, "mean_entry_weights", nothing), "mobility_nn.mean_entry_weights")
    anchor_rms_entry_weights = parse_penalty_entry_weights(get(mobility_nn_cfg, "anchor_rms_entry_weights", nothing), "mobility_nn.anchor_rms_entry_weights")

    params = FitDMParams(
        String(data_cfg["input_hdf5"]),
        String(data_cfg["plain_score_bson"]),
        String(data_cfg["joint_score_bson"]),
        Float64(data_cfg["burnin_fraction"]),
        Float64(data_cfg["tau_min"]),
        Bool(get(data_cfg, "force_recompute_phi", false)),
        Int(eval_cfg["lag_stride"]),
        Bool(get(eval_cfg, "use_all_observed_pairs", true)),
        Float64(eval_cfg["cphi_spline_smoothing"]),
        Float64.(get(eval_cfg, "cphi_smoothing_rel_grid", [0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0])),
        Float64(get(eval_cfg, "cphi_smoothing_rmse_tolerance", 0.02)),
        Bool(get(eval_cfg, "use_all_sphi_pairs", true)),
        Bool(get(eval_cfg, "use_all_a_pairs", true)),
        Int(get(eval_cfg, "a_pairs_per_tau", 200000)),
        Int(eval_cfg["score_batch_size"]),
        Int(eval_cfg["joint_batch_size"]),
        raw_train_obs_indices,
        raw_train_coord_indices,
        train_pairs,
        raw_train_pair_weights,
        Int(get(mobility_cfg, "grid_nx", 160)),
        Int(get(mobility_cfg, "grid_ny", 160)),
        Float64(get(mobility_cfg, "ridge", 1.0e-6)),
        Float64(get(mobility_cfg, "grid_pad_fraction", 0.05)),
        Int(mobility_nn_cfg["pairs_per_tau"]),
        Int(mobility_nn_cfg["tau_batch_size"]),
        Int(get(mobility_nn_cfg, "anchor_points", 8192)),
        Int(mobility_nn_cfg["epochs"]),
        Int(mobility_nn_cfg["eval_every"]),
        Float64(mobility_nn_cfg["learning_rate"]),
        Float64(get(mobility_nn_cfg, "lag_weight_power", 0.0)),
        Float64(get(mobility_nn_cfg, "mean_penalty_weight", 0.0)),
        Float64(get(mobility_nn_cfg, "anchor_rms_penalty_weight", 0.0)),
        Float64(get(mobility_nn_cfg, "mean_penalty_final_scale", 1.0)),
        Float64(get(mobility_nn_cfg, "anchor_rms_penalty_final_scale", 1.0)),
        Float64(get(mobility_nn_cfg, "weight_decay", 0.0)),
        String(get(mobility_nn_cfg, "checkpoint_metric", "normalized_mse")),
        Int.(get(mobility_nn_cfg, "validation_pair_seeds", [20260421])),
        String(get(mobility_nn_cfg, "model", "mlp")),
        String(get(mobility_nn_cfg, "reliability_weight_mode", "none")),
        Int(get(mobility_nn_cfg, "reliability_splits", 4)),
        Float64(get(mobility_nn_cfg, "reliability_fraction", 0.5)),
        Float64(get(mobility_nn_cfg, "reliability_min_weight", 0.25)),
        Float64(get(mobility_nn_cfg, "reliability_strength", 1.0)),
        mean_entry_weights,
        anchor_rms_entry_weights,
        Float64(get(mobility_nn_cfg, "psd_jitter", 1.0e-5)),
        Float64(get(mobility_nn_cfg, "scale_floor", 1.0e-4)),
        Int.(mobility_nn_cfg["widths"]),
        Int(fig_cfg["width"]),
        Int(fig_cfg["height"]),
        String(out_cfg["a_png"]),
        String(out_cfg["cphi_png"]),
        String(out_cfg["mobility_png"]),
        String(out_cfg["training_png"]),
        String(out_cfg["mobility_bson"]),
        String(out_cfg["metrics_txt"]),
        String(out_cfg["phi_bson"]),
        String(out_cfg["artifact_bson"]),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0,1).")
    require_condition(params.tau_min > 0.0, "tau_min must be positive.")
    require_condition(params.lag_stride >= 1, "lag_stride must be >= 1.")
    require_condition(params.cphi_spline_smoothing >= 0.0, "cphi_spline_smoothing must be nonnegative.")
    require_condition(!isempty(params.cphi_smoothing_rel_grid), "cphi_smoothing_rel_grid must not be empty.")
    require_condition(all(x -> x >= 0.0, params.cphi_smoothing_rel_grid), "cphi_smoothing_rel_grid must contain nonnegative entries.")
    require_condition(params.cphi_smoothing_rmse_tolerance >= 0.0, "cphi_smoothing_rmse_tolerance must be nonnegative.")
    require_condition(params.a_pairs_per_tau >= 100, "a_pairs_per_tau must be >= 100.")
    require_condition(params.score_batch_size >= 16, "score_batch_size must be >= 16.")
    require_condition(params.joint_batch_size >= 16, "joint_batch_size must be >= 16.")
    validate_training_indices(params.train_obs_indices, length(observable_labels()), "train_obs_indices")
    validate_training_indices(params.train_coord_indices, length(observable_labels()), "train_coord_indices")
    validate_training_pairs(params.train_pairs, length(observable_labels()), "train_pairs")
    validate_training_weights(params.train_pair_weights, length(params.train_pairs))
    require_condition(params.mobility_grid_nx >= 8 && params.mobility_grid_ny >= 8, "mobility grid is too small.")
    require_condition(params.mobility_ridge > 0.0, "mobility ridge must be positive.")
    require_condition(params.grid_pad_fraction >= 0.0, "grid_pad_fraction must be nonnegative.")
    require_condition(params.mobility_nn_pairs_per_tau >= 128, "mobility_nn_pairs_per_tau must be >= 128.")
    require_condition(params.mobility_nn_tau_batch >= 1, "mobility_nn_tau_batch must be >= 1.")
    require_condition(params.mobility_nn_anchor_points >= 256, "mobility_nn_anchor_points must be >= 256.")
    require_condition(params.mobility_nn_epochs >= 1, "mobility_nn_epochs must be >= 1.")
    require_condition(params.mobility_nn_eval_every >= 1, "mobility_nn_eval_every must be >= 1.")
    require_condition(params.mobility_nn_learning_rate > 0.0, "mobility_nn_learning_rate must be positive.")
    require_condition(params.mobility_nn_lag_weight_power >= 0.0, "mobility_nn_lag_weight_power must be nonnegative.")
    require_condition(params.mobility_nn_mean_penalty_weight >= 0.0, "mobility_nn_mean_penalty_weight must be nonnegative.")
    require_condition(params.mobility_nn_anchor_rms_penalty_weight >= 0.0, "mobility_nn_anchor_rms_penalty_weight must be nonnegative.")
    require_condition(params.mobility_nn_mean_penalty_final_scale >= 0.0, "mobility_nn_mean_penalty_final_scale must be nonnegative.")
    require_condition(params.mobility_nn_anchor_rms_penalty_final_scale >= 0.0, "mobility_nn_anchor_rms_penalty_final_scale must be nonnegative.")
    require_condition(params.mobility_nn_weight_decay >= 0.0, "mobility_nn_weight_decay must be nonnegative.")
    require_condition(params.mobility_nn_checkpoint_metric in ("normalized_mse", "regularized_objective"),
        "mobility_nn.checkpoint_metric must be one of: normalized_mse, regularized_objective.")
    require_condition(!isempty(params.mobility_nn_validation_pair_seeds), "mobility_nn.validation_pair_seeds must not be empty.")
    require_condition(length(unique(params.mobility_nn_validation_pair_seeds)) == length(params.mobility_nn_validation_pair_seeds),
        "mobility_nn.validation_pair_seeds must not contain duplicates.")
    require_condition(params.mobility_nn_model in ("mlp", "affine_skip", "entry_affine"),
        "mobility_nn.model must be one of: mlp, affine_skip, entry_affine.")
    require_condition(params.mobility_nn_reliability_mode in ("none", "split_half"),
        "mobility_nn.reliability_weight_mode must be one of: none, split_half.")
    require_condition(params.mobility_nn_reliability_splits >= 2, "mobility_nn.reliability_splits must be at least 2.")
    require_condition(0.0 < params.mobility_nn_reliability_fraction <= 1.0,
        "mobility_nn.reliability_fraction must lie in (0, 1].")
    require_condition(0.0 < params.mobility_nn_reliability_min_weight <= 1.0,
        "mobility_nn.reliability_min_weight must lie in (0, 1].")
    require_condition(0.0 <= params.mobility_nn_reliability_strength <= 1.0,
        "mobility_nn.reliability_strength must lie in [0, 1].")
    require_condition(params.mobility_nn_psd_jitter > 0.0, "mobility_nn_psd_jitter must be positive.")
    require_condition(params.mobility_nn_scale_floor > 0.0, "mobility_nn_scale_floor must be positive.")
    require_condition(all(width -> width >= 1, params.mobility_nn_widths), "mobility_nn_widths entries must be positive.")
    require_condition(params.figure_width >= 1200 && params.figure_height >= 800, "figure dimensions are too small.")
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

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function load_state_tensor(path::AbstractString)
    times = h5read(path, "/trajectories/time")
    states = h5read(path, "/trajectories/states")
    require_condition(ndims(states) == 3, "Expected /trajectories/states to be rank-3.")
    require_condition(size(states, 2) == 2, "Expected state dimension 2.")
    require_condition(size(states, 1) == length(times), "State tensor does not match time axis.")
    return times, states
end

function build_pair_sampler(path::AbstractString, burnin_fraction::Float64, tau_min::Float64, lag_stride::Int)
    times, states = load_state_tensor(path)
    t_decorrelation = read(h5open(path, "r")["/statistics/correlations/t_decorrelation"])
    save_dt = length(times) > 1 ? (times[2] - times[1]) : 0.0
    require_condition(save_dt > 0.0, "Saved dt must be positive.")
    start_idx = burnin_start_index(length(times), burnin_fraction)
    tau_max = min(t_decorrelation, times[end] - times[start_idx])
    require_condition(tau_max >= tau_min, "tau_min exceeds available lag range.")
    min_lag = max(1, ceil(Int, tau_min / save_dt - 1e-9))
    max_lag = min(length(times) - start_idx, floor(Int, tau_max / save_dt + 1e-9))
    lag_steps = collect(min_lag:lag_stride:max_lag)
    require_condition(!isempty(lag_steps), "No lag steps remain after applying lag_stride.")
    lag_times = lag_steps .* save_dt
    return PairSampler(states, times, start_idx, lag_steps, lag_times)
end

function load_affine_model_metadata(path::AbstractString)
    h5open(path, "r") do file
        require_condition(haskey(file, "/metadata/B0"), "Expected affine-noise metadata /metadata/B0 in the simulation file.")
        require_condition(haskey(file, "/metadata/B1"), "Expected affine-noise metadata /metadata/B1 in the simulation file.")
        require_condition(haskey(file, "/metadata/B2"), "Expected affine-noise metadata /metadata/B2 in the simulation file.")
        omega = haskey(file, "/metadata/omega") ? read(file["/metadata/omega"]) : 0.35
        B0 = Matrix{Float64}(read(file["/metadata/B0"]))
        B1 = Matrix{Float64}(read(file["/metadata/B1"]))
        B2 = Matrix{Float64}(read(file["/metadata/B2"]))
        return AffineModelMetadata(Float64(omega), B0, B1, B2)
    end
end

function load_models(plain_path::AbstractString, joint_path::AbstractString, device::DeviceConfig)
    plain_data = BSON.load(plain_path)
    joint_data = BSON.load(joint_path)
    plain_model = to_device(plain_data[:host_model], device)
    joint_model = to_device(joint_data[:host_model], device)
    joint_meta = joint_data[:metadata]
    return plain_model, joint_model, joint_meta
end

function build_observable_context(states::Array{Float64, 3}, start_idx::Int)
    return ObservableContext(
        mean(@view states[start_idx:end, 1, :]),
        mean(@view states[start_idx:end, 2, :]),
    )
end

function monomial_label(px::Int, py::Int)
    parts = String[]
    if px > 0
        push!(parts, px == 1 ? "x" : "x^" * string(px))
    end
    if py > 0
        push!(parts, py == 1 ? "y" : "y^" * string(py))
    end
    return join(parts, "")
end

function build_observable_specs(max_degree::Int)
    specs = ObservableSpec[]
    for degree in 1:max_degree
        for py in 0:degree
            px = degree - py
            push!(specs, ObservableSpec(monomial_label(px, py), px, py))
        end
    end
    return specs
end

const OBSERVABLE_SPECS = build_observable_specs(MAX_OBSERVABLE_TOTAL_DEGREE)
const OBSERVABLE_LABELS = [spec.label for spec in OBSERVABLE_SPECS]

function observable_labels()
    return OBSERVABLE_LABELS
end

function observable_basis(x::AbstractVector{Float64}, y::AbstractVector{Float64},
        obs_indices::AbstractVector{Int}=collect(1:length(OBSERVABLE_SPECS)))
    n = length(x)
    max_degree = MAX_OBSERVABLE_TOTAL_DEGREE
    xpow = Matrix{Float64}(undef, max_degree + 1, n)
    ypow = Matrix{Float64}(undef, max_degree + 1, n)
    @views xpow[1, :] .= 1.0
    @views ypow[1, :] .= 1.0
    @inbounds for degree in 1:max_degree
        @views xpow[degree + 1, :] .= xpow[degree, :] .* x
        @views ypow[degree + 1, :] .= ypow[degree, :] .* y
    end
    basis = Matrix{Float64}(undef, length(obs_indices), n)
    @inbounds for (local_idx, obs_idx) in enumerate(obs_indices)
        spec = OBSERVABLE_SPECS[obs_idx]
        @views basis[local_idx, :] .= xpow[spec.px + 1, :] .* ypow[spec.py + 1, :]
    end
    return basis
end

function observable_values(index::Int, x, y, ctx::ObservableContext)
    require_condition(1 <= index <= length(OBSERVABLE_SPECS), "Unsupported observable index: $(index)")
    spec = OBSERVABLE_SPECS[index]
    xpart = spec.px == 0 ? one.(x) : x .^ spec.px
    ypart = spec.py == 0 ? one.(y) : y .^ spec.py
    return xpart .* ypart
end

function observable_grad_hess(index::Int, x::AbstractVector{Float64}, y::AbstractVector{Float64}, ctx::ObservableContext)
    n = length(x)
    grad = Matrix{Float64}(undef, 2, n)
    h11 = Vector{Float64}(undef, n)
    h12 = Vector{Float64}(undef, n)
    h22 = Vector{Float64}(undef, n)
    require_condition(1 <= index <= length(OBSERVABLE_SPECS), "Unsupported observable index: $(index)")
    spec = OBSERVABLE_SPECS[index]
    px = spec.px
    py = spec.py
    @inbounds for idx in 1:n
        xv = x[idx]
        yv = y[idx]
        grad[1, idx] = px == 0 ? 0.0 : px * xv^(px - 1) * yv^py
        grad[2, idx] = py == 0 ? 0.0 : py * xv^px * yv^(py - 1)
        h11[idx] = px <= 1 ? 0.0 : px * (px - 1) * xv^(px - 2) * yv^py
        h12[idx] = (px == 0 || py == 0) ? 0.0 : px * py * xv^(px - 1) * yv^(py - 1)
        h22[idx] = py <= 1 ? 0.0 : py * (py - 1) * xv^px * yv^(py - 2)
    end
    return grad, h11, h12, h22
end

function state_domain(states::Array{Float64, 3}, start_idx::Int, pad_fraction::Float64)
    x = vec(@view states[start_idx:end, 1, :])
    y = vec(@view states[start_idx:end, 2, :])
    xmin, xmax = minimum(x), maximum(x)
    ymin, ymax = minimum(y), maximum(y)
    xpad = max((xmax - xmin) * pad_fraction, 1e-3)
    ypad = max((ymax - ymin) * pad_fraction, 1e-3)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)
end

function load_or_estimate_phi_from_data(path::AbstractString, force_recompute::Bool,
        states::Array{Float64, 3}, times::Vector{Float64}, start_idx::Int)
    if !force_recompute && isfile(path)
        data = BSON.load(path)
        if get(data, :phi_source, "") == "observed_short_lag_coordinate_correlation_v3_notranspose"
            @printf("Loaded Phi estimate from %s\n", path)
            return Matrix{Float64}(data[:phi_est])
        end
    end
    save_dt = length(times) > 1 ? (times[2] - times[1]) : 0.0
    require_condition(save_dt > 0.0, "Need at least two saved times to estimate Phi.")
    max_fit_time = min(0.1, times[end] - times[start_idx])
    max_lag = max(3, round(Int, max_fit_time / save_dt))
    max_lag = min(max_lag, size(states, 1) - start_idx - 1)
    require_condition(max_lag >= 1, "Not enough short-lag samples to estimate Phi.")

    lags = collect(0:max_lag)
    cphi = estimate_observed_cphi(states, start_idx, lags, ObservableContext(0.0, 0.0), true, 0, MersenneTwister(0), [1, 2])
    tfit = save_dt .* lags
    degree = min(3, length(tfit) - 1)
    X = hcat([tfit .^ k for k in 0:degree]...)
    cdot0_est = zeros(Float64, 2, 2)
    for m in 1:2, n in 1:2
        β = X \ vec(@view cphi[:, m, n])
        cdot0_est[m, n] = β[2]
    end
    phi_est = -cdot0_est
    bson(path, Dict{Symbol, Any}(
        :phi_source => "observed_short_lag_coordinate_correlation_v3_notranspose",
        :phi_est => phi_est,
    ))
    @printf("Saved Phi estimate from observed short-lag correlations to %s\n", path)
    return phi_est
end

function evaluate_plain_score(model, points::AbstractMatrix{Float32}, batch_size::Int, device::DeviceConfig)
    out = Matrix{Float32}(undef, 2, size(points, 2))
    for start in 1:batch_size:size(points, 2)
        stop = min(start + batch_size - 1, size(points, 2))
        batch = @view points[:, start:stop]
        out[:, start:stop] .= Float32.(to_host(model(to_device(batch, device))))
    end
    return out
end

function evaluate_conditional_score(plain_model, joint_model, x0::AbstractMatrix{Float32}, xt::AbstractMatrix{Float32}, tnorm::Float32,
        score_batch_size::Int, joint_batch_size::Int, device::DeviceConfig)
    n = size(x0, 2)
    stat_score = evaluate_plain_score(plain_model, x0, score_batch_size, device)
    cond = Matrix{Float32}(undef, 2, n)
    for start in 1:joint_batch_size:n
        stop = min(start + joint_batch_size - 1, n)
        batch_n = stop - start + 1
        joint_input = Matrix{Float32}(undef, 5, batch_n)
        @views begin
            joint_input[1:2, :] .= x0[:, start:stop]
            joint_input[3:4, :] .= xt[:, start:stop]
        end
        joint_input[5, :] .= tnorm
        joint_score = Float32.(to_host(joint_model(to_device(joint_input, device))))
        cond[:, start:stop] .= joint_score[1:2, :] .- stat_score[:, start:stop]
    end
    return cond
end

function normalize_tau(tau::Float64, joint_meta)
    tau_min_model = Float64(joint_meta[:tau_min])
    tau_max_model = Float64(joint_meta[:tau_max])
    require_condition(tau_min_model <= tau <= tau_max_model + 1e-10, "Requested tau lies outside the joint-score training range.")
    denom = max(tau_max_model - tau_min_model, eps())
    return Float32((tau - tau_min_model) / denom)
end

function sample_pairs(states::Array{Float64, 3}, start_idx::Int, lag::Int, npairs::Int, rng::AbstractRNG)
    nt, _, ntraj = size(states)
    upper = nt - lag
    require_condition(upper >= start_idx, "Lag exceeds post-burnin window.")
    x0 = Matrix{Float32}(undef, 2, npairs)
    xt = Matrix{Float32}(undef, 2, npairs)
    @inbounds for n in 1:npairs
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, start_idx:upper)
        x0[1, n] = Float32(states[time_idx, 1, traj_idx])
        x0[2, n] = Float32(states[time_idx, 2, traj_idx])
        xt[1, n] = Float32(states[time_idx + lag, 1, traj_idx])
        xt[2, n] = Float32(states[time_idx + lag, 2, traj_idx])
    end
    return x0, xt
end

function mean_product(a, b)
    total = 0.0
    n = length(a)
    @inbounds @simd for idx in eachindex(a, b)
        total += a[idx] * b[idx]
    end
    return total / n
end

function estimate_observed_cphi(states::Array{Float64, 3}, start_idx::Int, lag_steps::Vector{Int},
        obs_ctx::ObservableContext, use_all_pairs::Bool, npairs::Int, rng::AbstractRNG,
        obs_indices::AbstractVector{Int})
    nt, _, _ = size(states)
    nobs = length(obs_indices)
    cphi = zeros(Float64, length(lag_steps), nobs, nobs)

    if use_all_pairs
        Threads.@threads for lag_idx in eachindex(lag_steps)
            lag = lag_steps[lag_idx]
            upper = nt - lag
            @views begin
                x0x = Float64.(vec(states[start_idx:upper, 1, :]))
                x0y = Float64.(vec(states[start_idx:upper, 2, :]))
                xtx = Float64.(vec(states[(start_idx + lag):nt, 1, :]))
                xty = Float64.(vec(states[(start_idx + lag):nt, 2, :]))
            end
            obs0 = observable_basis(x0x, x0y, obs_indices)
            obst = observable_basis(xtx, xty, obs_indices)
            cphi[lag_idx, :, :] .= (obst * obs0') ./ length(x0x)
        end
        return cphi
    end

    for (lag_idx, lag) in enumerate(lag_steps)
        x0, xt = sample_pairs(states, start_idx, lag, npairs, rng)
        obs0 = observable_basis(Float64.(x0[1, :]), Float64.(x0[2, :]), obs_indices)
        obst = observable_basis(Float64.(xt[1, :]), Float64.(xt[2, :]), obs_indices)
        cphi[lag_idx, :, :] .= (obst * obs0') ./ size(x0, 2)
    end
    return cphi
end

function for_each_all_pairs_batch(fn!::F, states::Array{Float64, 3}, start_idx::Int, lag::Int, batch_pairs::Int) where {F}
    nt, _, ntraj = size(states)
    upper = nt - lag
    require_condition(upper >= start_idx, "Lag exceeds post-burnin window.")
    require_condition(batch_pairs >= 1, "batch_pairs must be positive.")
    x0 = Matrix{Float32}(undef, 2, batch_pairs)
    xt = Matrix{Float32}(undef, 2, batch_pairs)
    cursor = 0
    total = 0
    @inbounds for traj_idx in 1:ntraj
        for time_idx in start_idx:upper
            cursor += 1
            x0[1, cursor] = Float32(states[time_idx, 1, traj_idx])
            x0[2, cursor] = Float32(states[time_idx, 2, traj_idx])
            xt[1, cursor] = Float32(states[time_idx + lag, 1, traj_idx])
            xt[2, cursor] = Float32(states[time_idx + lag, 2, traj_idx])
            if cursor == batch_pairs
                fn!(@view(x0[:, 1:cursor]), @view(xt[:, 1:cursor]))
                total += cursor
                cursor = 0
            end
        end
    end
    if cursor > 0
        fn!(@view(x0[:, 1:cursor]), @view(xt[:, 1:cursor]))
        total += cursor
    end
    return total
end

function estimate_gamma_term(states::Array{Float64, 3}, start_idx::Int, lag_steps::Vector{Int},
        obs_ctx::ObservableContext, phi_est::Matrix{Float64}, use_all_pairs::Bool, npairs::Int, rng::AbstractRNG,
        plain_model, score_batch_size::Int, device::DeviceConfig, obs_indices::AbstractVector{Int})
    nobs = length(obs_indices)
    gamma_term = zeros(Float64, length(lag_steps), nobs, nobs)
    phi11 = phi_est[1, 1]
    phi12 = phi_est[1, 2]
    phi21 = phi_est[2, 1]
    phi22 = phi_est[2, 2]
    for (lag_idx, lag) in enumerate(lag_steps)
        if lag_idx == 1 || lag_idx % 25 == 0 || lag_idx == length(lag_steps)
            @printf("Estimating gamma term %d / %d\n", lag_idx, length(lag_steps))
        end
        if use_all_pairs
            sums = zeros(Float64, nobs, nobs)
            total = for_each_all_pairs_batch(states, start_idx, lag, score_batch_size) do x0_batch, xt_batch
                stat_score = evaluate_plain_score(plain_model, x0_batch, score_batch_size, device)
                x0x = Float64.(x0_batch[1, :])
                x0y = Float64.(x0_batch[2, :])
                xtx = Float64.(xt_batch[1, :])
                xty = Float64.(xt_batch[2, :])
                s1 = Float64.(stat_score[1, :])
                s2 = Float64.(stat_score[2, :])
                obst = observable_basis(xtx, xty, obs_indices)
                gamma = Matrix{Float64}(undef, nobs, length(s1))
                for (local_n, global_n) in enumerate(obs_indices)
                    grad, h11, h12, h22 = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                    g1 = @view grad[1, :]
                    g2 = @view grad[2, :]
                    u1 = phi11 .* g1 .+ phi12 .* g2
                    u2 = phi21 .* g1 .+ phi22 .* g2
                    hess_trace = phi11 .* h11 .+ (phi12 .+ phi21) .* h12 .+ phi22 .* h22
                    gamma[local_n, :] .= hess_trace .+ u1 .* s1 .+ u2 .* s2
                end
                sums .+= obst * gamma'
            end
            gamma_term[lag_idx, :, :] .= sums ./ total
        else
            x0, xt = sample_pairs(states, start_idx, lag, npairs, rng)
            stat_score = evaluate_plain_score(plain_model, x0, score_batch_size, device)
            x0x = Float64.(x0[1, :])
            x0y = Float64.(x0[2, :])
            xtx = Float64.(xt[1, :])
            xty = Float64.(xt[2, :])
            s1 = Float64.(stat_score[1, :])
            s2 = Float64.(stat_score[2, :])
            obst = observable_basis(xtx, xty, obs_indices)
            gamma = Matrix{Float64}(undef, nobs, size(x0, 2))
            for (local_n, global_n) in enumerate(obs_indices)
                grad, h11, h12, h22 = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                g1 = @view grad[1, :]
                g2 = @view grad[2, :]
                u1 = phi11 .* g1 .+ phi12 .* g2
                u2 = phi21 .* g1 .+ phi22 .* g2
                hess_trace = phi11 .* h11 .+ (phi12 .+ phi21) .* h12 .+ phi22 .* h22
                gamma[local_n, :] .= hess_trace .+ u1 .* s1 .+ u2 .* s2
            end
            gamma_term[lag_idx, :, :] .= (obst * gamma') ./ size(x0, 2)
        end
    end
    return gamma_term
end

function central_time_derivative(values::Matrix{Float64}, dt::Float64)
    n, m = size(values)
    out = similar(values)
    require_condition(n >= 2, "Need at least two lag points to differentiate.")
    if n == 2
        out[1, :] .= (values[2, :] .- values[1, :]) ./ dt
        out[2, :] .= out[1, :]
        return out
    end
    out[1, :] .= (-3 .* values[1, :] .+ 4 .* values[2, :] .- values[3, :]) ./ (2dt)
    for i in 2:(n - 1)
        out[i, :] .= (values[i + 1, :] .- values[i - 1, :]) ./ (2dt)
    end
    out[n, :] .= (3 .* values[n, :] .- 4 .* values[n - 1, :] .+ values[n - 2, :]) ./ (2dt)
    return out
end

function smooth_cphi_and_derivative(taus::Vector{Float64}, values::Vector{Float64}, smoothing::Float64)
    n = length(taus)
    require_condition(length(values) == n, "Spline inputs must have matching lengths.")
    if n < 4 || smoothing == 0.0
        return copy(values), vec(central_time_derivative(reshape(values, n, 1), taus[2] - taus[1]))
    end
    for scale in (1.0, 10.0, 100.0, 1000.0)
        try
            spline = Spline1D(taus, values; k=min(3, n - 1), s=smoothing * scale)
            smoothed = evaluate(spline, taus)
            deriv = derivative(spline, taus; nu=1)
            return Float64.(smoothed), Float64.(deriv)
        catch err
            if !(err isa ErrorException)
                rethrow()
            end
        end
    end
    return copy(values), vec(central_time_derivative(reshape(values, n, 1), taus[2] - taus[1]))
end

function derivative_roughness(values::Vector{Float64})
    return length(values) <= 1 ? 0.0 : sqrt(mean(diff(values) .^ 2))
end

function smoothing_candidates(values::Vector{Float64}, fixed_smoothing::Float64, rel_grid::Vector{Float64})
    centered = values .- mean(values)
    scale = max(sum(centered .^ 2), eps(Float64))
    return unique(sort(vcat([fixed_smoothing], scale .* rel_grid)))
end

function refine_smoothing_candidates(candidates::Vector{Float64}, choices::Vector{SmoothingSelection})
    best_idx = argmin(choice.rmse for choice in choices)
    best_s = choices[best_idx].smoothing
    best_s > 0.0 || return candidates

    positive = sort(filter(>(0.0), candidates))
    pos_idx = findfirst(==(best_s), positive)
    pos_idx === nothing && return candidates
    left = pos_idx == 1 ? best_s / 10.0 : positive[pos_idx - 1]
    right = pos_idx == length(positive) ? best_s * 10.0 : positive[pos_idx + 1]
    left = max(left, best_s / 100.0, eps(Float64))
    right = max(right, left * 1.01)
    refined = exp.(range(log(left), log(right), length=9))
    return unique(sort(vcat(candidates, refined)))
end

function select_smoothing_protocol_dataonly(taus::Vector{Float64}, values::Vector{Float64},
        fixed_smoothing::Float64, rel_grid::Vector{Float64}, fit_tolerance::Float64)
    candidates = smoothing_candidates(values, fixed_smoothing, rel_grid)
    choices = SmoothingSelection[]
    smooth_store = Dict{Float64, Vector{Float64}}()
    deriv_store = Dict{Float64, Vector{Float64}}()

    function evaluate_candidates!(candidate_values::Vector{Float64})
        for smoothing in candidate_values
            haskey(smooth_store, smoothing) && continue
            smoothed, deriv = smooth_cphi_and_derivative(taus, values, smoothing)
            fit_rmse = sqrt(mean((smoothed .- values) .^ 2))
            smooth_store[smoothing] = smoothed
            deriv_store[smoothing] = deriv
            push!(choices, SmoothingSelection(
                smoothing,
                fit_rmse,
                derivative_roughness(deriv),
            ))
        end
    end

    evaluate_candidates!(candidates)
    candidates = refine_smoothing_candidates(candidates, choices)
    evaluate_candidates!(candidates)
    sort!(choices; by=choice -> choice.smoothing)

    best_fit = minimum(choice.rmse for choice in choices)
    cutoff = best_fit * (1.0 + fit_tolerance) + 1e-12
    eligible = findall(choice -> choice.rmse <= cutoff, choices)
    selected_idx = eligible[1]
    for idx in eligible[2:end]
        if choices[idx].roughness < choices[selected_idx].roughness - 1e-12 ||
                (isapprox(choices[idx].roughness, choices[selected_idx].roughness; atol=1e-12) &&
                 choices[idx].smoothing > choices[selected_idx].smoothing)
            selected_idx = idx
        end
    end
    selected_s = choices[selected_idx].smoothing
    return smooth_store[selected_s], deriv_store[selected_s], choices[selected_idx]
end

function finite_difference_axis(data::Matrix{Float64}, grid::Vector{Float64}, axis::Int)
    n1, n2 = size(data)
    out = zeros(Float64, size(data))
    if axis == 1
        require_condition(n1 >= 3, "Need at least 3 x-grid points.")
        dx = grid[2] - grid[1]
        @inbounds for j in 1:n2
            out[1, j] = (-3data[1, j] + 4data[2, j] - data[3, j]) / (2dx)
            for i in 2:(n1 - 1)
                out[i, j] = (data[i + 1, j] - data[i - 1, j]) / (2dx)
            end
            out[n1, j] = (3data[n1, j] - 4data[n1 - 1, j] + data[n1 - 2, j]) / (2dx)
        end
    else
        require_condition(n2 >= 3, "Need at least 3 y-grid points.")
        dy = grid[2] - grid[1]
        @inbounds for i in 1:n1
            out[i, 1] = (-3data[i, 1] + 4data[i, 2] - data[i, 3]) / (2dy)
            for j in 2:(n2 - 1)
                out[i, j] = (data[i, j + 1] - data[i, j - 1]) / (2dy)
            end
            out[i, n2] = (3data[i, n2] - 4data[i, n2 - 1] + data[i, n2 - 2]) / (2dy)
        end
    end
    return out
end

function derivative_stencil(i::Int, n::Int, h::Float64)
    if i == 1
        return ([1, 2, 3], [-3.0 / (2h), 4.0 / (2h), -1.0 / (2h)])
    elseif i == n
        return ([n, n - 1, n - 2], [3.0 / (2h), -4.0 / (2h), 1.0 / (2h)])
    else
        return ([i + 1, i - 1], [1.0 / (2h), -1.0 / (2h)])
    end
end

@inline linear_index(i::Int, j::Int, nx::Int) = i + (j - 1) * nx

function affine_noise_matrix(x::Float64, y::Float64, meta::AffineModelMetadata)
    return meta.B0 .+ x .* meta.B1 .+ y .* meta.B2
end

@inline function affine_symmetric_mobility_entries(x::Float64, y::Float64, meta::AffineModelMetadata)
    b11 = meta.B0[1, 1] + x * meta.B1[1, 1] + y * meta.B2[1, 1]
    b12 = meta.B0[1, 2] + x * meta.B1[1, 2] + y * meta.B2[1, 2]
    b21 = meta.B0[2, 1] + x * meta.B1[2, 1] + y * meta.B2[2, 1]
    b22 = meta.B0[2, 2] + x * meta.B1[2, 2] + y * meta.B2[2, 2]
    s11 = 0.5 * (b11 * b11 + b12 * b12)
    s12 = 0.5 * (b11 * b21 + b12 * b22)
    s22 = 0.5 * (b21 * b21 + b22 * b22)
    return s11, s12, s22
end

function affine_model_drift(x::Float64, y::Float64, meta::AffineModelMetadata)
    return (
        -x^3 - x * y^2 - x - meta.omega * y,
        -y^3 - x^2 * y - y + meta.omega * x,
    )
end

function estimate_r_field(plain_model, states::Array{Float64, 3}, start_idx::Int,
        meta::AffineModelMetadata, grid_nx::Int, grid_ny::Int, ridge::Float64, pad_fraction::Float64,
        score_batch_size::Int, device::DeviceConfig)
    xrange, yrange = state_domain(states, start_idx, pad_fraction)
    xgrid = collect(range(xrange[1], xrange[2], length=grid_nx))
    ygrid = collect(range(yrange[1], yrange[2], length=grid_ny))
    nx = length(xgrid)
    ny = length(ygrid)
    X = repeat(xgrid, 1, ny)
    Y = repeat(reshape(ygrid, 1, ny), nx, 1)

    points = Matrix{Float32}(undef, 2, nx * ny)
    cursor = 1
    @inbounds for j in 1:ny, i in 1:nx
        points[1, cursor] = Float32(X[i, j])
        points[2, cursor] = Float32(Y[i, j])
        cursor += 1
    end
    scores = reshape(evaluate_plain_score(plain_model, points, score_batch_size, device), 2, nx, ny)
    s1 = Array{Float64}(undef, nx, ny)
    s2 = Array{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        s1[i, j] = scores[1, i, j]
        s2[i, j] = scores[2, i, j]
    end

    f1 = zeros(Float64, nx, ny)
    f2 = zeros(Float64, nx, ny)
    s11 = zeros(Float64, nx, ny)
    s12 = zeros(Float64, nx, ny)
    s22 = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        x = X[i, j]
        y = Y[i, j]
        f1[i, j], f2[i, j] = affine_model_drift(x, y, meta)
        s11[i, j], s12[i, j], s22[i, j] = affine_symmetric_mobility_entries(x, y, meta)
    end

    divs1 = finite_difference_axis(s11, xgrid, 1) .+ finite_difference_axis(s12, ygrid, 2)
    divs2 = finite_difference_axis(s12, xgrid, 1) .+ finite_difference_axis(s22, ygrid, 2)
    g1 = f1 .- (s11 .* s1 .+ s12 .* s2) .- divs1
    g2 = f2 .- (s12 .* s1 .+ s22 .* s2) .- divs2

    n = nx * ny
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros(Float64, 2n)
    dx = xgrid[2] - xgrid[1]
    dy = ygrid[2] - ygrid[1]

    row = 1
    @inbounds for j in 1:ny, i in 1:nx
        k = linear_index(i, j, nx)
        y_idx, y_coeff = derivative_stencil(j, ny, dy)
        for (jj, coeff) in zip(y_idx, y_coeff)
            push!(rows, row)
            push!(cols, linear_index(i, jj, nx))
            push!(vals, coeff)
        end
        push!(rows, row)
        push!(cols, k)
        push!(vals, s2[i, j])
        b[row] = g1[i, j]
        row += 1

        x_idx, x_coeff = derivative_stencil(i, nx, dx)
        for (ii, coeff) in zip(x_idx, x_coeff)
            push!(rows, row)
            push!(cols, linear_index(ii, j, nx))
            push!(vals, -coeff)
        end
        push!(rows, row)
        push!(cols, k)
        push!(vals, -s1[i, j])
        b[row] = g2[i, j]
        row += 1
    end

    Aop = sparse(rows, cols, vals, 2n, n)
    H = Aop' * Aop + ridge * spdiagm(0 => ones(Float64, n))
    rhs = Aop' * b
    rvec = H \ rhs
    rgrid = reshape(rvec, nx, ny)
    return MobilityField(xgrid, ygrid, rgrid, meta)
end

function interpolate_r(field::MobilityField, x::Float64, y::Float64)
    xclamp = clamp(x, field.xgrid[1], field.xgrid[end])
    yclamp = clamp(y, field.ygrid[1], field.ygrid[end])
    ix_hi = searchsortedfirst(field.xgrid, xclamp)
    iy_hi = searchsortedfirst(field.ygrid, yclamp)
    ix_hi = clamp(ix_hi, 2, length(field.xgrid))
    iy_hi = clamp(iy_hi, 2, length(field.ygrid))
    ix_lo = ix_hi - 1
    iy_lo = iy_hi - 1
    x1, x2 = field.xgrid[ix_lo], field.xgrid[ix_hi]
    y1, y2 = field.ygrid[iy_lo], field.ygrid[iy_hi]
    tx = isapprox(x2, x1; atol=1e-12) ? 0.0 : (xclamp - x1) / (x2 - x1)
    ty = isapprox(y2, y1; atol=1e-12) ? 0.0 : (yclamp - y1) / (y2 - y1)
    r11 = field.rgrid[ix_lo, iy_lo]
    r21 = field.rgrid[ix_hi, iy_lo]
    r12 = field.rgrid[ix_lo, iy_hi]
    r22 = field.rgrid[ix_hi, iy_hi]
    return (1 - tx) * (1 - ty) * r11 + tx * (1 - ty) * r21 + (1 - tx) * ty * r12 + tx * ty * r22
end

function true_mobility_entries(x0::AbstractMatrix{Float32}, field::MobilityField)
    n = size(x0, 2)
    m11 = Vector{Float64}(undef, n)
    m12 = Vector{Float64}(undef, n)
    m21 = Vector{Float64}(undef, n)
    m22 = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        x = Float64(x0[1, k])
        y = Float64(x0[2, k])
        s11, s12, s22 = affine_symmetric_mobility_entries(x, y, field.meta)
        r = interpolate_r(field, x, y)
        m11[k] = s11
        m12[k] = s12 + r
        m21[k] = s12 - r
        m22[k] = s22
    end
    return m11, m12, m21, m22
end

function true_delta_m_entries(x0::AbstractMatrix{Float32}, field::MobilityField, phi_est::Matrix{Float64})
    m11, m12, m21, m22 = true_mobility_entries(x0, field)
    return m11 .- phi_est[1, 1], m12 .- phi_est[1, 2], m21 .- phi_est[2, 1], m22 .- phi_est[2, 2]
end

function mean_true_mobility(field::MobilityField, states::Array{Float64, 3}, start_idx::Int)
    npoints = length(@view states[start_idx:end, 1, :])
    points = Matrix{Float32}(undef, 2, npoints)
    points[1, :] .= Float32.(vec(@view states[start_idx:end, 1, :]))
    points[2, :] .= Float32.(vec(@view states[start_idx:end, 2, :]))
    m11, m12, m21, m22 = true_mobility_entries(points, field)
    return [
        mean(m11) mean(m12)
        mean(m21) mean(m22)
    ]
end

function estimate_true_reference_operators(states::Array{Float64, 3}, start_idx::Int, lag_steps::Vector{Int}, lag_times::Vector{Float64},
        obs_ctx::ObservableContext, use_all_pairs::Bool, npairs::Int, rng::AbstractRNG,
        plain_model, joint_model, joint_meta, field::MobilityField,
        score_batch_size::Int, joint_batch_size::Int, device::DeviceConfig,
        phi_est::Matrix{Float64}, obs_indices::AbstractVector{Int})
    nobs = length(obs_indices)
    a_true = zeros(Float64, length(lag_steps), nobs, nobs)
    cdot_true = zeros(Float64, length(lag_steps), nobs, nobs)
    phi11 = phi_est[1, 1]
    phi12 = phi_est[1, 2]
    phi21 = phi_est[2, 1]
    phi22 = phi_est[2, 2]
    batch_pairs = max(1, min(score_batch_size, joint_batch_size))
    for (lag_idx, (lag, tau)) in enumerate(zip(lag_steps, lag_times))
        if lag_idx == 1 || lag_idx % 25 == 0 || lag_idx == length(lag_steps)
            @printf("Evaluating true reference operators %d / %d : tau = %.3f using %s\n",
                lag_idx, length(lag_steps), tau, use_all_pairs ? "all pairs" : "sampled pairs")
        end
        tnorm = normalize_tau(tau, joint_meta)
        sums_a = zeros(Float64, nobs, nobs)
        sums_c = zeros(Float64, nobs, nobs)
        if use_all_pairs
            total = for_each_all_pairs_batch(states, start_idx, lag, batch_pairs) do x0_batch, xt_batch
                scond = evaluate_conditional_score(plain_model, joint_model, x0_batch, xt_batch, tnorm,
                    score_batch_size, joint_batch_size, device)
                m11, m12, m21, m22 = true_mobility_entries(x0_batch, field)
                x0x = Float64.(x0_batch[1, :])
                x0y = Float64.(x0_batch[2, :])
                xtx = Float64.(xt_batch[1, :])
                xty = Float64.(xt_batch[2, :])
                s1 = Float64.(scond[1, :])
                s2 = Float64.(scond[2, :])
                obst = observable_basis(xtx, xty, obs_indices)
                ka = Matrix{Float64}(undef, nobs, length(s1))
                kc = Matrix{Float64}(undef, nobs, length(s1))
                for (local_n, global_n) in enumerate(obs_indices)
                    grad, _, _, _ = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                    g1 = @view grad[1, :]
                    g2 = @view grad[2, :]
                    a1 = (m11 .- phi11) .* g1 .+ (m12 .- phi12) .* g2
                    a2 = (m21 .- phi21) .* g1 .+ (m22 .- phi22) .* g2
                    c1 = m11 .* g1 .+ m12 .* g2
                    c2 = m21 .* g1 .+ m22 .* g2
                    ka[local_n, :] .= a1 .* s1 .+ a2 .* s2
                    kc[local_n, :] .= c1 .* s1 .+ c2 .* s2
                end
                sums_a .+= obst * ka'
                sums_c .-= obst * kc'
            end
            a_true[lag_idx, :, :] .= sums_a ./ total
            cdot_true[lag_idx, :, :] .= sums_c ./ total
        else
            x0, xt = sample_pairs(states, start_idx, lag, npairs, rng)
            scond = evaluate_conditional_score(plain_model, joint_model, x0, xt, tnorm,
                score_batch_size, joint_batch_size, device)
            m11, m12, m21, m22 = true_mobility_entries(x0, field)
            x0x = Float64.(x0[1, :])
            x0y = Float64.(x0[2, :])
            xtx = Float64.(xt[1, :])
            xty = Float64.(xt[2, :])
            s1 = Float64.(scond[1, :])
            s2 = Float64.(scond[2, :])
            obst = observable_basis(xtx, xty, obs_indices)
            ka = Matrix{Float64}(undef, nobs, size(x0, 2))
            kc = Matrix{Float64}(undef, nobs, size(x0, 2))
            for (local_n, global_n) in enumerate(obs_indices)
                grad, _, _, _ = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                g1 = @view grad[1, :]
                g2 = @view grad[2, :]
                a1 = (m11 .- phi11) .* g1 .+ (m12 .- phi12) .* g2
                a2 = (m21 .- phi21) .* g1 .+ (m22 .- phi22) .* g2
                c1 = m11 .* g1 .+ m12 .* g2
                c2 = m21 .* g1 .+ m22 .* g2
                ka[local_n, :] .= a1 .* s1 .+ a2 .* s2
                kc[local_n, :] .= c1 .* s1 .+ c2 .* s2
            end
            a_true[lag_idx, :, :] .= (obst * ka') ./ size(x0, 2)
            cdot_true[lag_idx, :, :] .= -(obst * kc') ./ size(x0, 2)
        end
    end
    return a_true, cdot_true
end

function extract_training_channels(values::Array{Float64, 3}, train_pairs::Vector{Tuple{Int, Int}})
    nlags = size(values, 1)
    out = zeros(Float64, length(train_pairs), 1, nlags)
    for (pair_pos, (obs_idx, coord_idx)) in enumerate(train_pairs)
        out[pair_pos, 1, :] .= values[:, obs_idx, coord_idx]
    end
    return out
end

function estimate_reliability_pair_weights(states::Array{Float64, 3}, start_idx::Int,
        lag_steps::Vector{Int}, lag_times::Vector{Float64}, obs_ctx::ObservableContext,
        phi_est::Matrix{Float64}, target_train::Array{Float64, 3}, train_pairs::Vector{Tuple{Int, Int}},
        obs_indices::AbstractVector{Int}, params::FitDMParams, plain_model, device::DeviceConfig)
    nchannels = length(train_pairs)
    if params.mobility_nn_reliability_mode == "none"
        return ones(Float64, nchannels), zeros(Float64, nchannels)
    end

    ntraj = size(states, 3)
    subset_size = clamp(round(Int, params.mobility_nn_reliability_fraction * ntraj), 1, ntraj)
    require_condition(subset_size < ntraj || ntraj == 1,
        "mobility_nn.reliability_fraction must leave at least one trajectory out for split_half weighting.")

    nobs = length(obs_indices)
    nlags = length(lag_steps)
    split_targets = Array{Float64}(undef, nchannels, nlags, params.mobility_nn_reliability_splits)
    target_scale = vec(Float64.(sqrt.(mean(target_train .^ 2; dims=3))[:, 1, 1]))
    target_scale .= max.(target_scale, params.mobility_nn_scale_floor)

    @printf("Estimating channel reliability from %d split-half A-data reconstructions\n", params.mobility_nn_reliability_splits)
    for split_idx in 1:params.mobility_nn_reliability_splits
        traj_rng = MersenneTwister(20260500 + split_idx)
        traj_indices = sort(randperm(traj_rng, ntraj)[1:subset_size])
        split_states = states[:, :, traj_indices]
        cphi_split = estimate_observed_cphi(split_states, start_idx, lag_steps,
            obs_ctx, params.use_all_observed_pairs, params.a_pairs_per_tau, traj_rng, obs_indices)
        cdot_split = zeros(Float64, nlags, nobs, nobs)
        for m in 1:nobs, n in 1:nobs
            _, deriv_vals, _ = select_smoothing_protocol_dataonly(
                lag_times,
                vec(cphi_split[:, m, n]),
                params.cphi_spline_smoothing,
                params.cphi_smoothing_rel_grid,
                params.cphi_smoothing_rmse_tolerance,
            )
            cdot_split[:, m, n] .= deriv_vals
        end
        gamma_rng = MersenneTwister(20260600 + split_idx)
        gamma_split = estimate_gamma_term(split_states, start_idx, lag_steps,
            obs_ctx, phi_est, params.use_all_sphi_pairs, params.a_pairs_per_tau, gamma_rng,
            plain_model, params.score_batch_size, device, obs_indices)
        split_targets[:, :, split_idx] .= extract_training_channels(gamma_split .- cdot_split, train_pairs)[:, 1, :]
    end

    instability = normalized_channel_instability(split_targets, target_scale)
    return reliability_weights_from_instability(instability, params.mobility_nn_reliability_min_weight,
        params.mobility_nn_reliability_strength), instability
end

function training_channel_labels(labels::Vector{String}, train_pairs::Vector{Tuple{Int, Int}})
    out = String[]
    for (obs_idx, coord_idx) in train_pairs
        push!(out, "A_" * labels[obs_idx] * "," * labels[coord_idx])
    end
    return out
end

function flatten_training_channel_matrix(values::AbstractMatrix{<:Real})
    out = Float64[]
    for obs_idx in axes(values, 1), coord_idx in axes(values, 2)
        push!(out, Float64(values[obs_idx, coord_idx]))
    end
    return out
end

function channel_rmse_matrix(target::Array{Float64, 3}, pred::Array{Float64, 3})
    return sqrt.(mean((target .- pred) .^ 2; dims=3))[:, :, 1]
end

function channel_normalized_rmse_matrix(target::Array{Float64, 3}, pred::Array{Float64, 3}, scale::AbstractArray{Float64})
    return sqrt.(mean((((target .- pred) ./ scale) .^ 2); dims=3))[:, :, 1]
end

function observed_state_points(states::Array{Float64, 3}, start_idx::Int)
    npoints = length(@view states[start_idx:end, 1, :])
    points = Matrix{Float32}(undef, 2, npoints)
    points[1, :] .= Float32.(vec(@view states[start_idx:end, 1, :]))
    points[2, :] .= Float32.(vec(@view states[start_idx:end, 2, :]))
    return points
end

function nearest_grid_index(grid::Vector{Float64}, value::Float64)
    hi = searchsortedfirst(grid, value)
    if hi <= 1
        return 1
    elseif hi > length(grid)
        return length(grid)
    end
    lo = hi - 1
    return abs(value - grid[lo]) <= abs(grid[hi] - value) ? lo : hi
end

function observed_support_mask(field::MobilityField, states::Array{Float64, 3}, start_idx::Int)
    nx = length(field.xgrid)
    ny = length(field.ygrid)
    mask = falses(nx, ny)
    xvals = vec(@view states[start_idx:end, 1, :])
    yvals = vec(@view states[start_idx:end, 2, :])
    @inbounds for idx in eachindex(xvals, yvals)
        ix = nearest_grid_index(field.xgrid, xvals[idx])
        iy = nearest_grid_index(field.ygrid, yvals[idx])
        for j in max(1, iy - 1):min(ny, iy + 1), i in max(1, ix - 1):min(nx, ix + 1)
            mask[i, j] = true
        end
    end
    return mask
end

function masked_rmse(ref::Matrix{Float64}, pred::Matrix{Float64}, mask::BitMatrix)
    total = 0.0
    count = 0
    @inbounds for j in axes(ref, 2), i in axes(ref, 1)
        if mask[i, j]
            delta = pred[i, j] - ref[i, j]
            total += delta * delta
            count += 1
        end
    end
    require_condition(count > 0, "Observed-support mask is empty.")
    return sqrt(total / count)
end

function pointwise_delta_rmse(model, field::MobilityField, points::Matrix{Float32}, phi_est::Matrix{Float64},
        μ::Vector{Float32}, σ::Vector{Float32}, psd_jitter::Float64, batch_size::Int, device::DeviceConfig)
    true11, true12, true21, true22 = true_delta_m_entries(points, field, phi_est)
    pred = evaluate_mobility_matrices_nn(model, points, μ, σ, psd_jitter, batch_size, device)
    pred11 = Float64.(pred[1, :]) .- phi_est[1, 1]
    pred12 = Float64.(pred[2, :]) .- phi_est[1, 2]
    pred21 = Float64.(pred[3, :]) .- phi_est[2, 1]
    pred22 = Float64.(pred[4, :]) .- phi_est[2, 2]
    component_rmse = [
        sqrt(mean((pred11 .- true11) .^ 2)),
        sqrt(mean((pred12 .- true12) .^ 2)),
        sqrt(mean((pred21 .- true21) .^ 2)),
        sqrt(mean((pred22 .- true22) .^ 2)),
    ]
    true_stack = vcat(true11, true12, true21, true22)
    pred_stack = vcat(pred11, pred12, pred21, pred22)
    total_rmse = sqrt(mean((pred_stack .- true_stack) .^ 2))
    relative_rmse = total_rmse / max(sqrt(mean(true_stack .^ 2)), eps(Float64))
    return component_rmse, total_rmse, relative_rmse
end

function sample_anchor_states(states::Array{Float64, 3}, start_idx::Int, npoints::Int, rng::AbstractRNG)
    nt, _, ntraj = size(states)
    points = Matrix{Float32}(undef, 2, npoints)
    @inbounds for idx in 1:npoints
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, start_idx:nt)
        points[1, idx] = Float32(states[time_idx, 1, traj_idx])
        points[2, idx] = Float32(states[time_idx, 2, traj_idx])
    end
    return points
end

function input_normalization_stats(states::Array{Float64, 3}, start_idx::Int)
    μ = Float32[
        mean(@view states[start_idx:end, 1, :]),
        mean(@view states[start_idx:end, 2, :]),
    ]
    σ = Float32[
        std(@view states[start_idx:end, 1, :]),
        std(@view states[start_idx:end, 2, :]),
    ]
    σ .= max.(σ, Float32(1.0e-4))
    return μ, σ
end

function normalize_points(points, μ, σ)
    tail = ntuple(_ -> 1, max(ndims(points) - 1, 0))
    shape = (length(μ), tail...)
    return (points .- reshape(μ, shape...)) ./ reshape(σ, shape...)
end

function build_residual_mlp(widths::Vector{Int})
    layers = Any[]
    in_dim = 2
    for width in widths
        push!(layers, Dense(in_dim, width, tanh))
        in_dim = width
    end
    push!(layers, Dense(in_dim, 4))
    return Chain(layers...)
end

function build_mobility_model(widths::Vector{Int}, model_kind::String)
    if model_kind == "mlp"
        return build_residual_mlp(widths)
    elseif model_kind == "affine_skip"
        return RawAffineResidualMobilityModel(Dense(2, 4), build_residual_mlp(widths))
    elseif model_kind == "entry_affine"
        return DirectEntryAffineMobilityModel(Dense(2, 4), zeros(Float32, 4))
    end
    error("Unsupported mobility_nn.model = $(model_kind)")
end

function softplus_inverse(x::Float32)
    return x > 20.0f0 ? x : log(expm1(x))
end

function mobility_phi_raw_bias(phi_est::Matrix{Float64}, psd_jitter::Float64)
    sym = 0.5 .* (phi_est .+ phi_est')
    min_eig = minimum(eigvals(Symmetric(sym)))
    if min_eig <= psd_jitter
        sym = sym + (psd_jitter - min_eig + 1.0e-6) .* Matrix{Float64}(I, 2, 2)
    end
    chol = cholesky(Symmetric(sym + psd_jitter .* Matrix{Float64}(I, 2, 2))).L
    skew12 = 0.5 * (phi_est[1, 2] - phi_est[2, 1])
    return Float32[
        softplus_inverse(Float32(max(chol[1, 1], sqrt(psd_jitter)))),
        Float32(chol[2, 1]),
        softplus_inverse(Float32(max(chol[2, 2], sqrt(psd_jitter)))),
        Float32(skew12),
    ]
end

function initialize_mobility_model!(model::Chain, phi_est::Matrix{Float64}, psd_jitter::Float64)
    final_layer = model.layers[end]
    final_layer.weight .= zero(eltype(final_layer.weight))
    final_layer.bias .= mobility_phi_raw_bias(phi_est, psd_jitter)
    return model
end

function initialize_mobility_model!(model::RawAffineResidualMobilityModel, phi_est::Matrix{Float64}, psd_jitter::Float64)
    model.affine.weight .= zero(eltype(model.affine.weight))
    model.affine.bias .= mobility_phi_raw_bias(phi_est, psd_jitter)
    residual_final = model.residual.layers[end]
    residual_final.weight .= zero(eltype(residual_final.weight))
    residual_final.bias .= zero(eltype(residual_final.bias))
    return model
end

function initialize_mobility_model!(model::DirectEntryAffineMobilityModel, phi_est::Matrix{Float64}, psd_jitter::Float64)
    model.delta_model.weight .= zero(eltype(model.delta_model.weight))
    model.delta_model.bias .= zero(eltype(model.delta_model.bias))
    model.phi_entries .= Float32[phi_est[1, 1], phi_est[1, 2], phi_est[2, 1], phi_est[2, 2]]
    return model
end

function mobility_outputs_from_raw(raw, psd_jitter::Real)
    T = eltype(raw)
    jitter = T(psd_jitter)
    l11 = Flux.softplus.(@view(raw[1, :]))
    l21 = @view(raw[2, :])
    l22 = Flux.softplus.(@view(raw[3, :]))
    a12 = @view(raw[4, :])

    d11 = l11 .^ 2 .+ jitter
    d12 = l11 .* l21
    d22 = l21 .^ 2 .+ l22 .^ 2 .+ jitter
    m11 = d11
    m12 = d12 .+ a12
    m21 = d12 .- a12
    m22 = d22
    return m11, m12, m21, m22, d11, d12, d22
end

function mobility_outputs_from_direct_entries(entries, psd_jitter::Real)
    T = eltype(entries)
    jitter = T(psd_jitter)
    m11 = @view(entries[1, :])
    m12 = @view(entries[2, :])
    m21 = @view(entries[3, :])
    m22 = @view(entries[4, :])
    d11 = m11
    d12 = (m12 .+ m21) .* T(0.5)
    d22 = m22
    skew12 = (m12 .- m21) .* T(0.5)
    half_diff = (d11 .- d22) .* T(0.5)
    rad = sqrt.(half_diff .^ 2 .+ d12 .^ 2)
    λmin = (d11 .+ d22) .* T(0.5) .- rad
    shift = Flux.softplus.(jitter .- λmin)
    d11_repaired = d11 .+ shift
    d22_repaired = d22 .+ shift
    m11_repaired = d11_repaired
    m12_repaired = d12 .+ skew12
    m21_repaired = d12 .- skew12
    m22_repaired = d22_repaired
    return m11_repaired, m12_repaired, m21_repaired, m22_repaired, d11_repaired, d12, d22_repaired
end

function evaluate_mobility_outputs(model::Chain, normalized_points, psd_jitter::Real)
    return mobility_outputs_from_raw(model(normalized_points), psd_jitter)
end

function evaluate_mobility_outputs(model::RawAffineResidualMobilityModel, normalized_points, psd_jitter::Real)
    return mobility_outputs_from_raw(model(normalized_points), psd_jitter)
end

function evaluate_mobility_outputs(model::DirectEntryAffineMobilityModel, normalized_points, psd_jitter::Real)
    return mobility_outputs_from_direct_entries(model(normalized_points), psd_jitter)
end

function build_mobility_training_cache(sampler::PairSampler, obs_ctx::ObservableContext,
        plain_model, joint_model, joint_meta, params::FitDMParams, device::DeviceConfig;
        pair_seed::Int=20260408, anchor_seed::Int=20260411)
    nlags = length(sampler.lag_steps)
    npairs = params.mobility_nn_pairs_per_tau
    x0_cache = Array{Float32}(undef, 2, npairs, nlags)
    scond_cache = Array{Float32}(undef, 2, npairs, nlags)
    phi_cache = Array{Float32}(undef, length(params.train_pairs), npairs, nlags)
    grad_cache = Array{Float32}(undef, 2, length(params.train_pairs), npairs, nlags)
    mean_rng = MersenneTwister(anchor_seed)
    mean_x0 = sample_anchor_states(sampler.states, sampler.start_idx, params.mobility_nn_anchor_points, mean_rng)
    rng = MersenneTwister(pair_seed)

    for (lag_idx, (lag, tau)) in enumerate(zip(sampler.lag_steps, sampler.lag_times))
        if lag_idx == 1 || lag_idx % 25 == 0 || lag_idx == nlags
            @printf("Building mobility cache %d / %d : tau = %.3f\n", lag_idx, nlags, tau)
        end
        x0, xt = sample_pairs(sampler.states, sampler.start_idx, lag, npairs, rng)
        tnorm = normalize_tau(tau, joint_meta)
        scond = evaluate_conditional_score(plain_model, joint_model, x0, xt, tnorm,
            params.score_batch_size, params.joint_batch_size, device)
        x0_cache[:, :, lag_idx] .= x0
        scond_cache[:, :, lag_idx] .= scond
        x0x = Float64.(x0[1, :])
        x0y = Float64.(x0[2, :])
        xtx = Float64.(xt[1, :])
        xty = Float64.(xt[2, :])
        for (pair_pos, (obs_idx, coord_idx)) in enumerate(params.train_pairs)
            phi_cache[pair_pos, :, lag_idx] .= Float32.(observable_values(obs_idx, xtx, xty, obs_ctx))
            grad, _, _, _ = observable_grad_hess(coord_idx, x0x, x0y, obs_ctx)
            grad_cache[1, pair_pos, :, lag_idx] .= Float32.(@view grad[1, :])
            grad_cache[2, pair_pos, :, lag_idx] .= Float32.(@view grad[2, :])
        end
    end
    return MobilityNNCache(x0_cache, scond_cache, phi_cache, grad_cache, mean_x0, sampler.lag_times)
end

function evaluate_mobility_raw(model, x0::AbstractMatrix{Float32}, μ::Vector{Float32}, σ::Vector{Float32},
        batch_size::Int, device::DeviceConfig)
    out = Matrix{Float32}(undef, 4, size(x0, 2))
    model_dev = to_device(model, device)
    μ_dev = to_device(μ, device)
    σ_dev = to_device(σ, device)
    for start in 1:batch_size:size(x0, 2)
        stop = min(start + batch_size - 1, size(x0, 2))
        batch = x0[:, start:stop]
        out[:, start:stop] .= Float32.(to_host(model_dev(normalize_points(to_device(batch, device), μ_dev, σ_dev))))
    end
    return out
end

function lag_weights(taus::Vector{Float64}, power::Float64)
    if power == 0.0
        return ones(Float32, length(taus))
    end
    τref = minimum(taus)
    weights = Float32.((τref ./ taus) .^ power)
    weights ./= Float32(mean(weights))
    return weights
end

function annealed_penalty_weight(base_weight::Float64, final_scale::Float64, epoch::Int, total_epochs::Int)
    total_epochs <= 1 && return base_weight * final_scale
    frac = (epoch - 1) / (total_epochs - 1)
    scale = 1.0 + (final_scale - 1.0) * frac
    return base_weight * scale
end

function predict_training_chunk(model, x0_chunk, scond_chunk, phi_chunk, grad_chunk,
        phi11::Float32, phi12::Float32, phi21::Float32, phi22::Float32,
        μ, σ, psd_jitter::Float64)
    npairs = size(x0_chunk, 2)
    nbatch = size(x0_chunk, 3)
    nchannels = size(phi_chunk, 1)
    xflat = reshape(x0_chunk, 2, :)
    m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(model, normalize_points(xflat, μ, σ), psd_jitter)
    s1 = reshape(scond_chunk[1, :, :], npairs, nbatch)
    s2 = reshape(scond_chunk[2, :, :], npairs, nbatch)
    delta11 = reshape(m11, npairs, nbatch) .- phi11
    delta12 = reshape(m12, npairs, nbatch) .- phi12
    delta21 = reshape(m21, npairs, nbatch) .- phi21
    delta22 = reshape(m22, npairs, nbatch) .- phi22
    row1 = s1 .* delta11 .+ s2 .* delta21
    row2 = s1 .* delta12 .+ s2 .* delta22
    inv_npairs = one(eltype(row1)) / npairs
    channel_components = map(1:nchannels) do channel_pos
        phi_vals = reshape(@view(phi_chunk[channel_pos, :, :]), npairs, nbatch)
        g1 = reshape(@view(grad_chunk[1, channel_pos, :, :]), npairs, nbatch)
        g2 = reshape(@view(grad_chunk[2, channel_pos, :, :]), npairs, nbatch)
        reshape(sum(phi_vals .* (row1 .* g1 .+ row2 .* g2); dims=1) .* inv_npairs, 1, 1, nbatch)
    end
    return cat(channel_components...; dims=1)
end

function evaluate_training_model_on_cache(model, cache::MobilityNNCache, μ::Vector{Float32}, σ::Vector{Float32},
        phi_est::Matrix{Float32}, psd_jitter::Float64, tau_batch::Int, device::DeviceConfig)
    nlags = size(cache.x0, 3)
    pred = zeros(Float64, size(cache.phi, 1), 1, nlags)
    x0_dev = to_device(cache.x0, device)
    scond_dev = to_device(cache.scond, device)
    phi_dev = to_device(cache.phi, device)
    grad_dev = to_device(cache.grad, device)
    phi11, phi12 = Float32(phi_est[1, 1]), Float32(phi_est[1, 2])
    phi21, phi22 = Float32(phi_est[2, 1]), Float32(phi_est[2, 2])
    μ_dev = to_device(μ, device)
    σ_dev = to_device(σ, device)
    model_dev = to_device(model, device)
    for start in 1:tau_batch:nlags
        stop = min(start + tau_batch - 1, nlags)
        chunk = start:stop
        chunk_pred = predict_training_chunk(model_dev, @view(x0_dev[:, :, chunk]), @view(scond_dev[:, :, chunk]),
            @view(phi_dev[:, :, chunk]), @view(grad_dev[:, :, :, chunk]),
            phi11, phi12, phi21, phi22, μ_dev, σ_dev, psd_jitter)
        pred[:, :, chunk] .= Float64.(to_host(chunk_pred))
    end
    return pred
end

function mobility_mean_delta_entries(model, mean_x0,
        phi11::Float32, phi12::Float32, phi21::Float32, phi22::Float32,
        μ, σ, psd_jitter::Float64)
    m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(model, normalize_points(mean_x0, μ, σ), psd_jitter)
    return mean(m11) - phi11, mean(m12) - phi12, mean(m21) - phi21, mean(m22) - phi22
end

function mobility_mean_delta(model, mean_x0,
        phi11::Float32, phi12::Float32, phi21::Float32, phi22::Float32,
        μ, σ, psd_jitter::Float64)
    δ11, δ12, δ21, δ22 = mobility_mean_delta_entries(model, mean_x0, phi11, phi12, phi21, phi22, μ, σ, psd_jitter)
    return [
        Float64(δ11) Float64(δ12)
        Float64(δ21) Float64(δ22)
    ]
end

function mean_penalty_from_delta(mean_delta::AbstractMatrix{<:Real},
        phi_scale11::Real, phi_scale12::Real, phi_scale21::Real, phi_scale22::Real,
        entry_weights::AbstractMatrix{<:Real})
    weight_sum = sum(entry_weights)
    return (
        entry_weights[1, 1] * (mean_delta[1, 1] / phi_scale11)^2 +
        entry_weights[1, 2] * (mean_delta[1, 2] / phi_scale12)^2 +
        entry_weights[2, 1] * (mean_delta[2, 1] / phi_scale21)^2 +
        entry_weights[2, 2] * (mean_delta[2, 2] / phi_scale22)^2
    ) / weight_sum
end

function mobility_anchor_delta_stats(model, anchor_x0,
        phi11::Float32, phi12::Float32, phi21::Float32, phi22::Float32,
        μ, σ, psd_jitter::Float64,
    phi_scale11::Real, phi_scale12::Real, phi_scale21::Real, phi_scale22::Real,
    entry_weights::AbstractMatrix{<:Real})
    m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(model, normalize_points(anchor_x0, μ, σ), psd_jitter)
    δ11 = m11 .- phi11
    δ12 = m12 .- phi12
    δ21 = m21 .- phi21
    δ22 = m22 .- phi22
    mean_delta = [
        Float64(mean(δ11)) Float64(mean(δ12))
        Float64(mean(δ21)) Float64(mean(δ22))
    ]
    weight_sum = sum(entry_weights)
    rms_penalty = mean(
        entry_weights[1, 1] .* (δ11 ./ phi_scale11) .^ 2 .+
        entry_weights[1, 2] .* (δ12 ./ phi_scale12) .^ 2 .+
        entry_weights[2, 1] .* (δ21 ./ phi_scale21) .^ 2 .+
        entry_weights[2, 2] .* (δ22 ./ phi_scale22) .^ 2
    ) / weight_sum
    return mean_delta, rms_penalty
end

function model_weight_decay(model)
    total = nothing
    count = 0
    for p in Flux.trainables(model)
        count += length(p)
        total = total === nothing ? sum(abs2, p) : total + sum(abs2, p)
    end
    total === nothing && return 0.0f0
    return total / count
end

function train_mobility_model(cache::MobilityNNCache, validation_caches::Vector{MobilityNNCache},
        target::Array{Float64, 3}, phi_est::Matrix{Float64},
        μ::Vector{Float32}, σ::Vector{Float32}, params::FitDMParams,
        effective_pair_weights::Vector{Float64}, device::DeviceConfig)
    require_condition(!isempty(validation_caches), "Need at least one validation cache.")
    validate_training_weights(effective_pair_weights, length(params.train_pairs))
    nlags = size(cache.x0, 3)
    x0_dev = to_device(cache.x0, device)
    scond_dev = to_device(cache.scond, device)
    phi_dev = to_device(cache.phi, device)
    grad_dev = to_device(cache.grad, device)
    mean_x0_dev = to_device(cache.mean_x0, device)
    target32 = Float32.(target)
    target_dev = to_device(target32, device)
    target_scale = Float64.(sqrt.(mean(target .^ 2; dims=3))[:, :, 1])
    target_scale .= max.(target_scale, params.mobility_nn_scale_floor)
    target_scale_dev = to_device(reshape(Float32.(target_scale), size(target_scale, 1), size(target_scale, 2), 1), device)
    pair_weights = Float32.(reshape(effective_pair_weights, length(effective_pair_weights), 1, 1))
    pair_weights_dev = to_device(pair_weights, device)
    phi_mat32 = Float32.(phi_est)
    phi11, phi12 = Float32(phi_mat32[1, 1]), Float32(phi_mat32[1, 2])
    phi21, phi22 = Float32(phi_mat32[2, 1]), Float32(phi_mat32[2, 2])
    phi_scale11 = max(abs(phi11), Float32(params.mobility_nn_scale_floor))
    phi_scale12 = max(abs(phi12), Float32(params.mobility_nn_scale_floor))
    phi_scale21 = max(abs(phi21), Float32(params.mobility_nn_scale_floor))
    phi_scale22 = max(abs(phi22), Float32(params.mobility_nn_scale_floor))
    μ_dev = to_device(μ, device)
    σ_dev = to_device(σ, device)
    lag_weights_dev = to_device(reshape(lag_weights(cache.taus, params.mobility_nn_lag_weight_power), 1, 1, :), device)

    model = build_mobility_model(params.mobility_nn_widths, params.mobility_nn_model)
    initialize_mobility_model!(model, phi_est, params.mobility_nn_psd_jitter)
    model = to_device(model, device)
    opt_state = Flux.setup(Flux.Adam(params.mobility_nn_learning_rate), model)
    rng = MersenneTwister(20260409)
    history = MobilityNNHistory(Int[], Float64[], Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[], Float64[], Float64[])
    best_metric = Inf
    best_host_model = cpu(model)

    for epoch in 1:params.mobility_nn_epochs
        current_mean_penalty_weight = annealed_penalty_weight(
            params.mobility_nn_mean_penalty_weight,
            params.mobility_nn_mean_penalty_final_scale,
            epoch,
            params.mobility_nn_epochs,
        )
        current_anchor_rms_penalty_weight = annealed_penalty_weight(
            params.mobility_nn_anchor_rms_penalty_weight,
            params.mobility_nn_anchor_rms_penalty_final_scale,
            epoch,
            params.mobility_nn_epochs,
        )
        perm = randperm(rng, nlags)
        epoch_loss = 0.0
        nbatches = 0
        for start in 1:params.mobility_nn_tau_batch:nlags
            stop = min(start + params.mobility_nn_tau_batch - 1, nlags)
            chunk = perm[start:stop]
            loss, grads = Flux.withgradient(model) do current_model
                pred = predict_training_chunk(current_model, @view(x0_dev[:, :, chunk]), @view(scond_dev[:, :, chunk]),
                    @view(phi_dev[:, :, chunk]), @view(grad_dev[:, :, :, chunk]),
                    phi11, phi12, phi21, phi22, μ_dev, σ_dev, params.mobility_nn_psd_jitter)
                chunk_weights = @view lag_weights_dev[:, :, chunk]
                data_loss = mean((((pred .- @view(target_dev[:, :, chunk])) ./ target_scale_dev) .^ 2) .* pair_weights_dev .* chunk_weights)
                mean_penalty = zero(data_loss)
                rms_penalty = zero(data_loss)
                if current_mean_penalty_weight > 0.0 || current_anchor_rms_penalty_weight > 0.0
                    mean_delta, rms_penalty = mobility_anchor_delta_stats(current_model, mean_x0_dev,
                        phi11, phi12, phi21, phi22, μ_dev, σ_dev, params.mobility_nn_psd_jitter,
                        phi_scale11, phi_scale12, phi_scale21, phi_scale22, params.mobility_nn_anchor_rms_entry_weights)
                    mean_penalty = mean_penalty_from_delta(mean_delta, phi_scale11, phi_scale12, phi_scale21, phi_scale22,
                        params.mobility_nn_mean_entry_weights)
                end
                reg_penalty = params.mobility_nn_weight_decay == 0.0 ? zero(data_loss) :
                    eltype(data_loss)(params.mobility_nn_weight_decay) * model_weight_decay(current_model)
                data_loss +
                    eltype(data_loss)(current_mean_penalty_weight) * mean_penalty +
                    eltype(data_loss)(current_anchor_rms_penalty_weight) * rms_penalty +
                    reg_penalty
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float64(loss)
            nbatches += 1
        end

        if epoch == 1 || epoch % params.mobility_nn_eval_every == 0 || epoch == params.mobility_nn_epochs
            host_model = cpu(model)
            physical_rmse_sum = 0.0
            normalized_rmse_sum = 0.0
            normalized_mse_sum = 0.0
            mean_delta_sum = zeros(Float64, 2, 2)
            eval_rms_penalty_sum = 0.0
            for validation_cache in validation_caches
                pred = evaluate_training_model_on_cache(host_model, validation_cache, μ, σ, phi_mat32,
                    params.mobility_nn_psd_jitter, params.mobility_nn_tau_batch, device)
                physical_rmse_sum += sqrt(mean((target .- pred) .^ 2))
                normalized_rmse_sum += sqrt(mean((((target .- pred) ./ target_scale) .^ 2)))
                normalized_mse_sum += mean((((target .- pred) ./ target_scale) .^ 2))
                cache_mean_delta, cache_eval_rms_penalty = mobility_anchor_delta_stats(host_model, validation_cache.mean_x0,
                    phi11, phi12, phi21, phi22, μ, σ, params.mobility_nn_psd_jitter,
                    phi_scale11, phi_scale12, phi_scale21, phi_scale22, params.mobility_nn_anchor_rms_entry_weights)
                mean_delta_sum .+= cache_mean_delta
                eval_rms_penalty_sum += Float64(cache_eval_rms_penalty)
            end
            inv_nvalidation = 1.0 / length(validation_caches)
            physical_rmse = physical_rmse_sum * inv_nvalidation
            normalized_rmse = normalized_rmse_sum * inv_nvalidation
            normalized_mse = normalized_mse_sum * inv_nvalidation
            mean_delta = mean_delta_sum .* inv_nvalidation
            eval_rms_penalty = eval_rms_penalty_sum * inv_nvalidation
            push!(history.epochs, epoch)
            push!(history.train_loss, epoch_loss / max(nbatches, 1))
            push!(history.normalized_rmse, normalized_rmse)
            push!(history.physical_rmse, physical_rmse)
            eval_mean_penalty = mean_penalty_from_delta(mean_delta, phi_scale11, phi_scale12, phi_scale21, phi_scale22,
                params.mobility_nn_mean_entry_weights)
            push!(history.mean_abs_delta, mean(abs.(mean_delta)))
            push!(history.mean_delta11, mean_delta[1, 1])
            push!(history.mean_delta12, mean_delta[1, 2])
            push!(history.mean_delta21, mean_delta[2, 1])
            push!(history.mean_delta22, mean_delta[2, 2])
            weight_l2 = Float64(model_weight_decay(host_model))
            push!(history.weight_l2, weight_l2)
            selection_metric = if params.mobility_nn_checkpoint_metric == "regularized_objective"
                normalized_mse +
                    params.mobility_nn_mean_penalty_weight * Float64(eval_mean_penalty) +
                    params.mobility_nn_anchor_rms_penalty_weight * Float64(eval_rms_penalty) +
                    params.mobility_nn_weight_decay * weight_l2
            else
                normalized_mse
            end
            @printf("M-NN epoch %4d | loss = %.6e | normalized RMSE = %.6e | physical RMSE = %.6e | |<ΔM>| = %.6e | select = %.6e\n",
                epoch, history.train_loss[end], normalized_rmse, physical_rmse, history.mean_abs_delta[end], selection_metric)
            if selection_metric < best_metric
                best_metric = selection_metric
                best_host_model = host_model
            end
        end
    end
    return best_host_model, history, target_scale
end

function evaluate_mobility_matrices_nn(model, x0::AbstractMatrix{Float32}, μ::Vector{Float32}, σ::Vector{Float32},
        psd_jitter::Float64, batch_size::Int, device::DeviceConfig)
    out = Matrix{Float32}(undef, 4, size(x0, 2))
    model_dev = to_device(model, device)
    μ_dev = to_device(μ, device)
    σ_dev = to_device(σ, device)
    for start in 1:batch_size:size(x0, 2)
        stop = min(start + batch_size - 1, size(x0, 2))
        batch = x0[:, start:stop]
        m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(model_dev,
            normalize_points(to_device(batch, device), μ_dev, σ_dev), psd_jitter)
        out[1, start:stop] .= Float32.(to_host(m11))
        out[2, start:stop] .= Float32.(to_host(m12))
        out[3, start:stop] .= Float32.(to_host(m21))
        out[4, start:stop] .= Float32.(to_host(m22))
    end
    return out
end

function mean_nn_mobility(model, states::Array{Float64, 3}, start_idx::Int, μ::Vector{Float32}, σ::Vector{Float32},
        psd_jitter::Float64, batch_size::Int, device::DeviceConfig)
    npoints = length(@view states[start_idx:end, 1, :])
    points = Matrix{Float32}(undef, 2, npoints)
    points[1, :] .= Float32.(vec(@view states[start_idx:end, 1, :]))
    points[2, :] .= Float32.(vec(@view states[start_idx:end, 2, :]))
    pred = evaluate_mobility_matrices_nn(model, points, μ, σ, psd_jitter, batch_size, device)
    return [
        mean(Float64.(pred[1, :])) mean(Float64.(pred[2, :]))
        mean(Float64.(pred[3, :])) mean(Float64.(pred[4, :]))
    ]
end

function estimate_nn_reference_operators(states::Array{Float64, 3}, start_idx::Int, lag_steps::Vector{Int}, lag_times::Vector{Float64},
        obs_ctx::ObservableContext, use_all_pairs::Bool, npairs::Int, rng::AbstractRNG,
        plain_model, joint_model, joint_meta, mobility_model,
        μ::Vector{Float32}, σ::Vector{Float32}, psd_jitter::Float64,
        score_batch_size::Int, joint_batch_size::Int, device::DeviceConfig,
        phi_est::Matrix{Float64}, obs_indices::AbstractVector{Int})
    nobs = length(obs_indices)
    a_nn = zeros(Float64, length(lag_steps), nobs, nobs)
    cdot_nn = zeros(Float64, length(lag_steps), nobs, nobs)
    phi11 = phi_est[1, 1]
    phi12 = phi_est[1, 2]
    phi21 = phi_est[2, 1]
    phi22 = phi_est[2, 2]
    batch_pairs = max(1, min(score_batch_size, joint_batch_size))
    for (lag_idx, (lag, tau)) in enumerate(zip(lag_steps, lag_times))
        if lag_idx == 1 || lag_idx % 25 == 0 || lag_idx == length(lag_steps)
            @printf("Evaluating NN operators %d / %d : tau = %.3f using %s\n",
                lag_idx, length(lag_steps), tau, use_all_pairs ? "all pairs" : "sampled pairs")
        end
        tnorm = normalize_tau(tau, joint_meta)
        sums_a = zeros(Float64, nobs, nobs)
        sums_c = zeros(Float64, nobs, nobs)
        if use_all_pairs
            total = for_each_all_pairs_batch(states, start_idx, lag, batch_pairs) do x0_batch, xt_batch
                scond = evaluate_conditional_score(plain_model, joint_model, x0_batch, xt_batch, tnorm,
                    score_batch_size, joint_batch_size, device)
                mnn = evaluate_mobility_matrices_nn(mobility_model, x0_batch, μ, σ, psd_jitter, score_batch_size, device)
                m11 = Float64.(mnn[1, :])
                m12 = Float64.(mnn[2, :])
                m21 = Float64.(mnn[3, :])
                m22 = Float64.(mnn[4, :])
                x0x = Float64.(x0_batch[1, :])
                x0y = Float64.(x0_batch[2, :])
                xtx = Float64.(xt_batch[1, :])
                xty = Float64.(xt_batch[2, :])
                s1 = Float64.(scond[1, :])
                s2 = Float64.(scond[2, :])
                obst = observable_basis(xtx, xty, obs_indices)
                ka = Matrix{Float64}(undef, nobs, length(s1))
                kc = Matrix{Float64}(undef, nobs, length(s1))
                for (local_n, global_n) in enumerate(obs_indices)
                    grad, _, _, _ = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                    g1 = @view grad[1, :]
                    g2 = @view grad[2, :]
                    a1 = (m11 .- phi11) .* g1 .+ (m12 .- phi12) .* g2
                    a2 = (m21 .- phi21) .* g1 .+ (m22 .- phi22) .* g2
                    c1 = m11 .* g1 .+ m12 .* g2
                    c2 = m21 .* g1 .+ m22 .* g2
                    ka[local_n, :] .= a1 .* s1 .+ a2 .* s2
                    kc[local_n, :] .= c1 .* s1 .+ c2 .* s2
                end
                sums_a .+= obst * ka'
                sums_c .-= obst * kc'
            end
            a_nn[lag_idx, :, :] .= sums_a ./ total
            cdot_nn[lag_idx, :, :] .= sums_c ./ total
        else
            x0, xt = sample_pairs(states, start_idx, lag, npairs, rng)
            scond = evaluate_conditional_score(plain_model, joint_model, x0, xt, tnorm,
                score_batch_size, joint_batch_size, device)
            mnn = evaluate_mobility_matrices_nn(mobility_model, x0, μ, σ, psd_jitter, score_batch_size, device)
            m11 = Float64.(mnn[1, :])
            m12 = Float64.(mnn[2, :])
            m21 = Float64.(mnn[3, :])
            m22 = Float64.(mnn[4, :])
            x0x = Float64.(x0[1, :])
            x0y = Float64.(x0[2, :])
            xtx = Float64.(xt[1, :])
            xty = Float64.(xt[2, :])
            s1 = Float64.(scond[1, :])
            s2 = Float64.(scond[2, :])
            obst = observable_basis(xtx, xty, obs_indices)
            ka = Matrix{Float64}(undef, nobs, size(x0, 2))
            kc = Matrix{Float64}(undef, nobs, size(x0, 2))
            for (local_n, global_n) in enumerate(obs_indices)
                grad, _, _, _ = observable_grad_hess(global_n, x0x, x0y, obs_ctx)
                g1 = @view grad[1, :]
                g2 = @view grad[2, :]
                a1 = (m11 .- phi11) .* g1 .+ (m12 .- phi12) .* g2
                a2 = (m21 .- phi21) .* g1 .+ (m22 .- phi22) .* g2
                c1 = m11 .* g1 .+ m12 .* g2
                c2 = m21 .* g1 .+ m22 .* g2
                ka[local_n, :] .= a1 .* s1 .+ a2 .* s2
                kc[local_n, :] .= c1 .* s1 .+ c2 .* s2
            end
            a_nn[lag_idx, :, :] .= (obst * ka') ./ size(x0, 2)
            cdot_nn[lag_idx, :, :] .= -(obst * kc') ./ size(x0, 2)
        end
    end
    return a_nn, cdot_nn
end

function a_channel_label(m::Int, n::Int, labels::Vector{String})
    return "A_" * labels[m] * labels[n]
end

function cdot_channel_label(m::Int, n::Int, labels::Vector{String})
    return "Cdot_" * labels[m] * labels[n]
end

function pair_channel_label(prefix::AbstractString, pair::Tuple{Int, Int}, labels::Vector{String})
    return prefix * "_" * labels[pair[1]] * "," * labels[pair[2]]
end

function panel_grid_dims(npanels::Int; max_cols::Int=MAX_FIT_PANEL_COLUMNS)
    require_condition(npanels >= 1, "Need at least one panel to plot.")
    require_condition(max_cols >= 1, "max_cols must be at least 1.")
    ncols = min(max_cols, ceil(Int, sqrt(npanels)))
    nrows = cld(npanels, ncols)
    return nrows, ncols
end

function a_panel(taus::Vector{Float64}, y_data::Vector{Float64}, y_ref::Vector{Float64}, y_nn::Vector{Float64},
        title::String, rmse_data_nn::Float64, rmse_true_nn::Float64; show_legend::Bool)
    p = plot(taus, y_data; xlabel="tau", ylabel="value", label="from data A", color=:black,
        legend=show_legend ? :topright : false,
        title=@sprintf("%s\nRMSE data/NN = %.3e | true/NN = %.3e", title, rmse_data_nn, rmse_true_nn))
    plot!(p, taus, y_ref; label="from true ΔM", color=:darkorange, linestyle=:dash)
    plot!(p, taus, y_nn; label="from NN ΔM", color=:royalblue3, linestyle=:dashdot)
    return p
end

function cdot_panel(taus::Vector{Float64}, y_data::Vector{Float64}, y_ref::Vector{Float64}, y_nn::Vector{Float64},
        title::String, rmse_data_true::Float64, rmse_true_nn::Float64; show_legend::Bool)
    p = plot(taus, y_data; xlabel="tau", ylabel="value", label="from data", color=:black,
        legend=show_legend ? :topright : false,
        title=@sprintf("%s\nRMSE data/true = %.3e | true/NN = %.3e", title, rmse_data_true, rmse_true_nn))
    plot!(p, taus, y_ref; label="from true M", color=:darkorange, linestyle=:dash)
    plot!(p, taus, y_nn; label="from NN M", color=:royalblue3, linestyle=:dashdot)
    return p
end

function create_a_figure(taus::Vector{Float64}, labels::Vector{String}, a_data::Array{Float64, 3}, a_ref::Array{Float64, 3}, a_nn::Array{Float64, 3},
        rmse_data_nn::Matrix{Float64}, rmse_true_nn::Matrix{Float64}, train_pairs::Vector{Tuple{Int, Int}},
        width::Int, height::Int, output_path::AbstractString)
    plots = Any[]
    for (pair_pos, pair) in enumerate(train_pairs)
        m, n = pair
        push!(plots, a_panel(taus, vec(a_data[:, m, n]), vec(a_ref[:, m, n]), vec(a_nn[:, m, n]),
            pair_channel_label("A", pair, labels), rmse_data_nn[m, n], rmse_true_nn[m, n];
            show_legend=pair_pos == 1))
    end
    nrows, ncols = panel_grid_dims(length(train_pairs))
    fig = plot(plots...; layout=(nrows, ncols),
        size=(width * ncols, height * nrows),
        margin=7Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function create_cdot_figure(taus::Vector{Float64}, labels::Vector{String}, cdot_data::Array{Float64, 3}, cdot_ref::Array{Float64, 3}, cdot_nn::Array{Float64, 3},
        rmse_data_true::Matrix{Float64}, rmse_true_nn::Matrix{Float64}, train_pairs::Vector{Tuple{Int, Int}},
        width::Int, height::Int, output_path::AbstractString)
    plots = Any[]
    for (pair_pos, pair) in enumerate(train_pairs)
        m, n = pair
        push!(plots, cdot_panel(taus, vec(cdot_data[:, m, n]), vec(cdot_ref[:, m, n]), vec(cdot_nn[:, m, n]),
            pair_channel_label("Cdot", pair, labels), rmse_data_true[m, n], rmse_true_nn[m, n];
            show_legend=pair_pos == 1))
    end
    nrows, ncols = panel_grid_dims(length(train_pairs))
    fig = plot(plots...; layout=(nrows, ncols),
        size=(width * ncols, height * nrows),
        margin=7Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function mobility_reference_matrices(field::MobilityField)
    nx = length(field.xgrid)
    ny = length(field.ygrid)
    mats = [zeros(Float64, nx, ny) for _ in 1:4]
    @inbounds for j in 1:ny, i in 1:nx
        x = field.xgrid[i]
        y = field.ygrid[j]
        s11, s12, s22 = affine_symmetric_mobility_entries(x, y, field.meta)
        r = field.rgrid[i, j]
        mats[1][i, j] = s11
        mats[2][i, j] = s12 + r
        mats[3][i, j] = s12 - r
        mats[4][i, j] = s22
    end
    return mats
end

function mobility_predicted_matrices(model, field::MobilityField, μ::Vector{Float32}, σ::Vector{Float32},
        psd_jitter::Float64, batch_size::Int, device::DeviceConfig)
    nx = length(field.xgrid)
    ny = length(field.ygrid)
    points = Matrix{Float32}(undef, 2, nx * ny)
    cursor = 1
    @inbounds for j in 1:ny, i in 1:nx
        points[1, cursor] = Float32(field.xgrid[i])
        points[2, cursor] = Float32(field.ygrid[j])
        cursor += 1
    end
    pred = evaluate_mobility_matrices_nn(model, points, μ, σ, psd_jitter, batch_size, device)
    return [reshape(Float64.(pred[k, :]), nx, ny) for k in 1:4]
end

function mobility_heatmap_panel(field::MobilityField, values::Matrix{Float64}, title::String; clims=nothing)
    kwargs = Dict{Symbol, Any}(
        :xlabel => "x",
        :ylabel => "y",
        :aspect_ratio => :equal,
        :colorbar => true,
        :title => title,
        :color => :balance,
    )
    if clims !== nothing
        kwargs[:clims] = clims
    end
    return heatmap(field.xgrid, field.ygrid, values'; kwargs...)
end

function create_mobility_heatmap_figure(field::MobilityField, ref_mats::Vector{Matrix{Float64}}, nn_mats::Vector{Matrix{Float64}},
        support_mask::BitMatrix, delta_component_rmse::Vector{Float64}, output_path::AbstractString)
    names = ["11", "12", "21", "22"]
    panels = Any[]
    for k in 1:4
        masked_ref = copy(ref_mats[k])
        masked_nn = copy(nn_mats[k])
        masked_ref[.!support_mask] .= NaN
        masked_nn[.!support_mask] .= NaN
        component_values = vcat(vec(ref_mats[k][support_mask]), vec(nn_mats[k][support_mask]))
        component_clim = maximum(abs, component_values)
        push!(panels, mobility_heatmap_panel(field, masked_ref, "True M" * names[k]; clims=(-component_clim, component_clim)))
        push!(panels, mobility_heatmap_panel(field, masked_nn, @sprintf("NN M%s | ΔM RMSE %.3e", names[k], delta_component_rmse[k]);
            clims=(-component_clim, component_clim)))
    end
    fig = plot(panels...; layout=(2, 4), size=(4800, 2200), margin=6Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function text_panel(lines::Vector{String})
    p = plot(; axis=nothing, framestyle=:none, xlim=(0, 1), ylim=(0, 1))
    y = 0.96
    for line in lines
        annotate!(p, 0.02, y, text(line, 9, :left, :top, "DejaVu Sans"))
        y -= 0.075
    end
    return p
end

function channel_bar_panel(values::Vector{Float64}, labels::Vector{String}, title::String, ylabel::String; yscale=:identity)
    return bar(1:length(labels), values; title=title, ylabel=ylabel, xlabel="channel",
        legend=false, color=:royalblue3, xticks=(1:length(labels), labels), xrotation=45, yscale=yscale)
end

function create_training_diagnostics_figure(history::MobilityNNHistory, channel_labels::Vector{String},
        rmse_phys::Matrix{Float64}, rmse_norm::Matrix{Float64}, target_scale::Matrix{Float64}, final_lines::Vector{String},
        output_path::AbstractString, width::Int, height::Int)
    loss_plot = plot(history.epochs, history.train_loss; xlabel="epoch", ylabel="loss", label="normalized loss",
        color=:black, title="Training Loss")
    rmse_plot = plot(history.epochs, history.physical_rmse; xlabel="epoch", ylabel="RMSE", label="physical",
        color=:royalblue3, title="Training RMSE")
    plot!(rmse_plot, history.epochs, history.normalized_rmse; label="normalized", color=:darkorange)
    mean_plot = plot(history.epochs, history.mean_delta11; xlabel="epoch", ylabel="<ΔM>", label="<ΔM11>",
        color=:black, title="Mean ΔM Entries")
    plot!(mean_plot, history.epochs, history.mean_delta12; label="<ΔM12>", color=:royalblue3)
    plot!(mean_plot, history.epochs, history.mean_delta21; label="<ΔM21>", color=:darkorange)
    plot!(mean_plot, history.epochs, history.mean_delta22; label="<ΔM22>", color=:seagreen4)
    hline!(mean_plot, [0.0]; color=:gray70, linestyle=:dot, label=false)
    weight_plot = plot(history.epochs, history.weight_l2; xlabel="epoch", ylabel="mean |W|^2", label="weights",
        color=:gray30, title="Weight Decay Trace")
    phys_bar = channel_bar_panel(flatten_training_channel_matrix(rmse_phys), channel_labels,
        "Final Channel RMSE", "RMSE")
    norm_bar = channel_bar_panel(flatten_training_channel_matrix(rmse_norm), channel_labels,
        "Final Channel Normalized RMSE", "normalized RMSE")
    scale_bar = channel_bar_panel(flatten_training_channel_matrix(target_scale), channel_labels,
        "Observable RMS Scales", "scale"; yscale=:log10)
    info = text_panel(final_lines)
    fig = plot(loss_plot, rmse_plot, mean_plot, weight_plot, phys_bar, norm_bar, scale_bar, info;
        layout=(4, 2), size=(max(width * 2, 4200), max(height * 2, 3600)), margin=6Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function history_to_dict(history::MobilityNNHistory)
    return Dict{Symbol, Any}(
        :epochs => copy(history.epochs),
        :train_loss => copy(history.train_loss),
        :normalized_rmse => copy(history.normalized_rmse),
        :physical_rmse => copy(history.physical_rmse),
        :mean_abs_delta => copy(history.mean_abs_delta),
        :mean_delta11 => copy(history.mean_delta11),
        :mean_delta12 => copy(history.mean_delta12),
        :mean_delta21 => copy(history.mean_delta21),
        :mean_delta22 => copy(history.mean_delta22),
        :weight_l2 => copy(history.weight_l2),
    )
end

function save_mobility_model(path::AbstractString, model, history::MobilityNNHistory, μ::Vector{Float32}, σ::Vector{Float32},
        mean_x0::Matrix{Float32}, phi_est::Matrix{Float64}, target_scale::Matrix{Float64},
    channel_labels::Vector{String}, train_pairs::Vector{Tuple{Int, Int}}, train_pair_weights::Vector{Float64})
    bson(path, Dict{Symbol, Any}(
        :model_kind => "full_mobility_affine_benchmark",
    :training_target_source => TRAINING_TARGET_SOURCE,
        :host_model => cpu(model),
        :history => history_to_dict(history),
        :input_mean => Array(μ),
        :input_scale => Array(σ),
        :mean_x0 => Array(mean_x0),
        :phi_est => copy(phi_est),
        :target_scale => copy(target_scale),
        :training_channel_labels => copy(channel_labels),
        :training_pairs => [(pair[1], pair[2]) for pair in train_pairs],
        :training_pair_weights => copy(train_pair_weights),
    ))
    return nothing
end

function save_artifacts(path::AbstractString, lag_times::Vector{Float64}, labels::Vector{String},
        a_data::Array{Float64, 3}, a_ref::Array{Float64, 3}, a_nn::Array{Float64, 3},
        cdot_data::Array{Float64, 3}, cdot_ref::Array{Float64, 3}, cdot_nn::Array{Float64, 3},
        target_scale::Matrix{Float64}, channel_labels::Vector{String}, train_pairs::Vector{Tuple{Int, Int}},
        train_pair_weights::Vector{Float64}, phi_est::Matrix{Float64},
        field_component_rmse::Vector{Float64}, field_total_rmse::Float64, selections::Matrix{SmoothingSelection})
    bson(path, Dict{Symbol, Any}(
    :training_target_source => TRAINING_TARGET_SOURCE,
        :lag_times => copy(lag_times),
        :observable_labels => copy(labels),
        :a_data => a_data,
        :a_true => a_ref,
        :a_nn => a_nn,
        :cdot_data => cdot_data,
        :cdot_true => cdot_ref,
        :cdot_nn => cdot_nn,
        :phi_est => copy(phi_est),
        :target_scale => copy(target_scale),
        :training_channel_labels => copy(channel_labels),
        :training_pairs => [(pair[1], pair[2]) for pair in train_pairs],
        :training_pair_weights => copy(train_pair_weights),
        :field_component_rmse => copy(field_component_rmse),
        :field_total_rmse => field_total_rmse,
        :selected_smoothing => [selections[m, n].smoothing for m in axes(selections, 1), n in axes(selections, 2)],
    ))
    return nothing
end

function training_pairs_line(train_pairs::Vector{Tuple{Int, Int}})
    return "Training pairs (m,n): [" * join([@sprintf("(%d,%d)", pair[1], pair[2]) for pair in train_pairs], ", ") * "]"
end

function training_pair_weights_line(train_pair_weights::Vector{Float64})
    return "Training pair weights: [" * join([@sprintf("%.3f", weight) for weight in train_pair_weights], ", ") * "]"
end

function matrix_to_lines(name::AbstractString, mat::AbstractMatrix{<:Real})
    return [
        name * " = [",
        @sprintf("  %.8e  %.8e", mat[1, 1], mat[1, 2]),
        @sprintf("  %.8e  %.8e", mat[2, 1], mat[2, 2]),
        "]",
    ]
end

function write_metrics_report(path::AbstractString, labels::Vector{String},
        phi_est::Matrix{Float64}, m_true_mean::Matrix{Float64}, m_nn_mean::Matrix{Float64},
        delta_true_mean::Matrix{Float64}, delta_nn_mean::Matrix{Float64},
        a_rmse_data_true::Matrix{Float64}, a_rmse_data_nn::Matrix{Float64}, a_rmse_true_nn::Matrix{Float64},
        cdot_rmse_data_true::Matrix{Float64}, cdot_rmse_true_nn::Matrix{Float64},
    channel_labels::Vector{String}, train_pairs::Vector{Tuple{Int, Int}}, train_pair_weights::Vector{Float64},
        train_rmse_data_phys::Matrix{Float64}, train_rmse_data_norm::Matrix{Float64},
        train_rmse_true_phys::Matrix{Float64}, target_scale::Matrix{Float64},
        field_component_rmse::Vector{Float64}, field_total_rmse::Float64, field_relative_rmse::Float64, field_r2::Float64,
        point_component_rmse::Vector{Float64}, point_total_rmse::Float64, point_relative_rmse::Float64)
    open(path, "w") do io
        println(io, "fit_dm.jl metrics report")
        println(io, "Training target source: " * TRAINING_TARGET_SOURCE)
        println(io, "Training channel labels: " * join(channel_labels, ", "))
    println(io, training_pairs_line(train_pairs))
        println(io, training_pair_weights_line(train_pair_weights))
        println(io)
        for line in matrix_to_lines("Phi", phi_est)
            println(io, line)
        end
        println(io)
        for line in matrix_to_lines("<M> true", m_true_mean)
            println(io, line)
        end
        println(io)
        for line in matrix_to_lines("<M> NN", m_nn_mean)
            println(io, line)
        end
        println(io)
        for line in matrix_to_lines("<ΔM> true", delta_true_mean)
            println(io, line)
        end
        println(io)
        for line in matrix_to_lines("<ΔM> NN", delta_nn_mean)
            println(io, line)
        end
        println(io)
        println(io, @sprintf("||<ΔM>_NN-<ΔM>_true||_F = %.8e", norm(delta_nn_mean .- delta_true_mean)))
        println(io, @sprintf("Heatmap-on-support RMSE ΔM total = %.8e", field_total_rmse))
        println(io, @sprintf("Heatmap-on-support relative RMSE ΔM = %.8e", field_relative_rmse))
        println(io, @sprintf("Heatmap-on-support R2 ΔM = %.8e", field_r2))
        for (idx, name) in enumerate(("11", "12", "21", "22"))
            println(io, @sprintf("Heatmap-on-support component RMSE ΔM%s = %.8e", name, field_component_rmse[idx]))
        end
        println(io, @sprintf("Pointwise RMSE ΔM total = %.8e", point_total_rmse))
        println(io, @sprintf("Pointwise relative RMSE ΔM = %.8e", point_relative_rmse))
        for (idx, name) in enumerate(("11", "12", "21", "22"))
            println(io, @sprintf("Pointwise component RMSE ΔM%s = %.8e", name, point_component_rmse[idx]))
        end
        println(io)
        println(io, "Training-channel diagnostics")
        for (idx, label) in enumerate(channel_labels)
            println(io, @sprintf("%-12s | RMSE data/NN = %.8e | normalized = %.8e | RMSE true/NN = %.8e | scale = %.8e",
                label,
                train_rmse_data_phys[idx, 1],
                train_rmse_data_norm[idx, 1],
                train_rmse_true_phys[idx, 1],
                target_scale[idx, 1]))
        end
        println(io)
        println(io, @sprintf("Mean training-channel RMSE data/NN = %.8e", mean(train_rmse_data_phys)))
        println(io, @sprintf("Mean training-channel normalized RMSE = %.8e", mean(train_rmse_data_norm)))
        println(io, @sprintf("Mean training-channel RMSE true/NN = %.8e", mean(train_rmse_true_phys)))
        println(io)
        println(io, "Full A-operator RMSE table")
        for m in 1:length(labels), n in 1:length(labels)
            println(io, @sprintf("%-16s | data/true = %.8e | data/NN = %.8e | true/NN = %.8e",
                a_channel_label(m, n, labels), a_rmse_data_true[m, n], a_rmse_data_nn[m, n], a_rmse_true_nn[m, n]))
        end
        println(io)
        println(io, "Full Cdot RMSE table")
        for m in 1:length(labels), n in 1:length(labels)
            println(io, @sprintf("%-16s | data/true = %.8e | true/NN = %.8e",
                cdot_channel_label(m, n, labels), cdot_rmse_data_true[m, n], cdot_rmse_true_nn[m, n]))
        end
    end
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    plain_bson = resolve_path(base_dir, params.plain_score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    a_png = resolve_path(base_dir, params.output_a_png)
    cphi_png = resolve_path(base_dir, params.output_cphi_png)
    mobility_png = resolve_path(base_dir, params.output_mobility_png)
    training_png = resolve_path(base_dir, params.output_training_png)
    mobility_bson = resolve_path(base_dir, params.output_mobility_bson)
    metrics_txt = resolve_path(base_dir, params.output_metrics_txt)
    phi_bson = resolve_path(base_dir, params.output_phi_bson)
    artifact_bson = resolve_path(base_dir, params.output_artifact_bson)
    ensure_parent_dir(a_png)
    ensure_parent_dir(cphi_png)
    ensure_parent_dir(mobility_png)
    ensure_parent_dir(training_png)
    ensure_parent_dir(mobility_bson)
    ensure_parent_dir(metrics_txt)
    ensure_parent_dir(phi_bson)
    ensure_parent_dir(artifact_bson)

    device = detect_device()
    @printf("Analysis device: %s\n", device.name)

    plain_model, joint_model, joint_meta = load_models(plain_bson, joint_bson, device)
    sampler = build_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min, params.lag_stride)
    phi_est = load_or_estimate_phi_from_data(phi_bson, params.force_recompute_phi,
        sampler.states, sampler.times, sampler.start_idx)
    obs_ctx = build_observable_context(sampler.states, sampler.start_idx)
    full_labels = observable_labels()
    active_obs_indices, local_train_pairs, labels = active_observable_subset(params.train_pairs)
    channel_labels = training_channel_labels(full_labels, params.train_pairs)
    @printf("Phi      = [[%.6f, %.6f], [%.6f, %.6f]]\n",
        phi_est[1, 1], phi_est[1, 2], phi_est[2, 1], phi_est[2, 2])

    rng_obs = MersenneTwister(20260417)
    cphi = estimate_observed_cphi(sampler.states, sampler.start_idx, sampler.lag_steps,
        obs_ctx, params.use_all_observed_pairs, 10000, rng_obs, active_obs_indices)
    cdot_data = zeros(Float64, size(cphi))
    nobs = length(labels)
    selections = Matrix{SmoothingSelection}(undef, nobs, nobs)
    for m in 1:nobs, n in 1:nobs
        _, deriv_vals, selection = select_smoothing_protocol_dataonly(
            sampler.lag_times,
            vec(cphi[:, m, n]),
            params.cphi_spline_smoothing,
            params.cphi_smoothing_rel_grid,
            params.cphi_smoothing_rmse_tolerance,
        )
        cdot_data[:, m, n] .= deriv_vals
        selections[m, n] = selection
    end

    rng_sphi = MersenneTwister(20260418)
    gamma_term = estimate_gamma_term(sampler.states, sampler.start_idx, sampler.lag_steps,
        obs_ctx, phi_est, params.use_all_sphi_pairs, params.a_pairs_per_tau, rng_sphi,
        plain_model, params.score_batch_size, device, active_obs_indices)
    a_data = gamma_term .- cdot_data

    target_train = extract_training_channels(a_data, local_train_pairs)
    reliability_pair_weights, pair_instability = estimate_reliability_pair_weights(
        sampler.states, sampler.start_idx, sampler.lag_steps, sampler.lag_times,
        obs_ctx, phi_est, target_train, local_train_pairs, active_obs_indices, params, plain_model, device)
    effective_train_pair_weights = params.train_pair_weights .* reliability_pair_weights
    effective_train_pair_weights ./= mean(effective_train_pair_weights)
    μ, σ = input_normalization_stats(sampler.states, sampler.start_idx)
    @printf("Training channels: %s\n", join(channel_labels, ", "))
    if params.mobility_nn_reliability_mode != "none"
        @printf("Reliability weights: %s\n", join([@sprintf("%.3f", weight) for weight in reliability_pair_weights], ", "))
        @printf("Effective training weights: %s\n", join([@sprintf("%.3f", weight) for weight in effective_train_pair_weights], ", "))
    end
    @printf("Building NN training cache with %d pairs/tau\n", params.mobility_nn_pairs_per_tau)
    cache = build_mobility_training_cache(sampler, obs_ctx, plain_model, joint_model, joint_meta, params, device;
        pair_seed=20260408, anchor_seed=20260411)
    validation_caches = MobilityNNCache[]
    for (seed_idx, pair_seed) in enumerate(params.mobility_nn_validation_pair_seeds)
        push!(validation_caches, build_mobility_training_cache(sampler, obs_ctx, plain_model, joint_model, joint_meta, params, device;
            pair_seed=pair_seed, anchor_seed=pair_seed + 10_000 + seed_idx))
    end
    @printf("Training mobility NN on normalized A-channels from %s\n", TRAINING_TARGET_SOURCE)
    mobility_model, history, target_scale = train_mobility_model(cache, validation_caches, target_train, phi_est, μ, σ, params,
        effective_train_pair_weights, device)

    rng_nn = MersenneTwister(20260420)
    a_nn, cdot_nn = estimate_nn_reference_operators(sampler.states, sampler.start_idx, sampler.lag_steps, sampler.lag_times,
        obs_ctx, params.use_all_a_pairs, params.a_pairs_per_tau, rng_nn,
        plain_model, joint_model, joint_meta, mobility_model, μ, σ, params.mobility_nn_psd_jitter,
        params.score_batch_size, params.joint_batch_size, device, phi_est, active_obs_indices)

    meta = load_affine_model_metadata(input_hdf5)
    @printf("Estimating true mobility field on %d x %d grid\n", params.mobility_grid_nx, params.mobility_grid_ny)
    field = estimate_r_field(plain_model, sampler.states, sampler.start_idx, meta,
        params.mobility_grid_nx, params.mobility_grid_ny, params.mobility_ridge, params.grid_pad_fraction,
        params.score_batch_size, device)
    m_true_mean = mean_true_mobility(field, sampler.states, sampler.start_idx)
    delta_true_mean = m_true_mean .- phi_est
    @printf("<M> true = [[%.6f, %.6f], [%.6f, %.6f]]\n",
        m_true_mean[1, 1], m_true_mean[1, 2], m_true_mean[2, 1], m_true_mean[2, 2])

    rng_ref = MersenneTwister(20260419)
    a_ref, cdot_ref = estimate_true_reference_operators(sampler.states, sampler.start_idx, sampler.lag_steps, sampler.lag_times,
        obs_ctx, params.use_all_a_pairs, params.a_pairs_per_tau, rng_ref,
        plain_model, joint_model, joint_meta, field, params.score_batch_size, params.joint_batch_size, device, phi_est,
        active_obs_indices)
    split_a_ref = gamma_term .- cdot_ref
    @printf("Split consistency max |A_true - (gamma - Cdot_true)| = %.6e\n", maximum(abs.(a_ref .- split_a_ref)))

    a_rmse_data_true = zeros(Float64, nobs, nobs)
    a_rmse_data_nn = zeros(Float64, nobs, nobs)
    a_rmse_true_nn = zeros(Float64, nobs, nobs)
    cdot_rmse_data_true = zeros(Float64, nobs, nobs)
    cdot_rmse_true_nn = zeros(Float64, nobs, nobs)
    for m in 1:nobs, n in 1:nobs
        a_rmse_data_true[m, n] = sqrt(mean((a_data[:, m, n] .- a_ref[:, m, n]) .^ 2))
        a_rmse_data_nn[m, n] = sqrt(mean((a_data[:, m, n] .- a_nn[:, m, n]) .^ 2))
        a_rmse_true_nn[m, n] = sqrt(mean((a_ref[:, m, n] .- a_nn[:, m, n]) .^ 2))
        cdot_rmse_data_true[m, n] = sqrt(mean((cdot_data[:, m, n] .- cdot_ref[:, m, n]) .^ 2))
        cdot_rmse_true_nn[m, n] = sqrt(mean((cdot_ref[:, m, n] .- cdot_nn[:, m, n]) .^ 2))
    end

    train_pred = evaluate_training_model_on_cache(mobility_model, cache, μ, σ, Float32.(phi_est),
        params.mobility_nn_psd_jitter, params.mobility_nn_tau_batch, device)
    train_target_true = extract_training_channels(a_ref, local_train_pairs)
    train_rmse_data_phys = channel_rmse_matrix(target_train, train_pred)
    train_rmse_data_norm = channel_normalized_rmse_matrix(target_train, train_pred, target_scale)
    train_rmse_true_phys = channel_rmse_matrix(train_target_true, train_pred)

    @printf("Saving A figure to %s\n", a_png)
    create_a_figure(sampler.lag_times, labels, a_data, a_ref, a_nn,
        a_rmse_data_nn, a_rmse_true_nn, local_train_pairs, params.figure_width, params.figure_height, a_png)
    @printf("Saving Cdot figure to %s\n", cphi_png)
    create_cdot_figure(sampler.lag_times, labels, cdot_data, cdot_ref, cdot_nn,
        cdot_rmse_data_true, cdot_rmse_true_nn, local_train_pairs, params.figure_width, params.figure_height, cphi_png)

    ref_mats = mobility_reference_matrices(field)
    nn_mats = mobility_predicted_matrices(mobility_model, field, μ, σ, params.mobility_nn_psd_jitter,
        params.score_batch_size, device)
    delta_ref_mats = [ref_mats[1] .- phi_est[1, 1], ref_mats[2] .- phi_est[1, 2], ref_mats[3] .- phi_est[2, 1], ref_mats[4] .- phi_est[2, 2]]
    delta_nn_mats = [nn_mats[1] .- phi_est[1, 1], nn_mats[2] .- phi_est[1, 2], nn_mats[3] .- phi_est[2, 1], nn_mats[4] .- phi_est[2, 2]]
    support_mask = observed_support_mask(field, sampler.states, sampler.start_idx)
    field_component_rmse = [masked_rmse(delta_ref_mats[k], delta_nn_mats[k], support_mask) for k in 1:4]
    delta_ref_stack = vcat([vec(delta_ref_mats[k][support_mask]) for k in 1:4]...)
    delta_nn_stack = vcat([vec(delta_nn_mats[k][support_mask]) for k in 1:4]...)
    field_total_rmse = sqrt(mean((delta_nn_stack .- delta_ref_stack) .^ 2))
    field_relative_rmse = field_total_rmse / max(sqrt(mean(delta_ref_stack .^ 2)), eps(Float64))
    field_r2 = 1.0 - sum((delta_nn_stack .- delta_ref_stack) .^ 2) / max(sum((delta_ref_stack .- mean(delta_ref_stack)) .^ 2), eps(Float64))
    support_points = observed_state_points(sampler.states, sampler.start_idx)
    point_component_rmse, point_total_rmse, point_relative_rmse = pointwise_delta_rmse(
        mobility_model, field, support_points, phi_est, μ, σ,
        params.mobility_nn_psd_jitter, params.score_batch_size, device)
    m_nn_mean = mean_nn_mobility(mobility_model, sampler.states, sampler.start_idx, μ, σ,
        params.mobility_nn_psd_jitter, params.score_batch_size, device)
    delta_nn_mean = m_nn_mean .- phi_est

    for (idx, label) in enumerate(channel_labels)
        @printf("%-12s | RMSE data/NN = %.6e | normalized = %.6e | RMSE true/NN = %.6e | scale = %.6e\n",
            label,
            train_rmse_data_phys[idx, 1],
            train_rmse_data_norm[idx, 1],
            train_rmse_true_phys[idx, 1],
            target_scale[idx, 1])
    end
    @printf("Mean A RMSE data/NN over selected %dx%d subset = %.6e | true/NN = %.6e\n",
        nobs, nobs, mean(a_rmse_data_nn), mean(a_rmse_true_nn))
    @printf("Heatmap RMSE ΔM total = %.6e | relative = %.6e | R2 = %.6e\n",
        field_total_rmse, field_relative_rmse, field_r2)
    @printf("Pointwise RMSE ΔM total = %.6e | relative = %.6e\n",
        point_total_rmse, point_relative_rmse)

    @printf("Saving mobility heatmaps to %s\n", mobility_png)
    create_mobility_heatmap_figure(field, ref_mats, nn_mats, support_mask, field_component_rmse, mobility_png)

    final_lines = [
        "Training target = " * TRAINING_TARGET_SOURCE,
        @sprintf("Mean training-channel RMSE data/NN = %.3e", mean(train_rmse_data_phys)),
        @sprintf("Mean training-channel normalized RMSE = %.3e", mean(train_rmse_data_norm)),
        @sprintf("Mean training-channel RMSE true/NN = %.3e", mean(train_rmse_true_phys)),
        @sprintf("Heatmap ΔM RMSE total = %.3e", field_total_rmse),
        @sprintf("Heatmap ΔM relative RMSE = %.3e", field_relative_rmse),
        @sprintf("Heatmap-on-support ΔM R2 = %.3e", field_r2),
        @sprintf("Pointwise ΔM RMSE total = %.3e", point_total_rmse),
        @sprintf("Pointwise ΔM relative RMSE = %.3e", point_relative_rmse),
        @sprintf("||<ΔM>_NN-<ΔM>_true||_F = %.3e", norm(delta_nn_mean .- delta_true_mean)),
        @sprintf("Mean A data/NN RMSE over selected %dx%d subset = %.3e", nobs, nobs, mean(a_rmse_data_nn)),
        @sprintf("Mean A true/NN RMSE over selected %dx%d subset = %.3e", nobs, nobs, mean(a_rmse_true_nn)),
        @sprintf("Mean Cdot true/NN RMSE over selected %dx%d subset = %.3e", nobs, nobs, mean(cdot_rmse_true_nn)),
        @sprintf("pairs/tau = %d", params.mobility_nn_pairs_per_tau),
        @sprintf("tau batch = %d", params.mobility_nn_tau_batch),
        @sprintf("epochs = %d", params.mobility_nn_epochs),
        @sprintf("lr = %.2e", params.mobility_nn_learning_rate),
        @sprintf("mean penalty = %.2e", params.mobility_nn_mean_penalty_weight),
        @sprintf("anchor rms penalty = %.2e", params.mobility_nn_anchor_rms_penalty_weight),
        @sprintf("weight decay = %.2e", params.mobility_nn_weight_decay),
        "checkpoint metric = " * params.mobility_nn_checkpoint_metric,
        @sprintf("validation caches = %d", length(params.mobility_nn_validation_pair_seeds)),
        "model = " * params.mobility_nn_model,
        @sprintf("scale floor = %.2e", params.mobility_nn_scale_floor),
        isempty(params.mobility_nn_widths) ? "widths = affine-only" : "widths = " * join(params.mobility_nn_widths, " x "),
    ]
    if params.mobility_nn_reliability_mode != "none"
        push!(final_lines, "reliability weighting = " * params.mobility_nn_reliability_mode)
        push!(final_lines, @sprintf("reliability strength = %.2f", params.mobility_nn_reliability_strength))
        push!(final_lines, "reliability weights = [" * join([@sprintf("%.2f", weight) for weight in reliability_pair_weights], ", ") * "]")
        push!(final_lines, "pair instability = [" * join([@sprintf("%.2f", value) for value in pair_instability], ", ") * "]")
    end
    if params.mobility_nn_mean_penalty_final_scale != 1.0
        push!(final_lines, @sprintf("mean penalty final scale = %.2f", params.mobility_nn_mean_penalty_final_scale))
    end
    if params.mobility_nn_anchor_rms_penalty_final_scale != 1.0
        push!(final_lines, @sprintf("anchor penalty final scale = %.2f", params.mobility_nn_anchor_rms_penalty_final_scale))
    end
    if any(params.mobility_nn_mean_entry_weights .!= 1.0)
        push!(final_lines, @sprintf("mean entry weights = [%.2f %.2f; %.2f %.2f]",
            params.mobility_nn_mean_entry_weights[1, 1], params.mobility_nn_mean_entry_weights[1, 2],
            params.mobility_nn_mean_entry_weights[2, 1], params.mobility_nn_mean_entry_weights[2, 2]))
    end
    if any(params.mobility_nn_anchor_rms_entry_weights .!= 1.0)
        push!(final_lines, @sprintf("anchor entry weights = [%.2f %.2f; %.2f %.2f]",
            params.mobility_nn_anchor_rms_entry_weights[1, 1], params.mobility_nn_anchor_rms_entry_weights[1, 2],
            params.mobility_nn_anchor_rms_entry_weights[2, 1], params.mobility_nn_anchor_rms_entry_weights[2, 2]))
    end
    @printf("Saving training diagnostics to %s\n", training_png)
    create_training_diagnostics_figure(history, channel_labels, train_rmse_data_phys, train_rmse_data_norm, target_scale,
        final_lines, training_png, params.figure_width, params.figure_height)

    @printf("Saving mobility model to %s\n", mobility_bson)
    save_mobility_model(mobility_bson, mobility_model, history, μ, σ, cache.mean_x0, phi_est, target_scale,
        channel_labels, params.train_pairs, effective_train_pair_weights)
    @printf("Saving artifact bundle to %s\n", artifact_bson)
    save_artifacts(artifact_bson, sampler.lag_times, labels, a_data, a_ref, a_nn, cdot_data, cdot_ref, cdot_nn,
        target_scale, channel_labels, local_train_pairs, effective_train_pair_weights,
        phi_est, field_component_rmse, field_total_rmse, selections)
    @printf("Writing metrics report to %s\n", metrics_txt)
    write_metrics_report(metrics_txt, labels, phi_est, m_true_mean, m_nn_mean, delta_true_mean, delta_nn_mean,
        a_rmse_data_true, a_rmse_data_nn, a_rmse_true_nn, cdot_rmse_data_true, cdot_rmse_true_nn,
        channel_labels, local_train_pairs, effective_train_pair_weights,
        train_rmse_data_phys, train_rmse_data_norm, train_rmse_true_phys, target_scale,
        field_component_rmse, field_total_rmse, field_relative_rmse, field_r2,
        point_component_rmse, point_total_rmse, point_relative_rmse)

    @printf("Done. Mean training-channel RMSE data/NN = %.6e | normalized = %.6e | heatmap ΔM RMSE = %.6e | pointwise ΔM RMSE = %.6e\n",
        mean(train_rmse_data_phys), mean(train_rmse_data_norm), field_total_rmse, point_total_rmse)
    return nothing
end

include(joinpath(@__DIR__, "src", "mobility_forward_validation.jl"))

function string_any_dict(raw)::Dict{String, Any}
    return Dict{String, Any}(String(key) => deepcopy(value) for (key, value) in pairs(raw))
end

function merge_string_dicts(base::Dict{String, Any}, overlay_raw)
    merged = deepcopy(base)
    for (key, value) in pairs(overlay_raw)
        merged[String(key)] = deepcopy(value)
    end
    return merged
end

function repo_resolve_path(path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(@__DIR__, String(path)))
end

function managed_config(raw::Dict{String, Any})
    if haskey(raw, "run")
        return true
    end
    if haskey(raw, "forward_validation")
        forward_cfg = raw["forward_validation"]
        return forward_cfg isa AbstractDict && haskey(forward_cfg, "enabled")
    end
    return false
end

function forward_validation_enabled(raw::Dict{String, Any})
    forward_cfg = get(raw, "forward_validation", Dict{String, Any}())
    return forward_cfg isa AbstractDict && Bool(get(forward_cfg, "enabled", false))
end

function require_managed_table(raw::Dict{String, Any}, key::String)
    require_condition(haskey(raw, key), "Managed mobility config is missing [$(key)].")
    require_condition(raw[key] isa AbstractDict, "Managed mobility config section [$(key)] must be a TOML table.")
    return string_any_dict(raw[key])
end

function existing_run_dir_for_config(param_file::AbstractString)
    dir = dirname(abspath(param_file))
    runs_root = joinpath(@__DIR__, "runs")
    if dirname(dir) == runs_root && occursin(r"^run_\d{3}$", basename(dir))
        return dir
    end
    return nothing
end

function next_run_dir(runs_root::AbstractString)
    mkpath(runs_root)
    max_id = 0
    for entry in readdir(runs_root)
        match_obj = match(r"^run_(\d{3})$", entry)
        match_obj === nothing && continue
        max_id = max(max_id, parse(Int, match_obj.captures[1]))
    end
    return joinpath(runs_root, @sprintf("run_%03d", max_id + 1))
end

function managed_run_paths(param_file::AbstractString)
    existing_dir = existing_run_dir_for_config(param_file)
    reused_existing_dir = existing_dir !== nothing
    run_dir = reused_existing_dir ? existing_dir : next_run_dir(joinpath(@__DIR__, "runs"))
    figure_dir = joinpath(run_dir, "figures")
    data_dir = joinpath(run_dir, "data")
    mkpath(figure_dir)
    mkpath(data_dir)
    paths = ManagedRunPaths(
        run_dir,
        joinpath(run_dir, "config.toml"),
        joinpath(run_dir, "run_info.toml"),
        joinpath(run_dir, "fit_stage.toml"),
        joinpath(run_dir, "forward_validation_stage.toml"),
        figure_dir,
        data_dir,
        joinpath(figure_dir, "fit_A.png"),
        joinpath(figure_dir, "fit_cdot.png"),
        joinpath(figure_dir, "fit_M.png"),
        joinpath(figure_dir, "fit_training.png"),
        joinpath(data_dir, "mobility_model.bson"),
        joinpath(data_dir, "mobility_metrics.txt"),
        joinpath(data_dir, "mobility_phi.bson"),
        joinpath(data_dir, "mobility_artifacts.bson"),
        joinpath(figure_dir, "forward_validation_stats.png"),
        joinpath(figure_dir, "forward_validation_cphi.png"),
        joinpath(data_dir, "forward_validation_metrics.txt"),
        joinpath(data_dir, "forward_validation_artifacts.bson"),
        joinpath(data_dir, "forward_validation_trajectories.h5"),
    )
    return paths, reused_existing_dir
end

function copy_managed_config(param_file::AbstractString, paths::ManagedRunPaths)
    source = abspath(param_file)
    destination = paths.config_copy
    if source != destination
        cp(source, destination; force=true)
    end
    return nothing
end

function write_toml_file(path::AbstractString, raw::Dict{String, Any})
    ensure_parent_dir(path)
    open(path, "w") do io
        TOML.print(io, raw)
    end
    return nothing
end

function forward_validation_defaults()
    return Dict{String, Any}(
        "burnin_fraction" => 0.1,
        "simulation" => Dict{String, Any}(
            "dt" => 0.005,
            "save_stride" => 1,
            "total_time" => 32.0,
            "burnin_time" => 8.0,
            "ntrajectories" => 512,
            "seed" => 20260416,
            "use_common_random_numbers" => true,
            "eval_batch_size" => 2048,
            "clamp_eval_to_support" => true,
            "hard_clamp_state" => false,
            "support_pad_fraction" => 0.10,
            "mobility_psd_jitter" => 1.0e-5,
            "diffusion_floor" => 1.0e-5,
            "diffusion_cap_quantile" => 0.995,
            "diffusion_cap_multiplier" => 1.25,
        ),
        "analysis" => Dict{String, Any}(
            "pdf_bins" => 160,
            "pdf_max_samples" => 300000,
            "correlation_stride" => 4,
            "correlation_max_time" => 8.0,
            "correlation_threshold" => 0.05,
            "use_artifact_lags" => true,
            "cphi_max_time" => 1.5,
            "cphi_stride" => 1,
            "auxiliary_max_samples" => 150000,
        ),
        "divergence" => Dict{String, Any}(
            "method" => "forwarddiff_grid",
            "grid_nx" => 160,
            "grid_ny" => 160,
            "finite_difference_eps" => 1.0e-3,
        ),
        "figure" => Dict{String, Any}(
            "width" => 2400,
            "height" => 2100,
        ),
    )
end

function build_managed_fit_stage_config(raw::Dict{String, Any}, paths::ManagedRunPaths)
    fit_data_cfg = require_managed_table(raw, "data")
    fit_data_cfg["input_hdf5"] = repo_resolve_path(String(fit_data_cfg["input_hdf5"]))
    fit_data_cfg["plain_score_bson"] = repo_resolve_path(String(fit_data_cfg["plain_score_bson"]))
    fit_data_cfg["joint_score_bson"] = repo_resolve_path(String(fit_data_cfg["joint_score_bson"]))

    stage_raw = Dict{String, Any}(
        "data" => fit_data_cfg,
        "evaluation" => require_managed_table(raw, "evaluation"),
        "mobility" => require_managed_table(raw, "mobility"),
        "mobility_nn" => require_managed_table(raw, "mobility_nn"),
        "figure" => require_managed_table(raw, "figure"),
        "output" => Dict{String, Any}(
            "a_png" => paths.fit_a_png,
            "cphi_png" => paths.fit_cphi_png,
            "mobility_png" => paths.fit_mobility_png,
            "training_png" => paths.fit_training_png,
            "mobility_bson" => paths.fit_model_bson,
            "metrics_txt" => paths.fit_metrics_txt,
            "phi_bson" => paths.fit_phi_bson,
            "artifact_bson" => paths.fit_artifact_bson,
        ),
    )
    if haskey(raw, "training_observables")
        stage_raw["training_observables"] = string_any_dict(raw["training_observables"])
    end
    return stage_raw
end

function build_managed_forward_validation_stage_config(raw::Dict{String, Any}, paths::ManagedRunPaths)
    fit_data_cfg = require_managed_table(raw, "data")
    defaults = forward_validation_defaults()
    forward_cfg = haskey(raw, "forward_validation") ? string_any_dict(raw["forward_validation"]) : Dict{String, Any}()
    sim_cfg = merge_string_dicts(string_any_dict(defaults["simulation"]), get(forward_cfg, "simulation", Dict{String, Any}()))
    analysis_cfg = merge_string_dicts(string_any_dict(defaults["analysis"]), get(forward_cfg, "analysis", Dict{String, Any}()))
    divergence_cfg = merge_string_dicts(string_any_dict(defaults["divergence"]), get(forward_cfg, "divergence", Dict{String, Any}()))
    figure_cfg = merge_string_dicts(string_any_dict(defaults["figure"]), get(forward_cfg, "figure", Dict{String, Any}()))
    burnin_fraction = Float64(get(forward_cfg, "burnin_fraction", defaults["burnin_fraction"]))

    return Dict{String, Any}(
        "inputs" => Dict{String, Any}(
            "input_hdf5" => repo_resolve_path(String(fit_data_cfg["input_hdf5"])),
            "score_bson" => repo_resolve_path(String(fit_data_cfg["plain_score_bson"])),
            "mobility_model_bson" => paths.fit_model_bson,
            "mobility_artifact_bson" => paths.fit_artifact_bson,
            "phi_bson" => paths.fit_phi_bson,
            "burnin_fraction" => burnin_fraction,
        ),
        "simulation" => sim_cfg,
        "analysis" => analysis_cfg,
        "divergence" => divergence_cfg,
        "figure" => figure_cfg,
        "output" => Dict{String, Any}(
            "figure_stats_png" => paths.validation_stats_png,
            "figure_cphi_png" => paths.validation_cphi_png,
            "metrics_txt" => paths.validation_metrics_txt,
            "diagnostics_bson" => paths.validation_diagnostics_bson,
            "trajectories_hdf5" => paths.validation_trajectories_hdf5,
        ),
    )
end

function write_run_metadata(path::AbstractString, raw::Dict{String, Any}, source_config::AbstractString,
        paths::ManagedRunPaths, reused_existing_dir::Bool, enable_forward_validation::Bool)
    run_cfg = haskey(raw, "run") ? string_any_dict(raw["run"]) : Dict{String, Any}()
    metadata = Dict{String, Any}(
        "label" => String(get(run_cfg, "label", "")),
        "notes" => String(get(run_cfg, "notes", "")),
        "source_config" => abspath(source_config),
        "run_dir" => paths.run_dir,
        "reused_existing_run_dir" => reused_existing_dir,
        "forward_validation_enabled" => enable_forward_validation,
        "generated_files" => Dict{String, Any}(
            "config_copy" => paths.config_copy,
            "fit_stage_config" => paths.fit_stage_config,
            "forward_stage_config" => paths.forward_stage_config,
            "fit_metrics" => paths.fit_metrics_txt,
            "fit_model" => paths.fit_model_bson,
            "fit_artifacts" => paths.fit_artifact_bson,
            "fit_phi" => paths.fit_phi_bson,
            "forward_metrics" => paths.validation_metrics_txt,
            "forward_artifacts" => paths.validation_diagnostics_bson,
            "forward_trajectories" => paths.validation_trajectories_hdf5,
        ),
    )
    write_toml_file(path, metadata)
    return nothing
end

function run_managed_pipeline(param_file::AbstractString)
    raw = TOML.parsefile(param_file)
    if !managed_config(raw)
        return run_pipeline(param_file)
    end

    paths, reused_existing_dir = managed_run_paths(param_file)
    enable_forward_validation = forward_validation_enabled(raw)
    copy_managed_config(param_file, paths)
    write_run_metadata(paths.metadata_toml, raw, param_file, paths, reused_existing_dir, enable_forward_validation)

    fit_stage_raw = build_managed_fit_stage_config(raw, paths)
    write_toml_file(paths.fit_stage_config, fit_stage_raw)

    @printf("Run directory: %s\n", paths.run_dir)
    @printf("Saved managed config copy to %s\n", paths.config_copy)
    @printf("Launching mobility fit stage with %s\n", paths.fit_stage_config)
    run_pipeline(paths.fit_stage_config)

    if enable_forward_validation
        forward_stage_raw = build_managed_forward_validation_stage_config(raw, paths)
        write_toml_file(paths.forward_stage_config, forward_stage_raw)
        @printf("Launching forward validation stage with %s\n", paths.forward_stage_config)
        run_forward_validation(paths.forward_stage_config)
    else
        @printf("Forward validation disabled; run contains fit-stage outputs only.\n")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_managed_pipeline(param_file)
end
