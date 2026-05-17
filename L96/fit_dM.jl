#!/usr/bin/env julia

import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")
const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "fit_dM.toml")
const JOINT_STATE_CHANNELS = 2
const JOINT_INPUT_CHANNELS = 3
const OBSERVABLE_KEYS = (:coord, :var, :nn1, :nn2, :adv, :flux)

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

ensure_packages([
    "BSON",
    "CUDA",
    "CairoMakie",
    "cuDNN",
    "Dierckx",
    "FFTW",
    "Flux",
    "ForwardDiff",
    "Functors",
    "HDF5",
    "KernelDensity",
    "LaTeXStrings",
    "NNlib",
    "TOML",
])

using BSON
using CUDA
using cuDNN
using Dierckx
using FFTW
using Flux
using ForwardDiff
using Functors
using HDF5
using KernelDensity
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics
using TOML

include(joinpath(SCOREUNET_SRC, "Device.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "PeriodicConv.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "Blocks.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "UNet1D.jl"))
include(joinpath(SCOREUNET_SRC, "data", "DataPipeline.jl"))
include(joinpath(SCOREUNET_SRC, "training", "Trainer.jl"))

isfile(STYLE_FILE) || error("Shared figure style file not found: $(STYLE_FILE)")
const FIGURE_STYLE_LOADED = Ref(false)

function ensure_figure_support_loaded!()
    if !FIGURE_STYLE_LOADED[]
        @eval using CairoMakie
        include(STYLE_FILE)
        Base.invokelatest(CairoMakie.activate!)
        FIGURE_STYLE_LOADED[] = true
    end
    return nothing
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

struct JointScoreUNet{M}
    backbone::M
end

Functors.@functor JointScoreUNet (backbone,)

function (model::JointScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:JOINT_STATE_CHANNELS, :]
end

struct ObservableMeans
    var_mean::Float64
    nn1_mean::Float64
    nn2_mean::Float64
    adv_mean::Float64
    flux_mean::Float64
end

struct SmoothingSelection
    smoothing::Float64
    rmse::Float64
    roughness::Float64
end

struct FitDML96Params
    input_hdf5::String
    plain_score_bson::String
    joint_score_bson::String
    burnin_fraction::Float64
    lag_stride::Int
    pairs_per_tau::Int
    eval_batch_size::Int
    score_batch_size::Int
    joint_batch_size::Int
    cphi_spline_smoothing::Float64
    cphi_smoothing_rel_grid::Vector{Float64}
    cphi_smoothing_rmse_tolerance::Float64
    reference_mobility::String
    antisymmetric_offsets::Vector{Int}
    antisymmetric_feature_offsets::Vector{Int}
    antisymmetric_fit_samples::Int
    antisymmetric_fit_batch_size::Int
    antisymmetric_fit_ridge::Float64
    coordinate_separations::Vector{Int}
    nonlinear_separations::Vector{Int}
    plot_coordinate_separations::Vector{Int}
    plot_standard_nonlinear_separations::Vector{Int}
    plot_adv_flux_separations::Vector{Int}
    figure_width::Int
    figure_height::Int
    device_name::String
    seed::Int
    runs_root::String
    label::String
    notes::String
    mobility_nn_enabled::Bool
    mobility_nn_offsets::Vector{Int}
    mobility_nn_window_offsets::Vector{Int}
    mobility_nn_pairs_per_tau::Int
    mobility_nn_tau_batch::Int
    mobility_nn_anchor_states::Int
    mobility_nn_epochs::Int
    mobility_nn_eval_every::Int
    mobility_nn_learning_rate::Float64
    mobility_nn_lag_weight_power::Float64
    mobility_nn_zero_mean_penalty::Float64
    mobility_nn_zero_mean_penalty_final_scale::Float64
    mobility_nn_anchor_rms_penalty::Float64
    mobility_nn_anchor_rms_penalty_final_scale::Float64
    mobility_nn_current_action_penalty::Float64
    mobility_nn_weight_decay::Float64
    mobility_nn_checkpoint_metric::String
    mobility_nn_scale_floor::Float64
    mobility_nn_validation_pair_seeds::Vector{Int}
    mobility_nn_widths::Vector{Int}
    forward_validation_enabled::Bool
    forward_validation_dt::Float64
    forward_validation_save_stride::Int
    forward_validation_total_time::Float64
    forward_validation_burnin_time::Float64
    forward_validation_ntrajectories::Int
    forward_validation_seed::Int
    forward_validation_use_common_random_numbers::Bool
    forward_validation_clamp_eval_to_support::Bool
    forward_validation_hard_clamp_state::Bool
    forward_validation_support_pad_fraction::Float64
    forward_validation_pdf_bins::Int
    forward_validation_pdf_max_samples::Int
    forward_validation_correlation_stride::Int
    forward_validation_correlation_max_time::Float64
    forward_validation_correlation_threshold::Float64
    forward_validation_cphi_max_time::Float64
    forward_validation_cphi_stride::Int
    forward_validation_bivariate_offsets::Vector{Int}
    forward_validation_auxiliary_max_samples::Int
end

struct ManagedRunPaths
    run_dir::String
    config_copy::String
    metadata_toml::String
    figure_dir::String
    data_dir::String
    figure_png::String
    figure_pdf::String
    output_bson::String
    metrics_txt::String
    fit_a_png::String
    fit_cdot_png::String
    fit_training_png::String
    fit_mobility_png::String
    fit_model_bson::String
    forward_stats_png::String
    forward_channels_png::String
    forward_metrics_txt::String
    forward_diagnostics_bson::String
    forward_trajectories_hdf5::String
end

struct PairSampler
    times::Vector{Float64}
    states::Array{Float32, 3}
    start_idx::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    tau_min::Float64
    tau_max::Float64
    decorrelation_time::Float64
    save_dt::Float64
    F::Float64
    Q::Matrix{Float64}
end

struct LoadedModels
    score_model
    joint_model
    score_sigma::Float32
    joint_sigma::Float32
    score_mean::Vector{Float32}
    score_std::Vector{Float32}
    joint_mean::Vector{Float32}
    joint_std::Vector{Float32}
    tau_min::Float64
    tau_max::Float64
end

struct ReferenceMobilityResult
    active_mode::String
    q_only_matrix::Matrix{Float64}
    fitted_matrix::Union{Nothing, Matrix{Float64}}
    active_matrix::Matrix{Float64}
    antisymmetric_matrix::Union{Nothing, Matrix{Float64}}
    offsets::Vector{Int}
    feature_offsets::Vector{Int}
    quadratic_pairs::Vector{Tuple{Int, Int}}
    coefficients::Matrix{Float64}
    fit_condition::Float64
    fit_rms::Float64
    status::String
end

function require_condition(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(base_dir, String(path)))
end

function repo_resolve_path(path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(@__DIR__, String(path)))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function periodic(i::Int, K::Int)
    return mod1(i, K)
end

function to_host(x)
    return x isa AbstractArray && !(x isa Array) ? Array(x) : x
end

function describe_device(device::ExecutionDevice)
    if device isa GPUDevice
        return "GPU:" * join(device.ids, ",")
    end
    return "CPU"
end

function detect_device(name::AbstractString)
    normalized = uppercase(strip(name))
    normalized == "AUTO" && return CUDA.functional() ? select_device("GPU:0") : CPUDevice()
    normalized == "GPU" && return select_device("GPU:0")
    return select_device(name)
end

function string_any_dict(raw)::Dict{String, Any}
    return Dict{String, Any}(String(key) => deepcopy(value) for (key, value) in pairs(raw))
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

function managed_run_paths(param_file::AbstractString, runs_root::AbstractString)
    existing_dir = existing_run_dir_for_config(param_file)
    reused_existing_dir = existing_dir !== nothing
    run_dir = reused_existing_dir ? existing_dir : next_run_dir(runs_root)
    figure_dir = joinpath(run_dir, "figures")
    data_dir = joinpath(run_dir, "data")
    mkpath(figure_dir)
    mkpath(data_dir)
    return ManagedRunPaths(
        run_dir,
        joinpath(run_dir, "config.toml"),
        joinpath(run_dir, "run_info.toml"),
        figure_dir,
        data_dir,
        joinpath(figure_dir, "l96_cdot_reference_comparison.png"),
        joinpath(figure_dir, "l96_cdot_reference_comparison.pdf"),
        joinpath(data_dir, "l96_cdot_reference_outputs.bson"),
        joinpath(data_dir, "l96_cdot_reference_metrics.txt"),
        joinpath(figure_dir, "fit_A.png"),
        joinpath(figure_dir, "fit_cdot.png"),
        joinpath(figure_dir, "fit_training.png"),
        joinpath(figure_dir, "fit_mobility.png"),
        joinpath(data_dir, "mobility_model.bson"),
        joinpath(figure_dir, "forward_validation_stats.png"),
        joinpath(figure_dir, "forward_validation_channels.png"),
        joinpath(data_dir, "forward_validation_metrics.txt"),
        joinpath(data_dir, "forward_validation_artifacts.bson"),
        joinpath(data_dir, "forward_validation_trajectories.h5"),
    ), reused_existing_dir
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

function write_run_metadata(path::AbstractString, params::FitDML96Params, source_config::AbstractString,
        paths::ManagedRunPaths, reused_existing_dir::Bool)
    metadata = Dict{String, Any}(
        "label" => params.label,
        "notes" => params.notes,
        "source_config" => abspath(source_config),
        "run_dir" => paths.run_dir,
        "reused_existing_run_dir" => reused_existing_dir,
        "figure_png" => paths.figure_png,
        "figure_pdf" => paths.figure_pdf,
        "output_bson" => paths.output_bson,
        "metrics_txt" => paths.metrics_txt,
        "fit_a_png" => paths.fit_a_png,
        "fit_cdot_png" => paths.fit_cdot_png,
        "fit_training_png" => paths.fit_training_png,
        "fit_mobility_png" => paths.fit_mobility_png,
        "fit_model_bson" => paths.fit_model_bson,
        "forward_stats_png" => paths.forward_stats_png,
        "forward_channels_png" => paths.forward_channels_png,
        "forward_metrics_txt" => paths.forward_metrics_txt,
        "forward_diagnostics_bson" => paths.forward_diagnostics_bson,
        "forward_trajectories_hdf5" => paths.forward_trajectories_hdf5,
    )
    write_toml_file(path, metadata)
    return nothing
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data_cfg = raw["data"]
    eval_cfg = raw["evaluation"]
    ref_cfg = raw["reference"]
    mobility_nn_cfg = haskey(raw, "mobility_nn") ? raw["mobility_nn"] : Dict{String, Any}()
    forward_cfg = haskey(raw, "forward_validation") ? raw["forward_validation"] : Dict{String, Any}()
    fig_cfg = raw["figure"]
    run_cfg = haskey(raw, "run") ? raw["run"] : Dict{String, Any}()

    params = FitDML96Params(
        String(data_cfg["input_hdf5"]),
        String(data_cfg["plain_score_bson"]),
        String(data_cfg["joint_score_bson"]),
        Float64(data_cfg["burnin_fraction"]),
        Int(get(eval_cfg, "lag_stride", 1)),
        Int(get(eval_cfg, "pairs_per_tau", 60_000)),
        Int(get(eval_cfg, "eval_batch_size", 4096)),
        Int(get(eval_cfg, "score_batch_size", 8192)),
        Int(get(eval_cfg, "joint_batch_size", 4096)),
        Float64(get(eval_cfg, "cphi_spline_smoothing", 1.0)),
        Float64.(get(eval_cfg, "cphi_smoothing_rel_grid", [0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0])),
        Float64(get(eval_cfg, "cphi_smoothing_rmse_tolerance", 0.02)),
        String(get(ref_cfg, "reference_mobility", "local_antisymmetric_fit")),
        Int.(get(ref_cfg, "antisymmetric_offsets", [1, 2, 3])),
        Int.(get(ref_cfg, "antisymmetric_feature_offsets", [-3, -2, -1, 0, 1, 2, 3])),
        Int(get(ref_cfg, "antisymmetric_fit_samples", 150_000)),
        Int(get(ref_cfg, "antisymmetric_fit_batch_size", 4096)),
        Float64(get(ref_cfg, "antisymmetric_fit_ridge", 1.0e-4)),
        Int.(get(eval_cfg, "coordinate_separations", [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20])),
        Int.(get(eval_cfg, "nonlinear_separations", [-4, -3, -2, -1, 0, 1, 2, 3, 4])),
        Int.(get(eval_cfg, "plot_coordinate_separations", [0, 1, 2, 4, 8, 15, 20])),
        Int.(get(eval_cfg, "plot_standard_nonlinear_separations", [-2, -1, 0, 1, 2])),
        Int.(get(eval_cfg, "plot_adv_flux_separations", [-3, -2, -1, 0, 1, 2, 3])),
        Int(get(fig_cfg, "width", 3600)),
        Int(get(fig_cfg, "height", 3200)),
        String(get(run_cfg, "device", "AUTO")),
        Int(get(run_cfg, "seed", 20260428)),
        String(get(run_cfg, "runs_root", joinpath(@__DIR__, "runs"))),
        String(get(run_cfg, "label", "l96_cdot_reference_validation")),
        String(get(run_cfg, "notes", "Validate the conditional-score identity against direct time derivatives on stochastic L96.")),
        Bool(get(mobility_nn_cfg, "enabled", true)),
        Int.(get(mobility_nn_cfg, "offsets", [1, 2, 3])),
        Int.(get(mobility_nn_cfg, "window_offsets", [-3, -2, -1, 0, 1, 2, 3])),
        Int(get(mobility_nn_cfg, "pairs_per_tau", 2048)),
        Int(get(mobility_nn_cfg, "tau_batch_size", 4)),
        Int(get(mobility_nn_cfg, "anchor_states", 4096)),
        Int(get(mobility_nn_cfg, "epochs", 120)),
        Int(get(mobility_nn_cfg, "eval_every", 5)),
        Float64(get(mobility_nn_cfg, "learning_rate", 1.0e-3)),
        Float64(get(mobility_nn_cfg, "lag_weight_power", 0.0)),
        Float64(get(mobility_nn_cfg, "zero_mean_penalty", 1.0e-3)),
        Float64(get(mobility_nn_cfg, "zero_mean_penalty_final_scale", 1.0)),
        Float64(get(mobility_nn_cfg, "anchor_rms_penalty", 0.0)),
        Float64(get(mobility_nn_cfg, "anchor_rms_penalty_final_scale", 1.0)),
        Float64(get(mobility_nn_cfg, "current_action_penalty", 0.0)),
        Float64(get(mobility_nn_cfg, "weight_decay", 1.0e-6)),
        String(get(mobility_nn_cfg, "checkpoint_metric", "normalized_rmse")),
        Float64(get(mobility_nn_cfg, "scale_floor", 1.0e-4)),
        Int.(get(mobility_nn_cfg, "validation_pair_seeds", [20260511, 20260512])),
        Int.(get(mobility_nn_cfg, "widths", [64, 64])),
        Bool(get(forward_cfg, "enabled", true)),
        Float64(get(forward_cfg, "dt", 0.0025)),
        Int(get(forward_cfg, "save_stride", 4)),
        Float64(get(forward_cfg, "total_time", 16.0)),
        Float64(get(forward_cfg, "burnin_time", 4.0)),
        Int(get(forward_cfg, "ntrajectories", 128)),
        Int(get(forward_cfg, "seed", 20260531)),
        Bool(get(forward_cfg, "use_common_random_numbers", true)),
        Bool(get(forward_cfg, "clamp_eval_to_support", false)),
        Bool(get(forward_cfg, "hard_clamp_state", false)),
        Float64(get(forward_cfg, "support_pad_fraction", 0.10)),
        Int(get(forward_cfg, "pdf_bins", 160)),
        Int(get(forward_cfg, "pdf_max_samples", 250_000)),
        Int(get(forward_cfg, "correlation_stride", 4)),
        Float64(get(forward_cfg, "correlation_max_time", 8.0)),
        Float64(get(forward_cfg, "correlation_threshold", 0.05)),
        Float64(get(forward_cfg, "cphi_max_time", 1.5)),
        Int(get(forward_cfg, "cphi_stride", 1)),
        Int.(get(forward_cfg, "bivariate_offsets", [1, 2, 5, 10])),
        Int(get(forward_cfg, "auxiliary_max_samples", 100_000)),
    )

    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require_condition(params.lag_stride >= 1, "lag_stride must be >= 1.")
    require_condition(params.pairs_per_tau >= 1000, "pairs_per_tau must be at least 1000.")
    require_condition(params.eval_batch_size >= 64, "eval_batch_size must be at least 64.")
    require_condition(params.score_batch_size >= 64, "score_batch_size must be at least 64.")
    require_condition(params.joint_batch_size >= 64, "joint_batch_size must be at least 64.")
    require_condition(params.reference_mobility in ("Q_only", "local_antisymmetric_fit"),
        "reference_mobility must be either 'Q_only' or 'local_antisymmetric_fit'.")
    require_condition(all(offset -> 1 <= offset <= fld(40, 2), params.mobility_nn_offsets),
        "mobility_nn.offsets must contain positive local separations.")
    require_condition(0 in params.mobility_nn_window_offsets, "mobility_nn.window_offsets must contain 0.")
    require_condition(all(abs.(params.mobility_nn_window_offsets) .<= fld(40, 2)),
        "mobility_nn.window_offsets are too wide for the L96 ring.")
    require_condition(all(offset -> offset in params.mobility_nn_window_offsets, params.mobility_nn_offsets),
        "Each mobility_nn.offsets entry must also appear in mobility_nn.window_offsets.")
    require_condition(params.mobility_nn_pairs_per_tau >= 256, "mobility_nn.pairs_per_tau must be at least 256.")
    require_condition(params.mobility_nn_tau_batch >= 1, "mobility_nn.tau_batch_size must be positive.")
    require_condition(params.mobility_nn_anchor_states >= 256, "mobility_nn.anchor_states must be at least 256.")
    require_condition(params.mobility_nn_epochs >= 1, "mobility_nn.epochs must be positive.")
    require_condition(params.mobility_nn_eval_every >= 1, "mobility_nn.eval_every must be positive.")
    require_condition(params.mobility_nn_learning_rate > 0.0, "mobility_nn.learning_rate must be positive.")
    require_condition(params.mobility_nn_zero_mean_penalty >= 0.0, "mobility_nn.zero_mean_penalty must be nonnegative.")
    require_condition(params.mobility_nn_zero_mean_penalty_final_scale >= 0.0,
        "mobility_nn.zero_mean_penalty_final_scale must be nonnegative.")
    require_condition(params.mobility_nn_anchor_rms_penalty >= 0.0, "mobility_nn.anchor_rms_penalty must be nonnegative.")
    require_condition(params.mobility_nn_anchor_rms_penalty_final_scale >= 0.0,
        "mobility_nn.anchor_rms_penalty_final_scale must be nonnegative.")
    require_condition(params.mobility_nn_current_action_penalty >= 0.0,
        "mobility_nn.current_action_penalty must be nonnegative.")
    require_condition(params.mobility_nn_weight_decay >= 0.0, "mobility_nn.weight_decay must be nonnegative.")
    require_condition(params.mobility_nn_checkpoint_metric in ("normalized_rmse", "regularized_objective"),
        "mobility_nn.checkpoint_metric must be either 'normalized_rmse' or 'regularized_objective'.")
    require_condition(params.mobility_nn_scale_floor > 0.0, "mobility_nn.scale_floor must be positive.")
    require_condition(!isempty(params.mobility_nn_validation_pair_seeds), "mobility_nn.validation_pair_seeds must not be empty.")
    require_condition(all(width -> width >= 1, params.mobility_nn_widths), "mobility_nn.widths must be positive.")
    require_condition(params.forward_validation_dt > 0.0, "forward_validation.dt must be positive.")
    require_condition(params.forward_validation_save_stride >= 1, "forward_validation.save_stride must be positive.")
    require_condition(params.forward_validation_total_time > params.forward_validation_burnin_time >= 0.0,
        "forward_validation requires total_time > burnin_time >= 0.")
    require_condition(params.forward_validation_ntrajectories >= 32, "forward_validation.ntrajectories must be at least 32.")
    require_condition(params.forward_validation_support_pad_fraction >= 0.0,
        "forward_validation.support_pad_fraction must be nonnegative.")
    require_condition(params.forward_validation_pdf_bins >= 64, "forward_validation.pdf_bins must be at least 64.")
    require_condition(params.forward_validation_pdf_max_samples >= 10_000,
        "forward_validation.pdf_max_samples must be at least 10000.")
    require_condition(params.forward_validation_correlation_stride >= 1,
        "forward_validation.correlation_stride must be positive.")
    require_condition(params.forward_validation_correlation_max_time > 0.0,
        "forward_validation.correlation_max_time must be positive.")
    require_condition(params.forward_validation_correlation_threshold > 0.0,
        "forward_validation.correlation_threshold must be positive.")
    require_condition(params.forward_validation_cphi_max_time > 0.0,
        "forward_validation.cphi_max_time must be positive.")
    require_condition(params.forward_validation_cphi_stride >= 1,
        "forward_validation.cphi_stride must be positive.")
    require_condition(all(offset -> 1 <= offset <= 39, params.forward_validation_bivariate_offsets),
        "forward_validation.bivariate_offsets must lie in 1:(K-1).")
    require_condition(params.forward_validation_auxiliary_max_samples >= 5000,
        "forward_validation.auxiliary_max_samples must be at least 5000.")
    require_condition(all(r -> r >= 0, params.coordinate_separations), "coordinate_separations must be nonnegative.")
    require_condition(all(abs.(params.nonlinear_separations) .>= 0), "nonlinear_separations must be integers.")
    require_condition(params.figure_width >= 2200 && params.figure_height >= 1800, "Figure dimensions are too small.")
    return params, raw
end

function dict_get(d, key::Symbol)
    haskey(d, key) && return d[key]
    skey = String(key)
    haskey(d, skey) && return d[skey]
    error("Key $(key) not found.")
end

function stats_vector(raw_stats, field::Symbol, K::Int)
    raw = if raw_stats isa DataStats
        getfield(raw_stats, field)
    else
        dict_get(raw_stats, field)
    end
    vec_stats = vec(Float32.(to_host(raw)))
    if length(vec_stats) == 1
        return fill(vec_stats[1], K)
    end
    require_condition(length(vec_stats) == K, "Normalization stats must have length 1 or K.")
    return vec_stats
end

function score_from_joint_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function load_state_tensor(path::AbstractString)
    times = Float64.(h5read(path, "/trajectories/time"))
    states_raw = h5read(path, "/trajectories/states")
    require_condition(ndims(states_raw) == 3, "Expected /trajectories/states to be rank 3.")
    original_shape = size(states_raw)
    states = if size(states_raw, 1) == length(times)
        Float32.(states_raw)
    elseif size(states_raw, 2) == length(times)
        Float32.(permutedims(states_raw, (2, 1, 3)))
    else
        error("Could not infer the time axis of /trajectories/states from its shape $(original_shape).")
    end
    @printf("Loaded trajectory tensor with raw shape %s and normalized internal shape %s (time, dimension, trajectory)\n",
        string(original_shape), string(size(states)))
    return times, states
end

function load_models(score_path::AbstractString, joint_path::AbstractString, device::ExecutionDevice, K::Int)
    score_blob = BSON.load(score_path)
    joint_blob = BSON.load(joint_path)

    score_model = move_model(dict_get(score_blob, :host_model), device)
    joint_model = move_model(dict_get(joint_blob, :host_model), device)

    score_trainer = dict_get(score_blob, :trainer_cfg)
    joint_trainer = dict_get(joint_blob, :trainer_cfg)
    score_stats = dict_get(score_blob, :stats)
    joint_stats = dict_get(joint_blob, :stats)
    joint_meta = dict_get(joint_blob, :metadata)

    score_mean = stats_vector(score_stats, :mean, K)
    score_std = stats_vector(score_stats, :std, K)
    joint_mean = stats_vector(joint_stats, :mean, K)
    joint_std = stats_vector(joint_stats, :std, K)

    return LoadedModels(
        score_model,
        joint_model,
        Float32(score_trainer.sigma),
        Float32(joint_trainer.sigma),
        score_mean,
        score_std,
        joint_mean,
        joint_std,
        Float64(dict_get(joint_meta, :tau_min)),
        Float64(dict_get(joint_meta, :tau_max)),
    )
end

function build_pair_sampler(path::AbstractString, burnin_fraction::Float64, lag_stride::Int,
        tau_min::Float64, tau_max::Float64)
    times, states = load_state_tensor(path)
    start_idx = burnin_start_index(length(times), burnin_fraction)
    save_dt = length(times) > 1 ? times[2] - times[1] : 0.0
    require_condition(save_dt > 0.0, "Need a positive saved time step.")

    t_decorrelation = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    Q = Matrix{Float64}(h5read(path, "/diffusion/Q"))
    F = Float64(h5read(path, "/metadata/F"))

    min_lag = max(1, ceil(Int, tau_min / save_dt - 1e-9))
    max_lag = min(length(times) - start_idx, floor(Int, tau_max / save_dt + 1e-9))
    require_condition(max_lag >= min_lag, "No lag steps are available inside the joint-score training range.")

    lag_steps = collect(min_lag:lag_stride:max_lag)
    lag_times = lag_steps .* save_dt
    @printf("Validation lag window: [%.3f, %.3f] with %d positive lags; decorrelation time %.3f\n",
        tau_min, tau_max, length(lag_steps), t_decorrelation)
    return PairSampler(times, states, start_idx, lag_steps, lag_times, tau_min, tau_max, t_decorrelation, save_dt, F, Q)
end

function sample_pair_batch!(x0::AbstractMatrix{Float32}, xt::AbstractMatrix{Float32},
        sampler::PairSampler, lag::Int, rng::AbstractRNG)
    nt, K, ntraj = size(sampler.states)
    upper = nt - lag
    require_condition(upper >= sampler.start_idx, "Lag exceeds the available post-burn-in window.")
    @inbounds for sample_idx in 1:size(x0, 2)
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:upper)
        for mode_idx in 1:K
            x0[mode_idx, sample_idx] = sampler.states[time_idx, mode_idx, traj_idx]
            xt[mode_idx, sample_idx] = sampler.states[time_idx + lag, mode_idx, traj_idx]
        end
    end
    return nothing
end

function sample_state_batch!(x::AbstractMatrix{Float32}, sampler::PairSampler, rng::AbstractRNG)
    nt, K, ntraj = size(sampler.states)
    @inbounds for sample_idx in 1:size(x, 2)
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:nt)
        for mode_idx in 1:K
            x[mode_idx, sample_idx] = sampler.states[time_idx, mode_idx, traj_idx]
        end
    end
    return nothing
end

function standardize_batch!(dest::AbstractMatrix{Float32}, src::AbstractMatrix{Float32}, mu::Float64, sigma_x::Float64)
    inv_sigma = Float32(1.0 / sigma_x)
    mu32 = Float32(mu)
    @inbounds for j in 1:size(src, 2), i in 1:size(src, 1)
        dest[i, j] = (src[i, j] - mu32) * inv_sigma
    end
    return dest
end

function normalize_with_stats!(dest::AbstractArray{Float32, 3}, x::AbstractMatrix{Float32}, mean_vec::Vector{Float32}, std_vec::Vector{Float32})
    K = size(x, 1)
    B = size(x, 2)
    @inbounds for b in 1:B, i in 1:K
        dest[i, 1, b] = (x[i, b] - mean_vec[i]) / std_vec[i]
    end
    return dest
end

function normalize_pair_with_stats!(dest::AbstractArray{Float32, 3}, x0::AbstractMatrix{Float32}, xt::AbstractMatrix{Float32},
        tnorm::Float32, mean_vec::Vector{Float32}, std_vec::Vector{Float32})
    K = size(x0, 1)
    B = size(x0, 2)
    @inbounds for b in 1:B, i in 1:K
        dest[i, 1, b] = (x0[i, b] - mean_vec[i]) / std_vec[i]
        dest[i, 2, b] = (xt[i, b] - mean_vec[i]) / std_vec[i]
        dest[i, 3, b] = tnorm
    end
    return dest
end

function evaluate_stationary_score_x(models::LoadedModels, x_raw::AbstractMatrix{Float32}, batch_size::Int, device::ExecutionDevice)
    K, N = size(x_raw)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, 1, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        normalize_with_stats!(@view(scratch[:, :, 1:batch_n]), @view(x_raw[:, start:stop]), models.score_mean, models.score_std)
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_z = score_from_model(models.score_model, batch_dev, models.score_sigma)
        scores_x = reshape(to_host(scores_z), K, batch_n)
        @inbounds for b in 1:batch_n, i in 1:K
            out[i, start + b - 1] = scores_x[i, b] / models.score_std[i]
        end
    end
    return out
end

function evaluate_joint_score_x0(models::LoadedModels, x0_raw::AbstractMatrix{Float32}, xt_raw::AbstractMatrix{Float32},
        tnorm::Float32, batch_size::Int, device::ExecutionDevice)
    K, N = size(x0_raw)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, JOINT_INPUT_CHANNELS, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        normalize_pair_with_stats!(@view(scratch[:, :, 1:batch_n]),
            @view(x0_raw[:, start:stop]), @view(xt_raw[:, start:stop]), tnorm,
            models.joint_mean, models.joint_std)
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_z = score_from_joint_model(models.joint_model, batch_dev, models.joint_sigma)
        x0_scores = reshape(@view(scores_z[:, 1, :]), K, batch_n)
        x0_scores_host = to_host(x0_scores)
        @inbounds for b in 1:batch_n, i in 1:K
            out[i, start + b - 1] = x0_scores_host[i, b] / models.joint_std[i]
        end
    end
    return out
end

function evaluate_conditional_score_x0(models::LoadedModels, x0_raw::AbstractMatrix{Float32}, xt_raw::AbstractMatrix{Float32},
        tau::Float64, score_batch_size::Int, joint_batch_size::Int, device::ExecutionDevice)
    require_condition(models.tau_min <= tau <= models.tau_max + 1e-10,
        "Requested tau lies outside the joint-score training range.")
    denom = max(models.tau_max - models.tau_min, eps())
    tnorm = Float32((tau - models.tau_min) / denom)
    stat_score = evaluate_stationary_score_x(models, x0_raw, score_batch_size, device)
    joint_score = evaluate_joint_score_x0(models, x0_raw, xt_raw, tnorm, joint_batch_size, device)
    return joint_score .- stat_score
end

function l96_drift!(dest::AbstractMatrix{Float64}, x::AbstractMatrix{Float32}, F::Float64)
    K, B = size(x)
    @inbounds for b in 1:B, i in 1:K
        im2 = periodic(i - 2, K)
        im1 = periodic(i - 1, K)
        ip1 = periodic(i + 1, K)
        dest[i, b] = x[im1, b] * (x[ip1, b] - x[im2, b]) - x[i, b] + F
    end
    return dest
end

function compute_standardization_and_observable_means(states::Array{Float32, 3}, start_idx::Int)
    post = @view states[start_idx:end, :, :]
    mu = Float64(mean(post))
    sigma_x = Float64(std(post))
    sigma_x = max(sigma_x, sqrt(eps(Float64)))

    nthreads = Threads.nthreads()
    partial = [zeros(Float64, 5) for _ in 1:nthreads]
    partial_count = zeros(Int, nthreads)
    nt, K, ntraj = size(states)

    Threads.@threads for traj_idx in 1:ntraj
        thread_id = Threads.threadid()
        sums = partial[thread_id]
        local_count = 0
        u = Vector{Float64}(undef, K)
        for time_idx in start_idx:nt
            @inbounds for i in 1:K
                u[i] = (states[time_idx, i, traj_idx] - mu) / sigma_x
            end
            @inbounds for i in 1:K
                im2 = periodic(i - 2, K)
                im1 = periodic(i - 1, K)
                ip1 = periodic(i + 1, K)
                ip2 = periodic(i + 2, K)
                ui = u[i]
                nn1 = ui * u[ip1]
                nn2 = ui * u[ip2]
                adv = u[im1] * (u[ip1] - u[im2])
                flux = ui * u[im1] * (u[ip1] - u[im2])
                sums[1] += ui * ui
                sums[2] += nn1
                sums[3] += nn2
                sums[4] += adv
                sums[5] += flux
                local_count += 1
            end
        end
        partial_count[thread_id] += local_count
    end

    total_sums = zeros(Float64, 5)
    total_count = sum(partial_count)
    for sums in partial
        total_sums .+= sums
    end

    means = ObservableMeans(
        total_sums[1] / total_count,
        total_sums[2] / total_count,
        total_sums[3] / total_count,
        total_sums[4] / total_count,
        total_sums[5] / total_count,
    )
    @printf("Post-burnin standardization: mu = %.6f, sigma_x = %.6f\n", mu, sigma_x)
    @printf("Observable means: <u^2> = %.6f, <uu1> = %.6f, <uu2> = %.6f, <adv> = %.6f, <flux> = %.6f\n",
        means.var_mean, means.nn1_mean, means.nn2_mean, means.adv_mean, means.flux_mean)
    return mu, sigma_x, means
end

function compute_observable_batches(u::AbstractMatrix{Float32}, means::ObservableMeans)
    K, B = size(u)
    var = Matrix{Float32}(undef, K, B)
    nn1 = Matrix{Float32}(undef, K, B)
    nn2 = Matrix{Float32}(undef, K, B)
    adv = Matrix{Float32}(undef, K, B)
    flux = Matrix{Float32}(undef, K, B)
    var_mean = Float32(means.var_mean)
    nn1_mean = Float32(means.nn1_mean)
    nn2_mean = Float32(means.nn2_mean)
    adv_mean = Float32(means.adv_mean)
    flux_mean = Float32(means.flux_mean)
    @inbounds for b in 1:B, i in 1:K
        im2 = periodic(i - 2, K)
        im1 = periodic(i - 1, K)
        ip1 = periodic(i + 1, K)
        ip2 = periodic(i + 2, K)
        ui = u[i, b]
        var[i, b] = ui * ui - var_mean
        nn1_val = ui * u[ip1, b]
        nn2_val = ui * u[ip2, b]
        adv_val = u[im1, b] * (u[ip1, b] - u[im2, b])
        nn1[i, b] = nn1_val - nn1_mean
        nn2[i, b] = nn2_val - nn2_mean
        adv[i, b] = adv_val - adv_mean
        flux[i, b] = ui * adv_val - flux_mean
    end
    return Dict(
        :coord => u,
        :var => var,
        :nn1 => nn1,
        :nn2 => nn2,
        :adv => adv,
        :flux => flux,
    )
end

function smoothing_candidates(values::AbstractVector{Float64}, fixed_smoothing::Float64, rel_grid::Vector{Float64})
    centered = values .- mean(values)
    scale = max(sum(centered .^ 2), eps(Float64))
    return unique(sort(vcat([fixed_smoothing], scale .* rel_grid)))
end

function derivative_roughness(values::AbstractVector{Float64})
    return length(values) <= 1 ? 0.0 : sqrt(mean(diff(values) .^ 2))
end

function central_time_derivative(values::AbstractMatrix{Float64}, taus::Vector{Float64})
    n, m = size(values)
    require_condition(n >= 2, "Need at least two lag points to differentiate.")
    out = similar(values)
    if n == 2
        dt = taus[2] - taus[1]
        out[1, :] .= (values[2, :] .- values[1, :]) ./ dt
        out[2, :] .= out[1, :]
        return out
    end
    @inbounds for j in 1:m
        out[1, j] = ((values[2, j] - values[1, j]) / (taus[2] - taus[1]) +
            (values[3, j] - values[1, j]) / (taus[3] - taus[1])) / 2
        for i in 2:(n - 1)
            out[i, j] = (values[i + 1, j] - values[i - 1, j]) / (taus[i + 1] - taus[i - 1])
        end
        out[n, j] = ((values[n, j] - values[n - 1, j]) / (taus[n] - taus[n - 1]) +
            (values[n, j] - values[n - 2, j]) / (taus[n] - taus[n - 2])) / 2
    end
    return out
end

function smooth_cphi_and_derivative(taus::Vector{Float64}, values::AbstractVector{Float64}, smoothing::Float64)
    n = length(taus)
    require_condition(length(values) == n, "Spline inputs must have matching lengths.")
    if n < 4 || smoothing == 0.0
        deriv = vec(central_time_derivative(reshape(values, n, 1), taus))
        return copy(values), deriv
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
    deriv = vec(central_time_derivative(reshape(values, n, 1), taus))
    return copy(values), deriv
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

function select_smoothing_protocol_dataonly(taus::Vector{Float64}, values::AbstractVector{Float64},
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
            push!(choices, SmoothingSelection(smoothing, fit_rmse, derivative_roughness(deriv)))
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

function shift_indices(separations::Vector{Int}, K::Int)
    base = collect(1:K)
    return [mod1.(base .+ sep, K) for sep in separations]
end

function accumulate_translation_channels!(accum::Vector{Float64}, phi::AbstractMatrix{<:Real}, signal::AbstractMatrix{<:Real},
        index_sets::Vector{Vector{Int}})
    K, B = size(phi)
    @inbounds for (sep_idx, idxs) in enumerate(index_sets)
        total = 0.0
        for b in 1:B, i in 1:K
            total += phi[i, b] * signal[idxs[i], b]
        end
        accum[sep_idx] += total / K
    end
    return nothing
end

function compute_generator_observable_batches(u::AbstractMatrix{Float32}, drift::AbstractMatrix{Float64},
        Q::AbstractMatrix{Float64}, sigma_x::Float64)
    K, B = size(u)
    coord = Matrix{Float64}(undef, K, B)
    var = Matrix{Float64}(undef, K, B)
    nn1 = Matrix{Float64}(undef, K, B)
    nn2 = Matrix{Float64}(undef, K, B)
    adv = Matrix{Float64}(undef, K, B)
    flux = Matrix{Float64}(undef, K, B)
    inv_sigma = 1.0 / sigma_x
    inv_sigma_sq = inv_sigma^2
    @inbounds for b in 1:B, i in 1:K
        im2 = periodic(i - 2, K)
        im1 = periodic(i - 1, K)
        ip1 = periodic(i + 1, K)
        ip2 = periodic(i + 2, K)

        ui = Float64(u[i, b])
        uim2 = Float64(u[im2, b])
        uim1 = Float64(u[im1, b])
        uip1 = Float64(u[ip1, b])
        uip2 = Float64(u[ip2, b])

        fi = drift[i, b]
        fim2 = drift[im2, b]
        fim1 = drift[im1, b]
        fip1 = drift[ip1, b]
        fip2 = drift[ip2, b]

        coord[i, b] = fi * inv_sigma
        var[i, b] = 2.0 * ui * fi * inv_sigma + 2.0 * Q[i, i] * inv_sigma_sq
        nn1[i, b] = (fi * uip1 + fip1 * ui) * inv_sigma + 2.0 * Q[i, ip1] * inv_sigma_sq
        nn2[i, b] = (fi * uip2 + fip2 * ui) * inv_sigma + 2.0 * Q[i, ip2] * inv_sigma_sq
        adv[i, b] = (fim1 * (uip1 - uim2) + uim1 * (fip1 - fim2)) * inv_sigma +
            2.0 * (Q[im1, ip1] - Q[im1, im2]) * inv_sigma_sq
        flux[i, b] = (fi * uim1 * (uip1 - uim2) +
            fim1 * ui * (uip1 - uim2) +
            fip1 * ui * uim1 -
            fim2 * ui * uim1) * inv_sigma +
            2.0 * ((uip1 - uim2) * Q[i, im1] +
            uim1 * Q[i, ip1] -
            uim1 * Q[i, im2] +
            ui * Q[im1, ip1] -
            ui * Q[im1, im2]) * inv_sigma_sq
    end
    return Dict(
        :coord => coord,
        :var => var,
        :nn1 => nn1,
        :nn2 => nn2,
        :adv => adv,
        :flux => flux,
    )
end

function local_antisymmetric_feature_action!(dest::AbstractMatrix{Float64}, u::AbstractMatrix{Float32},
        score::AbstractMatrix{Float32}, offset::Int, feature_idx::Int,
        feature_offsets::Vector{Int}, quadratic_pairs::Vector{Tuple{Int, Int}}, sigma_x::Float64;
        include_divergence::Bool=false)
    K, B = size(score)
    if feature_idx == 1
        @inbounds for b in 1:B, i in 1:K
            dest[i, b] = Float64(score[periodic(i + offset, K), b]) - Float64(score[periodic(i - offset, K), b])
        end
        return dest
    end

    linear_count = length(feature_offsets)
    inv_sigma = 1.0 / sigma_x
    if feature_idx <= 1 + linear_count
        shift = feature_offsets[feature_idx - 1]
        div_constant = include_divergence ? (((shift == offset) ? inv_sigma : 0.0) - ((shift == 0) ? inv_sigma : 0.0)) : 0.0
        @inbounds for b in 1:B, i in 1:K
            dest[i, b] = Float64(u[periodic(i + shift, K), b]) * Float64(score[periodic(i + offset, K), b]) -
                Float64(u[periodic(i - offset + shift, K), b]) * Float64(score[periodic(i - offset, K), b]) + div_constant
        end
        return dest
    end

    shift_a, shift_b = quadratic_pairs[feature_idx - 1 - linear_count]
    @inbounds for b in 1:B, i in 1:K
        beta_forward = Float64(u[periodic(i + shift_a, K), b]) * Float64(u[periodic(i + shift_b, K), b])
        beta_backward = Float64(u[periodic(i - offset + shift_a, K), b]) * Float64(u[periodic(i - offset + shift_b, K), b])
        div_term = 0.0
        if include_divergence
            if shift_a == offset
                div_term += Float64(u[periodic(i + shift_b, K), b]) * inv_sigma
            end
            if shift_b == offset
                div_term += Float64(u[periodic(i + shift_a, K), b]) * inv_sigma
            end
            if shift_a == 0
                div_term -= Float64(u[periodic(i - offset + shift_b, K), b]) * inv_sigma
            end
            if shift_b == 0
                div_term -= Float64(u[periodic(i - offset + shift_a, K), b]) * inv_sigma
            end
        end
        dest[i, b] = beta_forward * Float64(score[periodic(i + offset, K), b]) -
            beta_backward * Float64(score[periodic(i - offset, K), b]) + div_term
    end
    return nothing
end

function apply_local_antisymmetric_action!(dest::AbstractMatrix{Float64}, scratch::AbstractMatrix{Float64},
        u::AbstractMatrix{Float32}, score::AbstractMatrix{Float32}, offsets::Vector{Int},
        feature_offsets::Vector{Int}, quadratic_pairs::Vector{Tuple{Int, Int}},
        coefficients::AbstractMatrix{Float64}, sigma_x::Float64;
        include_divergence::Bool=false)
    fill!(dest, 0.0)
    isempty(coefficients) && return dest
    nfeatures = size(coefficients, 2)
    @inbounds for (offset_idx, offset) in enumerate(offsets)
        for feature_idx in 1:nfeatures
            coeff = coefficients[offset_idx, feature_idx]
            abs(coeff) <= 1.0e-14 && continue
            local_antisymmetric_feature_action!(scratch, u, score, offset, feature_idx, feature_offsets, quadratic_pairs, sigma_x;
                include_divergence=include_divergence)
            @. dest += coeff * scratch
        end
    end
    return dest
end

function mean_local_antisymmetric_matrix(K::Int, offsets::Vector{Int}, coefficients::AbstractMatrix{Float64},
        feature_means::AbstractVector{Float64})
    R = zeros(Float64, K, K)
    isempty(coefficients) && return R
    @inbounds for (offset_idx, offset) in enumerate(offsets)
        for feature_idx in 1:size(coefficients, 2)
            coeff = coefficients[offset_idx, feature_idx] * feature_means[feature_idx]
            abs(coeff) <= 1.0e-14 && continue
            for i in 1:K
                j = periodic(i + offset, K)
                R[i, j] += coeff
                R[j, i] -= coeff
            end
        end
    end
    return R
end

function fit_local_antisymmetric_reference(models::LoadedModels, sampler::PairSampler, params::FitDML96Params,
    mu::Float64, sigma_x::Float64, device::ExecutionDevice)
    K = size(sampler.states, 2)
    offsets = unique(sort(filter(r -> 1 <= r <= fld(K, 2), params.antisymmetric_offsets)))
    feature_offsets = unique(sort(filter(r -> abs(r) <= fld(K, 2), params.antisymmetric_feature_offsets)))
    quadratic_pairs = Tuple{Int, Int}[(a, b) for (ia, a) in enumerate(feature_offsets) for b in feature_offsets[ia:end]]
    if isempty(offsets)
        return ReferenceMobilityResult("Q_only", sampler.Q, nothing, sampler.Q, nothing, Int[], Int[], Tuple{Int, Int}[], zeros(Float64, 0, 0), 1.0, 0.0,
            "No antisymmetric offsets configured; using Q-only reference.")
    end

    nfeatures = 1 + length(feature_offsets) + length(quadratic_pairs)
    nbasis = length(offsets) * nfeatures
    G = zeros(Float64, nbasis, nbasis)
    h = zeros(Float64, nbasis)
    feature_sums = zeros(Float64, nfeatures)
    feature_count = 0
    x_batch = Matrix{Float32}(undef, K, params.antisymmetric_fit_batch_size)
    u_batch = Matrix{Float32}(undef, K, params.antisymmetric_fit_batch_size)
    drift_batch = Matrix{Float64}(undef, K, params.antisymmetric_fit_batch_size)
    action_cache = [Matrix{Float64}(undef, K, params.antisymmetric_fit_batch_size) for _ in 1:nbasis]
    rng = MersenneTwister(params.seed + 91)

    remaining = params.antisymmetric_fit_samples
    while remaining > 0
        batch_n = min(remaining, params.antisymmetric_fit_batch_size)
        sample_state_batch!(@view(x_batch[:, 1:batch_n]), sampler, rng)
        standardize_batch!(@view(u_batch[:, 1:batch_n]), @view(x_batch[:, 1:batch_n]), mu, sigma_x)
        u_view = @view(u_batch[:, 1:batch_n])
        batch_entry_count = K * batch_n
        feature_sums[1] += batch_entry_count
        linear_sum = sum(u_view)
        for feature_idx in 1:length(feature_offsets)
            feature_sums[1 + feature_idx] += linear_sum
        end
        for (pair_idx, (shift_a, shift_b)) in enumerate(quadratic_pairs)
            total = 0.0
            @inbounds for b in 1:batch_n, i in 1:K
                total += Float64(u_view[periodic(i + shift_a, K), b]) * Float64(u_view[periodic(i + shift_b, K), b])
            end
            feature_sums[1 + length(feature_offsets) + pair_idx] += total
        end
        feature_count += batch_entry_count
        l96_drift!(@view(drift_batch[:, 1:batch_n]), @view(x_batch[:, 1:batch_n]), sampler.F)
        stat_score = evaluate_stationary_score_x(models, @view(x_batch[:, 1:batch_n]), params.score_batch_size, device)
        residual = @view(drift_batch[:, 1:batch_n]) .- sampler.Q * Float64.(stat_score)
        for (offset_idx, offset) in enumerate(offsets)
            for feature_idx in 1:nfeatures
                basis_idx = (offset_idx - 1) * nfeatures + feature_idx
                local_antisymmetric_feature_action!(@view(action_cache[basis_idx][:, 1:batch_n]), u_view, stat_score,
                    offset, feature_idx, feature_offsets, quadratic_pairs, sigma_x; include_divergence=true)
            end
        end
        for a in 1:nbasis
            feat_a = @view(action_cache[a][:, 1:batch_n])
            h[a] += sum(feat_a .* residual)
            for b in a:nbasis
                feat_b = @view(action_cache[b][:, 1:batch_n])
                G[a, b] += sum(feat_a .* feat_b)
            end
        end
        remaining -= batch_n
    end
    for a in 2:nbasis, b in 1:(a - 1)
        G[a, b] = G[b, a]
    end

    H = G + params.antisymmetric_fit_ridge * I
    coeff_vec = H \ h
    coeffs = reshape(coeff_vec, nfeatures, length(offsets))'
    cond_est = cond(Matrix(H))
    feature_means = feature_sums ./ max(feature_count, 1)

    R = mean_local_antisymmetric_matrix(K, offsets, coeffs, feature_means)
    Mfit = sampler.Q + R

    x_batch_val = Matrix{Float32}(undef, K, params.antisymmetric_fit_batch_size)
    u_batch_val = Matrix{Float32}(undef, K, params.antisymmetric_fit_batch_size)
    drift_val = Matrix{Float64}(undef, K, params.antisymmetric_fit_batch_size)
    antisym_val = Matrix{Float64}(undef, K, params.antisymmetric_fit_batch_size)
    scratch_val = Matrix{Float64}(undef, K, params.antisymmetric_fit_batch_size)
    rng_val = MersenneTwister(params.seed + 191)
    residual_norm = 0.0
    residual_count = 0
    remaining = min(20_000, params.antisymmetric_fit_samples)
    while remaining > 0
        batch_n = min(remaining, params.antisymmetric_fit_batch_size)
        sample_state_batch!(@view(x_batch_val[:, 1:batch_n]), sampler, rng_val)
        standardize_batch!(@view(u_batch_val[:, 1:batch_n]), @view(x_batch_val[:, 1:batch_n]), mu, sigma_x)
        l96_drift!(@view(drift_val[:, 1:batch_n]), @view(x_batch_val[:, 1:batch_n]), sampler.F)
        stat_score = evaluate_stationary_score_x(models, @view(x_batch_val[:, 1:batch_n]), params.score_batch_size, device)
        apply_local_antisymmetric_action!(@view(antisym_val[:, 1:batch_n]), @view(scratch_val[:, 1:batch_n]),
            @view(u_batch_val[:, 1:batch_n]), stat_score, offsets, feature_offsets, quadratic_pairs, coeffs, sigma_x;
            include_divergence=true)
        residual = @view(drift_val[:, 1:batch_n]) .- sampler.Q * Float64.(stat_score) .- @view(antisym_val[:, 1:batch_n])
        residual_norm += sum(abs2, residual)
        residual_count += length(residual)
        remaining -= batch_n
    end
    fit_rms = sqrt(residual_norm / max(residual_count, 1))
    status = @sprintf("Fitted translation-equivariant local antisymmetric ansatz with pair offsets %s, linear feature offsets %s, and %d quadratic local features.",
        string(offsets), string(feature_offsets), length(quadratic_pairs))

    if !all(isfinite, coeffs) || !all(isfinite, Mfit) || cond_est > 1.0e12
        status = @sprintf("Antisymmetric fit unstable (cond = %.3e); falling back to Q-only reference.", cond_est)
        return ReferenceMobilityResult("Q_only", sampler.Q, Mfit, sampler.Q, R, offsets, feature_offsets, quadratic_pairs, Matrix{Float64}(coeffs), cond_est, fit_rms, status)
    end

    active_mode = params.reference_mobility == "local_antisymmetric_fit" ? "local_antisymmetric_fit" : "Q_only"
    active_matrix = active_mode == "local_antisymmetric_fit" ? Mfit : sampler.Q
    return ReferenceMobilityResult(active_mode, sampler.Q, Mfit, active_matrix, R, offsets, feature_offsets, quadratic_pairs, Matrix{Float64}(coeffs), cond_est, fit_rms, status)
end

function observable_title(key::Symbol)
    return Dict(
        :coord => "Coordinate",
        :var => "Local Variance",
        :nn1 => "Nearest-Neighbor Product",
        :nn2 => "Next-Nearest Product",
        :adv => "Advective Stencil",
        :flux => "Energy-Transfer Flux",
    )[key]
end

function evaluate_c_and_generator_cdot_channels(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mu::Float64, sigma_x::Float64, means::ObservableMeans, reference::ReferenceMobilityResult,
        device::ExecutionDevice)
    K = size(sampler.states, 2)
    all_coord_offsets = collect(0:(K - 1))
    nonlinear_offsets = params.nonlinear_separations
    coord_index_sets = shift_indices(all_coord_offsets, K)
    nonlinear_index_sets = shift_indices(nonlinear_offsets, K)

    data_taus = vcat(0.0, sampler.lag_times)
    ntaus = length(data_taus)
    positive_count = length(sampler.lag_times)

    C_coord_full = zeros(Float64, ntaus, length(all_coord_offsets))
    C_data = Dict{Symbol, Matrix{Float64}}(
        :var => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :nn1 => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :nn2 => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :adv => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :flux => zeros(Float64, ntaus, length(nonlinear_offsets)),
    )
    Cdot_data_coord_full = zeros(Float64, ntaus, length(all_coord_offsets))
    Cdot_data = Dict{Symbol, Matrix{Float64}}(
        :var => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :nn1 => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :nn2 => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :adv => zeros(Float64, ntaus, length(nonlinear_offsets)),
        :flux => zeros(Float64, ntaus, length(nonlinear_offsets)),
    )

    Cdot_ref_q_coord_full = fill(NaN, positive_count, length(all_coord_offsets))
    Cdot_ref_q = Dict{Symbol, Matrix{Float64}}(
        :var => fill(NaN, positive_count, length(nonlinear_offsets)),
        :nn1 => fill(NaN, positive_count, length(nonlinear_offsets)),
        :nn2 => fill(NaN, positive_count, length(nonlinear_offsets)),
        :adv => fill(NaN, positive_count, length(nonlinear_offsets)),
        :flux => fill(NaN, positive_count, length(nonlinear_offsets)),
    )

    Cdot_ref_fit_coord_full = fill(NaN, positive_count, length(all_coord_offsets))
    Cdot_ref_fit = Dict{Symbol, Matrix{Float64}}(
        :var => fill(NaN, positive_count, length(nonlinear_offsets)),
        :nn1 => fill(NaN, positive_count, length(nonlinear_offsets)),
        :nn2 => fill(NaN, positive_count, length(nonlinear_offsets)),
        :adv => fill(NaN, positive_count, length(nonlinear_offsets)),
        :flux => fill(NaN, positive_count, length(nonlinear_offsets)),
    )

    x0_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    u0_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    ut_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    drift_t_batch = Matrix{Float64}(undef, K, params.eval_batch_size)
    antisym_batch = Matrix{Float64}(undef, K, params.eval_batch_size)
    antisym_scratch = Matrix{Float64}(undef, K, params.eval_batch_size)

    q_only_matrix = reference.q_only_matrix
    fitted_matrix = reference.fitted_matrix
    has_fitted = fitted_matrix !== nothing && !isempty(reference.coefficients)

    for tau_idx in 1:ntaus
        tau = data_taus[tau_idx]
        lag = tau_idx == 1 ? 0 : sampler.lag_steps[tau_idx - 1]
        tau_rng = MersenneTwister(params.seed + 1000 * tau_idx)

        coord_full_sum = zeros(Float64, length(all_coord_offsets))
        cdot_coord_full_sum = zeros(Float64, length(all_coord_offsets))
        channel_sums = Dict(key => zeros(Float64, length(nonlinear_offsets)) for key in (:var, :nn1, :nn2, :adv, :flux))
        cdot_sums = Dict(key => zeros(Float64, length(nonlinear_offsets)) for key in (:var, :nn1, :nn2, :adv, :flux))
        ref_q_coord_sum = zeros(Float64, length(all_coord_offsets))
        ref_q_sums = Dict(key => zeros(Float64, length(nonlinear_offsets)) for key in (:var, :nn1, :nn2, :adv, :flux))
        ref_fit_coord_sum = zeros(Float64, length(all_coord_offsets))
        ref_fit_sums = Dict(key => zeros(Float64, length(nonlinear_offsets)) for key in (:var, :nn1, :nn2, :adv, :flux))

        remaining = params.pairs_per_tau
        total_pairs = 0
        while remaining > 0
            batch_n = min(remaining, params.eval_batch_size)
            sample_pair_batch!(@view(x0_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), sampler, lag, tau_rng)
            standardize_batch!(@view(u0_batch[:, 1:batch_n]), @view(x0_batch[:, 1:batch_n]), mu, sigma_x)
            standardize_batch!(@view(ut_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), mu, sigma_x)
            l96_drift!(@view(drift_t_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), sampler.F)
            obs = compute_observable_batches(@view(ut_batch[:, 1:batch_n]), means)
            generator_obs = compute_generator_observable_batches(@view(ut_batch[:, 1:batch_n]), @view(drift_t_batch[:, 1:batch_n]), sampler.Q, sigma_x)

            accumulate_translation_channels!(coord_full_sum, obs[:coord], @view(u0_batch[:, 1:batch_n]), coord_index_sets)
            accumulate_translation_channels!(cdot_coord_full_sum, generator_obs[:coord], @view(u0_batch[:, 1:batch_n]), coord_index_sets)
            for key in (:var, :nn1, :nn2, :adv, :flux)
                accumulate_translation_channels!(channel_sums[key], obs[key], @view(u0_batch[:, 1:batch_n]), nonlinear_index_sets)
                accumulate_translation_channels!(cdot_sums[key], generator_obs[key], @view(u0_batch[:, 1:batch_n]), nonlinear_index_sets)
            end

            if tau_idx > 1
                scond = evaluate_conditional_score_x0(models,
                    @view(x0_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), tau,
                    params.score_batch_size, params.joint_batch_size, device)
                q_signal = transpose(q_only_matrix) * Float64.(scond)
                projected_q = Float32.(q_signal ./ sigma_x)
                accumulate_translation_channels!(ref_q_coord_sum, obs[:coord], projected_q, coord_index_sets)
                for key in (:var, :nn1, :nn2, :adv, :flux)
                    accumulate_translation_channels!(ref_q_sums[key], obs[key], projected_q, nonlinear_index_sets)
                end

                if has_fitted
                    apply_local_antisymmetric_action!(@view(antisym_batch[:, 1:batch_n]), @view(antisym_scratch[:, 1:batch_n]),
                        @view(u0_batch[:, 1:batch_n]), scond, reference.offsets, reference.feature_offsets,
                        reference.quadratic_pairs,
                        reference.coefficients, sigma_x; include_divergence=false)
                    projected_fit = Float32.((q_signal .- @view(antisym_batch[:, 1:batch_n])) ./ sigma_x)
                    accumulate_translation_channels!(ref_fit_coord_sum, obs[:coord], projected_fit, coord_index_sets)
                    for key in (:var, :nn1, :nn2, :adv, :flux)
                        accumulate_translation_channels!(ref_fit_sums[key], obs[key], projected_fit, nonlinear_index_sets)
                    end
                end
            end

            total_pairs += batch_n
            remaining -= batch_n
        end

        C_coord_full[tau_idx, :] .= coord_full_sum ./ total_pairs
        Cdot_data_coord_full[tau_idx, :] .= cdot_coord_full_sum ./ total_pairs
        for key in (:var, :nn1, :nn2, :adv, :flux)
            C_data[key][tau_idx, :] .= channel_sums[key] ./ total_pairs
            Cdot_data[key][tau_idx, :] .= cdot_sums[key] ./ total_pairs
        end
        if tau_idx > 1
            pos_idx = tau_idx - 1
            Cdot_ref_q_coord_full[pos_idx, :] .= .-(ref_q_coord_sum ./ total_pairs)
            for key in (:var, :nn1, :nn2, :adv, :flux)
                Cdot_ref_q[key][pos_idx, :] .= .-(ref_q_sums[key] ./ total_pairs)
            end
            if has_fitted
                Cdot_ref_fit_coord_full[pos_idx, :] .= .-(ref_fit_coord_sum ./ total_pairs)
                for key in (:var, :nn1, :nn2, :adv, :flux)
                    Cdot_ref_fit[key][pos_idx, :] .= .-(ref_fit_sums[key] ./ total_pairs)
                end
            end
        end

        if tau_idx == 1 || tau_idx == ntaus || tau_idx % 20 == 0
            @printf("Estimated C/Cdot reference channels for tau index %d / %d (tau = %.3f)\n", tau_idx, ntaus, tau)
        end
    end

    C_data[:coord_full] = C_coord_full
    C_data[:coord] = C_coord_full[:, params.coordinate_separations .+ 1]
    Cdot_data[:coord_full] = Cdot_data_coord_full
    Cdot_data[:coord] = Cdot_data_coord_full[:, params.coordinate_separations .+ 1]

    Cdot_ref_q = merge(Cdot_ref_q, Dict(
        :coord_full => Cdot_ref_q_coord_full,
        :coord => Cdot_ref_q_coord_full[:, params.coordinate_separations .+ 1],
    ))
    Cdot_ref_fit = merge(Cdot_ref_fit, Dict(
        :coord_full => Cdot_ref_fit_coord_full,
        :coord => Cdot_ref_fit_coord_full[:, params.coordinate_separations .+ 1],
    ))

    return data_taus, C_data, Cdot_data, Cdot_ref_q, Cdot_ref_fit, all_coord_offsets
end

function smooth_all_channels(taus::Vector{Float64}, C_data::Dict{Symbol, Matrix{Float64}}, params::FitDML96Params)
    C_smooth = Dict{Symbol, Matrix{Float64}}()
    Cdot = Dict{Symbol, Matrix{Float64}}()
    selections = Dict{String, SmoothingSelection}()
    for (key, values) in pairs(C_data)
        smoothed = zeros(Float64, size(values))
        deriv = zeros(Float64, size(values))
        for channel_idx in 1:size(values, 2)
            smooth_vals, deriv_vals, selection = select_smoothing_protocol_dataonly(
                taus,
                vec(@view values[:, channel_idx]),
                params.cphi_spline_smoothing,
                params.cphi_smoothing_rel_grid,
                params.cphi_smoothing_rmse_tolerance,
            )
            smoothed[:, channel_idx] .= smooth_vals
            deriv[:, channel_idx] .= deriv_vals
            selections[string(key) * ":" * string(channel_idx)] = selection
        end
        C_smooth[key] = smoothed
        Cdot[key] = deriv
    end
    return C_smooth, Cdot, selections
end

function circulant_matrix_from_profile(profile::Vector{Float64})
    K = length(profile)
    out = Matrix{Float64}(undef, K, K)
    @inbounds for i in 1:K, j in 1:K
        out[i, j] = profile[mod(j - i, K) + 1]
    end
    return out
end

function circulant_profile(M::AbstractMatrix{<:Real})
    K = size(M, 1)
    profile = zeros(Float64, K)
    @inbounds for r in 0:(K - 1)
        total = 0.0
        for i in 1:K
            total += M[i, periodic(i + r, K)]
        end
        profile[r + 1] = total / K
    end
    return profile
end

rmse(a, b) = sqrt(mean((a .- b) .^ 2))
rms(a) = sqrt(mean(abs2, a))

function build_rmse_tables(Cdot_data::Dict{Symbol, Matrix{Float64}}, Cdot_ref_active::Dict{Symbol, Matrix{Float64}})
    rmse_by_observable = Dict{Symbol, Float64}()
    rel_by_observable = Dict{Symbol, Float64}()
    rmse_by_channel = Dict{String, Float64}()
    floor_scale = 1.0e-12
    for key in OBSERVABLE_KEYS
        data_vals = Cdot_data[key][2:end, :]
        ref_vals = Cdot_ref_active[key]
        obs_rmse = rmse(data_vals, ref_vals)
        obs_rel = obs_rmse / max(rms(data_vals), floor_scale)
        rmse_by_observable[key] = obs_rmse
        rel_by_observable[key] = obs_rel
        for idx in 1:size(data_vals, 2)
            rmse_by_channel[string(key) * ":" * string(idx)] = rmse(@view(data_vals[:, idx]), @view(ref_vals[:, idx]))
        end
    end
    all_data = vcat([vec(Cdot_data[key][2:end, :]) for key in OBSERVABLE_KEYS]...)
    all_ref = vcat([vec(Cdot_ref_active[key]) for key in OBSERVABLE_KEYS]...)
    global_rmse = rmse(all_data, all_ref)
    global_rel = global_rmse / max(rms(all_data), floor_scale)
    return rmse_by_observable, rel_by_observable, rmse_by_channel, global_rmse, global_rel
end

function _create_diagnostics_figure_impl(paths::ManagedRunPaths, taus::Vector{Float64}, Cdot_data::Dict{Symbol, Matrix{Float64}},
        Cdot_ref_active::Dict{Symbol, Matrix{Float64}}, Cdot_ref_q::Dict{Symbol, Matrix{Float64}},
        reference::ReferenceMobilityResult, params::FitDML96Params, rmse_by_observable::Dict{Symbol, Float64},
        Phi_data_profile::Vector{Float64}, Phi_ref_profile::Vector{Float64}, Phi_q_profile::Vector{Float64}, Q::Matrix{Float64})
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(size=(params.figure_width, params.figure_height))
        figure_title!(fig, "L96 Conditional-Score Identity Validation";
            subtitle=@sprintf("Reference mode: %s | taus in [%.3f, %.3f] | pairs/tau = %d",
                reference.active_mode, first(taus), last(taus), params.pairs_per_tau))

        panel_specs = [
            (:coord, params.plot_coordinate_separations, fig[1, 1]),
            (:var, params.plot_standard_nonlinear_separations, fig[1, 2]),
            (:nn1, params.plot_standard_nonlinear_separations, fig[2, 1]),
            (:nn2, params.plot_standard_nonlinear_separations, fig[2, 2]),
            (:adv, params.plot_adv_flux_separations, fig[3, 1]),
            (:flux, params.plot_adv_flux_separations, fig[3, 2]),
        ]

        positive_taus = taus[2:end]
        coord_offsets = params.coordinate_separations
        nonlinear_offsets = params.nonlinear_separations

        for (key, plotted_seps, parent) in panel_specs
            ax = Axis(parent;
                xlabel="tau",
                ylabel="Cdot",
                title=@sprintf("%s | RMSE = %.3e", observable_title(key), rmse_by_observable[key]))
            sep_list = key == :coord ? coord_offsets : nonlinear_offsets
            sep_to_idx = Dict(sep => idx for (idx, sep) in enumerate(sep_list))
            colors = Makie.wong_colors()
            legend_entries = Any[]
            legend_labels = String[]
            for (local_idx, sep) in enumerate(plotted_seps)
                haskey(sep_to_idx, sep) || continue
                idx = sep_to_idx[sep]
                color = colors[mod1(local_idx, length(colors))]
                data_line = lines!(ax, taus, Cdot_data[key][:, idx]; color=color, linewidth=curve_linewidth())
                ref_line = lines!(ax, positive_taus, Cdot_ref_active[key][:, idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9))
                push!(legend_entries, data_line)
                push!(legend_labels, @sprintf("r = %d data/ref", sep))
                _ = ref_line
            end
            axislegend(ax, legend_entries, legend_labels; position=:rb, framevisible=false, nbanks=2)
        end

        ax_phi = Axis(fig[4, 1]; xlabel="separation r", ylabel="Phi profile (u-units)", title="Phi profile comparison")
        separations = collect(0:(length(Phi_data_profile) - 1))
        lines!(ax_phi, separations, Phi_data_profile; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
        lines!(ax_phi, separations, Phi_q_profile; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9), label="Q-only reference")
        if reference.active_mode == "local_antisymmetric_fit" && reference.fitted_matrix !== nothing
            lines!(ax_phi, separations, Phi_ref_profile; color=STYLE_PRIMARY, linestyle=:dashdot, linewidth=curve_linewidth(emphasis=0.9), label="fitted antisymmetric reference")
        else
            lines!(ax_phi, separations, Phi_ref_profile; color=STYLE_PRIMARY, linestyle=:dashdot, linewidth=curve_linewidth(emphasis=0.9), label="active reference")
        end
        axislegend(ax_phi; position=:rb, framevisible=false)

        ax_q = Axis(fig[4, 2]; xlabel="j", ylabel="i", title="Diffusion tensor Q")
        hm = heatmap!(ax_q, collect(1:size(Q, 2)), collect(1:size(Q, 1)), Q; colormap=STYLE_SEQUENTIAL_BLUE)
        Colorbar(fig[4, 3], hm; label="Q_{ij}")

        save_figure(paths.figure_png, fig)
        save(paths.figure_pdf, fig)
    end
    return nothing
end

function create_diagnostics_figure(paths::ManagedRunPaths, taus::Vector{Float64}, Cdot_data::Dict{Symbol, Matrix{Float64}},
        Cdot_ref_active::Dict{Symbol, Matrix{Float64}}, Cdot_ref_q::Dict{Symbol, Matrix{Float64}},
        reference::ReferenceMobilityResult, params::FitDML96Params, rmse_by_observable::Dict{Symbol, Float64},
        Phi_data_profile::Vector{Float64}, Phi_ref_profile::Vector{Float64}, Phi_q_profile::Vector{Float64}, Q::Matrix{Float64})
    ensure_figure_support_loaded!()
    return Base.invokelatest(_create_diagnostics_figure_impl, paths, taus, Cdot_data, Cdot_ref_active, Cdot_ref_q,
        reference, params, rmse_by_observable, Phi_data_profile, Phi_ref_profile, Phi_q_profile, Q)
end

function save_outputs(path::AbstractString, taus::Vector{Float64}, observable_names::Vector{String},
        separation_lists::Dict{String, Vector{Int}}, C_data::Dict{Symbol, Matrix{Float64}},
        C_smooth::Dict{Symbol, Matrix{Float64}}, Cdot_data::Dict{Symbol, Matrix{Float64}},
        Cdot_data_spline::Dict{Symbol, Matrix{Float64}},
        Cdot_ref_active::Dict{Symbol, Matrix{Float64}}, Cdot_ref_q::Dict{Symbol, Matrix{Float64}},
        Cdot_ref_fit::Dict{Symbol, Matrix{Float64}}, Phi_data::Matrix{Float64}, Phi_ref::Matrix{Float64},
        Phi_q::Matrix{Float64}, Q::Matrix{Float64}, mu::Float64, sigma_x::Float64,
        rmse_by_observable::Dict{Symbol, Float64}, rmse_by_channel::Dict{String, Float64},
        rel_by_observable::Dict{Symbol, Float64}, parameter_dictionary, smoothing_selections,
        reference::ReferenceMobilityResult)
    BSON.@save path taus observable_names separation_lists C_data C_smooth Cdot_data Cdot_data_spline Cdot_ref_active Cdot_ref_q Cdot_ref_fit Phi_data Phi_ref Phi_q Q mu sigma_x rmse_by_observable rmse_by_channel rel_by_observable parameter_dictionary smoothing_selections reference
    return nothing
end

function write_metrics_report(path::AbstractString, params::FitDML96Params, sampler::PairSampler,
    input_hdf5::AbstractString,
        score_path::AbstractString, joint_path::AbstractString, reference::ReferenceMobilityResult,
        rmse_by_observable::Dict{Symbol, Float64}, rel_by_observable::Dict{Symbol, Float64},
        global_rmse::Float64, global_rel::Float64, qonly_rmse::Float64, qonly_rel::Float64,
        phi_rmse::Float64, phi_rel::Float64, sign_test_passed::Bool, sign_rmse::Float64, sign_flip_rmse::Float64,
        coordinate_separations::Vector{Int}, nonlinear_separations::Vector{Int})
    open(path, "w") do io
        println(io, "L96 conditional-score reference validation")
        println(io, "Cdot_data target = generator-based Lphi correlation")
        println(io, @sprintf("Global RMSE (active reference) = %.8e", global_rmse))
        println(io, @sprintf("Global relative RMSE (active reference) = %.8e", global_rel))
        println(io, @sprintf("Global RMSE (Q-only baseline) = %.8e", qonly_rmse))
        println(io, @sprintf("Global relative RMSE (Q-only baseline) = %.8e", qonly_rel))
        println(io, @sprintf("Phi RMSE (u-units profile) = %.8e", phi_rmse))
        println(io, @sprintf("Phi relative RMSE (u-units profile) = %.8e", phi_rel))
        println(io, @sprintf("Conditional-score sign test passed = %s", string(sign_test_passed)))
        println(io, @sprintf("RMSE with configured sign = %.8e", sign_rmse))
        println(io, @sprintf("RMSE with flipped sign = %.8e", sign_flip_rmse))
        println(io)
        println(io, "RMSE by observable family")
        for key in OBSERVABLE_KEYS
            println(io, @sprintf("%-6s | RMSE = %.8e | rel_RMSE = %.8e", String(key), rmse_by_observable[key], rel_by_observable[key]))
        end
        println(io)
        println(io, "Reference mobility")
        println(io, "Configured mode = " * params.reference_mobility)
        println(io, "Active mode = " * reference.active_mode)
        println(io, "Status = " * reference.status)
        if !isempty(reference.coefficients)
            println(io, "Fitted pair offsets = " * string(reference.offsets))
            println(io, "Fitted feature offsets = " * string(reference.feature_offsets))
            println(io, "Quadratic local feature count = " * string(length(reference.quadratic_pairs)))
            for (offset_idx, offset) in enumerate(reference.offsets)
                coeff_row = reference.coefficients[offset_idx, :]
                println(io, "coeffs[offset=" * string(offset) * "] = " * join([@sprintf("%.6e", c) for c in coeff_row], ", "))
            end
            println(io, @sprintf("Fit condition number = %.8e", reference.fit_condition))
            println(io, @sprintf("Fit RMS residual = %.8e", reference.fit_rms))
        end
        println(io)
        println(io, "Inputs")
        println(io, "Score model = " * score_path)
        println(io, "Joint score model = " * joint_path)
        println(io, "Data file = " * String(input_hdf5))
        println(io, "Number of sampled pairs per tau = " * string(params.pairs_per_tau))
        println(io, "Lag list = [" * join([@sprintf("%.3f", tau) for tau in sampler.lag_times], ", ") * "]")
        println(io, "Coordinate separations = " * string(coordinate_separations))
        println(io, "Nonlinear separations = " * string(nonlinear_separations))
    end
    return nothing
end

function agreement_is_acceptable(global_rel::Float64, phi_rel::Float64, sign_test_passed::Bool)
    return sign_test_passed && global_rel <= 0.5 && phi_rel <= 0.5
end

include(joinpath(@__DIR__, "src", "learned_mobility_pipeline.jl"))

function run_pipeline(param_file::AbstractString)
    params, raw = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.plain_score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)

    paths, reused_existing_dir = managed_run_paths(param_file, repo_resolve_path(params.runs_root))
    copy_managed_config(param_file, paths)
    write_run_metadata(paths.metadata_toml, params, param_file, paths, reused_existing_dir)

    times, states_preview = load_state_tensor(input_hdf5)
    K = size(states_preview, 2)

    device = detect_device(params.device_name)
    activate_device!(device)
    @printf("Analysis device: %s\n", describe_device(device))

    models = load_models(score_bson, joint_bson, device, K)
    sampler = build_pair_sampler(input_hdf5, params.burnin_fraction, params.lag_stride, models.tau_min, models.tau_max)

    score_mean_gap = maximum(abs.(models.score_mean .- models.joint_mean))
    score_std_gap = maximum(abs.(models.score_std .- models.joint_std))
    @printf("Score/joint normalization gap: max |mean gap| = %.3e, max |std gap| = %.3e\n", score_mean_gap, score_std_gap)

    mu, sigma_x, obs_means = compute_standardization_and_observable_means(sampler.states, sampler.start_idx)
    reference = fit_local_antisymmetric_reference(models, sampler, params, mu, sigma_x, device)
    @printf("Reference status: %s\n", reference.status)

    taus, C_data, Cdot_data, Cdot_ref_q, Cdot_ref_fit, coord_offsets_full = evaluate_c_and_generator_cdot_channels(
        sampler, models, params, mu, sigma_x, obs_means, reference, device)
    C_smooth, Cdot_data_spline, smoothing_selections = smooth_all_channels(taus, C_data, params)

    Cdot_ref_active = reference.active_mode == "local_antisymmetric_fit" && reference.fitted_matrix !== nothing ? Cdot_ref_fit : Cdot_ref_q

    rmse_by_observable, rel_by_observable, rmse_by_channel, global_rmse, global_rel = build_rmse_tables(Cdot_data, Cdot_ref_active)
    _, _, _, qonly_rmse, qonly_rel = build_rmse_tables(Cdot_data, Cdot_ref_q)
    sign_rmse = global_rmse
    sign_flip_rmse = rmse(vcat([vec(Cdot_data[key][2:end, :]) for key in OBSERVABLE_KEYS]...),
        vcat([vec(-Cdot_ref_active[key]) for key in OBSERVABLE_KEYS]...))
    sign_test_passed = sign_rmse <= sign_flip_rmse
    if !sign_test_passed
        @warn "The conditional-score sign test prefers the flipped sign. Check score conventions carefully." sign_rmse sign_flip_rmse
    end

    Phi_data_profile = -vec(Cdot_data[:coord_full][1, :])
    Phi_data = circulant_matrix_from_profile(Phi_data_profile)
    Phi_q = sampler.Q ./ (sigma_x^2)
    Phi_q_profile = circulant_profile(Phi_q)
    Phi_ref_matrix = reference.active_matrix ./ (sigma_x^2)
    Phi_ref_profile = circulant_profile(Phi_ref_matrix)
    Phi_ref = Phi_ref_matrix
    phi_rmse = rmse(Phi_data_profile, Phi_ref_profile)
    phi_rel = phi_rmse / max(rms(Phi_data_profile), 1.0e-12)

    create_diagnostics_figure(paths, taus, Cdot_data, Cdot_ref_active, Cdot_ref_q, reference, params,
        rmse_by_observable, Phi_data_profile, Phi_ref_profile, Phi_q_profile, sampler.Q)

    observable_names = [observable_title(key) for key in OBSERVABLE_KEYS]
    separation_lists = Dict(
        "coordinate" => params.coordinate_separations,
        "coordinate_full" => coord_offsets_full,
        "variance" => params.nonlinear_separations,
        "nn1" => params.nonlinear_separations,
        "nn2" => params.nonlinear_separations,
        "adv" => params.nonlinear_separations,
        "flux" => params.nonlinear_separations,
    )
    save_outputs(paths.output_bson, taus, observable_names, separation_lists, C_data, C_smooth, Cdot_data,
        Cdot_data_spline,
        Cdot_ref_active, Cdot_ref_q, Cdot_ref_fit, Phi_data, Phi_ref, Phi_q, sampler.Q, mu, sigma_x,
        rmse_by_observable, rmse_by_channel, rel_by_observable, raw, smoothing_selections, reference)

    write_metrics_report(paths.metrics_txt, params, sampler, input_hdf5, score_bson, joint_bson, reference,
        rmse_by_observable, rel_by_observable, global_rmse, global_rel, qonly_rmse, qonly_rel,
        phi_rmse, phi_rel, sign_test_passed, sign_rmse, sign_flip_rmse,
        params.coordinate_separations, params.nonlinear_separations)

    learned_summary = nothing
    if params.mobility_nn_enabled
        learned_summary = run_learned_mobility_pipeline(paths, sampler, models, params, input_hdf5, device,
            mu, sigma_x, obs_means, reference, taus, C_data, Cdot_data, Cdot_ref_q, Cdot_ref_fit,
            Phi_data_profile, raw, paths.metrics_txt)
    end

    acceptable = learned_summary === nothing ?
        agreement_is_acceptable(global_rel, phi_rel, sign_test_passed) : learned_summary.acceptable
    println()
    println("Summary")
    println("Run directory: " * paths.run_dir)
    println(@sprintf("Global RMSE = %.6e", global_rmse))
    for key in OBSERVABLE_KEYS
        println(@sprintf("%s RMSE = %.6e | rel = %.6e", observable_title(key), rmse_by_observable[key], rel_by_observable[key]))
    end
    println(@sprintf("Phi RMSE = %.6e | rel = %.6e", phi_rmse, phi_rel))
    if learned_summary !== nothing
        println(@sprintf("Learned-M mean channel nRMSE = %.6e", learned_summary.fit_mean_normalized_rmse))
        println(@sprintf("Learned-M mean Cdot RMSE = %.6e", learned_summary.fit_mean_cdot_rmse))
        if learned_summary.forward_cphi_mean_rmse !== nothing
            println(@sprintf("Forward-validation Cphi RMSE full/Phi = %.6e / %.6e",
                learned_summary.forward_cphi_mean_rmse, learned_summary.forward_phi_cphi_mean_rmse))
        end
    end
    println("Conditional-score sign test passed: " * string(sign_test_passed))
    println("Reference mode used: " * reference.active_mode)
    println("Agreement acceptable: " * string(acceptable))
    return paths.run_dir, acceptable
end

function run_managed_pipeline(param_file::AbstractString)
    return run_pipeline(param_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_managed_pipeline(param_file)
end
