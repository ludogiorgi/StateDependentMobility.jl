#!/usr/bin/env julia

import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")
const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "fit_dM_test.toml")
const JOINT_STATE_CHANNELS = 2
const JOINT_INPUT_CHANNELS = 3

function ensure_packages(packages::Vector{String})
    project_deps = Pkg.project().dependencies
    missing = String[]
    for pkg in packages
        haskey(project_deps, pkg) || push!(missing, pkg)
    end
    isempty(missing) || Pkg.add(missing)
    return nothing
end

ensure_packages(["BSON", "CUDA", "cuDNN", "Flux", "Functors", "GLMakie", "HDF5", "KernelDensity", "LaTeXStrings", "NNlib", "TOML"])

using BSON
using CUDA
using cuDNN
using Flux
using Functors
using HDF5
using KernelDensity
using LaTeXStrings
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
include(joinpath(SCOREUNET_SRC, "EnsembleIntegrator.jl"))

isfile(STYLE_FILE) || error("Shared figure style file not found: $(STYLE_FILE)")
include(STYLE_FILE)
GLMakie.activate!()

Base.@kwdef struct TestParams
    input_hdf5::String
    plain_score_bson::String
    joint_score_bson::String
    burnin_fraction::Float64
    lag_stride::Int
    pairs_per_tau::Int
    eval_batch_size::Int
    max_tau::Float64
    stein_samples::Int
    stein_ridge::Float64
    antisymmetric_scale::Float64
    dt::Float64
    save_stride::Int
    total_time::Float64
    burnin_time::Float64
    ntrajectories::Int
    forward_seed::Int
    score_batch_size::Int
    use_observed_initial_conditions::Bool
    clamp_eval_to_support::Bool
    hard_clamp_state::Bool
    support_pad_fraction::Float64
    pdf_bins::Int
    pdf_max_samples::Int
    correlation_max_time::Float64
    correlation_threshold::Float64
    cross_offsets::Vector{Int}
    mobility_nn_enabled::Bool
    mobility_nn_offsets::Vector{Int}
    mobility_nn_factor_offsets::Vector{Int}
    mobility_nn_window_offsets::Vector{Int}
    mobility_nn_coordinate_separations::Vector{Int}
    mobility_nn_quadratic_offsets::Vector{Int}
    mobility_nn_cubic_offsets::Vector{Int}
    mobility_nn_quartic_offsets::Vector{Int}
    mobility_nn_l96_channels::Vector{String}
    mobility_nn_select_channels::Bool
    mobility_nn_selection_min_channels::Int
    mobility_nn_selection_max_channels::Int
    mobility_nn_selection_min_amplitude_fraction::Float64
    mobility_nn_selection_max_noise_ratio::Float64
    mobility_nn_selection_max_roughness_ratio::Float64
    mobility_nn_selection_mandatory_channels::Vector{String}
    mobility_nn_pairs_per_tau::Int
    mobility_nn_tau_batch_size::Int
    mobility_nn_anchor_states::Int
    mobility_nn_epochs::Int
    mobility_nn_min_epochs::Int
    mobility_nn_eval_every::Int
    mobility_nn_plateau_patience::Int
    mobility_nn_plateau_rtol::Float64
    mobility_nn_learning_rate::Float64
    mobility_nn_lag_weight_power::Float64
    mobility_nn_zero_mean_penalty::Float64
    mobility_nn_zero_mean_penalty_final_scale::Float64
    mobility_nn_anchor_rms_penalty::Float64
    mobility_nn_current_action_penalty::Float64
    mobility_nn_current_action_samples::Int
    mobility_nn_current_action_batch_size::Int
    mobility_nn_factor_rms_penalty::Float64
    mobility_nn_skew_rms_penalty::Float64
    mobility_nn_weight_decay::Float64
    mobility_nn_scale_floor::Float64
    mobility_nn_forward_scale::Float64
    mobility_nn_psd_jitter::Float64
    mobility_nn_validation_pair_seeds::Vector{Int}
    mobility_nn_widths::Vector{Int}
    output_dir::String
    figure_png::String
    diagnostics_bson::String
    metrics_txt::String
    figure_width::Int
    figure_height::Int
    device_name::String
    seed::Int
    run_mode::String
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

struct PairPdfResult
    offset::Int
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
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

struct LoadedScoreModel
    model
    sigma::Float32
    mean::Vector{Float32}
    std::Vector{Float32}
end

struct JointScoreUNet{M}
    backbone::M
end

Functors.@functor JointScoreUNet (backbone,)

function (model::JointScoreUNet)(x)
    preds = model.backbone(x)
    return @view preds[:, 1:JOINT_STATE_CHANNELS, :]
end

struct LoadedJointScoreModel
    model
    sigma::Float32
    mean::Vector{Float32}
    std::Vector{Float32}
    tau_min::Float64
    tau_max::Float64
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

struct PairSampler
    times::Vector{Float64}
    states::Array{Float32, 3}
    start_idx::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    save_dt::Float64
end

struct PdfReference
    centers::Vector{Float64}
    density::Vector{Float64}
    boundary::Tuple{Float64, Float64}
end

struct PdfDiagnostics
    univariate_observed::PdfReference
    univariate_generated::PdfReference
    pair_observed::Vector{PairPdfResult}
    pair_generated::Vector{PairPdfResult}
    univariate_kl::Float64
    pair_kls::Vector{Float64}
    mean_kl::Float64
end

struct CorrelationResult
    lags::Vector{Float64}
    acf_mean::Vector{Float64}
    cross_offsets::Vector{Int}
    cross_mean::Matrix{Float64}
    t_decorrelation::Float64
    mean_value::Float64
    variance_value::Float64
end

struct MobilityNNCache
    windows::Array{Float32, 4}
    scond_u::Array{Float32, 3}
    observables::Array{Float32, 3}
    correlations::Matrix{Float64}
    data_cdot::Matrix{Float64}
    phi_cdot::Matrix{Float64}
    anchor_windows::Matrix{Float32}
    taus::Vector{Float64}
    channel_names::Vector{String}
end

struct MobilityNNHistory
    epochs::Vector{Int}
    train_loss::Vector{Float64}
    normalized_rmse::Vector{Float64}
    physical_rmse::Vector{Float64}
    mean_abs_coeff::Vector{Float64}
    anchor_rms::Vector{Float64}
    weight_l2::Vector{Float64}
end

struct CurrentActionCache
    windows::Array{Float32, 3}
    score_u::Matrix{Float32}
    target::Matrix{Float32}
    scale::Float32
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

function latex_text(s::AbstractString)
    escaped = replace(String(s),
        "\\" => "\\textbackslash{}",
        "_" => "\\_",
        "%" => "\\%",
        "&" => "\\&",
        "#" => "\\#",
        " " => "\\;")
    return latexstring("\\mathrm{" * escaped * "}")
end

function latex_channel_name(name::AbstractString)
    m = match(r"^u\[r=(\d+)\]$", name)
    m !== nothing && return latexstring("u_{i+" * m.captures[1] * "}")
    m = match(r"^u\*u\[q=(\d+)\]$", name)
    m !== nothing && return latexstring("u_i u_{i+" * m.captures[1] * "}")
    m = match(r"^u\^2\*u\[q=(\d+)\]$", name)
    m !== nothing && return latexstring("u_i^2 u_{i+" * m.captures[1] * "}")
    m = match(r"^u\^2\*u\^2\[q=(\d+)\]$", name)
    m !== nothing && return latexstring("u_i^2 u_{i+" * m.captures[1] * "}^2")
    name == "adv" && return latexstring("u_{i-1}(u_{i+1}-u_{i-2})")
    name == "flux" && return latexstring("u_i u_{i-1}(u_{i+1}-u_{i-2})")
    return latex_text(name)
end

function dict_get(d, key::Symbol)
    haskey(d, key) && return d[key]
    skey = String(key)
    haskey(d, skey) && return d[skey]
    error("Key $(key) not found.")
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
    return device isa GPUDevice ? "GPU:" * join(device.ids, ",") : "CPU"
end

function stats_vector(raw_stats, field::Symbol, K::Int)
    raw = raw_stats isa DataStats ? getfield(raw_stats, field) : dict_get(raw_stats, field)
    vec_stats = vec(Float32.(to_host(raw)))
    length(vec_stats) == 1 && return fill(vec_stats[1], K)
    require_condition(length(vec_stats) == K, "Normalization stats must have length 1 or K.")
    return vec_stats
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    cdot = raw["cdot"]
    mobility = get(raw, "mobility", Dict{String, Any}())
    fwd = raw["forward_validation"]
    mobnn = get(raw, "mobility_nn", Dict{String, Any}())
    out = raw["output"]
    fig = raw["figure"]
    run = get(raw, "run", Dict{String, Any}())
    params = TestParams(
        input_hdf5=String(data["input_hdf5"]),
        plain_score_bson=String(data["plain_score_bson"]),
        joint_score_bson=String(get(data, "joint_score_bson", "outputs/joint_score.bson")),
        burnin_fraction=Float64(data["burnin_fraction"]),
        lag_stride=Int(cdot["lag_stride"]),
        pairs_per_tau=Int(cdot["pairs_per_tau"]),
        eval_batch_size=Int(cdot["eval_batch_size"]),
        max_tau=Float64(cdot["max_tau"]),
        stein_samples=Int(get(cdot, "stein_samples", cdot["pairs_per_tau"])),
        stein_ridge=Float64(get(cdot, "stein_ridge", 1.0e-6)),
        antisymmetric_scale=Float64(get(mobility, "antisymmetric_scale", 0.0)),
        dt=Float64(fwd["dt"]),
        save_stride=Int(fwd["save_stride"]),
        total_time=Float64(fwd["total_time"]),
        burnin_time=Float64(fwd["burnin_time"]),
        ntrajectories=Int(fwd["ntrajectories"]),
        forward_seed=Int(fwd["seed"]),
        score_batch_size=Int(fwd["score_batch_size"]),
        use_observed_initial_conditions=Bool(get(fwd, "use_observed_initial_conditions", true)),
        clamp_eval_to_support=Bool(get(fwd, "clamp_eval_to_support", false)),
        hard_clamp_state=Bool(get(fwd, "hard_clamp_state", false)),
        support_pad_fraction=Float64(get(fwd, "support_pad_fraction", 0.1)),
        pdf_bins=Int(fwd["pdf_bins"]),
        pdf_max_samples=Int(fwd["pdf_max_samples"]),
        correlation_max_time=Float64(fwd["correlation_max_time"]),
        correlation_threshold=Float64(fwd["correlation_threshold"]),
        cross_offsets=Int.(fwd["cross_offsets"]),
        mobility_nn_enabled=Bool(get(mobnn, "enabled", false)),
        mobility_nn_offsets=Int.(get(mobnn, "offsets", [1, 2, 3])),
        mobility_nn_factor_offsets=Int.(get(mobnn, "factor_offsets", [0, 1, 2, 3])),
        mobility_nn_window_offsets=Int.(get(mobnn, "window_offsets", [-3, -2, -1, 0, 1, 2, 3])),
        mobility_nn_coordinate_separations=Int.(get(mobnn, "coordinate_separations", [0, 1, 2, 3, 4, 5, 8, 10, 15, 20])),
        mobility_nn_quadratic_offsets=Int.(get(mobnn, "quadratic_offsets", Int[])),
        mobility_nn_cubic_offsets=Int.(get(mobnn, "cubic_offsets", Int[])),
        mobility_nn_quartic_offsets=Int.(get(mobnn, "quartic_offsets", Int[])),
        mobility_nn_l96_channels=String.(get(mobnn, "l96_channels", String[])),
        mobility_nn_select_channels=Bool(get(mobnn, "select_channels", true)),
        mobility_nn_selection_min_channels=Int(get(mobnn, "selection_min_channels", 6)),
        mobility_nn_selection_max_channels=Int(get(mobnn, "selection_max_channels", 14)),
        mobility_nn_selection_min_amplitude_fraction=Float64(get(mobnn, "selection_min_amplitude_fraction", 0.08)),
        mobility_nn_selection_max_noise_ratio=Float64(get(mobnn, "selection_max_noise_ratio", 0.70)),
        mobility_nn_selection_max_roughness_ratio=Float64(get(mobnn, "selection_max_roughness_ratio", 1.50)),
        mobility_nn_selection_mandatory_channels=String.(get(mobnn, "selection_mandatory_channels", String[])),
        mobility_nn_pairs_per_tau=Int(get(mobnn, "pairs_per_tau", 4096)),
        mobility_nn_tau_batch_size=Int(get(mobnn, "tau_batch_size", 4)),
        mobility_nn_anchor_states=Int(get(mobnn, "anchor_states", 4096)),
        mobility_nn_epochs=Int(get(mobnn, "epochs", 80)),
        mobility_nn_min_epochs=Int(get(mobnn, "min_epochs", get(mobnn, "epochs", 80))),
        mobility_nn_eval_every=Int(get(mobnn, "eval_every", 5)),
        mobility_nn_plateau_patience=Int(get(mobnn, "plateau_patience", 0)),
        mobility_nn_plateau_rtol=Float64(get(mobnn, "plateau_rtol", 1.0e-3)),
        mobility_nn_learning_rate=Float64(get(mobnn, "learning_rate", 5.0e-4)),
        mobility_nn_lag_weight_power=Float64(get(mobnn, "lag_weight_power", 0.0)),
        mobility_nn_zero_mean_penalty=Float64(get(mobnn, "zero_mean_penalty", 1.0e-2)),
        mobility_nn_zero_mean_penalty_final_scale=Float64(get(mobnn, "zero_mean_penalty_final_scale", 1.0)),
        mobility_nn_anchor_rms_penalty=Float64(get(mobnn, "anchor_rms_penalty", 0.0)),
        mobility_nn_current_action_penalty=Float64(get(mobnn, "current_action_penalty", 0.0)),
        mobility_nn_current_action_samples=Int(get(mobnn, "current_action_samples", get(mobnn, "anchor_states", 4096))),
        mobility_nn_current_action_batch_size=Int(get(mobnn, "current_action_batch_size", 1024)),
        mobility_nn_factor_rms_penalty=Float64(get(mobnn, "factor_rms_penalty", get(mobnn, "anchor_rms_penalty", 0.0))),
        mobility_nn_skew_rms_penalty=Float64(get(mobnn, "skew_rms_penalty", get(mobnn, "anchor_rms_penalty", 0.0))),
        mobility_nn_weight_decay=Float64(get(mobnn, "weight_decay", 1.0e-6)),
        mobility_nn_scale_floor=Float64(get(mobnn, "scale_floor", 1.0e-4)),
        mobility_nn_forward_scale=Float64(get(mobnn, "forward_scale", 1.0)),
        mobility_nn_psd_jitter=Float64(get(mobnn, "psd_jitter", 1.0e-8)),
        mobility_nn_validation_pair_seeds=Int.(get(mobnn, "validation_pair_seeds", [20260511, 20260512])),
        mobility_nn_widths=Int.(get(mobnn, "widths", [96, 96])),
        output_dir=String(out["output_dir"]),
        figure_png=String(out["figure_png"]),
        diagnostics_bson=String(out["diagnostics_bson"]),
        metrics_txt=String(out["metrics_txt"]),
        figure_width=Int(fig["width"]),
        figure_height=Int(fig["height"]),
        device_name=String(get(run, "device", "AUTO")),
        seed=Int(get(run, "seed", 20260429)),
        run_mode=String(get(run, "mode", "full")),
    )
    require_condition(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must lie in [0, 1).")
    require_condition(params.lag_stride >= 1, "lag_stride must be positive.")
    require_condition(params.pairs_per_tau >= 1, "pairs_per_tau must be positive.")
    require_condition(params.eval_batch_size >= 1, "eval_batch_size must be positive.")
    require_condition(params.max_tau > 0.0, "max_tau must be positive.")
    require_condition(params.stein_samples >= 1, "stein_samples must be positive.")
    require_condition(params.stein_ridge >= 0.0, "stein_ridge must be nonnegative.")
    require_condition(all(offset -> offset > 0, params.mobility_nn_offsets), "mobility_nn.offsets must be positive.")
    require_condition(all(offset -> offset >= 0, params.mobility_nn_factor_offsets), "mobility_nn.factor_offsets must be nonnegative.")
    require_condition(0 in params.mobility_nn_factor_offsets, "mobility_nn.factor_offsets must contain 0.")
    require_condition(0 in params.mobility_nn_window_offsets, "mobility_nn.window_offsets must contain 0.")
    require_condition(all(offset -> offset in params.mobility_nn_window_offsets, params.mobility_nn_offsets),
        "Each mobility_nn.offset must also appear in mobility_nn.window_offsets.")
    require_condition(all(offset -> offset in params.mobility_nn_window_offsets, params.mobility_nn_factor_offsets),
        "Each mobility_nn.factor_offset must also appear in mobility_nn.window_offsets.")
    require_condition(all(offset -> offset >= 0, params.mobility_nn_quadratic_offsets), "mobility_nn.quadratic_offsets must be nonnegative.")
    require_condition(all(offset -> offset >= 0, params.mobility_nn_cubic_offsets), "mobility_nn.cubic_offsets must be nonnegative.")
    require_condition(all(offset -> offset >= 0, params.mobility_nn_quartic_offsets), "mobility_nn.quartic_offsets must be nonnegative.")
    require_condition(all(name -> name in ("adv", "flux"), params.mobility_nn_l96_channels),
        "mobility_nn.l96_channels can contain only \"adv\" and \"flux\".")
    require_condition(params.mobility_nn_selection_min_channels >= 1, "mobility_nn.selection_min_channels must be positive.")
    require_condition(params.mobility_nn_selection_max_channels >= params.mobility_nn_selection_min_channels,
        "mobility_nn.selection_max_channels must be >= selection_min_channels.")
    require_condition(params.mobility_nn_selection_min_amplitude_fraction >= 0.0,
        "mobility_nn.selection_min_amplitude_fraction must be non-negative.")
    require_condition(params.mobility_nn_selection_max_noise_ratio > 0.0,
        "mobility_nn.selection_max_noise_ratio must be positive.")
    require_condition(params.mobility_nn_selection_max_roughness_ratio > 0.0,
        "mobility_nn.selection_max_roughness_ratio must be positive.")
    require_condition(params.mobility_nn_pairs_per_tau >= 256, "mobility_nn.pairs_per_tau must be at least 256.")
    require_condition(params.mobility_nn_tau_batch_size >= 1, "mobility_nn.tau_batch_size must be positive.")
    require_condition(params.mobility_nn_anchor_states >= 256, "mobility_nn.anchor_states must be at least 256.")
    require_condition(params.mobility_nn_epochs >= 1, "mobility_nn.epochs must be positive.")
    require_condition(params.mobility_nn_min_epochs >= 1, "mobility_nn.min_epochs must be positive.")
    require_condition(params.mobility_nn_min_epochs <= params.mobility_nn_epochs,
        "mobility_nn.min_epochs must be <= mobility_nn.epochs.")
    require_condition(params.mobility_nn_eval_every >= 1, "mobility_nn.eval_every must be positive.")
    require_condition(params.mobility_nn_plateau_patience >= 0, "mobility_nn.plateau_patience must be non-negative.")
    require_condition(params.mobility_nn_plateau_rtol >= 0.0, "mobility_nn.plateau_rtol must be non-negative.")
    require_condition(params.mobility_nn_learning_rate > 0.0, "mobility_nn.learning_rate must be positive.")
    require_condition(params.mobility_nn_anchor_rms_penalty >= 0.0, "mobility_nn.anchor_rms_penalty must be non-negative.")
    require_condition(params.mobility_nn_current_action_penalty >= 0.0, "mobility_nn.current_action_penalty must be non-negative.")
    require_condition(params.mobility_nn_current_action_samples >= 0, "mobility_nn.current_action_samples must be non-negative.")
    require_condition(params.mobility_nn_current_action_batch_size >= 1, "mobility_nn.current_action_batch_size must be positive.")
    require_condition(params.mobility_nn_factor_rms_penalty >= 0.0, "mobility_nn.factor_rms_penalty must be non-negative.")
    require_condition(params.mobility_nn_skew_rms_penalty >= 0.0, "mobility_nn.skew_rms_penalty must be non-negative.")
    require_condition(params.mobility_nn_scale_floor > 0.0, "mobility_nn.scale_floor must be positive.")
    require_condition(isfinite(params.mobility_nn_forward_scale), "mobility_nn.forward_scale must be finite.")
    require_condition(params.mobility_nn_psd_jitter > 0.0, "mobility_nn.psd_jitter must be positive.")
    require_condition(params.total_time > params.burnin_time >= 0.0, "Need total_time > burnin_time >= 0.")
    require_condition(params.save_stride >= 1, "save_stride must be positive.")
    return params
end

function load_state_tensor(path::AbstractString)
    times = Float64.(h5read(path, "/trajectories/time"))
    states_raw = h5read(path, "/trajectories/states")
    states = if size(states_raw, 1) == length(times)
        Float32.(states_raw)
    elseif size(states_raw, 2) == length(times)
        Float32.(permutedims(states_raw, (2, 1, 3)))
    else
        error("Could not infer time axis for /trajectories/states.")
    end
    @printf("Loaded states with shape %s (time, dimension, trajectory)\n", string(size(states)))
    return times, states
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function build_pair_sampler(path::AbstractString, params::TestParams)
    times, states = load_state_tensor(path)
    save_dt = times[2] - times[1]
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    max_lag = min(length(times) - start_idx - 1, floor(Int, params.max_tau / save_dt + 1e-9))
    require_condition(max_lag >= 1, "No positive lag is available for the requested max_tau.")
    lag_steps = collect(1:params.lag_stride:max_lag)
    return PairSampler(times, states, start_idx, lag_steps, lag_steps .* save_dt, save_dt)
end

function load_l96_generator_metadata(path::AbstractString)
    forcing = Float64(h5read(path, "/metadata/F"))
    Q = Float64.(h5read(path, "/diffusion/Q"))
    return forcing, Q
end

periodic(i::Int, K::Int) = mod1(i, K)

function l96_drift_batch!(dest::AbstractMatrix{Float64}, x::AbstractMatrix{Float32}, forcing::Float64)
    K, B = size(x)
    @inbounds for b in 1:B, i in 1:K
        dest[i, b] = Float64(x[periodic(i - 1, K), b]) *
            (Float64(x[periodic(i + 1, K), b]) - Float64(x[periodic(i - 2, K), b])) -
            Float64(x[i, b]) + forcing
    end
    return dest
end

function load_score_model(path::AbstractString, device::ExecutionDevice, K::Int)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    trainer = dict_get(blob, :trainer_cfg)
    stats = dict_get(blob, :stats)
    return LoadedScoreModel(model, Float32(trainer.sigma), stats_vector(stats, :mean, K), stats_vector(stats, :std, K))
end

function score_from_joint_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function load_joint_score_model(path::AbstractString, device::ExecutionDevice, K::Int)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    trainer = dict_get(blob, :trainer_cfg)
    stats = dict_get(blob, :stats)
    meta = dict_get(blob, :metadata)
    return LoadedJointScoreModel(model, Float32(trainer.sigma), stats_vector(stats, :mean, K),
        stats_vector(stats, :std, K), Float64(dict_get(meta, :tau_min)), Float64(dict_get(meta, :tau_max)))
end

function normalize_with_stats!(dest::AbstractArray{Float32, 3}, x::AbstractMatrix{Float32},
        mean_vec::Vector{Float32}, std_vec::Vector{Float32})
    K, B = size(x)
    @inbounds for b in 1:B, i in 1:K
        dest[i, 1, b] = (x[i, b] - mean_vec[i]) / std_vec[i]
    end
    return dest
end

function normalize_pair_with_stats!(dest::AbstractArray{Float32, 3}, x0::AbstractMatrix{Float32},
        xt::AbstractMatrix{Float32}, tnorm::Float32, mean_vec::Vector{Float32}, std_vec::Vector{Float32})
    K, B = size(x0)
    @inbounds for b in 1:B, i in 1:K
        dest[i, 1, b] = (x0[i, b] - mean_vec[i]) / std_vec[i]
        dest[i, 2, b] = (xt[i, b] - mean_vec[i]) / std_vec[i]
        dest[i, 3, b] = tnorm
    end
    return dest
end

function evaluate_stationary_score_x(model::LoadedScoreModel, x_raw::AbstractMatrix{Float32},
        batch_size::Int, device::ExecutionDevice)
    K, N = size(x_raw)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, 1, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        normalize_with_stats!(@view(scratch[:, :, 1:batch_n]), @view(x_raw[:, start:stop]), model.mean, model.std)
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_z = score_from_model(model.model, batch_dev, model.sigma)
        scores_x = reshape(to_host(scores_z), K, batch_n)
        @inbounds for b in 1:batch_n, i in 1:K
            out[i, start + b - 1] = scores_x[i, b] / model.std[i]
        end
    end
    return out
end

function evaluate_stationary_score_u(model::LoadedScoreModel, u::AbstractMatrix{Float32},
        batch_size::Int, device::ExecutionDevice)
    K, N = size(u)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, 1, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        @inbounds for b in 1:batch_n, i in 1:K
            scratch[i, 1, b] = u[i, start + b - 1]
        end
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_u = score_from_model(model.model, batch_dev, model.sigma)
        out[:, start:stop] .= reshape(to_host(scores_u), K, batch_n)
    end
    return out
end

function evaluate_stationary_score_x(model::LoadedScoreModel, x_raw::AbstractMatrix{Float32},
        batch_size::Int, device::ExecutionDevice)
    K, N = size(x_raw)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, 1, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        normalize_with_stats!(@view(scratch[:, :, 1:batch_n]), @view(x_raw[:, start:stop]), model.mean, model.std)
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_z = score_from_model(model.model, batch_dev, model.sigma)
        scores_x = reshape(to_host(scores_z), K, batch_n)
        @inbounds for b in 1:batch_n, i in 1:K
            out[i, start + b - 1] = scores_x[i, b] / model.std[i]
        end
    end
    return out
end

function evaluate_joint_score_x0(joint_model::LoadedJointScoreModel, x0_raw::AbstractMatrix{Float32},
        xt_raw::AbstractMatrix{Float32}, tnorm::Float32, batch_size::Int, device::ExecutionDevice)
    K, N = size(x0_raw)
    out = Matrix{Float32}(undef, K, N)
    scratch = Array{Float32}(undef, K, JOINT_INPUT_CHANNELS, min(batch_size, N))
    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        batch_n = stop - start + 1
        normalize_pair_with_stats!(@view(scratch[:, :, 1:batch_n]),
            @view(x0_raw[:, start:stop]), @view(xt_raw[:, start:stop]), tnorm,
            joint_model.mean, joint_model.std)
        batch_dev = move_array(@view(scratch[:, :, 1:batch_n]), device)
        scores_z = score_from_joint_model(joint_model.model, batch_dev, joint_model.sigma)
        x0_scores = reshape(@view(scores_z[:, 1, :]), K, batch_n)
        x0_scores_host = to_host(x0_scores)
        @inbounds for b in 1:batch_n, i in 1:K
            out[i, start + b - 1] = x0_scores_host[i, b] / joint_model.std[i]
        end
    end
    return out
end

function evaluate_conditional_score_u(score_model::LoadedScoreModel, joint_model::LoadedJointScoreModel,
        x0_raw::AbstractMatrix{Float32}, xt_raw::AbstractMatrix{Float32}, tau::Float64,
        score_batch_size::Int, joint_batch_size::Int, sigma_x::Float64, device::ExecutionDevice)
    require_condition(joint_model.tau_min <= tau <= joint_model.tau_max + 1.0e-10,
        @sprintf("Requested tau %.6f is outside joint-score range [%.6f, %.6f].",
            tau, joint_model.tau_min, joint_model.tau_max))
    denom = max(joint_model.tau_max - joint_model.tau_min, eps(Float64))
    tnorm = Float32((tau - joint_model.tau_min) / denom)
    stat_score_x = evaluate_stationary_score_x(score_model, x0_raw, score_batch_size, device)
    joint_score_x = evaluate_joint_score_x0(joint_model, x0_raw, xt_raw, tnorm, joint_batch_size, device)
    return Float32(sigma_x) .* (joint_score_x .- stat_score_x)
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

function sample_pair_batch!(x0::AbstractMatrix{Float32}, xt::AbstractMatrix{Float32},
        sampler::PairSampler, lag::Int, rng::AbstractRNG)
    nt, K, ntraj = size(sampler.states)
    upper = nt - lag
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

function sample_pair_increment_batch!(x0::AbstractMatrix{Float32}, xt::AbstractMatrix{Float32},
        xt_next::AbstractMatrix{Float32}, sampler::PairSampler, lag::Int, rng::AbstractRNG)
    nt, K, ntraj = size(sampler.states)
    upper = nt - lag - 1
    @inbounds for sample_idx in 1:size(x0, 2)
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, sampler.start_idx:upper)
        for mode_idx in 1:K
            x0[mode_idx, sample_idx] = sampler.states[time_idx, mode_idx, traj_idx]
            xt[mode_idx, sample_idx] = sampler.states[time_idx + lag, mode_idx, traj_idx]
            xt_next[mode_idx, sample_idx] = sampler.states[time_idx + lag + 1, mode_idx, traj_idx]
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

function shift_indices(separations::Vector{Int}, K::Int)
    base = collect(1:K)
    return [mod1.(base .+ sep, K) for sep in separations]
end

function accumulate_translation_channels!(accum::Vector{Float64}, phi::AbstractMatrix{<:Real},
        signal::AbstractMatrix{<:Real}, index_sets::Vector{Vector{Int}})
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

function estimate_standardization(states::Array{Float32, 3}, start_idx::Int)
    post = @view states[start_idx:end, :, :]
    mu = mean(Float64, post)
    n = length(post)
    variance = mean(x -> abs2(Float64(x) - mu), post) * n / max(n - 1, 1)
    sigma_x = sqrt(variance)
    return mu, max(sigma_x, sqrt(eps(Float64)))
end

function estimate_coordinate_cdot_and_phi(sampler::PairSampler, params::TestParams, mu::Float64, sigma_x::Float64)
    K = size(sampler.states, 2)
    offsets = collect(0:(K - 1))
    index_sets = shift_indices(offsets, K)
    taus = vcat(0.0, sampler.lag_times)
    lag_steps = vcat(0, sampler.lag_steps)
    C = zeros(Float64, length(taus), K)
    Cdot = zeros(Float64, length(taus), K)
    x0 = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt = similar(x0)
    xt_next = similar(x0)
    u0 = similar(x0)
    ut = similar(x0)
    ut_next = similar(x0)
    dudt = Matrix{Float64}(undef, K, params.eval_batch_size)
    for (tau_idx, lag) in enumerate(lag_steps)
        rng = MersenneTwister(params.seed + 1000 * tau_idx)
        csum = zeros(Float64, K)
        cdotsum = zeros(Float64, K)
        total_pairs = 0
        remaining = params.pairs_per_tau
        while remaining > 0
            batch_n = min(remaining, params.eval_batch_size)
            sample_pair_increment_batch!(@view(x0[:, 1:batch_n]), @view(xt[:, 1:batch_n]),
                @view(xt_next[:, 1:batch_n]), sampler, lag, rng)
            standardize_batch!(@view(u0[:, 1:batch_n]), @view(x0[:, 1:batch_n]), mu, sigma_x)
            standardize_batch!(@view(ut[:, 1:batch_n]), @view(xt[:, 1:batch_n]), mu, sigma_x)
            standardize_batch!(@view(ut_next[:, 1:batch_n]), @view(xt_next[:, 1:batch_n]), mu, sigma_x)
            @. dudt[:, 1:batch_n] = (ut_next[:, 1:batch_n] - ut[:, 1:batch_n]) / sampler.save_dt
            accumulate_translation_channels!(csum, @view(ut[:, 1:batch_n]), @view(u0[:, 1:batch_n]), index_sets)
            accumulate_translation_channels!(cdotsum, @view(dudt[:, 1:batch_n]), @view(u0[:, 1:batch_n]), index_sets)
            total_pairs += batch_n
            remaining -= batch_n
        end
        C[tau_idx, :] .= csum ./ total_pairs
        Cdot[tau_idx, :] .= cdotsum ./ total_pairs
        @printf("Estimated coordinate C and data-increment Cdot at tau %.4f (%d/%d)\n", taus[tau_idx], tau_idx, length(taus))
    end
    phi_profile_u = -vec(Cdot[1, :])
    Phi_u = circulant_matrix_from_profile(phi_profile_u)
    Phi_raw = sigma_x^2 .* Phi_u
    return taus, C, Cdot, phi_profile_u, Phi_u, Phi_raw
end

function estimate_phi_from_short_lag_correlations(sampler::PairSampler, params::TestParams,
        mu::Float64, sigma_x::Float64; max_lag_fit::Int=8, poly_degree::Int=2)
    K = size(sampler.states, 2)
    nt = size(sampler.states, 1)
    max_available_lag = nt - sampler.start_idx
    L = min(max_lag_fit, max_available_lag)
    require_condition(L >= 1, "Need at least one saved lag for short-lag Phi estimation.")
    degree = min(poly_degree, L)
    offsets = collect(0:(K - 1))
    index_sets = shift_indices(offsets, K)
    lag_steps = collect(0:L)
    taus_fit = lag_steps .* sampler.save_dt
    C_profiles = zeros(Float64, length(lag_steps), K)
    x0 = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt = similar(x0)
    u0 = similar(x0)
    ut = similar(x0)
    for (lag_idx, lag) in enumerate(lag_steps)
        rng = MersenneTwister(params.seed + 70_000 + 1000 * lag_idx)
        csum = zeros(Float64, K)
        total_pairs = 0
        remaining = params.pairs_per_tau
        while remaining > 0
            batch_n = min(remaining, params.eval_batch_size)
            sample_pair_batch!(@view(x0[:, 1:batch_n]), @view(xt[:, 1:batch_n]), sampler, lag, rng)
            standardize_batch!(@view(u0[:, 1:batch_n]), @view(x0[:, 1:batch_n]), mu, sigma_x)
            standardize_batch!(@view(ut[:, 1:batch_n]), @view(xt[:, 1:batch_n]), mu, sigma_x)
            accumulate_translation_channels!(csum, @view(ut[:, 1:batch_n]), @view(u0[:, 1:batch_n]), index_sets)
            total_pairs += batch_n
            remaining -= batch_n
        end
        C_profiles[lag_idx, :] .= csum ./ total_pairs
        @printf("Estimated short-lag C profile at lag %d, tau %.4f (%d/%d)\n",
            lag, taus_fit[lag_idx], lag_idx, length(lag_steps))
    end
    design = hcat([taus_fit .^ p for p in 0:degree]...)
    Cdot0_profile_fit = zeros(Float64, K)
    for r in 1:K
        coeffs = design \ C_profiles[:, r]
        Cdot0_profile_fit[r] = coeffs[2]
    end
    Phi_u_fit = circulant_matrix_from_profile(-Cdot0_profile_fit)
    Phi_x_fit = sigma_x^2 .* Phi_u_fit
    return taus_fit, C_profiles, Cdot0_profile_fit, Phi_u_fit, Phi_x_fit
end

function estimate_saved_increment_effective_diffusion(sampler::PairSampler, params::TestParams)
    K = size(sampler.states, 2)
    offsets = collect(0:(K - 1))
    index_sets = shift_indices(offsets, K)
    x0 = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt = similar(x0)
    xt_next = similar(x0)
    dx = Matrix{Float64}(undef, K, params.eval_batch_size)
    inc_cov_profile_sum = zeros(Float64, K)
    rng = MersenneTwister(params.seed + 280_001)
    remaining = params.pairs_per_tau
    total_pairs = 0
    while remaining > 0
        batch_n = min(remaining, params.eval_batch_size)
        sample_pair_increment_batch!(@view(x0[:, 1:batch_n]), @view(xt[:, 1:batch_n]),
            @view(xt_next[:, 1:batch_n]), sampler, 0, rng)
        @. dx[:, 1:batch_n] = xt_next[:, 1:batch_n] - xt[:, 1:batch_n]
        accumulate_translation_channels!(inc_cov_profile_sum, @view(dx[:, 1:batch_n]),
            @view(dx[:, 1:batch_n]), index_sets)
        total_pairs += batch_n
        remaining -= batch_n
    end
    inc_cov_profile = inc_cov_profile_sum ./ total_pairs
    D_eff = circulant_matrix_from_profile(inc_cov_profile ./ (2.0 * sampler.save_dt))
    return inc_cov_profile, D_eff
end

function estimate_clean_stein_matrix(sampler::PairSampler, score_model::LoadedScoreModel,
        params::TestParams, mu::Float64, sigma_x::Float64, device::ExecutionDevice)
    K = size(sampler.states, 2)
    offsets = collect(0:(K - 1))
    index_sets = shift_indices(offsets, K)
    x = Matrix{Float32}(undef, K, params.eval_batch_size)
    u = Matrix{Float32}(undef, K, params.eval_batch_size)
    v_profile_sum = zeros(Float64, K)
    rng = MersenneTwister(params.seed + 91_337)
    remaining = params.stein_samples
    total = 0
    while remaining > 0
        batch_n = min(remaining, params.eval_batch_size)
        sample_state_batch!(@view(x[:, 1:batch_n]), sampler, rng)
        standardize_batch!(@view(u[:, 1:batch_n]), @view(x[:, 1:batch_n]), mu, sigma_x)
        score = evaluate_stationary_score_u(score_model, @view(u[:, 1:batch_n]), params.score_batch_size, device)
        accumulate_translation_channels!(v_profile_sum, @view(score[:, 1:batch_n]), @view(u[:, 1:batch_n]), index_sets)
        total += batch_n
        remaining -= batch_n
    end
    # V = -E[s_model(u_i) u_j]. For an exact score and matching coordinates this is I.
    v_profile = .-(v_profile_sum ./ total)
    V = circulant_matrix_from_profile(v_profile)
    if params.stein_ridge > 0
        @inbounds for i in 1:K
            V[i, i] += params.stein_ridge
        end
    end
    return V, v_profile
end

function circulant_matrix_from_profile(profile::Vector{Float64})
    K = length(profile)
    out = Matrix{Float64}(undef, K, K)
    @inbounds for i in 1:K, j in 1:K
        out[i, j] = profile[mod(j - i, K) + 1]
    end
    return out
end

function build_window_tensor(u::AbstractMatrix{Float32}, window_offsets::Vector{Int})
    K, B = size(u)
    windows = Array{Float32}(undef, length(window_offsets), K, B)
    @inbounds for b in 1:B, i in 1:K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, i, b] = u[periodic(i + shift, K), b]
        end
    end
    return windows
end

function build_window_tensor(u::AbstractArray{Float32, 3}, window_offsets::Vector{Int})
    K, B, T = size(u)
    windows = Array{Float32}(undef, length(window_offsets), K, B, T)
    @inbounds for t in 1:T, b in 1:B, i in 1:K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, i, b, t] = u[periodic(i + shift, K), b]
        end
    end
    return windows
end

function build_window_matrix64(u::AbstractMatrix{Float64}, window_offsets::Vector{Int})
    K, B = size(u)
    windows = Matrix{Float64}(undef, length(window_offsets), K * B)
    @inbounds for b in 1:B, i in 1:K
        col = i + (b - 1) * K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, col] = u[periodic(i + shift, K), b]
        end
    end
    return windows
end

function build_local_mobility_model(window_size::Int, widths::Vector{Int}, nfactor_offsets::Int, nskew_offsets::Int)
    layers = Any[]
    in_dim = window_size
    for width in widths
        push!(layers, Dense(in_dim, width, tanh))
        in_dim = width
    end
    push!(layers, Dense(in_dim, nfactor_offsets + nskew_offsets))
    model = Chain(layers...)
    final_layer = model.layers[end]
    final_layer.weight .= zero(eltype(final_layer.weight))
    final_layer.bias .= zero(eltype(final_layer.bias))
    return model
end

function evaluate_local_mobility_coeffs(model, windows::AbstractArray{Float32, 4},
        nfactor_offsets::Int, nskew_offsets::Int)
    coeff_flat = model(reshape(windows, size(windows, 1), :))
    coeffs = reshape(coeff_flat, nfactor_offsets + nskew_offsets, size(windows, 2), size(windows, 3), size(windows, 4))
    factor_coeffs = @view coeffs[1:nfactor_offsets, :, :, :]
    skew_coeffs = @view coeffs[(nfactor_offsets + 1):(nfactor_offsets + nskew_offsets), :, :, :]
    return factor_coeffs, skew_coeffs
end

function local_factor_action(coeffs::AbstractArray, signal::AbstractArray, offsets::Vector{Int})
    terms = map(enumerate(offsets)) do (offset_idx, offset)
        coeff_slice = @view coeffs[offset_idx, :, :, :]
        coeff_slice .* circshift(signal, (-offset, 0, 0))
    end
    return reduce(+, terms)
end

function local_factor_transpose_action(coeffs::AbstractArray, signal::AbstractArray, offsets::Vector{Int})
    terms = map(enumerate(offsets)) do (offset_idx, offset)
        coeff_slice = @view coeffs[offset_idx, :, :, :]
        circshift(coeff_slice .* signal, (offset, 0, 0))
    end
    return reduce(+, terms)
end

function local_skew_action(coeffs::AbstractArray, signal::AbstractArray, offsets::Vector{Int})
    terms = map(enumerate(offsets)) do (offset_idx, offset)
        coeff_slice = @view coeffs[offset_idx, :, :, :]
        coeff_slice .* circshift(signal, (-offset, 0, 0)) .-
            circshift(coeff_slice .* signal, (offset, 0, 0))
    end
    return reduce(+, terms)
end

function local_factor_action(coeffs::AbstractArray{Float64, 3}, signal::AbstractMatrix{Float64}, offsets::Vector{Int})
    K, B = size(signal)
    out = zeros(Float64, K, B)
    @inbounds for b in 1:B, i in 1:K
        accum = 0.0
        for (offset_idx, offset) in enumerate(offsets)
            accum += coeffs[offset_idx, i, b] * signal[periodic(i + offset, K), b]
        end
        out[i, b] = accum
    end
    return out
end

function local_factor_transpose_action(coeffs::AbstractArray{Float64, 3}, signal::AbstractMatrix{Float64}, offsets::Vector{Int})
    K, B = size(signal)
    out = zeros(Float64, K, B)
    @inbounds for b in 1:B, i in 1:K
        accum = 0.0
        for (offset_idx, offset) in enumerate(offsets)
            back_site = periodic(i - offset, K)
            accum += coeffs[offset_idx, back_site, b] * signal[back_site, b]
        end
        out[i, b] = accum
    end
    return out
end

function local_skew_action(coeffs::AbstractArray{Float64, 3}, signal::AbstractMatrix{Float64}, offsets::Vector{Int})
    K, B = size(signal)
    out = zeros(Float64, K, B)
    @inbounds for b in 1:B, i in 1:K
        accum = 0.0
        for (offset_idx, offset) in enumerate(offsets)
            forward_idx = periodic(i + offset, K)
            backward_idx = periodic(i - offset, K)
            accum += coeffs[offset_idx, i, b] * signal[forward_idx, b]
            accum -= coeffs[offset_idx, backward_idx, b] * signal[backward_idx, b]
        end
        out[i, b] = accum
    end
    return out
end

function mean_offset_coefficients(model, anchor_windows::AbstractMatrix{Float32})
    return vec(mean(model(anchor_windows); dims=2))
end

function split_mobility_outputs(raw, nfactor_offsets::Int, nskew_offsets::Int)
    return @view(raw[1:nfactor_offsets, :]), @view(raw[(nfactor_offsets + 1):(nfactor_offsets + nskew_offsets), :])
end

function anchor_rms_penalty_value(model, anchor_windows::AbstractMatrix{Float32})
    coeffs = model(anchor_windows)
    return sum(abs2, coeffs) / length(coeffs)
end

function split_anchor_rms_penalty_values(model, anchor_windows::AbstractMatrix{Float32},
        nfactor_offsets::Int, nskew_offsets::Int)
    coeffs = model(anchor_windows)
    factor_rms = nfactor_offsets == 0 ? zero(eltype(coeffs)) :
        sum(abs2, @view(coeffs[1:nfactor_offsets, :])) / (nfactor_offsets * size(coeffs, 2))
    skew_rms = nskew_offsets == 0 ? zero(eltype(coeffs)) :
        sum(abs2, @view(coeffs[(nfactor_offsets + 1):(nfactor_offsets + nskew_offsets), :])) /
        (nskew_offsets * size(coeffs, 2))
    return factor_rms, skew_rms
end

function activation_is_tanh(layer)
    return layer.σ === tanh || string(layer.σ) == "tanh"
end

function activation_is_identity(layer)
    return layer.σ === identity || string(layer.σ) == "identity"
end

function manual_chain_output_and_jacobian(model::Chain, x::AbstractMatrix{Float64})
    nsamples = size(x, 2)
    ninputs = size(x, 1)
    activ = x
    jac = zeros(Float64, size(activ, 1), ninputs, nsamples)
    @inbounds for sample_idx in 1:nsamples, input_idx in 1:ninputs
        jac[input_idx, input_idx, sample_idx] = 1.0
    end
    for layer in model.layers
        weights = Float64.(layer.weight)
        bias = Float64.(layer.bias)
        z = weights * activ .+ bias
        next_jac = Array{Float64}(undef, size(weights, 1), ninputs, nsamples)
        @inbounds for sample_idx in 1:nsamples
            next_jac[:, :, sample_idx] .= weights * @view jac[:, :, sample_idx]
        end
        if activation_is_tanh(layer)
            activ = tanh.(z)
            deriv = 1.0 .- activ .^ 2
            jac = next_jac .* reshape(deriv, size(deriv, 1), 1, nsamples)
        elseif activation_is_identity(layer)
            activ = z
            jac = next_jac
        else
            error("Unsupported activation in local mobility model.")
        end
    end
    return activ, jac
end

function local_mobility_coefficients_and_divergence_u(model, u::AbstractMatrix{Float64},
        factor_offsets::Vector{Int}, skew_offsets::Vector{Int}, window_offsets::Vector{Int},
        sym_factor::Matrix{Float64}; coefficient_scale::Float64=1.0)
    K, B = size(u)
    windows = build_window_matrix64(u, window_offsets)
    coeff_flat, jac = manual_chain_output_and_jacobian(model, windows)
    if coefficient_scale != 1.0
        coeff_flat = coefficient_scale .* coeff_flat
        jac = coefficient_scale .* jac
    end
    nfactor = length(factor_offsets)
    nskew = length(skew_offsets)
    factor_coeffs = reshape(@view(coeff_flat[1:nfactor, :]), nfactor, K, B)
    skew_coeffs = reshape(@view(coeff_flat[(nfactor + 1):(nfactor + nskew), :]), nskew, K, B)
    shift_lookup = Dict(shift => idx for (idx, shift) in enumerate(window_offsets))
    center_idx = shift_lookup[0]
    div_factor = zeros(Float64, K, B)
    div_skew = zeros(Float64, K, B)
    @inbounds for b in 1:B, i in 1:K
        sample_offset = (b - 1) * K
        for (factor_idx, factor_offset) in enumerate(factor_offsets)
            factor_col = periodic(i + factor_offset, K)
            for (shift, shift_idx) in shift_lookup
                deriv_site = periodic(i + shift, K)
                a_deriv_col = sym_factor[deriv_site, factor_col]
                for (other_factor_idx, other_offset) in enumerate(factor_offsets)
                    if periodic(deriv_site + other_offset, K) == factor_col
                        a_deriv_col += factor_coeffs[other_factor_idx, deriv_site, b]
                    end
                end
                div_factor[i, b] += jac[factor_idx, shift_idx, i + sample_offset] * a_deriv_col
            end
        end
        for deriv_site in 1:K
            for (factor_idx, factor_offset) in enumerate(factor_offsets)
                col = periodic(deriv_site + factor_offset, K)
                a_i_col = sym_factor[i, col]
                for (other_factor_idx, other_offset) in enumerate(factor_offsets)
                    if periodic(i + other_offset, K) == col
                        a_i_col += factor_coeffs[other_factor_idx, i, b]
                    end
                end
                div_factor[i, b] += a_i_col * jac[factor_idx, center_idx, deriv_site + sample_offset]
            end
        end
        for (offset_idx, offset) in enumerate(skew_offsets)
            if haskey(shift_lookup, offset)
                div_skew[i, b] += jac[nfactor + offset_idx, shift_lookup[offset], i + sample_offset]
            end
            back_site = periodic(i - offset, K)
            div_skew[i, b] -= jac[nfactor + offset_idx, center_idx, back_site + sample_offset]
        end
    end
    return factor_coeffs, skew_coeffs, div_factor, div_skew
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

function mobility_lag_weights(taus::Vector{Float64}, power::Float64)
    power == 0.0 && return ones(Float32, length(taus))
    tau_ref = minimum(taus)
    weights = Float32.((tau_ref ./ taus) .^ power)
    weights ./= Float32(mean(weights))
    return weights
end

function annealed_penalty_weight(base_weight::Float64, final_scale::Float64, epoch::Int, total_epochs::Int)
    total_epochs <= 1 && return base_weight * final_scale
    frac = (epoch - 1) / (total_epochs - 1)
    return base_weight * (1.0 + (final_scale - 1.0) * frac)
end

function symmetric_factor(mat::Matrix{Float64})
    sym = 0.5 .* (mat .+ mat')
    return Matrix(cholesky(Symmetric(sym); check=true).L)
end

function psd_symmetric_mobility(mat::Matrix{Float64})
    sym = 0.5 .* (mat .+ mat')
    eig = eigen(Symmetric(sym))
    values = clamp.(eig.values, 1.0e-10, Inf)
    return eig.vectors * Diagonal(values) * eig.vectors'
end

function support_bounds(states::Array{Float32, 3}, start_idx::Int, pad_fraction::Float64)
    post = @view states[start_idx:end, :, :]
    xmin = Float64(minimum(post))
    xmax = Float64(maximum(post))
    span = max(xmax - xmin, 1.0e-6)
    pad = max(span * pad_fraction, 1.0e-3)
    return xmin - pad, xmax + pad
end

function clamp_states_for_eval!(points::AbstractMatrix{Float64}, xmin::Float64, xmax::Float64)
    outside = 0
    @inbounds for idx in eachindex(points)
        if points[idx] < xmin || points[idx] > xmax
            outside += 1
            points[idx] = clamp(points[idx], xmin, xmax)
        end
    end
    return outside
end

function integrate_phi_only(sampler::PairSampler, score_model::LoadedScoreModel, params::TestParams,
        Phi_u::Matrix{Float64}, mu::Float64, sigma_x::Float64, device::ExecutionDevice)
    total_steps = round(Int, params.total_time / params.dt)
    burnin_steps = round(Int, params.burnin_time / params.dt)
    require_condition(isapprox(total_steps * params.dt, params.total_time; atol=1e-10), "total_time must be an integer multiple of dt.")
    require_condition(isapprox(burnin_steps * params.dt, params.burnin_time; atol=1e-10), "burnin_time must be an integer multiple of dt.")
    K = size(sampler.states, 2)
    state32 = Matrix{Float32}(undef, K, params.ntrajectories)
    sample_state_batch!(state32, sampler, MersenneTwister(params.forward_seed + 17))
    x0 = Float32.((Float64.(state32) .- mu) ./ sigma_x)
    sigma_u = Float32.(symmetric_factor(Phi_u))
    sym_phi = 0.5 .* (Phi_u .+ Phi_u')
    eig_phi = eigvals(Symmetric(sym_phi))
    require_condition(!params.clamp_eval_to_support && !params.hard_clamp_state,
        "The EnsembleIntegrator path expects clamp_eval_to_support=false and hard_clamp_state=false.")
    wrapper = ScoreWrapper(score_model.model, score_model.sigma, K, 1, K)
    traj = EnsembleIntegrator.evolve_sde_snapshots(wrapper, x0, Float32.(Phi_u), sigma_u;
        dt=params.dt,
        n_steps=total_steps,
        burn_in=burnin_steps,
        resolution=params.save_stride,
        device=is_gpu(device) ? "gpu" : "cpu",
        boundary=nothing,
        progress=false,
        progress_desc="L96 Phi-only score Langevin")
    nsaved = size(traj, 2)
    times = collect(1:nsaved) .* (params.dt * params.save_stride)
    states = Array{Float64}(undef, nsaved, K, params.ntrajectories)
    @inbounds for ens in 1:params.ntrajectories, snap in 1:nsaved, i in 1:K
        states[snap, i, ens] = mu + sigma_x * Float64(traj[i, snap, ens])
    end
    stats = Dict{Symbol, Any}(
        :saved_dt => params.dt * params.save_stride,
        :sym_phi_lambda_min => minimum(eig_phi),
        :sym_phi_lambda_max => maximum(eig_phi),
        :eval_clamp_fraction => 0.0,
        :hard_clamp_fraction => 0.0,
    )
    return times, states, stats
end

function integrate_learned_and_phi(sampler::PairSampler, score_model::LoadedScoreModel, params::TestParams,
        mobility_model, Phi_u::Matrix{Float64}, mu::Float64, sigma_x::Float64, device::ExecutionDevice;
        forward_scale::Float64=params.mobility_nn_forward_scale,
        total_time::Float64=params.total_time,
        burnin_time::Float64=params.burnin_time,
        ntrajectories::Int=params.ntrajectories)
    total_steps = round(Int, total_time / params.dt)
    burnin_steps = round(Int, burnin_time / params.dt)
    K = size(sampler.states, 2)
    init32 = Matrix{Float32}(undef, K, ntrajectories)
    sample_state_batch!(init32, sampler, MersenneTwister(params.forward_seed + 17))
    u0 = (Float64.(init32) .- mu) ./ sigma_x
    state_m = copy(u0)
    state_phi = copy(u0)
    noise_base = similar(state_m)
    noise_corr = similar(state_m)
    noise_corr_m = similar(state_m)
    sqrt_phi = symmetric_factor(Phi_u)
    sym_phi = 0.5 .* (Phi_u .+ Phi_u')
    eig_phi = eigvals(Symmetric(sym_phi))
    nsaved = fld(total_steps - burnin_steps, params.save_stride)
    times = collect(1:nsaved) .* (params.dt * params.save_stride)
    states_m = Array{Float64}(undef, nsaved, K, ntrajectories)
    states_phi = similar(states_m)
    rng = MersenneTwister(params.forward_seed + 991)
    save_idx = 0
    sqrt_2dt = sqrt(2.0 * params.dt)
    for step in 1:total_steps
        score_m = Float64.(evaluate_stationary_score_u(score_model, Float32.(state_m), params.score_batch_size, device))
        factor_coeffs, skew_coeffs, div_factor, div_skew = local_mobility_coefficients_and_divergence_u(
            mobility_model, state_m, params.mobility_nn_factor_offsets,
            params.mobility_nn_offsets, params.mobility_nn_window_offsets, sqrt_phi;
            coefficient_scale=forward_scale)
        aT_score = sqrt_phi' * score_m .+
            local_factor_transpose_action(factor_coeffs, score_m, params.mobility_nn_factor_offsets)
        d_signal = sqrt_phi * aT_score .+
            local_factor_action(factor_coeffs, aT_score, params.mobility_nn_factor_offsets) .-
            sym_phi * score_m
        r_signal = local_skew_action(skew_coeffs, score_m, params.mobility_nn_offsets)
        drift_m = Phi_u * score_m .+ d_signal .+ r_signal .+ div_factor .+ div_skew
        score_phi = Float64.(evaluate_stationary_score_u(score_model, Float32.(state_phi), params.score_batch_size, device))
        drift_phi = Phi_u * score_phi
        randn!(rng, noise_base)
        mul!(noise_corr, sqrt_phi, noise_base)
        noise_corr_m .= noise_corr .+ local_factor_action(factor_coeffs, noise_base, params.mobility_nn_factor_offsets)
        state_m .+= params.dt .* drift_m .+ sqrt_2dt .* noise_corr_m
        state_phi .+= params.dt .* drift_phi .+ sqrt_2dt .* noise_corr
        require_condition(all(isfinite, state_m), "Learned-M integration produced non-finite states.")
        require_condition(all(isfinite, state_phi), "Phi integration produced non-finite states.")
        if step > burnin_steps && (step - burnin_steps) % params.save_stride == 0
            save_idx += 1
            @inbounds for ens in 1:ntrajectories, i in 1:K
                states_m[save_idx, i, ens] = mu + sigma_x * state_m[i, ens]
                states_phi[save_idx, i, ens] = mu + sigma_x * state_phi[i, ens]
            end
        end
        if step % 500 == 0
            @printf("Forward learned/Phi integration step %d/%d\n", step, total_steps)
        end
    end
    stats = Dict{Symbol, Any}(
        :saved_dt => params.dt * params.save_stride,
        :sym_phi_lambda_min => minimum(eig_phi),
        :sym_phi_lambda_max => maximum(eig_phi),
        :eval_clamp_fraction => 0.0,
        :hard_clamp_fraction => 0.0,
    )
    return times, states_m, states_phi, stats
end

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function draw_univariate_samples(states::Array{Float64, 3}, start_idx::Int, max_samples::Int, rng::AbstractRNG)
    nt, K, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        mode_idx = (tmp % K) + 1
        traj_idx = (tmp ÷ K) + 1
        values[sample_idx] = states[start_idx + time_local, mode_idx, traj_idx]
    end
    return values
end

function compute_pdf_reference_from_hdf5(path::AbstractString)
    centers = Float64.(h5read(path, "/statistics/pdf/univariate_centers"))
    density = Float64.(h5read(path, "/statistics/pdf/univariate_density"))
    return PdfReference(centers, density, (first(centers), last(centers)))
end

function read_pair_pdf(path::AbstractString, offset::Int)
    base = @sprintf("/statistics/pdf/bivariate/offset_%d", offset)
    x_grid = Float64.(h5read(path, string(base, "/x_grid")))
    y_grid = Float64.(h5read(path, string(base, "/y_grid")))
    density = Float64.(h5read(path, string(base, "/density")))
    return PairPdfResult(offset, x_grid, y_grid, density)
end

function compute_univariate_pdf_on_reference(states::Array{Float64, 3}, start_idx::Int,
        reference::PdfReference, max_samples::Int, seed::Int)
    samples = draw_univariate_samples(states, start_idx, max_samples, MersenneTwister(seed))
    kde_result = kde(samples; npoints=length(reference.centers), boundary=reference.boundary)
    return PdfReference(reference.centers, collect(kde_result.density), reference.boundary)
end

function flatten_state_tensor(states::Array{Float64, 3}, start_idx::Int)
    post = @view states[start_idx:end, :, :]
    return reshape(permutedims(post, (2, 1, 3)), size(states, 2), :)
end

function draw_pair_samples(samples::AbstractMatrix{<:Real}, offset::Int, max_samples::Int, rng::AbstractRNG)
    K, nsnaps = size(samples)
    total = K * nsnaps
    nsamples = max_samples <= 0 ? total : min(max_samples, total)
    x_values = Vector{Float64}(undef, nsamples)
    y_values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = nsamples == total ? sample_idx - 1 : rand(rng, 0:(total - 1))
        mode_idx = (linear % K) + 1
        snap_idx = (linear ÷ K) + 1
        paired_idx = periodic(mode_idx + offset, K)
        x_values[sample_idx] = samples[mode_idx, snap_idx]
        y_values[sample_idx] = samples[paired_idx, snap_idx]
    end
    return x_values, y_values
end

function kde_on_grid_2d(x_values::AbstractVector{<:Real}, y_values::AbstractVector{<:Real},
        x_grid::Vector{Float64}, y_grid::Vector{Float64})
    result = kde((Float64.(x_values), Float64.(y_values));
        npoints=(length(x_grid), length(y_grid)),
        boundary=((x_grid[1], x_grid[end]), (y_grid[1], y_grid[end])))
    return Float64.(result.density)
end

function compute_pair_pdf_on_reference(samples::AbstractMatrix{<:Real}, reference::PairPdfResult,
        max_samples::Int, seed::Int)
    x_values, y_values = draw_pair_samples(samples, reference.offset, max_samples, MersenneTwister(seed + reference.offset))
    density = kde_on_grid_2d(x_values, y_values, reference.x_grid, reference.y_grid)
    return PairPdfResult(reference.offset, reference.x_grid, reference.y_grid, density)
end

function tensor_mean_and_variance(data::Array{Float64, 3})
    total = 0.0
    count = 0
    @inbounds for idx in eachindex(data)
        total += data[idx]
        count += 1
    end
    mean_value = total / count
    sumsq = 0.0
    @inbounds for idx in eachindex(data)
        delta = data[idx] - mean_value
        sumsq += delta * delta
    end
    return mean_value, sumsq / count
end

function estimate_decorrelation_time(acf::Vector{Float64}, lags::Vector{Float64}, threshold::Float64)
    envelope = abs.(acf)
    running_max = copy(envelope)
    for idx in (length(envelope) - 1):-1:1
        running_max[idx] = max(running_max[idx], running_max[idx + 1])
    end
    for idx in 2:length(lags)
        running_max[idx] <= threshold && return lags[idx]
    end
    return lags[end]
end

function compute_lattice_correlations(data::Array{Float64, 3}, saved_dt::Float64,
        max_time::Float64, threshold::Float64, cross_offsets::Vector{Int})
    ntime, K, ntraj = size(data)
    mean_value, variance_value = tensor_mean_and_variance(data)
    require_condition(variance_value > 0.0, "Empirical variance is zero.")
    max_lag = min(ntime - 1, floor(Int, max_time / saved_dt))
    require_condition(max_lag >= 1, "Correlation window is empty.")
    lags = collect(0:max_lag) .* saved_dt
    acf_mean = zeros(Float64, max_lag + 1)
    cross_mean = zeros(Float64, max_lag + 1, length(cross_offsets))
    Threads.@threads for lag in 0:max_lag
        sum_acf = 0.0
        sum_cross = zeros(Float64, length(cross_offsets))
        count = 0
        @inbounds for traj_idx in 1:ntraj, time_idx in 1:(ntime - lag), mode_idx in 1:K
            x0 = data[time_idx, mode_idx, traj_idx] - mean_value
            x_same = data[time_idx + lag, mode_idx, traj_idx] - mean_value
            sum_acf += x_same * x0
            for (offset_idx, offset) in enumerate(cross_offsets)
                paired_idx = periodic(mode_idx + offset, K)
                x_pair = data[time_idx + lag, paired_idx, traj_idx] - mean_value
                sum_cross[offset_idx] += x_pair * x0
            end
            count += 1
        end
        acf_mean[lag + 1] = sum_acf / (count * variance_value)
        for offset_idx in eachindex(cross_offsets)
            cross_mean[lag + 1, offset_idx] = sum_cross[offset_idx] / (count * variance_value)
        end
    end
    return CorrelationResult(lags, acf_mean, copy(cross_offsets), cross_mean,
        estimate_decorrelation_time(acf_mean, lags, threshold), mean_value, variance_value)
end

function sample_anchor_window_matrix(sampler::PairSampler, mu::Float64, sigma_x::Float64,
        window_offsets::Vector{Int}, nstates::Int, seed::Int)
    K = size(sampler.states, 2)
    x = Matrix{Float32}(undef, K, nstates)
    u = similar(x)
    sample_state_batch!(x, sampler, MersenneTwister(seed))
    standardize_batch!(u, x, mu, sigma_x)
    return reshape(build_window_tensor(u, window_offsets), length(window_offsets), :)
end

function mobility_channel_names(coordinate_separations::Vector{Int}, quadratic_offsets::Vector{Int},
        cubic_offsets::Vector{Int}, quartic_offsets::Vector{Int}=Int[], l96_channels::Vector{String}=String[])
    names = ["u[r=$(sep)]" for sep in coordinate_separations]
    append!(names, ["u*u[q=$(offset)]" for offset in quadratic_offsets])
    append!(names, ["u^2*u[q=$(offset)]" for offset in cubic_offsets])
    append!(names, ["u^2*u^2[q=$(offset)]" for offset in quartic_offsets])
    append!(names, l96_channels)
    return names
end

function fill_mobility_observable_channels!(dest::AbstractMatrix{Float32}, u::AbstractMatrix{Float32},
        coordinate_separations::Vector{Int}, quadratic_offsets::Vector{Int}, cubic_offsets::Vector{Int},
        quartic_offsets::Vector{Int}, second_moment_by_offset::Dict{Int, Float64},
        cubic_moment_by_offset::Dict{Int, Float64}, quartic_moment_by_offset::Dict{Int, Float64},
        l96_channels::Vector{String}, l96_moments::Dict{String, Float64})
    K, B = size(u)
    channel = 0
    @inbounds for sep in coordinate_separations
        channel += 1
        for b in 1:B, i in 1:K
            dest[channel, i + (b - 1) * K] = u[periodic(i + sep, K), b]
        end
    end
    @inbounds for offset in quadratic_offsets
        channel += 1
        center = Float32(get(second_moment_by_offset, offset, 0.0))
        for b in 1:B, i in 1:K
            dest[channel, i + (b - 1) * K] =
                u[i, b] * u[periodic(i + offset, K), b] - center
        end
    end
    @inbounds for offset in cubic_offsets
        channel += 1
        center = Float32(get(cubic_moment_by_offset, offset, 0.0))
        for b in 1:B, i in 1:K
            ui = u[i, b]
            dest[channel, i + (b - 1) * K] =
                ui * ui * u[periodic(i + offset, K), b] - center
        end
    end
    @inbounds for offset in quartic_offsets
        channel += 1
        center = Float32(get(quartic_moment_by_offset, offset, 0.0))
        for b in 1:B, i in 1:K
            ui = u[i, b]
            uj = u[periodic(i + offset, K), b]
            dest[channel, i + (b - 1) * K] = ui * ui * uj * uj - center
        end
    end
    @inbounds for name in l96_channels
        channel += 1
        center = Float32(get(l96_moments, name, 0.0))
        for b in 1:B, i in 1:K
            uim2 = u[periodic(i - 2, K), b]
            uim1 = u[periodic(i - 1, K), b]
            uip1 = u[periodic(i + 1, K), b]
            adv = uim1 * (uip1 - uim2)
            dest[channel, i + (b - 1) * K] = name == "adv" ? adv - center : u[i, b] * adv - center
        end
    end
    return dest
end

function fill_mobility_generator_channels!(dest::AbstractMatrix{Float64}, u::AbstractMatrix{Float32},
        drift_raw::AbstractMatrix{Float64}, Q::Matrix{Float64}, sigma_x::Float64,
        coordinate_separations::Vector{Int}, quadratic_offsets::Vector{Int}, cubic_offsets::Vector{Int},
        quartic_offsets::Vector{Int}, l96_channels::Vector{String})
    K, B = size(u)
    inv_sigma = 1.0 / sigma_x
    inv_sigma2 = inv_sigma * inv_sigma
    channel = 0
    @inbounds for sep in coordinate_separations
        channel += 1
        for b in 1:B, i in 1:K
            a = periodic(i + sep, K)
            dest[channel, i + (b - 1) * K] = drift_raw[a, b] * inv_sigma
        end
    end
    @inbounds for offset in quadratic_offsets
        channel += 1
        for b in 1:B, i in 1:K
            a = i
            c = periodic(i + offset, K)
            ua = Float64(u[a, b])
            uc = Float64(u[c, b])
            if offset == 0
                dest[channel, i + (b - 1) * K] =
                    2.0 * drift_raw[a, b] * ua * inv_sigma + 2.0 * Q[a, a] * inv_sigma2
            else
                dest[channel, i + (b - 1) * K] =
                    (drift_raw[a, b] * uc + drift_raw[c, b] * ua) * inv_sigma +
                    2.0 * Q[a, c] * inv_sigma2
            end
        end
    end
    @inbounds for offset in cubic_offsets
        channel += 1
        for b in 1:B, i in 1:K
            a = i
            c = periodic(i + offset, K)
            ua = Float64(u[a, b])
            uc = Float64(u[c, b])
            if offset == 0
                dest[channel, i + (b - 1) * K] =
                    3.0 * drift_raw[a, b] * ua * ua * inv_sigma +
                    6.0 * Q[a, a] * ua * inv_sigma2
            else
                dest[channel, i + (b - 1) * K] =
                    (2.0 * drift_raw[a, b] * ua * uc + drift_raw[c, b] * ua * ua) * inv_sigma +
                    (2.0 * Q[a, a] * uc + 4.0 * Q[a, c] * ua) * inv_sigma2
            end
        end
    end
    @inbounds for offset in quartic_offsets
        channel += 1
        for b in 1:B, i in 1:K
            a = i
            c = periodic(i + offset, K)
            ua = Float64(u[a, b])
            uc = Float64(u[c, b])
            if offset == 0
                dest[channel, i + (b - 1) * K] =
                    4.0 * drift_raw[a, b] * ua^3 * inv_sigma +
                    12.0 * Q[a, a] * ua^2 * inv_sigma2
            else
                dest[channel, i + (b - 1) * K] =
                    (2.0 * drift_raw[a, b] * ua * uc^2 + 2.0 * drift_raw[c, b] * uc * ua^2) * inv_sigma +
                    (2.0 * Q[a, a] * uc^2 + 2.0 * Q[c, c] * ua^2 + 8.0 * Q[a, c] * ua * uc) * inv_sigma2
            end
        end
    end
    @inbounds for name in l96_channels
        channel += 1
        for b in 1:B, i in 1:K
            im2 = periodic(i - 2, K)
            im1 = periodic(i - 1, K)
            ip1 = periodic(i + 1, K)
            u0 = Float64(u[i, b])
            um2 = Float64(u[im2, b])
            um1 = Float64(u[im1, b])
            up1 = Float64(u[ip1, b])
            L_adv =
                (drift_raw[im1, b] * (up1 - um2) +
                 um1 * (drift_raw[ip1, b] - drift_raw[im2, b])) * inv_sigma +
                2.0 * (Q[im1, ip1] - Q[im1, im2]) * inv_sigma2
            if name == "adv"
                dest[channel, i + (b - 1) * K] = L_adv
            else
                L_u_adv =
                    drift_raw[i, b] * um1 * (up1 - um2) * inv_sigma +
                    u0 * L_adv +
                    2.0 * (Q[i, im1] * (up1 - um2) +
                           um1 * (Q[i, ip1] - Q[i, im2])) * inv_sigma2
                dest[channel, i + (b - 1) * K] = L_u_adv
            end
        end
    end
    return dest
end

function estimate_observable_moment_by_offset(sampler::PairSampler, params::TestParams,
        mu::Float64, sigma_x::Float64, offsets::Vector{Int}, order::Int)
    out = Dict{Int, Float64}()
    isempty(offsets) && return out
    require_condition(order == 2 || order == 3 || order == 4, "Only second-, third-, and fourth-order observable centers are implemented.")
    K = size(sampler.states, 2)
    x = Matrix{Float32}(undef, K, params.eval_batch_size)
    u = similar(x)
    sums = Dict(offset => 0.0 for offset in offsets)
    total = 0
    rng = MersenneTwister(params.seed + 51_019)
    remaining = params.stein_samples
    while remaining > 0
        batch_n = min(remaining, params.eval_batch_size)
        sample_state_batch!(@view(x[:, 1:batch_n]), sampler, rng)
        standardize_batch!(@view(u[:, 1:batch_n]), @view(x[:, 1:batch_n]), mu, sigma_x)
        @inbounds for offset in offsets
            acc = 0.0
            for b in 1:batch_n, i in 1:K
                ui = Float64(u[i, b])
                uj = Float64(u[periodic(i + offset, K), b])
                acc += order == 2 ? ui * uj : order == 3 ? ui * ui * uj : ui * ui * uj * uj
            end
            sums[offset] += acc / K
        end
        total += batch_n
        remaining -= batch_n
    end
    for offset in offsets
        out[offset] = sums[offset] / total
    end
    return out
end

estimate_second_moment_by_offset(sampler::PairSampler, params::TestParams,
    mu::Float64, sigma_x::Float64, offsets::Vector{Int}) =
    estimate_observable_moment_by_offset(sampler, params, mu, sigma_x, offsets, 2)

estimate_cubic_moment_by_offset(sampler::PairSampler, params::TestParams,
    mu::Float64, sigma_x::Float64, offsets::Vector{Int}) =
    estimate_observable_moment_by_offset(sampler, params, mu, sigma_x, offsets, 3)

estimate_quartic_moment_by_offset(sampler::PairSampler, params::TestParams,
    mu::Float64, sigma_x::Float64, offsets::Vector{Int}) =
    estimate_observable_moment_by_offset(sampler, params, mu, sigma_x, offsets, 4)

function estimate_l96_channel_moments(sampler::PairSampler, params::TestParams,
        mu::Float64, sigma_x::Float64, channels::Vector{String})
    out = Dict{String, Float64}()
    isempty(channels) && return out
    K = size(sampler.states, 2)
    x = Matrix{Float32}(undef, K, params.eval_batch_size)
    u = similar(x)
    sums = Dict(name => 0.0 for name in channels)
    total = 0
    rng = MersenneTwister(params.seed + 71_019)
    remaining = params.stein_samples
    while remaining > 0
        batch_n = min(remaining, params.eval_batch_size)
        sample_state_batch!(@view(x[:, 1:batch_n]), sampler, rng)
        standardize_batch!(@view(u[:, 1:batch_n]), @view(x[:, 1:batch_n]), mu, sigma_x)
        @inbounds for name in channels
            acc = 0.0
            for b in 1:batch_n, i in 1:K
                adv = Float64(u[periodic(i - 1, K), b]) *
                    (Float64(u[periodic(i + 1, K), b]) - Float64(u[periodic(i - 2, K), b]))
                acc += name == "adv" ? adv : Float64(u[i, b]) * adv
            end
            sums[name] += acc / K
        end
        total += batch_n
        remaining -= batch_n
    end
    for name in channels
        out[name] = sums[name] / total
    end
    return out
end

function build_mobility_training_cache(sampler::PairSampler, score_model::LoadedScoreModel,
        joint_model::LoadedJointScoreModel, params::TestParams, mu::Float64, sigma_x::Float64,
        Phi_u::Matrix{Float64}, second_moment_by_offset::Dict{Int, Float64},
        cubic_moment_by_offset::Dict{Int, Float64}, quartic_moment_by_offset::Dict{Int, Float64},
        l96_moments::Dict{String, Float64}, forcing::Float64, Q::Matrix{Float64},
        device::ExecutionDevice; pair_seed::Int, anchor_seed::Int)
    lag_keep = findall(t -> joint_model.tau_min <= t <= min(joint_model.tau_max, params.max_tau) + 1.0e-10,
        sampler.lag_times)
    require_condition(!isempty(lag_keep), "No mobility-training lags lie inside the joint-score range.")
    K = size(sampler.states, 2)
    B = params.mobility_nn_pairs_per_tau
    T = length(lag_keep)
    x0 = Matrix{Float32}(undef, K, B)
    xt = similar(x0)
    u0 = similar(x0)
    ut_batch = similar(x0)
    drift_t = Matrix{Float64}(undef, K, B)
    windows = Array{Float32}(undef, length(params.mobility_nn_window_offsets), K, B, T)
    scond_u = Array{Float32}(undef, K, B, T)
    channel_names = mobility_channel_names(params.mobility_nn_coordinate_separations,
        params.mobility_nn_quadratic_offsets, params.mobility_nn_cubic_offsets,
        params.mobility_nn_quartic_offsets, params.mobility_nn_l96_channels)
    nchan = length(channel_names)
    observables = Array{Float32}(undef, nchan, K * B, T)
    generator_observables = Matrix{Float64}(undef, nchan, K * B)
    correlations = zeros(Float64, T, nchan)
    data_cdot = zeros(Float64, T, nchan)
    phi_cdot = zeros(Float64, T, nchan)
    rng = MersenneTwister(pair_seed)
    for (local_idx, lag_idx) in enumerate(lag_keep)
        lag = sampler.lag_steps[lag_idx]
        tau = sampler.lag_times[lag_idx]
        sample_pair_batch!(x0, xt, sampler, lag, rng)
        standardize_batch!(u0, x0, mu, sigma_x)
        standardize_batch!(ut_batch, xt, mu, sigma_x)
        l96_drift_batch!(drift_t, xt, forcing)
        windows[:, :, :, local_idx] .= build_window_tensor(u0, params.mobility_nn_window_offsets)
        scond_u[:, :, local_idx] .= evaluate_conditional_score_u(score_model, joint_model, x0, xt, tau,
            params.score_batch_size, params.eval_batch_size, sigma_x, device)
        fill_mobility_observable_channels!(@view(observables[:, :, local_idx]), ut_batch,
            params.mobility_nn_coordinate_separations, params.mobility_nn_quadratic_offsets,
            params.mobility_nn_cubic_offsets, params.mobility_nn_quartic_offsets,
            second_moment_by_offset, cubic_moment_by_offset, quartic_moment_by_offset,
            params.mobility_nn_l96_channels, l96_moments)
        fill_mobility_generator_channels!(generator_observables, ut_batch, drift_t, Q, sigma_x,
            params.mobility_nn_coordinate_separations, params.mobility_nn_quadratic_offsets,
            params.mobility_nn_cubic_offsets, params.mobility_nn_quartic_offsets,
            params.mobility_nn_l96_channels)
        projected_phi = Float32.(transpose(Phi_u) * Float64.(scond_u[:, :, local_idx]))
        @inbounds for chan in 1:nchan
            corr_sum = 0.0
            data_sum = 0.0
            phi_sum = 0.0
            for b in 1:B, i in 1:K
                col = i + (b - 1) * K
                corr_sum += Float64(observables[chan, col, local_idx]) * Float64(u0[i, b])
                data_sum += generator_observables[chan, col] * Float64(u0[i, b])
                phi_sum += Float64(observables[chan, col, local_idx]) * Float64(projected_phi[i, b])
            end
            correlations[local_idx, chan] = corr_sum / (K * B)
            data_cdot[local_idx, chan] = data_sum / (K * B)
            phi_cdot[local_idx, chan] = -phi_sum / (K * B)
        end
        if local_idx == 1 || local_idx == T || local_idx % 10 == 0
            @printf("Built mobility cache %d/%d at tau %.3f\n", local_idx, T, tau)
        end
    end
    return MobilityNNCache(windows, scond_u, observables, correlations, data_cdot, phi_cdot,
        sample_anchor_window_matrix(sampler, mu, sigma_x, params.mobility_nn_window_offsets,
            params.mobility_nn_anchor_states, anchor_seed),
        sampler.lag_times[lag_keep], channel_names)
end

function move_cache_chunk(cache::MobilityNNCache, tau_chunk, device::ExecutionDevice)
    return (
        windows = move_array(cache.windows[:, :, :, tau_chunk], device),
        scond_u = move_array(cache.scond_u[:, :, tau_chunk], device),
        observables = move_array(cache.observables[:, :, tau_chunk], device),
    )
end

function build_current_action_cache(sampler::PairSampler, score_model::LoadedScoreModel,
        params::TestParams, mu::Float64, sigma_x::Float64, Phi_u::Matrix{Float64},
        forcing::Float64, device::ExecutionDevice)
    nsamples = params.mobility_nn_current_action_samples
    nsamples == 0 && return nothing
    K = size(sampler.states, 2)
    x = Matrix{Float32}(undef, K, nsamples)
    u = similar(x)
    drift_raw = Matrix{Float64}(undef, K, nsamples)
    sample_state_batch!(x, sampler, MersenneTwister(params.seed + 98_731))
    standardize_batch!(u, x, mu, sigma_x)
    l96_drift_batch!(drift_raw, x, forcing)
    score_u = evaluate_stationary_score_u(score_model, u, params.score_batch_size, device)
    target = Float32.(drift_raw ./ sigma_x .- Phi_u * Float64.(score_u))
    scale = Float32(max(sqrt(mean(abs2, target)), params.mobility_nn_scale_floor))
    windows = build_window_tensor(u, params.mobility_nn_window_offsets)
    return CurrentActionCache(windows, score_u, target, scale)
end

function move_current_action_batch(cache::CurrentActionCache, idxs::AbstractVector{Int}, device::ExecutionDevice)
    return (
        windows = move_array(reshape(cache.windows[:, :, idxs], size(cache.windows, 1), size(cache.windows, 2), length(idxs), 1), device),
        score_u = move_array(reshape(cache.score_u[:, idxs], size(cache.score_u, 1), length(idxs), 1), device),
        target = move_array(reshape(cache.target[:, idxs], size(cache.target, 1), length(idxs), 1), device),
        scale = cache.scale,
    )
end

function predict_delta_cdot_chunk(model, windows::AbstractArray{Float32, 4},
        scond_u::AbstractArray{Float32, 3}, observables::AbstractArray{Float32, 3},
        factor_offsets::Vector{Int}, skew_offsets::Vector{Int}, sym_factor::AbstractMatrix)
    factor_coeffs, skew_coeffs = evaluate_local_mobility_coeffs(model, windows, length(factor_offsets), length(skew_offsets))
    aT_s = reshape(sym_factor' * reshape(scond_u, size(scond_u, 1), :), size(scond_u)) .+
        local_factor_transpose_action(factor_coeffs, scond_u, factor_offsets)
    d_signal = reshape(sym_factor * reshape(aT_s, size(aT_s, 1), :), size(aT_s)) .+
        local_factor_action(factor_coeffs, aT_s, factor_offsets) .-
        reshape((sym_factor * sym_factor') * reshape(scond_u, size(scond_u, 1), :), size(scond_u))
    r_signal = local_skew_action(skew_coeffs, scond_u, skew_offsets)
    delta_signal = r_signal .- d_signal
    K, B, T = size(delta_signal)
    nchan = size(observables, 1)
    obs = reshape(observables, nchan, K, B, T)
    return reshape(mean(obs .* reshape(delta_signal, 1, K, B, T); dims=(2, 3)), nchan, 1, T)
end

function predict_delta_drift_action_chunk(model, windows::AbstractArray{Float32, 4},
        score_u::AbstractArray{Float32, 3}, factor_offsets::Vector{Int},
        skew_offsets::Vector{Int}, sym_factor::AbstractMatrix)
    factor_coeffs, skew_coeffs = evaluate_local_mobility_coeffs(model, windows, length(factor_offsets), length(skew_offsets))
    aT_s = reshape(sym_factor' * reshape(score_u, size(score_u, 1), :), size(score_u)) .+
        local_factor_transpose_action(factor_coeffs, score_u, factor_offsets)
    d_signal = reshape(sym_factor * reshape(aT_s, size(aT_s, 1), :), size(aT_s)) .+
        local_factor_action(factor_coeffs, aT_s, factor_offsets) .-
        reshape((sym_factor * sym_factor') * reshape(score_u, size(score_u, 1), :), size(score_u))
    r_signal = local_skew_action(skew_coeffs, score_u, skew_offsets)
    return d_signal .+ r_signal
end

function evaluate_mobility_model_on_cache(model, cache::MobilityNNCache, params::TestParams,
        sym_factor::AbstractMatrix)
    pred = zeros(Float64, size(cache.observables, 1), 1, length(cache.taus))
    for start in 1:params.mobility_nn_tau_batch_size:length(cache.taus)
        stop = min(start + params.mobility_nn_tau_batch_size - 1, length(cache.taus))
        chunk = start:stop
        pred[:, :, chunk] .= Float64.(predict_delta_cdot_chunk(model,
            @view(cache.windows[:, :, :, chunk]), @view(cache.scond_u[:, :, chunk]),
            @view(cache.observables[:, :, chunk]), params.mobility_nn_factor_offsets,
            params.mobility_nn_offsets, sym_factor))
    end
    return pred
end

function mobility_target_from_cache(cache::MobilityNNCache)
    return reshape(permutedims(cache.data_cdot .- cache.phi_cdot, (2, 1)),
        size(cache.data_cdot, 2), 1, length(cache.taus))
end

function filter_mobility_cache(cache::MobilityNNCache, keep::Vector{Int})
    return MobilityNNCache(cache.windows, cache.scond_u, cache.observables[keep, :, :],
        cache.correlations[:, keep], cache.data_cdot[:, keep], cache.phi_cdot[:, keep],
        cache.anchor_windows, cache.taus, cache.channel_names[keep])
end

function roughness_ratio(values::AbstractVector{<:Real})
    amp = sqrt(mean(abs2, Float64.(values)))
    length(values) < 3 && return 0.0
    rough = sqrt(mean(abs2, diff(diff(Float64.(values)))))
    return rough / max(amp, eps(Float64))
end

function select_mobility_training_channels(train_cache::MobilityNNCache,
        validation_caches::Vector{MobilityNNCache}, params::TestParams)
    target = train_cache.data_cdot .- train_cache.phi_cdot
    nchan = size(target, 2)
    amplitude = [sqrt(mean(abs2, @view(target[:, idx]))) for idx in 1:nchan]
    data_amplitude = [sqrt(mean(abs2, @view(train_cache.data_cdot[:, idx]))) for idx in 1:nchan]
    roughness = [roughness_ratio(@view(target[:, idx])) for idx in 1:nchan]
    noise = zeros(Float64, nchan)
    if !isempty(validation_caches)
        for vcache in validation_caches
            vtarget = vcache.data_cdot .- vcache.phi_cdot
            @inbounds for idx in 1:nchan
                noise[idx] += sqrt(mean((vtarget[:, idx] .- target[:, idx]) .^ 2))
            end
        end
        noise ./= length(validation_caches)
    end
    noise_ratio = noise ./ max.(amplitude, eps(Float64))
    score = data_amplitude .* amplitude ./ (1.0 .+ noise_ratio .+ roughness)
    amp_floor = params.mobility_nn_selection_min_amplitude_fraction * maximum(amplitude)
    candidates = findall(idx ->
            amplitude[idx] >= amp_floor &&
            noise_ratio[idx] <= params.mobility_nn_selection_max_noise_ratio &&
            roughness[idx] <= params.mobility_nn_selection_max_roughness_ratio,
        1:nchan)
    ranked_all = sortperm(score; rev=true)
    if length(candidates) < params.mobility_nn_selection_min_channels
        candidates = ranked_all[1:min(params.mobility_nn_selection_min_channels, nchan)]
    else
        candidates = sort(candidates; by=idx -> score[idx], rev=true)
    end
    mandatory = Int[]
    for name in params.mobility_nn_selection_mandatory_channels
        idx = findfirst(==(name), train_cache.channel_names)
        idx === nothing || push!(mandatory, idx)
    end
    keep_ranked = unique(vcat(mandatory, candidates))
    if length(keep_ranked) > params.mobility_nn_selection_max_channels
        mandatory_set = Set(mandatory)
        optional = [idx for idx in candidates if !(idx in mandatory_set)]
        keep_ranked = vcat(mandatory, optional[1:max(params.mobility_nn_selection_max_channels - length(mandatory), 0)])
    end
    keep = sort(unique(keep_ranked))
    @printf("Selected %d/%d mobility-observable channels after amplitude/noise screening.\n", length(keep), nchan)
    @printf("%-18s %10s %10s %10s %10s %10s %s\n", "channel", "score", "amp", "data_amp", "noise/amp", "rough", "keep")
    for idx in ranked_all[1:min(nchan, max(params.mobility_nn_selection_max_channels, 18))]
        @printf("%-18s %10.3e %10.3e %10.3e %10.3e %10.3e %s\n",
            train_cache.channel_names[idx], score[idx], amplitude[idx], data_amplitude[idx],
            noise_ratio[idx], roughness[idx], idx in keep ? "*" : "")
    end
    report = Dict{Symbol, Any}(
        :keep => keep,
        :channel_names => train_cache.channel_names,
        :score => score,
        :amplitude => amplitude,
        :data_amplitude => data_amplitude,
        :noise_ratio => noise_ratio,
        :roughness => roughness,
    )
    return keep, report
end

function train_mobility_model(cache::MobilityNNCache, validation_caches::Vector{MobilityNNCache},
        target::Array{Float64, 3}, Phi_u::Matrix{Float64}, params::TestParams, device::ExecutionDevice;
        checkpoint_path::AbstractString="", current_action_cache=nothing)
    sym_phi = 0.5 .* (Phi_u .+ Phi_u')
    sym_factor_host = symmetric_factor(sym_phi .+ params.mobility_nn_psd_jitter .* Matrix{Float64}(I, size(Phi_u, 1), size(Phi_u, 2)))
    sym_factor = move_array(Float32.(sym_factor_host), device)
    model = build_local_mobility_model(size(cache.windows, 1), params.mobility_nn_widths,
        length(params.mobility_nn_factor_offsets), length(params.mobility_nn_offsets))
    model = move_model(model, device)
    opt_state = Flux.setup(Flux.Adam(params.mobility_nn_learning_rate), model)
    anchor = move_array(cache.anchor_windows, device)
    target_scale = Float64.(sqrt.(mean(target .^ 2; dims=3))[:, 1, 1])
    target_scale .= max.(target_scale, params.mobility_nn_scale_floor)
    target_scale_dev = move_array(reshape(Float32.(target_scale), :, 1, 1), device)
    lag_weights = reshape(mobility_lag_weights(cache.taus, params.mobility_nn_lag_weight_power), 1, 1, :)
    history = MobilityNNHistory(Int[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    best_metric = Inf
    best_plateau_metric = Inf
    last_validation_improvement_epoch = 0
    plateau_reached = params.mobility_nn_plateau_patience == 0
    best_model = move_model(model, CPUDevice())
    rng = MersenneTwister(params.seed + 881)
    current_rng = MersenneTwister(params.seed + 882)
    for epoch in 1:params.mobility_nn_epochs
        mean_weight = annealed_penalty_weight(params.mobility_nn_zero_mean_penalty,
            params.mobility_nn_zero_mean_penalty_final_scale, epoch, params.mobility_nn_epochs)
        perm = randperm(rng, length(cache.taus))
        epoch_loss = 0.0
        nbatches = 0
        for start in 1:params.mobility_nn_tau_batch_size:length(perm)
            stop = min(start + params.mobility_nn_tau_batch_size - 1, length(perm))
            chunk = perm[start:stop]
            c = move_cache_chunk(cache, chunk, device)
            target_chunk = move_array(Float32.(target[:, :, chunk]), device)
            chunk_weights = move_array(lag_weights[:, :, chunk], device)
            current_batch = nothing
            if current_action_cache !== nothing && params.mobility_nn_current_action_penalty > 0.0
                current_n = size(current_action_cache.score_u, 2)
                batch_n_current = min(params.mobility_nn_current_action_batch_size, current_n)
                current_idxs = rand(current_rng, 1:current_n, batch_n_current)
                current_batch = move_current_action_batch(current_action_cache, current_idxs, device)
            end
            loss, grads = Flux.withgradient(model) do current_model
                pred = predict_delta_cdot_chunk(current_model, c.windows, c.scond_u, c.observables,
                    params.mobility_nn_factor_offsets, params.mobility_nn_offsets, sym_factor)
                data_loss = mean((((pred .- target_chunk) ./ target_scale_dev) .^ 2) .* chunk_weights)
                current_loss = zero(data_loss)
                if current_batch !== nothing
                    pred_current = predict_delta_drift_action_chunk(current_model, current_batch.windows,
                        current_batch.score_u, params.mobility_nn_factor_offsets, params.mobility_nn_offsets,
                        sym_factor)
                    current_loss = mean(((pred_current .- current_batch.target) ./ current_batch.scale) .^ 2)
                end
                mean_coeff = mean_offset_coefficients(current_model, anchor)
                mean_penalty = sum(abs2, mean_coeff) / length(mean_coeff)
                rms_penalty = anchor_rms_penalty_value(current_model, anchor)
                factor_rms, skew_rms = split_anchor_rms_penalty_values(current_model, anchor,
                    length(params.mobility_nn_factor_offsets), length(params.mobility_nn_offsets))
                reg = Float32(params.mobility_nn_weight_decay) * model_weight_decay(current_model)
                data_loss + Float32(mean_weight) * mean_penalty +
                    Float32(params.mobility_nn_current_action_penalty) * current_loss +
                    Float32(params.mobility_nn_anchor_rms_penalty) * rms_penalty +
                    Float32(params.mobility_nn_factor_rms_penalty) * factor_rms +
                    Float32(params.mobility_nn_skew_rms_penalty) * skew_rms + reg
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float64(loss)
            nbatches += 1
        end
        if epoch == 1 || epoch % params.mobility_nn_eval_every == 0 || epoch == params.mobility_nn_epochs
            host_model = move_model(model, CPUDevice())
            nrmse = 0.0
            prmse = 0.0
            nmse = 0.0
            sym_factor_cpu = Float32.(sym_factor_host)
            for vcache in validation_caches
                vtarget = reshape(permutedims(vcache.data_cdot .- vcache.phi_cdot, (2, 1)),
                    size(vcache.data_cdot, 2), 1, length(vcache.taus))
                pred = evaluate_mobility_model_on_cache(host_model, vcache, params, sym_factor_cpu)
                nrmse += sqrt(mean((((vtarget .- pred) ./ reshape(target_scale, :, 1, 1)) .^ 2)))
                nmse += mean((((vtarget .- pred) ./ reshape(target_scale, :, 1, 1)) .^ 2))
                prmse += sqrt(mean((vtarget .- pred) .^ 2))
            end
            nval = max(length(validation_caches), 1)
            nrmse /= nval
            nmse /= nval
            prmse /= nval
            mean_coeff = Float64.(mean_offset_coefficients(host_model, cache.anchor_windows))
            anchor_rms = Float64(anchor_rms_penalty_value(host_model, cache.anchor_windows))
            factor_rms_host, skew_rms_host = split_anchor_rms_penalty_values(host_model, cache.anchor_windows,
                length(params.mobility_nn_factor_offsets), length(params.mobility_nn_offsets))
            push!(history.epochs, epoch)
            push!(history.train_loss, epoch_loss / max(nbatches, 1))
            push!(history.normalized_rmse, nrmse)
            push!(history.physical_rmse, prmse)
            push!(history.mean_abs_coeff, mean(abs.(mean_coeff)))
            push!(history.anchor_rms, anchor_rms)
            push!(history.weight_l2, Float64(model_weight_decay(host_model)))
            @printf("Mobility NN epoch %4d | loss %.6e | nRMSE %.6e | RMSE %.6e | |<delta>| %.6e\n",
                epoch, history.train_loss[end], nrmse, prmse, history.mean_abs_coeff[end])
            metric = nmse
            if metric < best_metric
                best_metric = metric
                best_model = deepcopy(host_model)
            end
            if !isempty(checkpoint_path)
                ensure_parent_dir(checkpoint_path)
                BSON.@save checkpoint_path epoch host_model best_model history target_scale best_metric best_plateau_metric last_validation_improvement_epoch
            end
            if isinf(best_plateau_metric)
                best_plateau_metric = metric
                last_validation_improvement_epoch = epoch
            else
                improvement_threshold = params.mobility_nn_plateau_rtol * max(abs(best_plateau_metric), 1.0)
                if metric < best_plateau_metric - improvement_threshold
                    best_plateau_metric = metric
                    last_validation_improvement_epoch = epoch
                elseif !plateau_reached && epoch >= params.mobility_nn_min_epochs &&
                        epoch - last_validation_improvement_epoch >= params.mobility_nn_plateau_patience
                    plateau_reached = true
                    @printf("Mobility NN validation plateau reached at epoch %d: best metric %.6e, no relative improvement > %.3e for %d epochs.\n",
                        epoch, best_plateau_metric, params.mobility_nn_plateau_rtol, params.mobility_nn_plateau_patience)
                    break
                end
            end
        end
    end
    require_condition(plateau_reached,
        @sprintf("Mobility NN did not reach a validation plateau within %d epochs; increase mobility_nn.epochs or relax plateau_rtol/patience.",
            params.mobility_nn_epochs))
    return best_model, history, target_scale
end

function load_observed_correlation(path::AbstractString, max_time::Float64, cross_offsets::Vector{Int})
    lags = Float64.(h5read(path, "/statistics/correlations/lags"))
    acf = Float64.(h5read(path, "/statistics/correlations/acf_mean"))
    offsets_all = Int.(h5read(path, "/statistics/correlations/cross_offsets"))
    cross_all = Float64.(h5read(path, "/statistics/correlations/cross_mean"))
    keep_lag = findall(<=(max_time), lags)
    idxs = [findfirst(==(offset), offsets_all) for offset in cross_offsets]
    require_condition(all(!isnothing, idxs), "Requested cross_offsets are not all present in the observed HDF5 diagnostics.")
    cross = cross_all[keep_lag, Int.(idxs)]
    return CorrelationResult(lags[keep_lag], acf[keep_lag], copy(cross_offsets), cross,
        Float64(h5read(path, "/statistics/correlations/t_decorrelation")),
        Float64(h5read(path, "/statistics/correlations/global_mean")),
        Float64(h5read(path, "/statistics/correlations/global_variance")))
end

rmse(a, b) = sqrt(mean((Float64.(a) .- Float64.(b)) .^ 2))

function kl_divergence_1d(ref::Vector{Float64}, pred::Vector{Float64}, width::Float64)
    eps_val = 1.0e-12
    p = ref .* width .+ eps_val
    q = pred .* width .+ eps_val
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function grid_spacing(grid::Vector{Float64})
    return length(grid) > 1 ? grid[2] - grid[1] : 1.0
end

function kl_divergence_2d(ref::Matrix{Float64}, pred::Matrix{Float64}, x_width::Float64, y_width::Float64)
    eps_val = 1.0e-12
    p = vec(ref .* (x_width * y_width)) .+ eps_val
    q = vec(pred .* (x_width * y_width)) .+ eps_val
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function compute_pdf_diagnostics(input_hdf5::AbstractString, pred_states::Array{Float64, 3}, params::TestParams)
    observed_pdf = compute_pdf_reference_from_hdf5(input_hdf5)
    pred_pdf = compute_univariate_pdf_on_reference(pred_states, 1, observed_pdf,
        params.pdf_max_samples, params.forward_seed + 401)
    pred_matrix = flatten_state_tensor(pred_states, 1)
    pair_observed = [read_pair_pdf(input_hdf5, offset) for offset in params.cross_offsets]
    pair_generated = [compute_pair_pdf_on_reference(pred_matrix, pair_ref,
        params.pdf_max_samples, params.forward_seed + 607) for pair_ref in pair_observed]
    uni_kl = kl_divergence_1d(observed_pdf.density, pred_pdf.density, grid_spacing(observed_pdf.centers))
    pair_kls = [
        kl_divergence_2d(pair_ref.density, pair_gen.density,
            grid_spacing(pair_ref.x_grid), grid_spacing(pair_ref.y_grid))
        for (pair_ref, pair_gen) in zip(pair_observed, pair_generated)
    ]
    return PdfDiagnostics(observed_pdf, pred_pdf, pair_observed, pair_generated,
        uni_kl, pair_kls, mean([uni_kl; pair_kls]))
end

function log_density(density::AbstractMatrix{<:Real})
    positive = Float64.(density)
    floor_value = max(maximum(positive) * 1.0e-8, 1.0e-12)
    return log10.(max.(positive, floor_value))
end

function robust_log_colorrange(log_values::AbstractVector{<:Real})
    finite_logs = filter(isfinite, Float64.(log_values))
    require_condition(!isempty(finite_logs), "Cannot determine color range from non-finite values.")
    lo = quantile(finite_logs, 0.01)
    hi = quantile(finite_logs, 0.995)
    hi <= lo && (hi = lo + 1.0)
    return (lo, hi)
end

function robust_pair_colorrange(pair::PairPdfResult)
    logs = vec(log_density(pair.density))
    finite_logs = filter(isfinite, logs)
    require_condition(!isempty(finite_logs), "Cannot determine pair-PDF color range from non-finite densities.")
    lo = quantile(finite_logs, 0.01)
    hi = quantile(finite_logs, 0.995)
    hi <= lo && (hi = lo + 1.0)
    return (lo, hi)
end

function render_pair_pdf_panel!(slot, pair::PairPdfResult; title::AbstractString, colorrange=nothing)
    layout = GridLayout(slot)
    ax = Axis(layout[1, 1]; title=title, xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pair.offset),
        aspect=DataAspect())
    hm = heatmap!(ax, pair.x_grid, pair.y_grid, log_density(pair.density);
        colormap=STYLE_SEQUENTIAL_BLUE, colorrange=colorrange)
    Colorbar(layout[1, 2], hm; label="log10 density", width=18)
    colgap!(layout, 8)
    return nothing
end

function render_forward_stats(path::AbstractString, params::TestParams, pdf_diag::PdfDiagnostics,
        observed_corr::CorrelationResult, pred_corr::CorrelationResult,
        taus::Vector{Float64}, C::Matrix{Float64}, Cdot::Matrix{Float64}, phi_profile_u::Vector{Float64},
        mobility_profile_u::Vector{Float64}, stats::Dict{Symbol, Any}, pdf_rmse_value::Float64,
        acf_rmse_value::Float64, cross_rmse_value::Float64)
    with_scaled_figure_style(params.figure_width, params.figure_height; scale_override=1.08) do _
        fig = Figure(size=(params.figure_width, params.figure_height);
            figure_padding=(36, 36, 72, 28))
        subtitle = @sprintf("full Stein-corrected Phi | antisymmetric scale %.2f | eig sym %.2e..%.2e",
            params.antisymmetric_scale, stats[:sym_phi_lambda_min], stats[:sym_phi_lambda_max])
        figure_title!(fig, "L96 Phi-Only Forward Validation"; subtitle=subtitle,
            fontsize=30, subtitle_fontsize=18)
        rowgap!(fig.layout, 24)
        colgap!(fig.layout, 30)

        ax_pdf = Axis(fig[1, 1]; title=@sprintf("Univariate PDF  KL %.2e", pdf_diag.univariate_kl), xlabel="x_i", ylabel="density")
        lines!(ax_pdf, pdf_diag.univariate_observed.centers, pdf_diag.univariate_observed.density; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
        lines!(ax_pdf, pdf_diag.univariate_generated.centers, pdf_diag.univariate_generated.density; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(), label="Phi only")
        axislegend(ax_pdf; position=:rt, framevisible=true, labelsize=15)

        ax_acf = Axis(fig[1, 2]; title=@sprintf("Autocorrelation  RMSE %.2e", acf_rmse_value), xlabel="lag tau", ylabel="C_0(tau)")
        hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linestyle=:dash)
        lines!(ax_acf, observed_corr.lags, observed_corr.acf_mean; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
        lines!(ax_acf, pred_corr.lags, pred_corr.acf_mean; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(), label="Phi only")
        axislegend(ax_acf; position=:rt, framevisible=true, labelsize=15)

        ax_cross = Axis(fig[1, 3]; title=@sprintf("Shifted Cross-Correlations  RMSE %.2e", cross_rmse_value), xlabel="lag tau", ylabel="C_r(tau)")
        colors = Makie.wong_colors()
        for (idx, offset) in enumerate(observed_corr.cross_offsets)
            color = colors[mod1(idx, length(colors))]
            lines!(ax_cross, observed_corr.lags, observed_corr.cross_mean[:, idx]; color=color, linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("obs r=%d", offset))
            lines!(ax_cross, pred_corr.lags, pred_corr.cross_mean[:, idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("Phi r=%d", offset))
        end
        axislegend(ax_cross; position=:rt, nbanks=2, framevisible=true, labelsize=14)

        ax_c = Axis(fig[2, 1]; title="Observed Coordinate Correlations", xlabel="tau", ylabel="C_r(tau)")
        for sep in (0, 1, 2, 5, 10)
            sep + 1 <= size(C, 2) || continue
            lines!(ax_c, taus, C[:, sep + 1]; linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("r=%d", sep))
        end
        axislegend(ax_c; position=:rt, framevisible=true, labelsize=14)

        ax_cdot = Axis(fig[2, 2]; title="Data-Increment Coordinate Cdot", xlabel="tau", ylabel="Cdot_r(tau)")
        for sep in (0, 1, 2, 5, 10)
            sep + 1 <= size(Cdot, 2) || continue
            lines!(ax_cdot, taus, Cdot[:, sep + 1]; linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("r=%d", sep))
        end
        axislegend(ax_cdot; position=:rt, framevisible=true, labelsize=14)

        ax_phi = Axis(fig[2, 3]; title="Phi Profile In Score Coordinates", xlabel="separation r", ylabel="Phi_r")
        seps = collect(0:(length(phi_profile_u) - 1))
        lines!(ax_phi, seps, phi_profile_u; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label="full -Cdot(0)")
        lines!(ax_phi, seps, mobility_profile_u; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label="Stein-corrected Phi used")
        axislegend(ax_phi; position=:rt, framevisible=true, labelsize=14)

        if !isempty(pdf_diag.pair_observed)
            observed_pair_colorrange = robust_pair_colorrange(pdf_diag.pair_observed[1])
            generated_pair_colorrange = robust_pair_colorrange(pdf_diag.pair_generated[1])
            render_pair_pdf_panel!(fig[3, 1], pdf_diag.pair_observed[1];
                title=@sprintf("Observed Pair PDF r=%d", pdf_diag.pair_observed[1].offset),
                colorrange=observed_pair_colorrange)
            render_pair_pdf_panel!(fig[3, 2], pdf_diag.pair_generated[1];
                title=@sprintf("Phi Pair PDF r=%d  KL %.2e", pdf_diag.pair_generated[1].offset, pdf_diag.pair_kls[1]),
                colorrange=generated_pair_colorrange)
            ax_kl = Axis(fig[3, 3]; title="Bivariate PDF Relative Entropy", xlabel="offset r", ylabel="KL(observed || Phi)")
            barplot!(ax_kl, 1:length(pdf_diag.pair_kls), pdf_diag.pair_kls; color=STYLE_SECONDARY, gap=0.25)
            ax_kl.xticks = (1:length(pdf_diag.pair_kls), string.(getfield.(pdf_diag.pair_observed, :offset)))
            ylims!(ax_kl, 0, 1.15 * maximum(pdf_diag.pair_kls))
        end

        for c in 1:3
            colsize!(fig.layout, c, Relative(1 / 3))
        end
        rowsize!(fig.layout, 1, Relative(0.34))
        rowsize!(fig.layout, 2, Relative(0.31))
        rowsize!(fig.layout, 3, Relative(0.35))

        save_figure(path, fig)
    end
    return nothing
end

function render_forward_comparison_stats(path::AbstractString, params::TestParams,
        pdf_m::PdfDiagnostics, pdf_phi::PdfDiagnostics,
        observed_corr::CorrelationResult, corr_m::CorrelationResult, corr_phi::CorrelationResult,
        training_taus::Vector{Float64}, target_delta::Array{Float64, 3}, pred_delta::Array{Float64, 3},
        stats::Dict{Symbol, Any})
    with_scaled_figure_style(params.figure_width, params.figure_height; scale_override=1.08) do _
        fig = Figure(size=(params.figure_width, params.figure_height); figure_padding=(36, 36, 72, 28))
        figure_title!(fig, "L96 Learned Mobility Forward Validation";
            subtitle=@sprintf("full Phi + %.2f zero-mean local full delta M (PSD symmetric factor + antisymmetric part) | eig sym %.2e..%.2e",
                params.mobility_nn_forward_scale, stats[:sym_phi_lambda_min], stats[:sym_phi_lambda_max]),
            fontsize=30, subtitle_fontsize=18)
        rowgap!(fig.layout, 24)
        colgap!(fig.layout, 30)

        ax_pdf = Axis(fig[1, 1]; title=@sprintf("Univariate PDF  KL M %.2e  Phi %.2e",
            pdf_m.univariate_kl, pdf_phi.univariate_kl), xlabel="x_i", ylabel="density")
        lines!(ax_pdf, pdf_m.univariate_observed.centers, pdf_m.univariate_observed.density;
            color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
        lines!(ax_pdf, pdf_m.univariate_generated.centers, pdf_m.univariate_generated.density;
            color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label="learned M")
        lines!(ax_pdf, pdf_phi.univariate_generated.centers, pdf_phi.univariate_generated.density;
            color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label="Phi only")
        axislegend(ax_pdf; position=:rt, framevisible=true, labelsize=14)

        shared_lag = min(length(observed_corr.lags), length(corr_m.lags), length(corr_phi.lags))
        acf_m = rmse(observed_corr.acf_mean[1:shared_lag], corr_m.acf_mean[1:shared_lag])
        acf_phi = rmse(observed_corr.acf_mean[1:shared_lag], corr_phi.acf_mean[1:shared_lag])
        ax_acf = Axis(fig[1, 2]; title=@sprintf("Autocorrelation  RMSE M %.2e  Phi %.2e", acf_m, acf_phi),
            xlabel="lag tau", ylabel="C_0(tau)")
        hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linestyle=:dash)
        lines!(ax_acf, observed_corr.lags, observed_corr.acf_mean; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
        lines!(ax_acf, corr_m.lags, corr_m.acf_mean; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label="learned M")
        lines!(ax_acf, corr_phi.lags, corr_phi.acf_mean; color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label="Phi only")
        axislegend(ax_acf; position=:rt, framevisible=true, labelsize=14)

        cross_m = rmse(observed_corr.cross_mean[1:shared_lag, :], corr_m.cross_mean[1:shared_lag, :])
        cross_phi = rmse(observed_corr.cross_mean[1:shared_lag, :], corr_phi.cross_mean[1:shared_lag, :])
        ax_cross = Axis(fig[1, 3]; title=@sprintf("Cross-Correlations  RMSE M %.2e  Phi %.2e", cross_m, cross_phi),
            xlabel="lag tau", ylabel="C_r(tau)")
        colors = Makie.wong_colors()
        for (idx, offset) in enumerate(observed_corr.cross_offsets)
            color = colors[mod1(idx, length(colors))]
            lines!(ax_cross, observed_corr.lags, observed_corr.cross_mean[:, idx]; color=color, linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("obs r=%d", offset))
            lines!(ax_cross, corr_m.lags, corr_m.cross_mean[:, idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("M r=%d", offset))
            lines!(ax_cross, corr_phi.lags, corr_phi.cross_mean[:, idx]; color=color, linestyle=:dot, linewidth=curve_linewidth(emphasis=0.75), label=@sprintf("Phi r=%d", offset))
        end
        axislegend(ax_cross; position=:rt, nbanks=2, framevisible=true, labelsize=12)

        ax_fit = Axis(fig[2, 1:2]; title="Coordinate Residual Fit", xlabel="tau", ylabel="Cdot residual")
        sep_count = min(size(target_delta, 1), 6)
        for idx in 1:sep_count
            color = colors[mod1(idx, length(colors))]
            lines!(ax_fit, training_taus, vec(target_delta[idx, 1, :]); color=color, linewidth=curve_linewidth(), label=@sprintf("target r=%d", params.mobility_nn_coordinate_separations[idx]))
            lines!(ax_fit, training_taus, vec(pred_delta[idx, 1, :]); color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.85), label=@sprintf("NN r=%d", params.mobility_nn_coordinate_separations[idx]))
        end
        axislegend(ax_fit; position=:rt, nbanks=2, framevisible=true, labelsize=12)

        ax_bars = Axis(fig[2, 3]; title="Bivariate PDF Relative Entropy", xlabel="offset r", ylabel="KL")
        xs = collect(1:length(pdf_m.pair_kls))
        barplot!(ax_bars, xs .- 0.18, pdf_m.pair_kls; color=STYLE_PRIMARY, width=0.34, label="learned M")
        barplot!(ax_bars, xs .+ 0.18, pdf_phi.pair_kls; color=STYLE_SECONDARY, width=0.34, label="Phi only")
        ax_bars.xticks = (xs, string.(getfield.(pdf_m.pair_observed, :offset)))
        axislegend(ax_bars; position=:rt, framevisible=true, labelsize=13)

        if !isempty(pdf_m.pair_observed)
            obs_range = robust_pair_colorrange(pdf_m.pair_observed[1])
            m_range = robust_pair_colorrange(pdf_m.pair_generated[1])
            phi_range = robust_pair_colorrange(pdf_phi.pair_generated[1])
            render_pair_pdf_panel!(fig[3, 1], pdf_m.pair_observed[1]; title=@sprintf("Observed Pair PDF r=%d", pdf_m.pair_observed[1].offset), colorrange=obs_range)
            render_pair_pdf_panel!(fig[3, 2], pdf_m.pair_generated[1]; title=@sprintf("Learned-M Pair PDF  KL %.2e", pdf_m.pair_kls[1]), colorrange=m_range)
            render_pair_pdf_panel!(fig[3, 3], pdf_phi.pair_generated[1]; title=@sprintf("Phi Pair PDF  KL %.2e", pdf_phi.pair_kls[1]), colorrange=phi_range)
        end

        for c in 1:3
            colsize!(fig.layout, c, Relative(1 / 3))
        end
        rowsize!(fig.layout, 1, Relative(0.34))
        rowsize!(fig.layout, 2, Relative(0.28))
        rowsize!(fig.layout, 3, Relative(0.38))
        save_figure(path, fig)
    end
    return nothing
end

function render_training_diagnostics(path::AbstractString, params::TestParams, history::MobilityNNHistory,
        taus::Vector{Float64}, target_delta::Array{Float64, 3}, pred_delta::Array{Float64, 3},
        target_scale::Vector{Float64}, channel_labels::Vector{String})
    with_scaled_figure_style(params.figure_width, params.figure_height; scale_override=1.08) do _
        fig = Figure(size=(params.figure_width, params.figure_height); figure_padding=(36, 36, 72, 28))
        figure_title!(fig, "L96 Mobility NN Training Diagnostics"; fontsize=30)
        rowgap!(fig.layout, 24)
        colgap!(fig.layout, 30)

        ax_loss = Axis(fig[1, 1]; title=L"\mathrm{Training\;objective}", xlabel=L"\mathrm{epoch}", ylabel=L"\mathcal{L}")
        lines!(ax_loss, history.epochs, history.train_loss; color=STYLE_REFERENCE, linewidth=curve_linewidth())
        ax_rmse = Axis(fig[1, 2]; title=L"\mathrm{Validation\;RMSE}", xlabel=L"\mathrm{epoch}", ylabel=L"\mathrm{RMSE}")
        lines!(ax_rmse, history.epochs, history.normalized_rmse; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label=L"\mathrm{normalized}")
        lines!(ax_rmse, history.epochs, history.physical_rmse; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label=L"\mathrm{physical}")
        axislegend(ax_rmse; position=:rt)
        ax_mean = Axis(fig[1, 3]; title=L"\mathrm{Mean\;and\;RMS\;\delta M}", xlabel=L"\mathrm{epoch}", ylabel=L"\mathrm{value}")
        lines!(ax_mean, history.epochs, history.mean_abs_coeff; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label=L"|\langle \delta M\rangle|")
        lines!(ax_mean, history.epochs, history.anchor_rms; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label=L"\mathrm{anchor\;RMS}")
        axislegend(ax_mean; position=:rt)

        channel_rmse = [rmse(vec(@view(target_delta[i, 1, :])), vec(@view(pred_delta[i, 1, :]))) for i in 1:size(target_delta, 1)]
        channel_nrmse = channel_rmse ./ target_scale
        ax_bar = Axis(fig[2, 1]; title=L"\mathrm{Final\;channel\;nRMSE}", xlabel=L"\phi_m",
            ylabel=L"\mathrm{nRMSE}", xticks=(1:length(channel_nrmse), latex_channel_name.(channel_labels)))
        barplot!(ax_bar, 1:length(channel_nrmse), channel_nrmse; color=STYLE_PRIMARY, gap=0.18)
        ax_bar.xticklabelrotation = pi / 5

        ax_target = Axis(fig[2, 2:3]; title=L"\mathrm{Residual\;targets\;and\;NN\;prediction}",
            xlabel=L"\tau", ylabel=L"\dot{C}_{mn}^{\mathrm{res}}(\tau)")
        colors = Makie.wong_colors()
        for idx in 1:min(size(target_delta, 1), 8)
            color = colors[mod1(idx, length(colors))]
            label = channel_labels[idx]
            lines!(ax_target, taus, vec(target_delta[idx, 1, :]); color=color, linewidth=curve_linewidth(), label=latex_text("target " * label))
            lines!(ax_target, taus, vec(pred_delta[idx, 1, :]); color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.85), label=latex_text("NN " * label))
        end
        axislegend(ax_target; position=:rt, nbanks=2, labelsize=12)

        summary = [
            @sprintf("best validation nRMSE = %.3e", minimum(history.normalized_rmse)),
            @sprintf("final validation nRMSE = %.3e", history.normalized_rmse[end]),
            @sprintf("final physical RMSE = %.3e", history.physical_rmse[end]),
            @sprintf("final |<delta>| = %.3e", history.mean_abs_coeff[end]),
            @sprintf("pairs/tau = %d", params.mobility_nn_pairs_per_tau),
            @sprintf("epochs = %d", params.mobility_nn_epochs),
            "offsets = [" * join(string.(params.mobility_nn_offsets), ", ") * "]",
            "window = [" * join(string.(params.mobility_nn_window_offsets), ", ") * "]",
            "widths = [" * join(string.(params.mobility_nn_widths), ", ") * "]",
        ]
        text_panel!(fig[3, 1:3], summary; title="Summary")
        rowsize!(fig.layout, 1, Relative(0.32))
        rowsize!(fig.layout, 2, Relative(0.40))
        rowsize!(fig.layout, 3, Relative(0.28))
        save_figure(path, fig)
    end
    return nothing
end

function channel_panel_title(label::AbstractString, idx::Int)
    return @sprintf("%02d  %s", idx, String(label))
end

function render_A_fit_figure(path::AbstractString, params::TestParams, taus::Vector{Float64},
        channel_names::Vector{String}, target_delta::Array{Float64, 3}, pred_delta::Array{Float64, 3})
    nchan = size(target_delta, 1)
    nrows, ncols = panel_grid_dims(nchan; max_cols=4)
    fw, fh = publication_panel_figure_size(nrows, ncols;
        base_w=params.figure_width, base_h=params.figure_height,
        panel_w=1080, panel_h=620, min_w=2400, min_h=1500, max_w=5600, max_h=5200)
    with_scaled_figure_style(fw, fh) do _
        fig = Figure(; size=(fw, fh))
        figure_title!(fig, "L96 mobility fit: A_mn(t)";
            subtitle="data residual  A = Cdot_data - Cdot_Phi  vs  conditional-score estimate from trained full M(x)")
        for idx in 1:nchan
            r, c = centered_panel_rc(idx, nchan, ncols)
            y_data = vec(target_delta[idx, 1, :])
            y_nn = vec(pred_delta[idx, 1, :])
            channel_rmse = rmse(y_data, y_nn)
            denom = max(sqrt(mean(abs2, y_data)), eps(Float64))
            ax = Axis(fig[r, c];
                xlabel=L"\tau",
                ylabel=L"A_{mn}(\tau)",
                title=latex_text(@sprintf("%s    RMSE %.3e  nRMSE %.3e", channel_panel_title(channel_names[idx], idx),
                    channel_rmse, channel_rmse / denom)))
            hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            lines!(ax, taus, y_data; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label=L"\mathrm{data}")
            lines!(ax, taus, y_nn; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label=L"\mathrm{NN}\;M")
            idx == 1 && axislegend(ax; position=:rt)
        end
        apply_publication_grid!(fig.layout, nrows, ncols; row_gap=34, col_gap=34)
        save_figure(path, fig)
    end
    return nothing
end

function contour_levels_for_pair(obs::PairPdfResult, pred::PairPdfResult; nlevels::Int=7)
    logs = vcat(vec(log_density(obs.density)), vec(log_density(pred.density)))
    finite_logs = sort(filter(isfinite, logs))
    require_condition(!isempty(finite_logs), "Cannot choose contour levels from empty pair PDF.")
    lo = quantile(finite_logs, 0.70)
    hi = quantile(finite_logs, 0.995)
    hi <= lo && (hi = lo + 1.0)
    return collect(range(lo, hi; length=nlevels))
end

function render_pair_contour_overlay!(parent, obs::PairPdfResult, pred::PairPdfResult, kl_value::Float64)
    ax = Axis(parent;
        xlabel=L"x_i",
        ylabel=latexstring("x_{i+" * string(obs.offset) * "}"),
        title=latexstring("\\mathrm{j=" * string(obs.offset) * ",\\;KL=" * @sprintf("%.3e", kl_value) * "}"),
        aspect=DataAspect())
    levels = contour_levels_for_pair(obs, pred)
    contour!(ax, obs.x_grid, obs.y_grid, log_density(obs.density);
        levels=levels, color=STYLE_REFERENCE, linewidth=curve_linewidth(emphasis=0.85), label=L"\mathrm{observed}")
    contour!(ax, pred.x_grid, pred.y_grid, log_density(pred.density);
        levels=levels, color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.85), label=L"\mathrm{NN}\;M")
    return ax
end

function render_forward_pdf_acf_figure(path::AbstractString, params::TestParams,
        pdf_m::PdfDiagnostics, observed_corr::CorrelationResult,
        corr_m::CorrelationResult, corr_phi::CorrelationResult)
    npairs = length(pdf_m.pair_observed)
    ncols = max(2, min(4, max(npairs, 1)))
    fw, fh = publication_panel_figure_size(2, ncols;
        base_w=params.figure_width, base_h=params.figure_height,
        panel_w=1000, panel_h=720, min_w=2600, min_h=1700, max_w=5600, max_h=3200)
    with_scaled_figure_style(fw, fh) do _
        fig = Figure(; size=(fw, fh))
        figure_title!(fig, "L96 forward validation from learned M(x)";
            subtitle="PDFs compare observations with learned-M Langevin; ACF panel also includes Phi baseline")

        ax_pdf = Axis(fig[1, 1:cld(ncols, 2)];
            xlabel=L"x_i", ylabel=L"\rho(x_i)",
            title=latexstring("\\mathrm{univariate\\;PDF,\\;KL=" * @sprintf("%.3e", pdf_m.univariate_kl) * "}"))
        lines!(ax_pdf, pdf_m.univariate_observed.centers, pdf_m.univariate_observed.density;
            color=STYLE_REFERENCE, linewidth=curve_linewidth(), label=L"\mathrm{observed}")
        lines!(ax_pdf, pdf_m.univariate_generated.centers, pdf_m.univariate_generated.density;
            color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label=L"\mathrm{NN}\;M")
        axislegend(ax_pdf; position=:rt)

        shared_m = min(length(observed_corr.lags), length(corr_m.lags))
        shared_phi = min(length(observed_corr.lags), length(corr_phi.lags))
        acf_m = rmse(observed_corr.acf_mean[1:shared_m], corr_m.acf_mean[1:shared_m])
        acf_phi = rmse(observed_corr.acf_mean[1:shared_phi], corr_phi.acf_mean[1:shared_phi])
        ax_acf = Axis(fig[1, (cld(ncols, 2) + 1):ncols];
            xlabel=L"\tau", ylabel=L"C_0(\tau)",
            title=latexstring("\\mathrm{average\\;ACF,\\;RMSE}_M=" * @sprintf("%.3e", acf_m) *
                "\\;\\mathrm{RMSE}_{\\Phi}=" * @sprintf("%.3e", acf_phi)))
        hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
        lines!(ax_acf, observed_corr.lags, observed_corr.acf_mean;
            color=STYLE_REFERENCE, linewidth=curve_linewidth(), label=L"\mathrm{observed}")
        lines!(ax_acf, corr_m.lags, corr_m.acf_mean;
            color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label=L"\mathrm{NN}\;M")
        lines!(ax_acf, corr_phi.lags, corr_phi.acf_mean;
            color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label=L"\Phi")
        axislegend(ax_acf; position=:rt)

        for idx in 1:npairs
            ax = render_pair_contour_overlay!(fig[2, idx], pdf_m.pair_observed[idx],
                pdf_m.pair_generated[idx], pdf_m.pair_kls[idx])
            idx == 1 && axislegend(ax; position=:rt)
        end
        apply_publication_grid!(fig.layout, 2, ncols; row_weights=[0.95, 1.15], row_gap=34, col_gap=32)
        save_figure(path, fig)
    end
    return nothing
end

function compute_training_observable_correlations(states::Array{Float64, 3}, saved_dt::Float64,
        taus::Vector{Float64}, coordinate_separations::Vector{Int}, quadratic_offsets::Vector{Int},
        cubic_offsets::Vector{Int}, quartic_offsets::Vector{Int},
        second_moment_by_offset::Dict{Int, Float64}, cubic_moment_by_offset::Dict{Int, Float64},
        quartic_moment_by_offset::Dict{Int, Float64}, l96_channels::Vector{String},
        l96_moments::Dict{String, Float64}, mu::Float64, sigma_x::Float64)
    ntime, K, ntraj = size(states)
    channel_names = mobility_channel_names(coordinate_separations, quadratic_offsets, cubic_offsets,
        quartic_offsets, l96_channels)
    out = zeros(Float64, length(taus), length(channel_names))
    lag_steps = round.(Int, taus ./ saved_dt)
    @inbounds for (tau_idx, lag) in enumerate(lag_steps)
        require_condition(0 <= lag < ntime, @sprintf("Training-correlation lag %.6f is outside rollout length.", taus[tau_idx]))
        accum = zeros(Float64, length(channel_names))
        count = 0
        for traj in 1:ntraj, time_idx in 1:(ntime - lag), i in 1:K
            u0 = (states[time_idx, i, traj] - mu) / sigma_x
            chan = 1
            for sep in coordinate_separations
                ut = (states[time_idx + lag, periodic(i + sep, K), traj] - mu) / sigma_x
                accum[chan] += ut * u0
                chan += 1
            end
            for offset in quadratic_offsets
                ui = (states[time_idx + lag, i, traj] - mu) / sigma_x
                uj = (states[time_idx + lag, periodic(i + offset, K), traj] - mu) / sigma_x
                accum[chan] += (ui * uj - second_moment_by_offset[offset]) * u0
                chan += 1
            end
            for offset in cubic_offsets
                ui = (states[time_idx + lag, i, traj] - mu) / sigma_x
                uj = (states[time_idx + lag, periodic(i + offset, K), traj] - mu) / sigma_x
                accum[chan] += (ui * ui * uj - cubic_moment_by_offset[offset]) * u0
                chan += 1
            end
            for offset in quartic_offsets
                ui = (states[time_idx + lag, i, traj] - mu) / sigma_x
                uj = (states[time_idx + lag, periodic(i + offset, K), traj] - mu) / sigma_x
                accum[chan] += (ui * ui * uj * uj - quartic_moment_by_offset[offset]) * u0
                chan += 1
            end
            for name in l96_channels
                uim2 = (states[time_idx + lag, periodic(i - 2, K), traj] - mu) / sigma_x
                uim1 = (states[time_idx + lag, periodic(i - 1, K), traj] - mu) / sigma_x
                uip1 = (states[time_idx + lag, periodic(i + 1, K), traj] - mu) / sigma_x
                adv = uim1 * (uip1 - uim2)
                value = name == "adv" ? adv : ((states[time_idx + lag, i, traj] - mu) / sigma_x) * adv
                accum[chan] += (value - l96_moments[name]) * u0
                chan += 1
            end
            count += 1
        end
        out[tau_idx, :] .= accum ./ count
    end
    return out
end

function render_training_correlation_validation_figure(path::AbstractString, params::TestParams,
        taus::Vector{Float64}, channel_names::Vector{String},
        cphi_data::Matrix{Float64}, cphi_m::Matrix{Float64}, cphi_phi::Matrix{Float64})
    nchan = length(channel_names)
    nrows, ncols = panel_grid_dims(nchan; max_cols=4)
    fw, fh = publication_panel_figure_size(nrows, ncols;
        base_w=params.figure_width, base_h=params.figure_height,
        panel_w=1080, panel_h=620, min_w=2400, min_h=1500, max_w=5600, max_h=5200)
    with_scaled_figure_style(fw, fh) do _
        fig = Figure(; size=(fw, fh))
        figure_title!(fig, "Forward validation of training correlations";
            subtitle="channels used to train M(x): data pairs vs learned-M Langevin vs Phi Langevin")
        for idx in 1:nchan
            r, c = centered_panel_rc(idx, nchan, ncols)
            shared_m = min(size(cphi_data, 1), size(cphi_m, 1))
            shared_phi = min(size(cphi_data, 1), size(cphi_phi, 1))
            err_m = rmse(cphi_data[1:shared_m, idx], cphi_m[1:shared_m, idx])
            err_phi = rmse(cphi_data[1:shared_phi, idx], cphi_phi[1:shared_phi, idx])
            ax = Axis(fig[r, c];
                xlabel=L"\tau", ylabel=L"C_{\phi}(\tau)",
                title=latex_text(@sprintf("%s    RMSE M %.3e  Phi %.3e", channel_panel_title(channel_names[idx], idx), err_m, err_phi)))
            hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            lines!(ax, taus, cphi_data[:, idx]; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label=L"\mathrm{data}")
            lines!(ax, taus, cphi_m[:, idx]; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label=L"\mathrm{NN}\;M")
            lines!(ax, taus, cphi_phi[:, idx]; color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label=L"\Phi")
            idx == 1 && axislegend(ax; position=:rt)
        end
        apply_publication_grid!(fig.layout, nrows, ncols; row_gap=34, col_gap=34)
        save_figure(path, fig)
    end
    return nothing
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

matrix_sym(A::AbstractMatrix{<:Real}) = 0.5 .* (A .+ A')
matrix_anti(A::AbstractMatrix{<:Real}) = 0.5 .* (A .- A')

function fit_phi_from_c_profiles(taus::AbstractVector{<:Real}, C_profiles::AbstractMatrix{<:Real},
        sigma_x::Float64, fit_window::Int, poly_degree::Int)
    require_condition(1 <= fit_window <= length(taus) - 1, "Invalid short-lag fit window.")
    degree = min(poly_degree, fit_window)
    fit_taus = Float64.(taus[1:(fit_window + 1)])
    design = hcat([fit_taus .^ p for p in 0:degree]...)
    K = size(C_profiles, 2)
    cdot = zeros(Float64, K)
    for r in 1:K
        coeffs = design \ Float64.(C_profiles[1:(fit_window + 1), r])
        cdot[r] = coeffs[2]
    end
    Phi_u = circulant_matrix_from_profile(-cdot)
    return cdot, Phi_u, sigma_x^2 .* Phi_u
end

function phi_diagnostic_stats(Phi_u::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real}, sigma_x::Float64)
    D_x = sigma_x^2 .* matrix_sym(Phi_u)
    A_x = sigma_x^2 .* matrix_anti(Phi_u)
    Phi_x = sigma_x^2 .* Phi_u
    sym_norm = max(norm(matrix_sym(Phi_x)), eps(Float64))
    return Dict{Symbol, Float64}(
        :relerr => norm(D_x .- Q) / norm(Q),
        :norm_ratio => norm(D_x) / norm(Q),
        :min_eig => minimum(eigvals(Symmetric(D_x))),
        :max_eig => maximum(eigvals(Symmetric(D_x))),
        :trace_ratio => tr(D_x) / tr(Q),
        :diag_mean_ratio => mean(diag(D_x)) / mean(diag(Q)),
        :anti_sym_ratio => norm(A_x) / sym_norm,
    )
end

function print_phi_diagnostic(name::AbstractString, Phi_u::AbstractMatrix{<:Real},
        Q::AbstractMatrix{<:Real}, sigma_x::Float64)
    stats = phi_diagnostic_stats(Phi_u, Q, sigma_x)
    println("=== Phi diagnostic: $(name) ===")
    @printf("||D_x - Q|| / ||Q||              = %.6e\n", stats[:relerr])
    @printf("||D_x|| / ||Q||                  = %.6e\n", stats[:norm_ratio])
    @printf("min eig(D_x)                     = %.6e\n", stats[:min_eig])
    @printf("max eig(D_x)                     = %.6e\n", stats[:max_eig])
    @printf("trace(D_x) / trace(Q)            = %.6e\n", stats[:trace_ratio])
    @printf("diag mean(D_x) / diag mean(Q)    = %.6e\n", stats[:diag_mean_ratio])
    @printf("||anti(Phi_x)|| / ||sym(Phi_x)|| = %.6e\n", stats[:anti_sym_ratio])
    return stats
end

function print_stein_diagnostic(V::AbstractMatrix{<:Real})
    Vsym = matrix_sym(V)
    I_K = Matrix{Float64}(I, size(V, 1), size(V, 2))
    println("=== Stein diagnostic ===")
    @printf("||V - I|| / ||I||       = %.6e\n", norm(V .- I_K) / norm(I_K))
    @printf("min eig(sym(V))         = %.6e\n", minimum(eigvals(Symmetric(Vsym))))
    @printf("max eig(sym(V))         = %.6e\n", maximum(eigvals(Symmetric(Vsym))))
    @printf("cond(V)                 = %.6e\n", cond(Matrix{Float64}(V)))
    return nothing
end

function print_saved_step_diagnostic(Phi_u_one_step::AbstractMatrix{<:Real}, D_eff::AbstractMatrix{<:Real},
        Q::AbstractMatrix{<:Real}, sigma_x::Float64, save_dt::Float64, simulator_dt::Float64)
    D_phi = sigma_x^2 .* matrix_sym(Phi_u_one_step)
    D_eff_sym = matrix_sym(D_eff)
    println("=== Saved-step finite-lag diagnostic ===")
    @printf("simulator dt                     = %.6e\n", simulator_dt)
    @printf("save_dt                          = %.6e\n", save_dt)
    @printf("save_dt / simulator dt           = %.6e\n", save_dt / simulator_dt)
    @printf("||D_phi_one_step - D_eff||/||D_eff|| = %.6e\n", norm(D_phi .- D_eff_sym) / max(norm(D_eff_sym), eps(Float64)))
    @printf("||D_eff - Q|| / ||Q||            = %.6e\n", norm(D_eff_sym .- Q) / norm(Q))
    @printf("trace(D_eff) / trace(Q)          = %.6e\n", tr(D_eff_sym) / tr(Q))
    @printf("min eig(D_eff)                   = %.6e\n", minimum(eigvals(Symmetric(D_eff_sym))))
    @printf("max eig(D_eff)                   = %.6e\n", maximum(eigvals(Symmetric(D_eff_sym))))
    return nothing
end

function render_phi_cdot0_diagnostic_figure(path::AbstractString, params::TestParams,
        Q::AbstractMatrix{<:Real}, candidates::Vector{Pair{String, Matrix{Float64}}},
        best_name::AbstractString, best_Phi_u::AbstractMatrix{<:Real}, sigma_x::Float64,
        sensitivity::Vector{Dict{Symbol, Float64}})
    fig = Figure(size=(params.figure_width, params.figure_height); figure_padding=(36, 36, 36, 28))
    D_best = sigma_x^2 .* matrix_sym(best_Phi_u)
    diff_best = D_best .- Q
    hm1 = heatmap!(Axis(fig[1, 1]; title="Q", aspect=DataAspect()), Matrix{Float64}(Q))
    Colorbar(fig[1, 2], hm1; width=18)
    hm2 = heatmap!(Axis(fig[1, 3]; title="Best estimate: $(best_name)", aspect=DataAspect()), D_best)
    Colorbar(fig[1, 4], hm2; width=18)
    hm3 = heatmap!(Axis(fig[1, 5]; title="Estimate - Q", aspect=DataAspect()), diff_best)
    Colorbar(fig[1, 6], hm3; width=18)

    ax_profile = Axis(fig[2, 1:2]; xlabel="offset", ylabel="circulant profile", title="Circulant Profiles")
    offsets = collect(0:(size(Q, 1) - 1))
    lines!(ax_profile, offsets, circulant_profile(Q); label="Q", linewidth=curve_linewidth(), color=STYLE_REFERENCE)
    colors = [STYLE_PRIMARY, STYLE_SECONDARY, STYLE_ACCENT, STYLE_MUTED]
    for (idx, candidate) in enumerate(candidates)
        D_x = sigma_x^2 .* matrix_sym(candidate.second)
        lines!(ax_profile, offsets, circulant_profile(D_x); label=candidate.first,
            linewidth=curve_linewidth(), color=colors[mod1(idx, length(colors))])
    end
    axislegend(ax_profile; position=:rt, nbanks=1, labelsize=12)

    ax_conv = Axis(fig[2, 3:4]; xlabel="fit window L", ylabel="relative error", title="Short-Lag Derivative Convergence")
    windows = [entry[:window] for entry in sensitivity]
    relerrs = [entry[:relerr] for entry in sensitivity]
    lines!(ax_conv, windows, relerrs; linewidth=curve_linewidth(), color=STYLE_PRIMARY)
    scatter!(ax_conv, windows, relerrs; marker=:circle, markersize=12, color=STYLE_PRIMARY)

    ax_eigs = Axis(fig[2, 5:6]; xlabel="eigenvalue index", ylabel="eigenvalue", title="Eigenvalues")
    q_eigs = sort(eigvals(Symmetric(Matrix{Float64}(Q))))
    d_eigs = sort(eigvals(Symmetric(D_best)))
    lines!(ax_eigs, eachindex(q_eigs), q_eigs; label="Q", linewidth=curve_linewidth(), color=STYLE_REFERENCE)
    lines!(ax_eigs, eachindex(d_eigs), d_eigs; label=best_name, linewidth=curve_linewidth(), color=STYLE_PRIMARY)
    axislegend(ax_eigs; position=:lt)
    apply_publication_grid!(fig.layout, 2, 6; row_gap=34, col_gap=20)
    save_figure(path, fig)
    return nothing
end

function run_phi_cdot0_test!(sampler::PairSampler, score_model::LoadedScoreModel,
        params::TestParams, mu::Float64, sigma_x::Float64, input_hdf5::AbstractString,
        output_dir::AbstractString, device::ExecutionDevice)
    println("Running phi_cdot0_test diagnostic only.")
    taus, C, Cdot_increment, phi_profile_u_increment, Phi_u_increment, Phi_x_increment =
        estimate_coordinate_cdot_and_phi(sampler, params, mu, sigma_x)
    taus_fit, C_profiles, Cdot0_profile_fit, Phi_u_fit, Phi_x_fit =
        estimate_phi_from_short_lag_correlations(sampler, params, mu, sigma_x; max_lag_fit=8, poly_degree=2)
    V, V_profile = estimate_clean_stein_matrix(sampler, score_model, params, mu, sigma_x, device)

    Phi_u_increment_raw = Phi_u_increment
    Phi_u_increment_corr = Phi_u_increment_raw / V
    Phi_u_fit_raw = Phi_u_fit
    Phi_u_fit_corr = Phi_u_fit_raw / V

    _, Q = load_l96_generator_metadata(input_hdf5)
    simulator_dt = Float64(h5read(input_hdf5, "/metadata/dt"))
    inc_cov_profile, D_eff_saved = estimate_saved_increment_effective_diffusion(sampler, params)
    print_stein_diagnostic(V)
    errors = Dict{Symbol, Dict{Symbol, Float64}}()
    errors[:one_step_raw] = print_phi_diagnostic("one-step raw", Phi_u_increment_raw, Q, sigma_x)
    errors[:one_step_stein_corrected] = print_phi_diagnostic("one-step Stein-corrected", Phi_u_increment_corr, Q, sigma_x)
    errors[:polynomial_fit_raw] = print_phi_diagnostic("polynomial-fit raw", Phi_u_fit_raw, Q, sigma_x)
    errors[:polynomial_fit_stein_corrected] = print_phi_diagnostic("polynomial-fit Stein-corrected", Phi_u_fit_corr, Q, sigma_x)
    print_saved_step_diagnostic(Phi_u_increment_raw, D_eff_saved, Q, sigma_x, sampler.save_dt, simulator_dt)

    println("=== Short-lag derivative sensitivity ===")
    @printf("fit lag indices used: %s\n", string(collect(0:(length(taus_fit) - 1))))
    @printf("fit lag times used: %s\n", string(round.(taus_fit; digits=6)))
    sensitivity = Dict{Symbol, Float64}[]
    for L in (1, 2, 3, 4, 6, 8)
        L <= length(taus_fit) - 1 || continue
        _, Phi_u_L, _ = fit_phi_from_c_profiles(taus_fit, C_profiles, sigma_x, L, 2)
        stats = phi_diagnostic_stats(Phi_u_L, Q, sigma_x)
        push!(sensitivity, Dict{Symbol, Float64}(:window => Float64(L),
            :relerr => stats[:relerr], :trace_ratio => stats[:trace_ratio], :min_eig => stats[:min_eig]))
        @printf("L = %d: relerr sym(Phi_x_fit_L) vs Q = %.6e, trace ratio = %.6e, min eig = %.6e\n",
            L, stats[:relerr], stats[:trace_ratio], stats[:min_eig])
    end

    candidates = [
        "one-step raw" => Matrix{Float64}(Phi_u_increment_raw),
        "one-step corrected" => Matrix{Float64}(Phi_u_increment_corr),
        "polynomial-fit raw" => Matrix{Float64}(Phi_u_fit_raw),
        "polynomial-fit corrected" => Matrix{Float64}(Phi_u_fit_corr),
    ]
    best_pair = first(candidates)
    best_relerr = Inf
    for candidate in candidates
        relerr = phi_diagnostic_stats(candidate.second, Q, sigma_x)[:relerr]
        if relerr < best_relerr
            best_relerr = relerr
            best_pair = candidate
        end
    end
    if best_relerr > 0.1 && sampler.save_dt > simulator_dt
        @printf("Phi diagnostic failure mechanism: finite saved-lag bias / insufficient saved temporal resolution. ")
        @printf("The smallest available lag is %.6g, which is %.1f simulator steps. ", sampler.save_dt, sampler.save_dt / simulator_dt)
        @printf("Regenerate data with save_dt close to simulator dt before using Cdot(0) to estimate Phi.\n")
    end
    figure_path = joinpath(output_dir, "phi_cdot0_diagnostic.png")
    render_phi_cdot0_diagnostic_figure(figure_path, params, Q, candidates, best_pair.first,
        best_pair.second, sigma_x, sensitivity)
    diag_path = joinpath(output_dir, "phi_cdot0_diagnostic.bson")
    BSON.@save diag_path mu sigma_x Q V V_profile taus C Cdot_increment taus_fit C_profiles Cdot0_profile_fit inc_cov_profile D_eff_saved Phi_x_increment Phi_x_fit Phi_u_increment_raw Phi_u_increment_corr Phi_u_fit_raw Phi_u_fit_corr errors sensitivity
    @printf("Saved phi_cdot0 diagnostic figure to %s\n", figure_path)
    @printf("Saved phi_cdot0 diagnostic data to %s\n", diag_path)
    return nothing
end

function write_metrics(path::AbstractString, metrics::Dict{Symbol, <:Any})
    open(path, "w") do io
        println(io, "L96 fit_dM_test validation")
        for key in sort(collect(keys(metrics)); by=String)
            println(io, string(key), " = ", metrics[key])
        end
    end
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.plain_score_bson)
    joint_bson = resolve_path(base_dir, params.joint_score_bson)
    output_dir = resolve_path(base_dir, params.output_dir)
    mkpath(output_dir)

    sampler = build_pair_sampler(input_hdf5, params)
    K = size(sampler.states, 2)
    device = detect_device(params.device_name)
    activate_device!(device)
    @printf("Using device: %s\n", describe_device(device))
    score_model = load_score_model(score_bson, device, K)

    data_mu, data_sigma_x = estimate_standardization(sampler.states, sampler.start_idx)
    mu = Float64(mean(score_model.mean))
    sigma_x = Float64(mean(score_model.std))
    mean_spread = maximum(abs.(Float64.(score_model.mean) .- mu))
    std_spread = maximum(abs.(Float64.(score_model.std) .- sigma_x))
    @printf("Post-burnin all-data standardization: mu = %.6f, sigma_x = %.6f\n", data_mu, data_sigma_x)
    @printf("Score-checkpoint standardization used for Phi rollout: mu = %.6f, sigma_x = %.6f (max spreads %.3e, %.3e)\n",
        mu, sigma_x, mean_spread, std_spread)

    if params.run_mode == "phi_cdot0_test"
        run_phi_cdot0_test!(sampler, score_model, params, mu, sigma_x, input_hdf5, output_dir, device)
        return nothing
    end

    forcing, Q_generator = load_l96_generator_metadata(input_hdf5)
    joint_model = params.mobility_nn_enabled ? load_joint_score_model(joint_bson, device, K) : nothing

    taus, C, Cdot, phi_profile_u_uncorrected, Phi_u_uncorrected, Phi_raw_uncorrected =
        estimate_coordinate_cdot_and_phi(sampler, params, mu, sigma_x)
    V_stein, V_stein_profile = estimate_clean_stein_matrix(sampler, score_model, params, mu, sigma_x, device)
    V_stein_raw = V_stein
    V_stein = 0.5 .* (V_stein_raw .+ V_stein_raw')
    Phi_u_full = Phi_u_uncorrected / V_stein
    Phi_raw_full = sigma_x^2 .* Phi_u_full
    phi_profile_u = circulant_profile(Phi_u_full)
    Phi_sym = 0.5 .* (Phi_u_full .+ Phi_u_full')
    Phi_anti = 0.5 .* (Phi_u_full .- Phi_u_full')
    Phi_u = Phi_sym .+ params.antisymmetric_scale .* Phi_anti
    Phi_raw = sigma_x^2 .* Phi_u
    phi_sym = 0.5 .* (Phi_u_full .+ Phi_u_full')
    phi_sym_eigs = eigvals(Symmetric(phi_sym))
    require_condition(minimum(phi_sym_eigs) > 0.0,
        @sprintf("Stein-corrected sym(Phi) is not positive definite: min eigenvalue %.6e", minimum(phi_sym_eigs)))
    phi_sym_norm = norm(phi_sym)
    phi_anti_norm = norm(0.5 .* (Phi_u_full .- Phi_u_full'))
    stein_diag_mean = mean(diag(V_stein))
    stein_offdiag_rms = sqrt(mean(abs2, V_stein .- Diagonal(diag(V_stein))))
    @printf("Clean Stein V stats: mean diag = %.6e, offdiag RMS = %.6e, cond = %.6e\n",
        stein_diag_mean, stein_offdiag_rms, cond(V_stein))
    @printf("Estimated Stein-corrected full Phi_u; ||sym|| = %.6e, ||anti|| = %.6e\n",
        phi_sym_norm, phi_anti_norm)
    @printf("Using full Phi_u with %.3f times antisym(Phi_u) in drift; diffusion uses exact chol(sym(Phi_u)).\n",
        params.antisymmetric_scale)

    observed_corr = load_observed_correlation(input_hdf5, params.correlation_max_time, params.cross_offsets)
    old_forward_figure_path = joinpath(output_dir, params.figure_png)
    a_figure_path = joinpath(output_dir, "fit_A.png")
    pdf_figure_path = joinpath(output_dir, "forward_validation_pdfs.png")
    cphi_figure_path = joinpath(output_dir, "forward_validation_cphi.png")
    training_figure_path = joinpath(output_dir, "mobility_training_diagnostics.png")
    mobility_checkpoint_path = joinpath(output_dir, "mobility_nn_checkpoint.bson")

    mobility_model = nothing
    mobility_history = nothing
    mobility_target_scale = Float64[]
    train_cache = nothing
    target_delta = Array{Float64, 3}(undef, 0, 0, 0)
    pred_delta = Array{Float64, 3}(undef, 0, 0, 0)
    second_moment_by_offset = Dict{Int, Float64}()
    cubic_moment_by_offset = Dict{Int, Float64}()
    quartic_moment_by_offset = Dict{Int, Float64}()
    l96_moments = Dict{String, Float64}()
    selected_channel_indices = Int[]
    channel_selection_report = Dict{Symbol, Any}()
    if params.mobility_nn_enabled
        second_moment_by_offset = estimate_second_moment_by_offset(sampler, params, mu, sigma_x,
            params.mobility_nn_quadratic_offsets)
        cubic_moment_by_offset = estimate_cubic_moment_by_offset(sampler, params, mu, sigma_x,
            params.mobility_nn_cubic_offsets)
        quartic_moment_by_offset = estimate_quartic_moment_by_offset(sampler, params, mu, sigma_x,
            params.mobility_nn_quartic_offsets)
        l96_moments = estimate_l96_channel_moments(sampler, params, mu, sigma_x,
            params.mobility_nn_l96_channels)
        @printf("Building mobility NN caches from joint conditional score (%d coordinate, %d quadratic, %d cubic, %d quartic, %d L96 channels).\n",
            length(params.mobility_nn_coordinate_separations), length(params.mobility_nn_quadratic_offsets),
            length(params.mobility_nn_cubic_offsets), length(params.mobility_nn_quartic_offsets),
            length(params.mobility_nn_l96_channels))
        train_cache = build_mobility_training_cache(sampler, score_model, joint_model, params, mu, sigma_x,
            Phi_u, second_moment_by_offset, cubic_moment_by_offset, quartic_moment_by_offset,
            l96_moments, forcing, Q_generator, device; pair_seed=params.seed + 411, anchor_seed=params.seed + 421)
        validation_caches = [
            build_mobility_training_cache(sampler, score_model, joint_model, params, mu, sigma_x,
                Phi_u, second_moment_by_offset, cubic_moment_by_offset, quartic_moment_by_offset,
                l96_moments, forcing, Q_generator, device; pair_seed=seed, anchor_seed=seed + 17)
            for seed in params.mobility_nn_validation_pair_seeds
        ]
        selected_channel_indices = collect(1:length(train_cache.channel_names))
        if params.mobility_nn_select_channels
            selected_channel_indices, channel_selection_report =
                select_mobility_training_channels(train_cache, validation_caches, params)
            train_cache = filter_mobility_cache(train_cache, selected_channel_indices)
            validation_caches = [filter_mobility_cache(vcache, selected_channel_indices) for vcache in validation_caches]
        end
        target_delta = mobility_target_from_cache(train_cache)
        current_action_cache = params.mobility_nn_current_action_penalty > 0.0 ?
            build_current_action_cache(sampler, score_model, params, mu, sigma_x, Phi_u, forcing, device) : nothing
        mobility_model, mobility_history, mobility_target_scale = train_mobility_model(train_cache,
            validation_caches, target_delta, Phi_u, params, device; checkpoint_path=mobility_checkpoint_path,
            current_action_cache=current_action_cache)
        pred_delta = evaluate_mobility_model_on_cache(mobility_model, train_cache, params,
            Float32.(symmetric_factor(0.5 .* (Phi_u .+ Phi_u') .+
                params.mobility_nn_psd_jitter .* Matrix{Float64}(I, K, K))))
        render_A_fit_figure(a_figure_path, params, train_cache.taus, train_cache.channel_names,
            target_delta, pred_delta)
        render_training_diagnostics(training_figure_path, params, mobility_history, train_cache.taus, target_delta,
            pred_delta, mobility_target_scale, train_cache.channel_names)
    end

    if params.mobility_nn_enabled
        isfile(old_forward_figure_path) && rm(old_forward_figure_path; force=true)
        rollout_times, pred_states_m, pred_states_phi, rollout_stats = integrate_learned_and_phi(sampler,
            score_model, params, mobility_model, Phi_u, mu, sigma_x, device)
        pdf_diag_m = compute_pdf_diagnostics(input_hdf5, pred_states_m, params)
        pdf_diag_phi = compute_pdf_diagnostics(input_hdf5, pred_states_phi, params)
        pred_corr_m = compute_lattice_correlations(pred_states_m, rollout_stats[:saved_dt],
            params.correlation_max_time, params.correlation_threshold, params.cross_offsets)
        pred_corr_phi = compute_lattice_correlations(pred_states_phi, rollout_stats[:saved_dt],
            params.correlation_max_time, params.correlation_threshold, params.cross_offsets)
        cphi_m = compute_training_observable_correlations(pred_states_m, rollout_stats[:saved_dt],
            train_cache.taus, params.mobility_nn_coordinate_separations, params.mobility_nn_quadratic_offsets,
            params.mobility_nn_cubic_offsets, params.mobility_nn_quartic_offsets,
            second_moment_by_offset, cubic_moment_by_offset, quartic_moment_by_offset,
            params.mobility_nn_l96_channels, l96_moments, mu, sigma_x)
        cphi_phi = compute_training_observable_correlations(pred_states_phi, rollout_stats[:saved_dt],
            train_cache.taus, params.mobility_nn_coordinate_separations, params.mobility_nn_quadratic_offsets,
            params.mobility_nn_cubic_offsets, params.mobility_nn_quartic_offsets,
            second_moment_by_offset, cubic_moment_by_offset, quartic_moment_by_offset,
            params.mobility_nn_l96_channels, l96_moments, mu, sigma_x)
        if params.mobility_nn_select_channels
            cphi_m = cphi_m[:, selected_channel_indices]
            cphi_phi = cphi_phi[:, selected_channel_indices]
        end
        render_forward_pdf_acf_figure(pdf_figure_path, params, pdf_diag_m,
            observed_corr, pred_corr_m, pred_corr_phi)
        render_training_correlation_validation_figure(cphi_figure_path, params, train_cache.taus,
            train_cache.channel_names, train_cache.correlations, cphi_m, cphi_phi)
        pred_states = pred_states_phi
        pdf_diag = pdf_diag_phi
        pred_corr = pred_corr_phi
        learned_pdf_diag = pdf_diag_m
        learned_pred_corr = pred_corr_m
    else
        rollout_times, pred_states, rollout_stats = integrate_phi_only(sampler, score_model, params, Phi_u, mu, sigma_x, device)
        pdf_diag = compute_pdf_diagnostics(input_hdf5, pred_states, params)
        pred_corr = compute_lattice_correlations(pred_states, rollout_stats[:saved_dt], params.correlation_max_time,
            params.correlation_threshold, params.cross_offsets)
        shared_lag_count_tmp = min(length(observed_corr.lags), length(pred_corr.lags))
        render_forward_stats(old_forward_figure_path, params, pdf_diag, observed_corr, pred_corr,
            taus, C, Cdot, phi_profile_u_uncorrected, circulant_profile(Phi_u), rollout_stats,
            rmse(pdf_diag.univariate_observed.density, pdf_diag.univariate_generated.density),
            rmse(observed_corr.acf_mean[1:shared_lag_count_tmp], pred_corr.acf_mean[1:shared_lag_count_tmp]),
            rmse(observed_corr.cross_mean[1:shared_lag_count_tmp, :], pred_corr.cross_mean[1:shared_lag_count_tmp, :]))
        learned_pdf_diag = nothing
        learned_pred_corr = nothing
    end

    shared_lag_count = min(length(observed_corr.lags), length(pred_corr.lags))
    pdf_rmse_value = rmse(pdf_diag.univariate_observed.density, pdf_diag.univariate_generated.density)
    acf_rmse_value = rmse(observed_corr.acf_mean[1:shared_lag_count], pred_corr.acf_mean[1:shared_lag_count])
    cross_rmse_value = rmse(observed_corr.cross_mean[1:shared_lag_count, :], pred_corr.cross_mean[1:shared_lag_count, :])

    diagnostics_path = joinpath(output_dir, params.diagnostics_bson)
    BSON.@save diagnostics_path params taus C Cdot phi_profile_u_uncorrected Phi_u_uncorrected Phi_raw_uncorrected V_stein_raw V_stein V_stein_profile phi_profile_u Phi_u_full Phi_raw_full Phi_u Phi_raw mu sigma_x data_mu data_sigma_x rollout_times rollout_stats pred_corr observed_corr pdf_diag learned_pdf_diag learned_pred_corr mobility_model mobility_history mobility_target_scale target_delta pred_delta selected_channel_indices channel_selection_report pdf_rmse_value acf_rmse_value cross_rmse_value phi_sym_norm phi_anti_norm stein_diag_mean stein_offdiag_rms a_figure_path pdf_figure_path cphi_figure_path training_figure_path
    selected_channel_names_for_metrics = params.mobility_nn_enabled && train_cache !== nothing ? train_cache.channel_names : String[]
    metrics = Dict{Symbol, Any}(
        :fit_A_figure_png => a_figure_path,
        :forward_pdf_figure_png => pdf_figure_path,
        :forward_cphi_figure_png => cphi_figure_path,
        :training_figure_png => training_figure_path,
        :selected_channel_count => length(selected_channel_indices),
        :selected_channels => join(selected_channel_names_for_metrics, ", "),
        :diagnostics_bson => diagnostics_path,
        :pdf_rmse => pdf_rmse_value,
        :pdf_kl_univariate => pdf_diag.univariate_kl,
        :pdf_kl_pair_mean => isempty(pdf_diag.pair_kls) ? NaN : mean(pdf_diag.pair_kls),
        :pdf_kl_mean => pdf_diag.mean_kl,
        :pdf_kl_pairs_by_offset => join([@sprintf("%d:%.8e", pair.offset, kl) for (pair, kl) in zip(pdf_diag.pair_observed, pdf_diag.pair_kls)], ", "),
        :acf_rmse => acf_rmse_value,
        :cross_rmse => cross_rmse_value,
        :observed_t_decorrelation => observed_corr.t_decorrelation,
        :phi_t_decorrelation => pred_corr.t_decorrelation,
        :phi_full_sym_norm => phi_sym_norm,
        :phi_full_antisym_norm => phi_anti_norm,
        :phi_full_antisym_to_sym_norm => phi_anti_norm / max(phi_sym_norm, eps(Float64)),
        :antisymmetric_scale => params.antisymmetric_scale,
        :stein_diag_mean => stein_diag_mean,
        :stein_offdiag_rms => stein_offdiag_rms,
        :stein_condition_number => cond(V_stein),
        :score_normalization_mean => mu,
        :score_normalization_std => sigma_x,
        :data_normalization_mean => data_mu,
        :data_normalization_std => data_sigma_x,
        :sym_phi_lambda_min => rollout_stats[:sym_phi_lambda_min],
        :sym_phi_lambda_max => rollout_stats[:sym_phi_lambda_max],
        :eval_clamp_fraction => rollout_stats[:eval_clamp_fraction],
        :hard_clamp_fraction => rollout_stats[:hard_clamp_fraction],
    )
    if params.mobility_nn_enabled && learned_pdf_diag !== nothing && learned_pred_corr !== nothing
        shared_m = min(length(observed_corr.lags), length(learned_pred_corr.lags))
        metrics[:learned_pdf_kl_univariate] = learned_pdf_diag.univariate_kl
        metrics[:learned_pdf_kl_mean] = learned_pdf_diag.mean_kl
        metrics[:learned_pdf_rmse] = rmse(learned_pdf_diag.univariate_observed.density,
            learned_pdf_diag.univariate_generated.density)
        metrics[:learned_acf_rmse] = rmse(observed_corr.acf_mean[1:shared_m],
            learned_pred_corr.acf_mean[1:shared_m])
        metrics[:learned_cross_rmse] = rmse(observed_corr.cross_mean[1:shared_m, :],
            learned_pred_corr.cross_mean[1:shared_m, :])
        metrics[:learned_t_decorrelation] = learned_pred_corr.t_decorrelation
        metrics[:mobility_best_validation_nrmse] = minimum(mobility_history.normalized_rmse)
        metrics[:mobility_final_validation_nrmse] = mobility_history.normalized_rmse[end]
        metrics[:mobility_final_mean_abs_delta] = mobility_history.mean_abs_coeff[end]
    end
    write_metrics(joinpath(output_dir, params.metrics_txt), metrics)

    println()
    println("Summary")
    params.mobility_nn_enabled && println("A-fit figure: " * a_figure_path)
    params.mobility_nn_enabled && println("Forward PDF/ACF figure: " * pdf_figure_path)
    params.mobility_nn_enabled && println("Forward Cphi figure: " * cphi_figure_path)
    params.mobility_nn_enabled && println("Training figure: " * training_figure_path)
    println(@sprintf("PDF RMSE / KL_1D / KL_mean = %.6e / %.6e / %.6e", pdf_rmse_value, pdf_diag.univariate_kl, pdf_diag.mean_kl))
    println(@sprintf("ACF RMSE = %.6e | cross RMSE = %.6e", acf_rmse_value, cross_rmse_value))
    println(@sprintf("t_decorr observed/Phi = %.6f / %.6f", observed_corr.t_decorrelation, pred_corr.t_decorrelation))
    return params.mobility_nn_enabled ? pdf_figure_path : old_forward_figure_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
