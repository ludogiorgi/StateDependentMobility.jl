import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")
const FHD_STATE_CHANNELS = 2

function ensure_packages(packages::Vector{String})
    project_deps = Pkg.project().dependencies
    missing = String[]
    for pkg in packages
        haskey(project_deps, pkg) || push!(missing, pkg)
    end
    isempty(missing) || Pkg.add(missing)
    return nothing
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ensure_packages(["BSON", "CUDA", "cuDNN", "Flux", "Functors", "GLMakie",
    "HDF5", "KernelDensity", "LaTeXStrings", "NNlib", "ProgressMeter", "TOML"])

function fhd_display_is_usable(display::AbstractString)
    xdpyinfo = Sys.which("xdpyinfo")
    xdpyinfo === nothing && return true
    try
        run(pipeline(`$xdpyinfo -display $display`, stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

function fhd_start_xvfb!()
    xvfb = Sys.which("Xvfb")
    xvfb === nothing && return nothing
    for display_id in 98:150
        display = ":" * string(display_id)
        if fhd_display_is_usable(display)
            ENV["DISPLAY"] = display
            ENV["STATEDEP_XVFB_DISPLAY"] = display
            return nothing
        end
    end
    for display_id in 101:150
        display = ":" * string(display_id)
        isfile("/tmp/.X$(display_id)-lock") && continue
        run(pipeline(`$xvfb $display -screen 0 1920x1200x24 -nolisten tcp`,
            stdout=devnull, stderr=devnull); wait=false)
        sleep(1.0)
        if fhd_display_is_usable(display)
            ENV["DISPLAY"] = display
            ENV["STATEDEP_XVFB_DISPLAY"] = display
            return nothing
        end
    end
    return nothing
end

if haskey(ENV, "DISPLAY") && !fhd_display_is_usable(ENV["DISPLAY"])
    delete!(ENV, "DISPLAY")
end
if !haskey(ENV, "DISPLAY") || isempty(ENV["DISPLAY"])
    fhd_start_xvfb!()
end

using BSON
using CUDA
using cuDNN
using Flux
using Functors
using GLMakie
using HDF5
using KernelDensity
using LaTeXStrings
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
include(STYLE_FILE)
GLMakie.activate!()

require_condition(condition::Bool, message::String) = condition || error(message)
resolve_path(base_dir::AbstractString, path::AbstractString) =
    isabspath(path) ? path : normpath(joinpath(base_dir, path))
ensure_parent_dir(path::AbstractString) = (mkpath(dirname(path)); nothing)
to_host(x) = x isa AbstractArray && !(x isa Array) ? Array(x) : x
periodic(i::Int, N::Int) = mod1(i, N)

Base.@kwdef struct FHDPhysicalParams
    N::Int
    L::Float64
    dx::Float64
    rho0::Float64
    cs::Float64
    theta::Float64
    eta0::Float64
    zeta::Float64
    velocity_density_floor::Float64
end

struct ProjectedScoreUNet{M}
    backbone::M
end
Functors.@functor ProjectedScoreUNet (backbone,)

function (model::ProjectedScoreUNet)(x)
    pred = model.backbone(x)
    y = @view pred[:, 1:FHD_STATE_CHANNELS, :]
    return project_zero_modes(y)
end

struct ReverseResidualScoreUNet{M}
    backbone::M
end
Functors.@functor ReverseResidualScoreUNet (backbone,)

function (model::ReverseResidualScoreUNet)(x)
    pred = model.backbone(x)
    y = @view pred[:, 1:FHD_STATE_CHANNELS, :]
    return project_zero_modes(y)
end

struct OutputScaledModel{M}
    model::M
    scale::Float32
end
Functors.@functor OutputScaledModel (model,)

function (model::OutputScaledModel)(x)
    return model.scale .* model.model(x)
end

struct NoiseEquivalentScoreModel{M}
    model::M
    sigma::Float32
end
Functors.@functor NoiseEquivalentScoreModel (model,)

function (model::NoiseEquivalentScoreModel)(x)
    return -model.sigma .* model.model(x)
end

struct LocalProjectedScoreMLP{M}
    network::M
end
Functors.@functor LocalProjectedScoreMLP (network,)

function (model::LocalProjectedScoreMLP)(x)
    K, C, B = size(x)
    features = reshape(permutedims(x, (2, 1, 3)), C, K * B)
    pred = model.network(features)
    y = permutedims(reshape(pred, C, K, B), (2, 1, 3))
    return project_zero_modes(y)
end

struct LocalFeatureProjectedScoreMLP{M}
    network::M
    rho_mean::Float32
    rho_std::Float32
    m_mean::Float32
    m_std::Float32
    rho_floor::Float32
    velocity_floor::Float32
end
Functors.@functor LocalFeatureProjectedScoreMLP (network,)

function (model::LocalFeatureProjectedScoreMLP)(x)
    K, _, B = size(x)
    rho_norm = @view x[:, 1:1, :]
    m_norm = @view x[:, 2:2, :]
    rho = rho_norm .* model.rho_std .+ model.rho_mean
    mom = m_norm .* model.m_std .+ model.m_mean
    rho_safe = max.(rho, model.rho_floor)
    vel = mom ./ max.(rho, model.velocity_floor)
    logrho = log.(rho_safe)
    features_tensor = cat(rho_norm, m_norm, logrho, vel; dims=2)
    features = reshape(permutedims(features_tensor, (2, 1, 3)), 4, K * B)
    pred = model.network(features)
    y = permutedims(reshape(pred, FHD_STATE_CHANNELS, K, B), (2, 1, 3))
    return project_zero_modes(y)
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

function normalization_from_string(name)
    s = lowercase(String(name))
    s == "batchnorm" && return :batchnorm
    s == "none" && return :none
    s == "groupnorm" && return :groupnorm
    error("Unsupported model.normalization=$(name); allowed: batchnorm, none, groupnorm.")
end

function normalize_time_features(name)
    s = lowercase(String(name))
    s in ("scalar", "fourier") || error("Unsupported time_features=$(name).")
    return s
end

time_feature_count(features::AbstractString, nfreq::Int) =
    features == "fourier" ? 1 + 2 * nfreq : 1

cond_input_channels(features::AbstractString, nfreq::Int; include_delta_input::Bool=true) =
    4 + (include_delta_input ? 2 : 0) + time_feature_count(features, nfreq)

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

function nvidia_smi_gpu_names()
    smi = Sys.which("nvidia-smi")
    smi === nothing && return Dict{Int, String}()
    try
        output = read(`$smi --query-gpu=index,name --format=csv,noheader`, String)
        result = Dict{Int, String}()
        for line in split(chomp(output), '\n')
            isempty(strip(line)) && continue
            parts = split(line, ","; limit=2)
            length(parts) == 2 || continue
            result[parse(Int, strip(parts[1]))] = strip(parts[2])
        end
        return result
    catch
        return Dict{Int, String}()
    end
end

function parse_single_gpu_request(req::AbstractString)
    normalized = uppercase(strip(req))
    startswith(normalized, "GPU:") || return nothing
    spec = split(normalized, ":", limit=2)[2]
    occursin(",", spec) && return nothing
    isempty(strip(spec)) && return nothing
    return parse(Int, strip(spec))
end

function cuda_ordinal_for_required_name(required_gpu_name::AbstractString)
    devices = collect(CUDA.devices())
    needle = lowercase(strip(required_gpu_name))
    for (idx, dev) in enumerate(devices)
        occursin(needle, lowercase(CUDA.name(dev))) && return idx - 1
    end
    return nothing
end

function detect_fhd_device(request::AbstractString, required_gpu_name::AbstractString="")
    req = strip(request)
    normalized = uppercase(req)
    normalized == "AUTO" && error("AUTO device selection is disabled for FHDChain; request the 5070 explicitly.")
    normalized == "GPU" && error("Ambiguous GPU request is disabled for FHDChain; request the 5070 explicitly.")
    if startswith(normalized, "GPU:")
        requested_index = parse_single_gpu_request(req)
        if requested_index !== nothing && !isempty(strip(required_gpu_name))
            smi_names = nvidia_smi_gpu_names()
            if haskey(smi_names, requested_index)
                smi_name = smi_names[requested_index]
                occursin(lowercase(strip(required_gpu_name)), lowercase(smi_name)) ||
                    error("Requested nvidia-smi GPU $(requested_index) is $(smi_name), which does not satisfy required_gpu_name=$(required_gpu_name).")
                cuda_id = cuda_ordinal_for_required_name(required_gpu_name)
                cuda_id === nothing && error("nvidia-smi GPU $(requested_index) is $(smi_name), but CUDA.jl did not expose a matching device.")
                return GPUDevice([cuda_id])
            end
        end
        return select_device(req)
    end
    if !isempty(strip(required_gpu_name))
        CUDA.has_cuda() || error("GPU required but CUDA is unavailable.")
        devices = collect(CUDA.devices())
        isempty(devices) && error("GPU required but no CUDA devices were detected.")
        needle = lowercase(strip(required_gpu_name))
        for (idx, dev) in enumerate(devices)
            zero_based = idx - 1
            name = lowercase(CUDA.name(dev))
            if occursin(needle, name)
                return GPUDevice([zero_based])
            end
        end
        names = [CUDA.name(d) for d in devices]
        error("No CUDA device name contains \"$(required_gpu_name)\". Detected CUDA devices: $(names)")
    end
    return select_device(req)
end

function nvidia_smi_index_for_cuda_name(cuda_name::AbstractString)
    smi_names = nvidia_smi_gpu_names()
    for (idx, name) in smi_names
        name == cuda_name && return idx
    end
    return nothing
end

function activate_and_describe_device!(device::ExecutionDevice, requested::AbstractString, required::AbstractString="")
    activate_device!(device)
    if device isa GPUDevice
        devices = collect(CUDA.devices())
        id = first(device.ids)
        name = CUDA.name(devices[id + 1])
        smi_idx = nvidia_smi_index_for_cuda_name(name)
        !isempty(required) && require_condition(occursin(lowercase(required), lowercase(name)),
            "Selected CUDA device $(id) is $(name), which does not satisfy required_gpu_name=$(required).")
        @printf("Training device request: %s\n", requested)
        if smi_idx === nothing
            @printf("Resolved CUDA device: ordinal %d, %s\n", id, name)
        else
            @printf("Resolved CUDA device: ordinal %d, %s (nvidia-smi index %d)\n", id, name, smi_idx)
        end
    else
        @printf("Resolved execution device: CPU\n")
    end
    return nothing
end

function load_fhd_physical_params(path::AbstractString)
    h5open(path, "r") do file
        return FHDPhysicalParams(
            N=Int(read(file["/metadata/N"])),
            L=Float64(read(file["/metadata/L"])),
            dx=Float64(read(file["/metadata/dx"])),
            rho0=Float64(read(file["/metadata/rho0"])),
            cs=Float64(read(file["/metadata/sound_speed"])),
            theta=Float64(read(file["/metadata/Theta"])),
            eta0=Float64(read(file["/metadata/eta0"])),
            zeta=Float64(read(file["/metadata/zeta"])),
            velocity_density_floor=Float64(read(file["/metadata/velocity_density_floor"])),
        )
    end
end

burnin_start_index(nsaved::Int, burnin_fraction::Float64) =
    clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)

function channel_shared_data_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    means = Array{Float32}(undef, C, K)
    stds = Array{Float32}(undef, C, K)
    for c in 1:C
        vals = @view samples[:, c, :]
        mu = mean(Float64, vals)
        n = length(vals)
        var = mean(x -> abs2(Float64(x) - mu), vals) * n / max(n - 1, 1)
        means[c, :] .= Float32(mu)
        stds[c, :] .= max(Float32(sqrt(var)), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function apply_fhd_stats(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return (samples .- mean_tensor) ./ std_tensor
end

function denormalize_fhd_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return samples .* std_tensor .+ mean_tensor
end

function project_zero_modes(x)
    return x .- (sum(x; dims=1) ./ eltype(x)(size(x, 1)))
end

function project_zero_modes!(x)
    x .-= sum(x; dims=1) ./ eltype(x)(size(x, 1))
    return x
end

function load_fhd_states(path::AbstractString)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    require_condition(ndims(states) == 4, "Expected /trajectories/states as time x site x channel x trajectory.")
    require_condition(size(states, 3) == FHD_STATE_CHANNELS, "Expected FHD channels (rho,m).")
    require_condition(size(states, 1) == length(times), "Time axis length mismatch.")
    return times, states
end

function sample_state_tensor(states::Array{Float32, 4}, start_idx::Int, nsamples::Int, rng::AbstractRNG)
    nt, K, C, ntraj = size(states)
    total = (nt - start_idx + 1) * ntraj
    n = nsamples <= 0 ? total : min(nsamples, total)
    tensor = Array{Float32}(undef, K, C, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tr = (linear ÷ (nt - start_idx + 1)) + 1
        tensor[:, :, s] .= states[t, :, :, tr]
    end
    return tensor
end

function load_fhd_dataset(path::AbstractString, burnin_fraction::Float64, max_samples::Int,
        rng::AbstractRNG)
    times, states = load_fhd_states(path)
    start_idx = burnin_start_index(length(times), burnin_fraction)
    raw = sample_state_tensor(states, start_idx, max_samples, rng)
    stats = channel_shared_data_stats(raw)
    normed = apply_fhd_stats(raw, stats)
    return NormalizedDataset(normed, stats), times, states, start_idx
end

function raw_flat_batch(z::Array{Float32, 3})
    K, _, B = size(z)
    out = Matrix{Float32}(undef, 2K, B)
    @views begin
        out[1:K, :] .= reshape(z[:, 1, :], K, B)
        out[K+1:2K, :] .= reshape(z[:, 2, :], K, B)
    end
    return out
end

function raw_flat_sample(z::AbstractMatrix{<:Real})
    K = size(z, 1)
    out = Vector{Float64}(undef, 2K)
    @inbounds for i in 1:K
        out[i] = Float64(z[i, 1])
        out[K + i] = Float64(z[i, 2])
    end
    return out
end

function std_flat(stats::DataStats)
    K = size(stats.std, 2)
    out = Vector{Float32}(undef, 2K)
    @views begin
        out[1:K] .= stats.std[1, :]
        out[K+1:2K] .= stats.std[2, :]
    end
    return out
end

function mean_flat(stats::DataStats)
    K = size(stats.mean, 2)
    out = Vector{Float32}(undef, 2K)
    @views begin
        out[1:K] .= stats.mean[1, :]
        out[K+1:2K] .= stats.mean[2, :]
    end
    return out
end

function physical_score_raw(samples::Array{Float32, 3}, phys::FHDPhysicalParams;
        velocity_floor::Float64=0.0, rho_floor::Float64=1.0e-12)
    K, _, B = size(samples)
    scores = Array{Float32}(undef, K, 2, B)
    @inbounds for b in 1:B
        a_mean = 0.0
        u_mean = 0.0
        a_vals = Vector{Float64}(undef, K)
        u_vals = Vector{Float64}(undef, K)
        for i in 1:K
            rho = max(Float64(samples[i, 1, b]), rho_floor)
            m = Float64(samples[i, 2, b])
            denom = velocity_floor > 0 ? max(rho, velocity_floor) : rho
            u = m / denom
            a = phys.cs^2 * log(rho / phys.rho0) - 0.5 * u^2
            a_vals[i] = a
            u_vals[i] = u
            a_mean += a
            u_mean += u
        end
        a_mean /= K
        u_mean /= K
        for i in 1:K
            scores[i, 1, b] = Float32(-(a_vals[i] - a_mean) / phys.theta)
            scores[i, 2, b] = Float32(-(u_vals[i] - u_mean) / phys.theta)
        end
    end
    return scores
end

function standardized_physical_score(norm_samples::Array{Float32, 3}, stats::DataStats,
        phys::FHDPhysicalParams; velocity_floor::Float64=0.0)
    raw = denormalize_fhd_tensor(norm_samples, stats)
    raw_score = physical_score_raw(raw, phys; velocity_floor=velocity_floor)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(raw_score, 1), size(raw_score, 2), 1)
    return raw_score .* std_tensor
end

function score_from_projected_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return project_zero_modes(preds)
end

score_from_residual_model(model, batch, sigma::Real) = score_from_projected_model(model, batch, sigma)

function fill_projected_noise!(noise, rng::AbstractRNG)
    if noise isa CUDA.CuArray
        Random.randn!(CUDA.default_rng(), noise)
    else
        randn!(rng, noise)
    end
    project_zero_modes!(noise)
    return noise
end

function parameter_norm(model)
    total = 0.0
    for p in Flux.trainables(model)
        total += sum(abs2, to_host(p))
    end
    return sqrt(total)
end

function projection_matrix(K::Int)
    Psite = Matrix{Float64}(I, K, K) .- 1.0 / K
    P = zeros(Float64, 2K, 2K)
    P[1:K, 1:K] .= Psite
    P[K+1:2K, K+1:2K] .= Psite
    return P
end

function nonzero_mode_basis(K::Int)
    D = 2K
    A = zeros(Float64, D, 2K - 2)
    col = 1
    for i in 1:(K - 1)
        A[i, col] = 1.0
        A[K, col] = -1.0
        col += 1
    end
    for i in 1:(K - 1)
        A[K + i, col] = 1.0
        A[2K, col] = -1.0
        col += 1
    end
    return Matrix(qr(A).Q)[:, 1:(2K - 2)]
end

function stein_matrix_from_scores(score_norm::Array{Float32, 3}, noisy_norm::Array{Float32, 3})
    K, _, B = size(score_norm)
    sflat = raw_flat_batch(score_norm)
    xflat = raw_flat_batch(noisy_norm)
    return -Matrix{Float64}(sflat * transpose(xflat)) ./ max(B, 1)
end

function evaluate_stationary_score_norm(model, norm_samples::Array{Float32, 3}, sigma::Float32,
        device::ExecutionDevice; batch_size::Int=8192)
    Flux.testmode!(model)
    K, C, B = size(norm_samples)
    out = Array{Float32}(undef, K, C, B)
    for lo in 1:batch_size:B
        hi = min(lo + batch_size - 1, B)
        batch = move_array(copy(@view norm_samples[:, :, lo:hi]), device)
        score = score_from_projected_model(model, batch, sigma)
        out[:, :, lo:hi] .= to_host(score)
    end
    return out
end

function normalized_score_to_raw(score_norm::Array{Float32, 3}, stats::DataStats)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(score_norm, 1), size(score_norm, 2), 1)
    return score_norm ./ std_tensor
end

function exact_score_metrics(model, dataset::NormalizedDataset, phys::FHDPhysicalParams,
        sigma::Float32, nsamples::Int, rng::AbstractRNG, device::ExecutionDevice;
        safe_density_min::Float64=0.05, batch_size::Int=8192)
    n = min(nsamples, length(dataset))
    keep = randperm(rng, length(dataset))[1:n]
    clean = dataset.data[:, :, keep]
    pred = evaluate_stationary_score_norm(model, clean, sigma, device; batch_size=batch_size)
    exact_ideal = standardized_physical_score(clean, dataset.stats, phys; velocity_floor=0.0)
    exact_sim = standardized_physical_score(clean, dataset.stats, phys; velocity_floor=phys.velocity_density_floor)
    raw = denormalize_fhd_tensor(clean, dataset.stats)
    safe = vec([minimum(@view raw[:, 1, b]) > safe_density_min for b in 1:n])
    return score_metric_dict(pred, exact_ideal, exact_sim, safe)
end

function score_metric_pair(pred, target)
    err = pred .- target
    rel = sqrt(sum(abs2, err) / max(sum(abs2, target), eps(Float64)))
    cosv = sum(pred .* target) / max(sqrt(sum(abs2, pred) * sum(abs2, target)), eps(Float64))
    return Float64(rel), Float64(cosv)
end

function score_metric_dict(pred, exact_ideal, exact_sim, safe::AbstractVector{Bool})
    rel_i, cos_i = score_metric_pair(pred, exact_ideal)
    rel_s, cos_s = score_metric_pair(pred, exact_sim)
    result = Dict{Symbol, Any}(
        :ideal_rel_rmse => rel_i,
        :ideal_cosine => cos_i,
        :sim_rel_rmse => rel_s,
        :sim_cosine => cos_s,
        :safe_count => count(identity, safe),
        :total_count => length(safe),
    )
    if any(safe)
        rel_is, cos_is = score_metric_pair(pred[:, :, safe], exact_ideal[:, :, safe])
        rel_ss, cos_ss = score_metric_pair(pred[:, :, safe], exact_sim[:, :, safe])
        result[:ideal_safe_rel_rmse] = rel_is
        result[:ideal_safe_cosine] = cos_is
        result[:sim_safe_rel_rmse] = rel_ss
        result[:sim_safe_cosine] = cos_ss
    else
        result[:ideal_safe_rel_rmse] = NaN
        result[:ideal_safe_cosine] = NaN
        result[:sim_safe_rel_rmse] = NaN
        result[:sim_safe_cosine] = NaN
    end
    return result
end

function build_score_unet(cfg::ScoreUNetConfig, norm::Symbol)
    adjusted = adjust_model_config_for_length(cfg, 8)
    return ProjectedScoreUNet(build_unet(adjusted; normalization=norm)), adjusted
end

function build_local_score_mlp(width::Int, nlayers::Int, activation)
    layers = Any[Dense(FHD_STATE_CHANNELS => width, activation)]
    for _ in 2:nlayers
        push!(layers, Dense(width => width, activation))
    end
    push!(layers, Dense(width => FHD_STATE_CHANNELS))
    return LocalProjectedScoreMLP(Chain(layers...))
end

function build_local_feature_score_mlp(stats::DataStats, phys::FHDPhysicalParams,
        width::Int, nlayers::Int, activation)
    layers = Any[Dense(4 => width, activation)]
    for _ in 2:nlayers
        push!(layers, Dense(width => width, activation))
    end
    push!(layers, Dense(width => FHD_STATE_CHANNELS))
    return LocalFeatureProjectedScoreMLP(Chain(layers...),
        Float32(mean(stats.mean[1, :])), Float32(mean(stats.std[1, :])),
        Float32(mean(stats.mean[2, :])), Float32(mean(stats.std[2, :])),
        1.0f-8, Float32(phys.velocity_density_floor))
end

function build_cond_unet(cfg::ScoreUNetConfig, norm::Symbol)
    adjusted = adjust_model_config_for_length(cfg, 8)
    return ReverseResidualScoreUNet(build_unet(adjusted; normalization=norm)), adjusted
end

function encode_time_features!(input, tval::Float32, site::Int, b::Int, first_channel::Int,
        features::AbstractString, nfreq::Int)
    input[site, first_channel, b] = tval
    if features == "fourier"
        ch = first_channel + 1
        t64 = Float64(tval)
        for freq in 1:nfreq
            angle = Float32(2.0 * pi * freq * t64)
            input[site, ch, b] = sin(angle)
            input[site, ch + 1, b] = cos(angle)
            ch += 2
        end
    end
    return nothing
end

function encode_cond_input!(input::AbstractArray{Float32, 3}, x0_norm::AbstractArray{Float32, 3},
        xt_norm::AbstractArray{Float32, 3}, tnorm::AbstractVector{Float32};
        time_features::AbstractString, time_fourier_frequencies::Int,
        include_delta_input::Bool)
    K, _, B = size(x0_norm)
    @inbounds for b in 1:B
        tv = tnorm[b]
        for i in 1:K
            input[i, 1, b] = x0_norm[i, 1, b]
            input[i, 2, b] = x0_norm[i, 2, b]
            input[i, 3, b] = xt_norm[i, 1, b]
            input[i, 4, b] = xt_norm[i, 2, b]
            first = 5
            if include_delta_input
                input[i, 5, b] = xt_norm[i, 1, b] - x0_norm[i, 1, b]
                input[i, 6, b] = xt_norm[i, 2, b] - x0_norm[i, 2, b]
                first = 7
            end
            encode_time_features!(input, tv, i, b, first, time_features, time_fourier_frequencies)
        end
    end
    return nothing
end

function refresh_delta_input_channels!(input::AbstractArray{Float32, 3})
    @views begin
        input[:, 5, :] .= input[:, 3, :] .- input[:, 1, :]
        input[:, 6, :] .= input[:, 4, :] .- input[:, 2, :]
    end
    return nothing
end

struct FHDPairSampler
    times::Vector{Float64}
    states::Array{Float32, 4}
    start_idx::Int
    save_dt::Float64
    K::Int
    D::Int
    lag_steps::Vector{Int}
    lag_times::Vector{Float64}
    lag_tnorm::Vector{Float32}
    tau_min::Float64
    tau_max::Float64
    decorrelation_time::Float64
end

function build_fhd_pair_sampler(path::AbstractString, burnin_fraction::Float64, tau_min::Float64,
        tau_max_decorrelation_multiples::Float64, lag_stride::Int)
    times, states = load_fhd_states(path)
    start_idx = burnin_start_index(length(times), burnin_fraction)
    save_dt = times[2] - times[1]
    tD = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
    effective_tau_min = tau_min <= 0.0 ? save_dt : tau_min
    tau_max = min(tau_max_decorrelation_multiples * tD, times[end] - times[start_idx])
    min_lag = max(1, ceil(Int, effective_tau_min / save_dt - 1e-9))
    max_lag = min(length(times) - start_idx - 1, floor(Int, tau_max / save_dt + 1e-9))
    require_condition(max_lag >= min_lag, "No lag steps available in requested tau range.")
    lag_steps = collect(min_lag:lag_stride:max_lag)
    lag_times = lag_steps .* save_dt
    denom = max(tau_max - effective_tau_min, eps(Float64))
    lag_tnorm = Float32.((lag_times .- effective_tau_min) ./ denom)
    K = size(states, 2)
    return FHDPairSampler(times, states, start_idx, save_dt, K, 2K, lag_steps, lag_times,
        lag_tnorm, effective_tau_min, tau_max, tD)
end

function sample_pair_batch!(x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tnorm::Vector{Float32}, sampler::FHDPairSampler, rng::AbstractRNG)
    nt, K, _, ntraj = size(sampler.states)
    nlags = length(sampler.lag_steps)
    B = size(x0, 3)
    @inbounds for b in 1:B
        lp = rand(rng, 1:nlags)
        lag = sampler.lag_steps[lp]
        tr = rand(rng, 1:ntraj)
        t = rand(rng, sampler.start_idx:(nt - lag))
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        tnorm[b] = sampler.lag_tnorm[lp]
    end
    return nothing
end

function random_lag_pairs(sampler::FHDPairSampler, lag::Int, npairs::Int, rng::AbstractRNG;
        centered_window::Int=0)
    nt, K, _, ntraj = size(sampler.states)
    lower = sampler.start_idx
    upper = nt - lag - centered_window
    lower = max(lower, sampler.start_idx + centered_window)
    require_condition(upper >= lower, "Requested lag/window exceeds available trajectory.")
    x0 = Array{Float32}(undef, K, 2, npairs)
    xt = Array{Float32}(undef, K, 2, npairs)
    xp = centered_window > 0 ? Array{Float32}(undef, K, 2, npairs) : Array{Float32}(undef, 0, 0, 0)
    xm = centered_window > 0 ? Array{Float32}(undef, K, 2, npairs) : Array{Float32}(undef, 0, 0, 0)
    @inbounds for b in 1:npairs
        tr = rand(rng, 1:ntraj)
        t = rand(rng, lower:upper)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        if centered_window > 0
            xp[:, :, b] .= sampler.states[t + lag + centered_window, :, :, tr]
            xm[:, :, b] .= sampler.states[t + lag - centered_window, :, :, tr]
        end
    end
    return x0, xt, xp, xm
end

function evaluate_transition_score_norm(model, x0_raw::Array{Float32, 3}, xt_raw::Array{Float32, 3},
        tnorm::Float32, stats::DataStats, sigma::Float32, device::ExecutionDevice;
        batch_size::Int, time_features::AbstractString, time_fourier_frequencies::Int,
        include_delta_input::Bool)
    Flux.testmode!(model)
    K, _, B = size(x0_raw)
    channels = cond_input_channels(time_features, time_fourier_frequencies;
        include_delta_input=include_delta_input)
    out = Array{Float32}(undef, K, 2, B)
    for lo in 1:batch_size:B
        hi = min(lo + batch_size - 1, B)
        x0n = apply_fhd_stats(copy(@view x0_raw[:, :, lo:hi]), stats)
        xtn = apply_fhd_stats(copy(@view xt_raw[:, :, lo:hi]), stats)
        inp = Array{Float32}(undef, K, channels, hi - lo + 1)
        encode_cond_input!(inp, x0n, xtn, fill(tnorm, hi - lo + 1);
            time_features=time_features,
            time_fourier_frequencies=time_fourier_frequencies,
            include_delta_input=include_delta_input)
        score = score_from_residual_model(model, move_array(inp, device), sigma)
        out[:, :, lo:hi] .= to_host(score)
    end
    return out
end

function load_stats_from_bson(blob)
    stats_obj = haskey(blob, :stats) ? blob[:stats] : blob["stats"]
    if stats_obj isa DataStats
        return stats_obj
    elseif stats_obj isa Dict
        return DataStats(Float32.(stats_obj[:mean]), Float32.(stats_obj[:std]))
    else
        error("Unsupported stats payload in BSON.")
    end
end

function dict_get(d, key::Symbol)
    haskey(d, key) && return d[key]
    sk = String(key)
    haskey(d, sk) && return d[sk]
    error("BSON key $(key) not found.")
end

function sympart(A::AbstractMatrix)
    return 0.5 .* (Matrix(A) .+ transpose(Matrix(A)))
end

function edge_viscosity(rho::AbstractVector{<:Real}, phys::FHDPhysicalParams)
    N = length(rho)
    eta = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        ip = periodic(i + 1, N)
        rho_edge = max(0.5 * (Float64(rho[i]) + Float64(rho[ip])), 1.0e-14)
        eta[i] = phys.eta0 * (rho_edge / phys.rho0)^phys.zeta
    end
    return eta
end

function true_D_action!(dest_m::AbstractVector{Float64}, rho::AbstractVector{<:Real},
        v_m::AbstractVector{<:Real}, phys::FHDPhysicalParams)
    N = length(rho)
    eta = edge_viscosity(rho, phys)
    h2 = phys.dx^2
    fill!(dest_m, 0.0)
    @inbounds for i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        dest_m[i] = phys.theta / h2 *
            ((eta[i] + eta[im]) * Float64(v_m[i]) - eta[i] * Float64(v_m[ip]) - eta[im] * Float64(v_m[im]))
    end
    return nothing
end

function Lh_action!(out_rho::AbstractVector{Float64}, out_m::AbstractVector{Float64},
        rho::AbstractVector{<:Real}, m::AbstractVector{<:Real},
        v_rho::AbstractVector{<:Real}, v_m::AbstractVector{<:Real}, phys::FHDPhysicalParams)
    N = length(rho)
    h = phys.dx
    @inbounds for i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        beta_im = Float64(v_m[im])
        beta_i = Float64(v_m[i])
        beta_ip = Float64(v_m[ip])
        alpha_im = Float64(v_rho[im])
        alpha_ip = Float64(v_rho[ip])
        rho_i = Float64(rho[i])
        rho_im = Float64(rho[im])
        rho_ip = Float64(rho[ip])
        m_i = Float64(m[i])
        m_im = Float64(m[im])
        m_ip = Float64(m[ip])
        out_rho[i] = -((rho_i * beta_i + rho_ip * beta_ip) -
                       (rho_im * beta_im + rho_i * beta_i)) / (2h)
        grad_alpha = (alpha_ip - alpha_im) / (2h)
        grad_beta = (beta_ip - beta_im) / (2h)
        flux_right = 0.5 * (m_i * beta_i + m_ip * beta_ip)
        flux_left = 0.5 * (m_im * beta_im + m_i * beta_i)
        out_m[i] = -rho_i * grad_alpha - m_i * grad_beta - (flux_right - flux_left) / h
    end
    return nothing
end

function true_mobility_transpose_action_sample(rho::AbstractVector{<:Real}, m::AbstractVector{<:Real},
        score::AbstractVector{<:Real}, phys::FHDPhysicalParams)
    N = length(rho)
    sr = @view score[1:N]
    sm = @view score[N+1:2N]
    d_m = zeros(Float64, N)
    true_D_action!(d_m, rho, sm, phys)
    l_r = zeros(Float64, N)
    l_m = zeros(Float64, N)
    Lh_action!(l_r, l_m, rho, m, sr, sm, phys)
    # M' = D - R and R = -Theta L_h, hence M' v = D v + Theta L_h v.
    out = Vector{Float64}(undef, 2N)
    @inbounds for i in 1:N
        out[i] = phys.theta * l_r[i]
        out[N + i] = d_m[i] + phys.theta * l_m[i]
    end
    return out
end

function true_mobility_matrix_sample(z::AbstractMatrix{<:Real}, phys::FHDPhysicalParams)
    N = size(z, 1)
    D = 2N
    M = Matrix{Float64}(undef, D, D)
    rho = Float64.(@view z[:, 1])
    m = Float64.(@view z[:, 2])
    for j in 1:D
        e = zeros(Float64, D)
        e[j] = 1.0
        Mt_e = true_mobility_transpose_action_sample(rho, m, e, phys)
        M[j, :] .= Mt_e
    end
    return M
end

function block_profile(A::AbstractMatrix{<:Real}, K::Int)
    prof = zeros(Float64, K, 2, 2)
    counts = zeros(Int, K, 2, 2)
    @inbounds for i in 1:K, r in 0:(K - 1), a in 1:2, b in 1:2
        jsite = periodic(i + r, K)
        row = a == 1 ? i : K + i
        col = b == 1 ? jsite : K + jsite
        prof[r + 1, a, b] += Float64(A[row, col])
        counts[r + 1, a, b] += 1
    end
    prof ./= counts
    return prof
end

function matrix_from_block_profile(prof::Array{Float64, 3})
    K = size(prof, 1)
    A = zeros(Float64, 2K, 2K)
    @inbounds for i in 1:K, r in 0:(K - 1), a in 1:2, b in 1:2
        jsite = periodic(i + r, K)
        row = a == 1 ? i : K + i
        col = b == 1 ? jsite : K + jsite
        A[row, col] = prof[r + 1, a, b]
    end
    return A
end

function project_block_circulant(A::AbstractMatrix{<:Real}, K::Int)
    return matrix_from_block_profile(block_profile(A, K))
end

function polynomial_derivative_at(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
        x0::Real, degree::Int)
    n = length(xs)
    deg = min(degree, n - 1)
    X = Matrix{Float64}(undef, n, deg + 1)
    @inbounds for i in 1:n
        dx = Float64(xs[i]) - Float64(x0)
        X[i, 1] = 1.0
        for p in 1:deg
            X[i, p + 1] = X[i, p] * dx
        end
    end
    coeff = X \ Float64.(ys)
    return deg >= 1 ? coeff[2] : 0.0
end

function covariance_derivative_phi(sampler::FHDPairSampler, stats::DataStats;
        pairs_per_lag::Int, phi_fit_max_lag::Int, phi_fit_degree::Int, seed::Int,
        include_zero_lag::Bool=false, zero_lag_samples::Int=pairs_per_lag)
    K = sampler.K
    D = sampler.D
    L = min(phi_fit_max_lag, length(sampler.lag_steps))
    nfit = L + (include_zero_lag ? 1 : 0)
    taus = Vector{Float64}(undef, nfit)
    C = Array{Float64}(undef, nfit, D, D)
    mu = Float64.(mean_flat(stats))
    rng = MersenneTwister(seed)
    offset = 0
    if include_zero_lag
        n0 = max(1, zero_lag_samples)
        raw = sample_state_tensor(sampler.states, sampler.start_idx, n0, rng)
        x0f = Float64.(raw_flat_batch(raw))
        x0f .-= mu
        taus[1] = 0.0
        C[1, :, :] .= (x0f * transpose(x0f)) ./ n0
        offset = 1
        @printf("Phi short-lag covariance tau 0 (zero lag, %d samples)\n", n0)
    end
    for ell in 1:L
        lag = sampler.lag_steps[ell]
        x0, xt, _, _ = random_lag_pairs(sampler, lag, pairs_per_lag, rng)
        x0f = Float64.(raw_flat_batch(x0))
        xtf = Float64.(raw_flat_batch(xt))
        x0f .-= mu
        xtf .-= mu
        taus[offset + ell] = sampler.lag_times[ell]
        C[offset + ell, :, :] .= (xtf * transpose(x0f)) ./ pairs_per_lag
        @printf("Phi short-lag covariance tau %.5g (%d/%d)\n", sampler.lag_times[ell], ell, L)
    end
    Cdot0 = zeros(Float64, D, D)
    @inbounds for row in 1:D, col in 1:D
        Cdot0[row, col] = polynomial_derivative_at(taus, C[:, row, col], 0.0, phi_fit_degree)
    end
    return taus, C, Cdot0, -Cdot0
end

function estimate_raw_stein_matrix(model, sampler::FHDPairSampler, stats::DataStats, sigma::Float32,
        device::ExecutionDevice; nsamples::Int, batch_size::Int, seed::Int)
    rng = MersenneTwister(seed)
    raw = sample_state_tensor(sampler.states, sampler.start_idx, nsamples, rng)
    normed = apply_fhd_stats(raw, stats)
    snorm = evaluate_stationary_score_norm(model, normed, sigma, device; batch_size=batch_size)
    sraw = Float64.(raw_flat_batch(normalized_score_to_raw(snorm, stats)))
    xraw = Float64.(raw_flat_batch(raw))
    mu = Float64.(mean_flat(stats))
    xraw .-= mu
    return -(sraw * transpose(xraw)) ./ size(xraw, 2)
end

function true_mean_mobility(sampler::FHDPairSampler, phys::FHDPhysicalParams;
        nsamples::Int, seed::Int)
    rng = MersenneTwister(seed)
    raw = sample_state_tensor(sampler.states, sampler.start_idx, nsamples, rng)
    D = sampler.D
    M = zeros(Float64, D, D)
    @inbounds for b in 1:nsamples
        M .+= true_mobility_matrix_sample(@view(raw[:, :, b]), phys)
    end
    return M ./ nsamples
end

function tangent_eigs(A::AbstractMatrix{<:Real}, K::Int)
    Q = nonzero_mode_basis(K)
    return eigen(Symmetric(Q' * sympart(A) * Q)).values
end

function agreement_metrics(reference::AbstractArray, estimate::AbstractArray)
    r = vec(Float64.(reference))
    e = vec(Float64.(estimate))
    mask = isfinite.(r) .& isfinite.(e)
    r = r[mask]
    e = e[mask]
    rel = sqrt(sum(abs2, e .- r) / max(sum(abs2, r), eps(Float64)))
    corr = cor(r, e)
    return Dict(:relative_rmse => rel, :correlation => corr)
end

function save_figure_checked(path::AbstractString, fig::Figure)
    ensure_parent_dir(path)
    save(path, fig)
    @printf("Saved figure to %s\n", path)
    return nothing
end
