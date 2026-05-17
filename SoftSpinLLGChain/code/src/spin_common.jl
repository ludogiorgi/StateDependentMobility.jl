import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SCOREUNET_PROJECT = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl"))
const SCOREUNET_SRC = joinpath(SCOREUNET_PROJECT, "src")
const SPIN_CHANNELS = 3

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

function ensure_packages(packages::Vector{String})
    deps = Pkg.project().dependencies
    missing = [pkg for pkg in packages if !haskey(deps, pkg)]
    isempty(missing) || Pkg.add(missing)
    return nothing
end

ensure_packages(["BSON", "CUDA", "cuDNN", "Flux", "Functors", "GLMakie",
    "HDF5", "KernelDensity", "NNlib", "ProgressMeter", "TOML"])

function display_is_usable(display::AbstractString)
    xdpyinfo = Sys.which("xdpyinfo")
    xdpyinfo === nothing && return true
    try
        run(pipeline(`$xdpyinfo -display $display`, stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

function start_xvfb!()
    xvfb = Sys.which("Xvfb")
    xvfb === nothing && return nothing
    for display_id in 151:190
        display = ":" * string(display_id)
        isfile("/tmp/.X$(display_id)-lock") && continue
        run(pipeline(`$xvfb $display -screen 0 1920x1200x24 -nolisten tcp`,
            stdout=devnull, stderr=devnull); wait=false)
        sleep(1.0)
        if display_is_usable(display)
            ENV["DISPLAY"] = display
            ENV["STATEDEP_XVFB_DISPLAY"] = display
            return nothing
        end
    end
    return nothing
end

if haskey(ENV, "DISPLAY") && !display_is_usable(ENV["DISPLAY"])
    delete!(ENV, "DISPLAY")
end
if !haskey(ENV, "DISPLAY") || isempty(ENV["DISPLAY"])
    start_xvfb!()
end

using BSON
using CUDA
using cuDNN
using Flux
using Functors
using GLMakie
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

const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
isfile(STYLE_FILE) && include(STYLE_FILE)
GLMakie.activate!()

if !isdefined(@__MODULE__, :STYLE_PRIMARY)
    const STYLE_PRIMARY = :dodgerblue4
    const STYLE_SECONDARY = :darkorange2
    const STYLE_ACCENT = :seagreen4
    const STYLE_HIGHLIGHT = :firebrick3
    const STYLE_REFERENCE = :gray35
    const STYLE_DIVERGING = :balance
end

require_condition(condition::Bool, message::String) = condition || error(message)
resolve_path(base::AbstractString, path::AbstractString) =
    isabspath(path) ? path : normpath(joinpath(base, path))
ensure_parent_dir(path::AbstractString) = (mkpath(dirname(path)); nothing)
to_host(x) = x isa AbstractArray && !(x isa Array) ? Array(x) : x
periodic(i::Int, N::Int) = mod1(i, N)

Base.@kwdef struct SpinParams
    N::Int = 12
    lambda::Float64 = 10.0
    mstar::Float64 = 1.0
    J::Float64 = 0.35
    K::Float64 = 0.50
    theta::Float64 = 0.20
    gamma::Float64 = 4.0
    alpha_perp::Float64 = 0.25
    alpha_parallel::Float64 = 0.03
    eps::Float64 = 1.0e-2
end

function spin_params_from_table(tbl)
    return SpinParams(
        N=Int(get(tbl, "N", 12)),
        lambda=Float64(get(tbl, "lambda", 10.0)),
        mstar=Float64(get(tbl, "mstar", 1.0)),
        J=Float64(get(tbl, "J", 0.35)),
        K=Float64(get(tbl, "K", 0.50)),
        theta=Float64(get(tbl, "Theta", 0.20)),
        gamma=Float64(get(tbl, "gamma", 4.0)),
        alpha_perp=Float64(get(tbl, "alpha_perp", 0.25)),
        alpha_parallel=Float64(get(tbl, "alpha_parallel", 0.03)),
        eps=Float64(get(tbl, "eps", 1.0e-2)),
    )
end

meq(p::SpinParams) = sqrt(p.mstar^2 + p.K / p.lambda)
state_dim(p::SpinParams) = SPIN_CHANNELS * p.N

function channel_names()
    return ["mx", "my", "mz"]
end

function flat_order(p::SpinParams)
    return ["site$(i)_$(channel_names()[c])" for i in 1:p.N for c in 1:SPIN_CHANNELS]
end

function flatten_state(x::AbstractMatrix{<:Real})
    N = size(x, 1)
    out = Vector{Float64}(undef, SPIN_CHANNELS * N)
    @inbounds for i in 1:N, c in 1:SPIN_CHANNELS
        out[(i - 1) * SPIN_CHANNELS + c] = Float64(x[i, c])
    end
    return out
end

function flatten_batch(x::Array{T, 3}) where {T<:Real}
    N, _, B = size(x)
    out = Matrix{T}(undef, SPIN_CHANNELS * N, B)
    @inbounds for b in 1:B, i in 1:N, c in 1:SPIN_CHANNELS
        out[(i - 1) * SPIN_CHANNELS + c, b] = x[i, c, b]
    end
    return out
end

function unflatten_batch(flat::AbstractMatrix{<:Real}, N::Int)
    B = size(flat, 2)
    out = Array{Float64}(undef, N, SPIN_CHANNELS, B)
    @inbounds for b in 1:B, i in 1:N, c in 1:SPIN_CHANNELS
        out[i, c, b] = Float64(flat[(i - 1) * SPIN_CHANNELS + c, b])
    end
    return out
end

function cross3(a1, a2, a3, b1, b2, b3)
    return (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)
end

function effective_field!(H::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, p::SpinParams)
    N = p.N
    @inbounds for i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        x1, x2, x3 = x[i, 1], x[i, 2], x[i, 3]
        r2 = x1 * x1 + x2 * x2 + x3 * x3
        common = -p.lambda * (r2 - p.mstar^2)
        H[i, 1] = common * x1 + p.J * (x[im, 1] + x[ip, 1] - 2.0 * x1)
        H[i, 2] = common * x2 + p.J * (x[im, 2] + x[ip, 2] - 2.0 * x2)
        H[i, 3] = common * x3 + p.J * (x[im, 3] + x[ip, 3] - 2.0 * x3) + p.K * x3
    end
    return H
end

function potential_energy(x::AbstractMatrix{<:Real}, p::SpinParams)
    U = 0.0
    @inbounds for i in 1:p.N
        ip = periodic(i + 1, p.N)
        r2 = sum(abs2, @view x[i, :])
        diff2 = sum(abs2, @view(x[ip, :]) .- @view(x[i, :]))
        U += 0.25 * p.lambda * (r2 - p.mstar^2)^2 + 0.5 * p.J * diff2 - 0.5 * p.K * x[i, 3]^2
    end
    return U
end

function apply_A(x1, x2, x3, v1, v2, v3, p::SpinParams)
    r2 = x1 * x1 + x2 * x2 + x3 * x3
    dotmv = x1 * v1 + x2 * v2 + x3 * v3
    base = p.eps + p.alpha_perp * r2
    coeff = p.alpha_parallel - p.alpha_perp
    return (base * v1 + coeff * x1 * dotmv,
            base * v2 + coeff * x2 * dotmv,
            base * v3 + coeff * x3 * dotmv)
end

function apply_sqrt_A(x1, x2, x3, z1, z2, z3, p::SpinParams)
    r2 = x1 * x1 + x2 * x2 + x3 * x3
    if r2 < 1.0e-24
        scale = sqrt(p.eps)
        return (scale * z1, scale * z2, scale * z3)
    end
    lam_perp = p.eps + p.alpha_perp * r2
    lam_para = p.eps + p.alpha_parallel * r2
    dotmz = x1 * z1 + x2 * z2 + x3 * z3
    invr2 = 1.0 / r2
    para1 = x1 * dotmz * invr2
    para2 = x2 * dotmz * invr2
    para3 = x3 * dotmz * invr2
    sperp = sqrt(lam_perp)
    spara = sqrt(lam_para)
    return (sperp * (z1 - para1) + spara * para1,
            sperp * (z2 - para2) + spara * para2,
            sperp * (z3 - para3) + spara * para3)
end

function drift!(out::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, p::SpinParams,
        H::AbstractMatrix{Float64})
    effective_field!(H, x, p)
    div_coeff = p.theta * (4.0 * p.alpha_parallel - 2.0 * p.alpha_perp)
    @inbounds for i in 1:p.N
        x1, x2, x3 = x[i, 1], x[i, 2], x[i, 3]
        h1, h2, h3 = H[i, 1], H[i, 2], H[i, 3]
        a1, a2, a3 = apply_A(x1, x2, x3, h1, h2, h3, p)
        c1, c2, c3 = cross3(x1, x2, x3, h1, h2, h3)
        out[i, 1] = a1 - p.gamma * c1 + div_coeff * x1
        out[i, 2] = a2 - p.gamma * c2 + div_coeff * x2
        out[i, 3] = a3 - p.gamma * c3 + div_coeff * x3
    end
    return out
end

function em_step!(x::AbstractMatrix{Float64}, p::SpinParams, dt::Float64,
        rng::AbstractRNG, work::NamedTuple)
    drift!(work.drift, x, p, work.field)
    sq = sqrt(2.0 * p.theta * dt)
    @inbounds for i in 1:p.N
        z1, z2, z3 = randn(rng), randn(rng), randn(rng)
        n1, n2, n3 = apply_sqrt_A(x[i, 1], x[i, 2], x[i, 3], z1, z2, z3, p)
        x[i, 1] += dt * work.drift[i, 1] + sq * n1
        x[i, 2] += dt * work.drift[i, 2] + sq * n2
        x[i, 3] += dt * work.drift[i, 3] + sq * n3
    end
    return x
end

function make_work(p::SpinParams)
    return (drift=zeros(Float64, p.N, SPIN_CHANNELS),
        field=zeros(Float64, p.N, SPIN_CHANNELS))
end

function analytic_score_raw(samples::Array{Float32, 3}, p::SpinParams)
    N, _, B = size(samples)
    out = Array{Float32}(undef, N, SPIN_CHANNELS, B)
    H = zeros(Float64, N, SPIN_CHANNELS)
    x = zeros(Float64, N, SPIN_CHANNELS)
    @inbounds for b in 1:B
        x .= Float64.(@view samples[:, :, b])
        effective_field!(H, x, p)
        out[:, :, b] .= Float32.(H ./ p.theta)
    end
    return out
end

function true_mobility_matrix(x::AbstractMatrix{<:Real}, p::SpinParams)
    D = state_dim(p)
    M = zeros(Float64, D, D)
    @inbounds for i in 1:p.N
        x1, x2, x3 = Float64(x[i, 1]), Float64(x[i, 2]), Float64(x[i, 3])
        r2 = x1 * x1 + x2 * x2 + x3 * x3
        A = p.eps * Matrix{Float64}(I, 3, 3) .+
            p.alpha_perp .* (r2 .* Matrix{Float64}(I, 3, 3) .- [x1, x2, x3] * transpose([x1, x2, x3])) .+
            p.alpha_parallel .* ([x1, x2, x3] * transpose([x1, x2, x3]))
        R = -p.gamma * p.theta .* [0.0 -x3 x2; x3 0.0 -x1; -x2 x1 0.0]
        block = p.theta .* A .+ R
        rows = ((i - 1) * 3 + 1):(i * 3)
        M[rows, rows] .= block
    end
    return M
end

function channel_shared_stats(samples::Array{Float32, 3})
    N, C, _ = size(samples)
    means = Array{Float32}(undef, C, N)
    stds = Array{Float32}(undef, C, N)
    for c in 1:C
        vals = @view samples[:, c, :]
        mu = mean(Float64, vals)
        n = length(vals)
        sig = sqrt(sum(x -> abs2(Float64(x) - mu), vals) / max(n - 1, 1))
        means[c, :] .= Float32(mu)
        stds[c, :] .= max(Float32(sig), sqrt(eps(Float32)))
    end
    return DataStats(means, stds)
end

function apply_stats_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return (samples .- mean_tensor) ./ std_tensor
end

function denormalize_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return samples .* std_tensor .+ mean_tensor
end

function std_flat(stats::DataStats)
    N = size(stats.std, 2)
    out = Vector{Float32}(undef, 3N)
    @inbounds for i in 1:N, c in 1:3
        out[(i - 1) * 3 + c] = stats.std[c, i]
    end
    return out
end

function mean_flat(stats::DataStats)
    N = size(stats.mean, 2)
    out = Vector{Float32}(undef, 3N)
    @inbounds for i in 1:N, c in 1:3
        out[(i - 1) * 3 + c] = stats.mean[c, i]
    end
    return out
end

function normalized_score_to_raw(score_norm::Array{Float32, 3}, stats::DataStats)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(score_norm, 1), size(score_norm, 2), 1)
    return score_norm ./ std_tensor
end

function standardized_analytic_score(norm_samples::Array{Float32, 3}, stats::DataStats, p::SpinParams)
    raw = denormalize_tensor(norm_samples, stats)
    raw_score = analytic_score_raw(raw, p)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(raw_score, 1), size(raw_score, 2), 1)
    return raw_score .* std_tensor
end

struct SpinScoreUNet{M}
    backbone::M
end
Functors.@functor SpinScoreUNet (backbone,)
(model::SpinScoreUNet)(x) = @view model.backbone(x)[:, 1:SPIN_CHANNELS, :]

struct DirectSpinScoreUNet{M}
    backbone::M
end
Functors.@functor DirectSpinScoreUNet (backbone,)
(model::DirectSpinScoreUNet)(x) = @view model.backbone(x)[:, 1:SPIN_CHANNELS, :]

function spin_r2_feature(x)
    r2 = sum(abs2, x; dims=2)
    return cat(x, r2; dims=2)
end

struct SpinR2ScoreUNet{M}
    backbone::M
end
Functors.@functor SpinR2ScoreUNet (backbone,)
(model::SpinR2ScoreUNet)(x) = @view model.backbone(spin_r2_feature(x))[:, 1:SPIN_CHANNELS, :]

struct DirectSpinR2ScoreUNet{M}
    backbone::M
end
Functors.@functor DirectSpinR2ScoreUNet (backbone,)
(model::DirectSpinR2ScoreUNet)(x) = @view model.backbone(spin_r2_feature(x))[:, 1:SPIN_CHANNELS, :]

struct PhysicalFeatureScore{T}
    coeff::T
    mean::T
    std::T
end
Functors.@functor PhysicalFeatureScore (coeff, mean, std)

function periodic_laplacian_sites(x)
    return circshift(x, (1, 0, 0)) .+ circshift(x, (-1, 0, 0)) .- 2 .* x
end

function (model::PhysicalFeatureScore)(z)
    N, C, B = size(z)
    mean_tensor = reshape(permutedims(model.mean, (2, 1)), N, C, 1)
    std_tensor = reshape(permutedims(model.std, (2, 1)), N, C, 1)
    raw = z .* std_tensor .+ mean_tensor
    r2 = sum(abs2, raw; dims=2)
    lap = periodic_laplacian_sites(raw)
    out = similar(z)
    @inbounds for c in 1:SPIN_CHANNELS
        rc = @view raw[:, c:c, :]
        lc = @view lap[:, c:c, :]
        sc = @view std_tensor[:, c:c, :]
        c1 = reshape(@view(model.coeff[c:c, 1:1]), 1, 1, 1)
        c2 = reshape(@view(model.coeff[c:c, 2:2]), 1, 1, 1)
        c3 = reshape(@view(model.coeff[c:c, 3:3]), 1, 1, 1)
        c4 = reshape(@view(model.coeff[c:c, 4:4]), 1, 1, 1)
        @views out[:, c:c, :] .= sc .* (
            c1 .* rc .+
            c2 .* (r2 .* rc) .+
            c3 .* lc .+
            c4 .* ((r2 .* r2) .* rc)
        )
    end
    return out
end

struct SpinConditionalResidualUNet{M}
    backbone::M
end
Functors.@functor SpinConditionalResidualUNet (backbone,)
(model::SpinConditionalResidualUNet)(x) = @view model.backbone(x)[:, 1:SPIN_CHANNELS, :]

struct PhysicalCondResidualMLP{M}
    mlp::M
    time_fourier_frequencies::Int
    include_delta_input::Bool
    include_tau_scalar::Bool
    output_scale::Float32
end
Functors.@functor PhysicalCondResidualMLP (mlp,)

function physical_cond_feature_dim(nfreq::Int; include_tau_scalar::Bool)
    return 44 + (include_tau_scalar ? 1 : 0) + 2nfreq
end

function cond_time_offsets(model::PhysicalCondResidualMLP)
    offset = 6 + (model.include_delta_input ? 3 : 0)
    tau_index = model.include_tau_scalar ? offset + 1 : 0
    time_offset = offset + (model.include_tau_scalar ? 1 : 0)
    return tau_index, time_offset
end

function normalized_cross_features(x0, xt)
    @views begin
        c1 = x0[:, 2:2, :] .* xt[:, 3:3, :] .- x0[:, 3:3, :] .* xt[:, 2:2, :]
        c2 = x0[:, 3:3, :] .* xt[:, 1:1, :] .- x0[:, 1:1, :] .* xt[:, 3:3, :]
        c3 = x0[:, 1:1, :] .* xt[:, 2:2, :] .- x0[:, 2:2, :] .* xt[:, 1:1, :]
        return cat(c1, c2, c3; dims=2)
    end
end

function physical_cond_features(model::PhysicalCondResidualMLP, input)
    x0 = @view input[:, 1:3, :]
    xt = @view input[:, 4:6, :]
    delta = model.include_delta_input ? @view(input[:, 7:9, :]) : xt .- x0
    tau_index, time_offset = cond_time_offsets(model)
    time = @view input[:, (time_offset + 1):(time_offset + 2 * model.time_fourier_frequencies), :]
    lap0 = periodic_laplacian_sites(x0)
    lapt = periodic_laplacian_sites(xt)
    lapd = periodic_laplacian_sites(delta)
    r20 = sum(abs2, x0; dims=2)
    r2t = sum(abs2, xt; dims=2)
    dr2 = r2t .- r20
    cross = normalized_cross_features(x0, xt)
    tau_for_scale = model.include_tau_scalar ?
        max.(@view(input[:, tau_index:tau_index, :]), eltype(input)(1.0f-3)) :
        (r20 .* zero(eltype(input)) .+ one(eltype(input)))
    inv_tau = one(eltype(input)) ./ tau_for_scale
    inv_sqrt_tau = one(eltype(input)) ./ sqrt.(tau_for_scale)
    if model.include_tau_scalar
        tau = @view input[:, tau_index:tau_index, :]
        return cat(x0, xt, delta,
            lap0, lapt, lapd,
            x0 .* r20, xt .* r2t, delta .* dr2,
            cross, r20, r2t, dr2,
            delta .* inv_tau, delta .* inv_sqrt_tau, lapd .* inv_tau,
            inv_tau, inv_sqrt_tau,
            tau, time; dims=2)
    else
        return cat(x0, xt, delta,
            lap0, lapt, lapd,
            x0 .* r20, xt .* r2t, delta .* dr2,
            cross, r20, r2t, dr2,
            delta .* inv_tau, delta .* inv_sqrt_tau, lapd .* inv_tau,
            inv_tau, inv_sqrt_tau,
            time; dims=2)
    end
end

function (model::PhysicalCondResidualMLP)(input)
    feats = physical_cond_features(model, input)
    N, F, B = size(feats)
    flat = reshape(permutedims(feats, (2, 1, 3)), F, N * B)
    y = model.mlp(flat)
    out = permutedims(reshape(y, SPIN_CHANNELS, N, B), (2, 1, 3))
    return out .* eltype(out)(model.output_scale)
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
    s == "none" && return :none
    s == "groupnorm" && return :groupnorm
    s == "layernorm" && return :layernorm
    s == "batchnorm" && error("BatchNorm is forbidden for this score benchmark.")
    error("Unsupported normalization=$(name).")
end

function max_compatible_unet_levels(length_dim::Int)
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

function score_input_feature_channels(input_features::Symbol)
    input_features == :spin && return SPIN_CHANNELS
    input_features == :spin_r2 && return SPIN_CHANNELS + 1
    error("Unsupported stationary score input_features=$(input_features).")
end

function build_spin_unet(cfg::ScoreUNetConfig, norm::Symbol, N::Int;
        output_mode::Symbol=:noise, input_features::Symbol=:spin)
    adjusted0 = ScoreUNetConfig(
        in_channels=score_input_feature_channels(input_features),
        base_channels=cfg.base_channels,
        channel_multipliers=cfg.channel_multipliers,
        kernel_size=cfg.kernel_size,
        periodic=cfg.periodic,
        activation=cfg.activation,
        final_activation=cfg.final_activation,
    )
    adjusted = adjust_model_config_for_length(adjusted0, N)
    backbone = build_unet(adjusted; normalization=norm)
    if input_features == :spin
        output_mode == :noise && return SpinScoreUNet(backbone), adjusted
        output_mode == :score && return DirectSpinScoreUNet(backbone), adjusted
    elseif input_features == :spin_r2
        output_mode == :noise && return SpinR2ScoreUNet(backbone), adjusted
        output_mode == :score && return DirectSpinR2ScoreUNet(backbone), adjusted
    end
    error("Unsupported score output mode $(output_mode).")
end

function score_from_dsm_model(model, batch, sigma::Real)
    pred_pos = model(batch)
    pred_neg = model(-batch)
    raw_pos = pred_pos .* (-one(eltype(pred_pos)) / sigma)
    raw_neg = pred_neg .* (-one(eltype(pred_neg)) / sigma)
    return (raw_pos .- raw_neg) .* eltype(raw_pos)(0.5)
end

function score_from_dsm_model(model::DirectSpinScoreUNet, batch, sigma::Real)
    pred_pos = model(batch)
    pred_neg = model(-batch)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function score_from_dsm_model(model::DirectSpinR2ScoreUNet, batch, sigma::Real)
    pred_pos = model(batch)
    pred_neg = model(-batch)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function score_from_dsm_model(model::PhysicalFeatureScore, batch, sigma::Real)
    pred_pos = model(batch)
    pred_neg = model(-batch)
    return (pred_pos .- pred_neg) .* eltype(pred_pos)(0.5)
end

function evaluate_score_norm(model, norm_samples::Array{Float32, 3}, sigma::Float32,
        device::ExecutionDevice; batch_size::Int=8192)
    Flux.testmode!(model)
    N, C, B = size(norm_samples)
    out = Array{Float32}(undef, N, C, B)
    for lo in 1:batch_size:B
        hi = min(lo + batch_size - 1, B)
        batch = move_array(copy(@view norm_samples[:, :, lo:hi]), device)
        score = score_from_dsm_model(model, batch, sigma)
        out[:, :, lo:hi] .= to_host(score)
    end
    return out
end

function score_metric_pair(pred, target)
    err = pred .- target
    rel = sqrt(sum(abs2, err) / max(sum(abs2, target), eps(Float64)))
    cosv = sum(pred .* target) / max(sqrt(sum(abs2, pred) * sum(abs2, target)), eps(Float64))
    return Float64(rel), Float64(cosv)
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
    CUDA.has_cuda() || return nothing
    needle = lowercase(replace(strip(required_gpu_name), " " => ""))
    for (idx, dev) in enumerate(collect(CUDA.devices()))
        name = lowercase(replace(CUDA.name(dev), " " => ""))
        occursin(needle, name) && return idx - 1
    end
    return nothing
end

function detect_spin_device(request::AbstractString, required_gpu_name::AbstractString)
    req = strip(request)
    up = uppercase(req)
    up in ("AUTO", "GPU") && error("Ambiguous GPU selection is disabled; request the 2080ti explicitly.")
    startswith(up, "GPU:") || error("This benchmark requires an explicit GPU:N request.")
    requested = parse_single_gpu_request(req)
    requested === nothing && error("Expected a single explicit GPU index, got $(request).")
    smi_names = nvidia_smi_gpu_names()
    if haskey(smi_names, requested)
        smi_name = smi_names[requested]
        occursin(lowercase(strip(required_gpu_name)), lowercase(replace(smi_name, " " => ""))) ||
            error("Requested nvidia-smi GPU $(requested) is $(smi_name), not required_gpu_name=$(required_gpu_name).")
    end
    cuda_id = cuda_ordinal_for_required_name(required_gpu_name)
    cuda_id === nothing && error("CUDA.jl did not expose a device matching $(required_gpu_name).")
    return GPUDevice([cuda_id])
end

function activate_and_describe_device!(device::ExecutionDevice, requested::AbstractString, required::AbstractString)
    activate_device!(device)
    devices = collect(CUDA.devices())
    id = first(device.ids)
    name = CUDA.name(devices[id + 1])
    compact_name = lowercase(replace(name, " " => ""))
    require_condition(occursin(lowercase(replace(required, " " => "")), compact_name),
        "Selected CUDA device $(id) is $(name), not required_gpu_name=$(required).")
    smi = nvidia_smi_gpu_names()
    smi_idx = nothing
    for (idx, smi_name) in smi
        if smi_name == name
            smi_idx = idx
            break
        end
    end
    @printf("Training device request: %s\n", requested)
    if smi_idx === nothing
        @printf("Resolved CUDA device: ordinal %d, %s\n", id, name)
    else
        @printf("Resolved CUDA device: ordinal %d, %s (nvidia-smi index %d)\n", id, name, smi_idx)
    end
    return nothing
end

function load_spin_states(path::AbstractString)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    require_condition(ndims(states) == 4, "Expected states as time x site x channel x trajectory.")
    require_condition(size(states, 3) == SPIN_CHANNELS, "Expected three spin channels.")
    return times, states
end

burnin_start_index(nsaved::Int, frac::Float64) = clamp(1 + floor(Int, frac * (nsaved - 1)), 1, nsaved)

function sample_state_tensor(states::Array{Float32, 4}, start_idx::Int, nsamples::Int, rng::AbstractRNG)
    nt, N, C, ntraj = size(states)
    total = (nt - start_idx + 1) * ntraj
    n = nsamples <= 0 ? total : min(nsamples, total)
    out = Array{Float32}(undef, N, C, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tr = (linear ÷ (nt - start_idx + 1)) + 1
        out[:, :, s] .= states[t, :, :, tr]
    end
    return out
end

function load_spin_dataset(path::AbstractString, burnin_fraction::Float64, max_samples::Int,
        rng::AbstractRNG)
    times, states = load_spin_states(path)
    start = burnin_start_index(length(times), burnin_fraction)
    raw = sample_state_tensor(states, start, max_samples, rng)
    stats = channel_shared_stats(raw)
    normed = apply_stats_tensor(raw, stats)
    return NormalizedDataset(normed, stats), times, states, start
end

function sympart(A::AbstractMatrix)
    return 0.5 .* (Matrix(A) .+ transpose(Matrix(A)))
end

function skewpart(A::AbstractMatrix)
    return 0.5 .* (Matrix(A) .- transpose(Matrix(A)))
end

function block_profile(A::AbstractMatrix{<:Real}, N::Int)
    prof = zeros(Float64, N, 3, 3)
    counts = zeros(Int, N, 3, 3)
    @inbounds for i in 1:N, r in 0:(N - 1), a in 1:3, b in 1:3
        j = periodic(i + r, N)
        row = (i - 1) * 3 + a
        col = (j - 1) * 3 + b
        prof[r + 1, a, b] += Float64(A[row, col])
        counts[r + 1, a, b] += 1
    end
    return prof ./ counts
end

function matrix_from_block_profile(prof::Array{Float64, 3})
    N = size(prof, 1)
    A = zeros(Float64, 3N, 3N)
    @inbounds for i in 1:N, r in 0:(N - 1), a in 1:3, b in 1:3
        j = periodic(i + r, N)
        A[(i - 1) * 3 + a, (j - 1) * 3 + b] = prof[r + 1, a, b]
    end
    return A
end

project_block_circulant(A::AbstractMatrix{<:Real}, N::Int) = matrix_from_block_profile(block_profile(A, N))

function project_soft_spin_phi(A::AbstractMatrix{<:Real}, N::Int)
    block = zeros(Float64, 3, 3)
    @inbounds for i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        block .+= Float64.(A[rows, rows])
    end
    block ./= N
    transverse = 0.5 * (block[1, 1] + block[2, 2])
    axial = block[3, 3]
    xy_skew = 0.5 * (block[2, 1] - block[1, 2])
    projected_block = [transverse -xy_skew 0.0;
                       xy_skew transverse 0.0;
                       0.0 0.0 axial]
    P = zeros(Float64, 3N, 3N)
    @inbounds for i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        P[rows, rows] .= projected_block
    end
    return P, projected_block
end

function psd_project_symmetric(S::AbstractMatrix{<:Real}; floor::Float64=1.0e-9)
    ev = eigen(Symmetric(sympart(S)))
    vals = max.(ev.values, floor)
    return ev.vectors * Diagonal(vals) * transpose(ev.vectors), vals
end

function polynomial_derivative_at(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, x0::Real, degree::Int)
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

function agreement_metrics(reference::AbstractArray, estimate::AbstractArray)
    r = vec(Float64.(reference))
    e = vec(Float64.(estimate))
    mask = isfinite.(r) .& isfinite.(e)
    r = r[mask]
    e = e[mask]
    rel = sqrt(sum(abs2, e .- r) / max(sum(abs2, r), eps(Float64)))
    corrv = length(r) > 2 ? cor(r, e) : NaN
    return Dict(:relative_rmse => rel, :correlation => corrv)
end

function save_figure_checked(path::AbstractString, fig::Figure)
    ensure_parent_dir(path)
    save(path, fig)
    @printf("Saved figure to %s\n", path)
    return nothing
end

function figure_title!(fig::Figure, title::AbstractString; subtitle::AbstractString="")
    Label(fig[0, :], isempty(subtitle) ? title : string(title, "\n", subtitle);
        fontsize=24, font=:bold, tellwidth=false)
    return nothing
end
