#!/usr/bin/env julia

using BSON
using CUDA
using Flux
using Functors
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SCOREUNET_SRC = normpath(joinpath(REPO_ROOT, "ScoreUNet1D.jl", "src"))
const STYLE_FILE = normpath(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))
const STATE_CHANNELS = 2

include(joinpath(SCOREUNET_SRC, "architecture", "PeriodicConv.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "Blocks.jl"))
include(joinpath(SCOREUNET_SRC, "architecture", "UNet1D.jl"))

require_condition(cond::Bool, msg::AbstractString) = cond || error(msg)
resolve_path(base::AbstractString, path::AbstractString) =
    isabspath(path) ? path : normpath(joinpath(base, path))
ensure_parent_dir(path::AbstractString) = (mkpath(dirname(path)); nothing)
to_host(x) = x isa CUDA.CuArray ? Array(x) : x
periodic(i::Int, N::Int) = mod1(i, N)

function save_figure_checked(path::AbstractString, fig)
    ensure_parent_dir(path)
    save(path, fig)
    @printf("Saved figure to %s\n", path)
    return nothing
end

Base.@kwdef struct FHDPhys
    N::Int
    L::Float64
    dx::Float64
    rho0::Float64
    cs::Float64
    theta::Float64
    eta0::Float64
    zeta::Float64
    kappa::Float64
    velocity_density_floor::Float64
end

struct DataStats
    mean::Array{Float32, 2}
    std::Array{Float32, 2}
end

struct NormalizedDataset
    data::Array{Float32, 3}
    stats::DataStats
end
Base.length(ds::NormalizedDataset) = size(ds.data, 3)

struct ProjectedScoreUNet{M}
    backbone::M
end
Functors.@functor ProjectedScoreUNet (backbone,)

function project_zero_modes(x)
    return x .- (sum(x; dims=1) ./ eltype(x)(size(x, 1)))
end

function project_zero_modes!(x)
    x .-= sum(x; dims=1) ./ eltype(x)(size(x, 1))
    return x
end

function (model::ProjectedScoreUNet)(x)
    y = model.backbone(x)
    return project_zero_modes(@view y[:, 1:STATE_CHANNELS, :])
end

function activation_from_string(name::AbstractString)
    s = lowercase(strip(name))
    s == "gelu" && return Flux.gelu
    s == "swish" && return Flux.swish
    s == "relu" && return Flux.relu
    s == "tanh" && return tanh
    s == "identity" && return identity
    error("Unsupported activation: $(name)")
end

function normalization_from_string(name)
    s = lowercase(String(name))
    s == "none" && return :none
    s == "groupnorm" && return :groupnorm
    s == "batchnorm" && return :batchnorm
    error("Unsupported normalization: $(name)")
end

function nvidia_smi_gpu_names()
    smi = Sys.which("nvidia-smi")
    smi === nothing && return Dict{Int, String}()
    out = try
        read(`$smi --query-gpu=index,name --format=csv,noheader`, String)
    catch
        return Dict{Int, String}()
    end
    names = Dict{Int, String}()
    for line in split(chomp(out), '\n')
        isempty(strip(line)) && continue
        parts = split(line, ","; limit=2)
        length(parts) == 2 || continue
        names[parse(Int, strip(parts[1]))] = strip(parts[2])
    end
    return names
end

function nvidia_smi_gpu_uuids()
    smi = Sys.which("nvidia-smi")
    smi === nothing && return Dict{Int, String}()
    out = try
        read(`$smi --query-gpu=index,uuid --format=csv,noheader`, String)
    catch
        return Dict{Int, String}()
    end
    uuids = Dict{Int, String}()
    for line in split(chomp(out), '\n')
        isempty(strip(line)) && continue
        parts = split(line, ","; limit=2)
        length(parts) == 2 || continue
        uuid = replace(strip(parts[2]), "GPU-" => "")
        uuids[parse(Int, strip(parts[1]))] = uuid
    end
    return uuids
end

function activate_device!(device::AbstractString, expected_smi_index::Int,
        visible_device_id::AbstractString, required_gpu_name::AbstractString)
    s = lowercase(strip(device))
    if s == "cpu"
        @printf("Using CPU execution.\n")
        return :cpu
    end
    require_condition(s == "gpu", "score.jl supports run.device=\"gpu\" or \"cpu\".")
    require_condition(CUDA.has_cuda(), "GPU requested but CUDA is unavailable.")
    visible = get(ENV, "CUDA_VISIBLE_DEVICES", "")
    require_condition(strip(visible) == strip(visible_device_id),
        "CUDA_VISIBLE_DEVICES must be exactly $(visible_device_id) for this run; got \"$(visible)\".")
    smi_names = nvidia_smi_gpu_names()
    smi_uuids = nvidia_smi_gpu_uuids()
    if haskey(smi_names, expected_smi_index)
        smi_name = smi_names[expected_smi_index]
        require_condition(occursin(lowercase(strip(required_gpu_name)), lowercase(smi_name)),
            "nvidia-smi GPU $(expected_smi_index) is $(smi_name), not $(required_gpu_name).")
    end
    CUDA.device!(0)
    cuda_name = CUDA.name(CUDA.device())
    require_condition(occursin(lowercase(strip(required_gpu_name)), lowercase(cuda_name)),
        "CUDA ordinal 0 is $(cuda_name), not $(required_gpu_name).")
    if haskey(smi_uuids, expected_smi_index)
        cuda_uuid = string(CUDA.uuid(CUDA.device()))
        require_condition(cuda_uuid == smi_uuids[expected_smi_index],
            "CUDA ordinal 0 UUID $(cuda_uuid) does not match nvidia-smi index $(expected_smi_index) UUID $(smi_uuids[expected_smi_index]).")
    end
    CUDA.allowscalar(false)
    @printf("Using CUDA_VISIBLE_DEVICES=%s -> CUDA ordinal 0: %s\n", visible, cuda_name)
    return :gpu
end

move_array(x, dev) = dev == :gpu ? cu(x) : x
move_model(m, dev) = dev == :gpu ? gpu(m) : cpu(m)

Base.@kwdef struct ScoreParams
    input_hdf5::String
    burnin_fraction::Float64
    max_samples::Int
    train_fraction::Float64
    model_config::ScoreUNetConfig
    model_normalization::Symbol
    init_seed::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float64
    sigma::Float32
    epoch_subset_size::Int
    seed::Int
    device::String
    expected_smi_index::Int
    visible_device_id::String
    required_gpu_name::String
    train::Bool
    evaluate::Bool
    render::Bool
    langevin_validate::Bool
    output_bson::String
    output_png::String
    output_langevin_hdf5::String
    output_langevin_png::String
    exact_score_samples::Int
    stein_samples::Int
    figure_width::Int
    figure_height::Int
    langevin_dt::Float64
    langevin_total_time::Float64
    langevin_burnin_time::Float64
    langevin_save_dt::Float64
    langevin_ntraj::Int
    langevin_score_clip::Float32
    langevin_max_pdf_samples::Int
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    model = raw["model"]
    training = raw["training"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    figure = get(raw, "figure", Dict{String, Any}())
    langevin = get(raw, "langevin", Dict{String, Any}())
    cfg = ScoreUNetConfig(
        in_channels=STATE_CHANNELS,
        base_channels=Int(get(model, "base_channels", 96)),
        channel_multipliers=Int.(get(model, "channel_multipliers", [1, 2, 4, 4])),
        kernel_size=Int(get(model, "kernel_size", 5)),
        periodic=Bool(get(model, "periodic", true)),
        activation=activation_from_string(String(get(model, "activation", "swish"))),
        final_activation=activation_from_string(String(get(model, "final_activation", "identity"))),
    )
    params = ScoreParams(
        input_hdf5=String(data["input_hdf5"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        max_samples=Int(get(data, "max_samples", 0)),
        train_fraction=Float64(get(data, "train_fraction", 0.85)),
        model_config=cfg,
        model_normalization=normalization_from_string(get(model, "normalization", "none")),
        init_seed=Int(get(model, "init_seed", 20260511)),
        batch_size=Int(get(training, "batch_size", 1024)),
        epochs=Int(get(training, "epochs", 500)),
        learning_rate=Float64(get(training, "learning_rate", 2.0e-4)),
        sigma=Float32(get(training, "sigma", 0.05)),
        epoch_subset_size=Int(get(training, "epoch_subset_size", 0)),
        seed=Int(get(training, "seed", 20260621)),
        device=String(get(run, "device", "gpu")),
        expected_smi_index=Int(get(run, "expected_smi_index", 0)),
        visible_device_id=String(get(run, "visible_device_id", string(Int(get(run, "expected_smi_index", 0))))),
        required_gpu_name=String(get(run, "required_gpu_name", "NVIDIA")),
        train=Bool(get(run, "train", true)),
        evaluate=Bool(get(run, "evaluate", true)),
        render=Bool(get(run, "render", false)),
        langevin_validate=Bool(get(run, "langevin_validate", false)),
        output_bson=String(output["model_bson"]),
        output_png=String(get(output, "figure_png", "../figures/score_diagnostics.png")),
        output_langevin_hdf5=String(get(output, "langevin_hdf5", "../data/score_langevin.h5")),
        output_langevin_png=String(get(output, "langevin_figure_png", "../figures/score_langevin_validation.png")),
        exact_score_samples=Int(get(figure, "exact_score_samples", 60000)),
        stein_samples=Int(get(figure, "stein_samples", 60000)),
        figure_width=Int(get(figure, "width", 3200)),
        figure_height=Int(get(figure, "height", 2300)),
        langevin_dt=Float64(get(langevin, "dt", 0.002)),
        langevin_total_time=Float64(get(langevin, "total_time", 80.0)),
        langevin_burnin_time=Float64(get(langevin, "burnin_time", 20.0)),
        langevin_save_dt=Float64(get(langevin, "save_dt", 0.05)),
        langevin_ntraj=Int(get(langevin, "ntrajectories", 256)),
        langevin_score_clip=Float32(get(langevin, "score_clip", 80.0)),
        langevin_max_pdf_samples=Int(get(langevin, "max_pdf_samples", 250000)),
    )
    require_condition(params.sigma == 0.05f0, "DSM sigma must remain 0.05.")
    require_condition(params.model_config.periodic, "Score U-Net must use periodic convolutions.")
    require_condition(params.model_normalization != :batchnorm, "BatchNorm is forbidden for this score model.")
    require_condition(0.0 < params.train_fraction < 1.0, "train_fraction must be in (0,1).")
    return params
end

function load_phys(path::AbstractString)
    h5open(path, "r") do h5
        return FHDPhys(
            N=Int(read(h5["/metadata/N"])),
            L=Float64(read(h5["/metadata/L"])),
            dx=Float64(read(h5["/metadata/dx"])),
            rho0=Float64(read(h5["/metadata/rho0"])),
            cs=Float64(read(h5["/metadata/sound_speed"])),
            theta=Float64(read(h5["/metadata/Theta"])),
            eta0=Float64(read(h5["/metadata/eta0"])),
            zeta=Float64(read(h5["/metadata/zeta"])),
            kappa=Float64(read(h5["/metadata/kappa"])),
            velocity_density_floor=Float64(read(h5["/metadata/velocity_density_floor"])),
        )
    end
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function load_states(path::AbstractString)
    times = Float64.(h5read(path, "/trajectories/time"))
    states = Float32.(h5read(path, "/trajectories/states"))
    require_condition(ndims(states) == 4, "Expected states as time x site x channel x trajectory.")
    require_condition(size(states, 3) == STATE_CHANNELS, "Expected channels (rho,m).")
    return times, states
end

function collect_postburnin_samples(states::Array{Float32, 4}, start_idx::Int,
        max_samples::Int, rng::AbstractRNG)
    nt, K, C, ntraj = size(states)
    total = (nt - start_idx + 1) * ntraj
    n = max_samples <= 0 ? total : min(max_samples, total)
    samples = Array{Float32}(undef, K, C, n)
    if n == total
        col = 0
        @inbounds for tr in 1:ntraj, t in start_idx:nt
            col += 1
            samples[:, :, col] .= states[t, :, :, tr]
        end
    else
        @inbounds for s in 1:n
            linear = rand(rng, 0:(total - 1))
            t = start_idx + (linear % (nt - start_idx + 1))
            tr = (linear ÷ (nt - start_idx + 1)) + 1
            samples[:, :, s] .= states[t, :, :, tr]
        end
    end
    return samples
end

function channel_shared_stats(samples::Array{Float32, 3})
    K, C, _ = size(samples)
    mu = Array{Float32}(undef, C, K)
    sd = Array{Float32}(undef, C, K)
    for c in 1:C
        vals = @view samples[:, c, :]
        m = mean(Float64, vals)
        v = mean(x -> abs2(Float64(x) - m), vals)
        mu[c, :] .= Float32(m)
        sd[c, :] .= Float32(max(sqrt(v), 1.0e-8))
    end
    return DataStats(mu, sd)
end

function apply_stats(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    out = (samples .- mean_tensor) ./ std_tensor
    project_zero_modes!(out)
    return out
end

function denormalize_tensor(samples::Array{Float32, 3}, stats::DataStats)
    mean_tensor = reshape(permutedims(stats.mean, (2, 1)), size(samples, 1), size(samples, 2), 1)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(samples, 1), size(samples, 2), 1)
    return samples .* std_tensor .+ mean_tensor
end

function std_flat(stats::DataStats)
    K = size(stats.std, 2)
    out = Vector{Float64}(undef, 2K)
    out[1:K] .= Float64.(@view stats.std[1, :])
    out[(K + 1):(2K)] .= Float64.(@view stats.std[2, :])
    return out
end

function mean_flat(stats::DataStats)
    K = size(stats.mean, 2)
    out = Vector{Float64}(undef, 2K)
    out[1:K] .= Float64.(@view stats.mean[1, :])
    out[(K + 1):(2K)] .= Float64.(@view stats.mean[2, :])
    return out
end

function raw_flat_batch(z::Array{Float32, 3})
    K, _, B = size(z)
    out = Matrix{Float32}(undef, 2K, B)
    @views begin
        out[1:K, :] .= reshape(z[:, 1, :], K, B)
        out[(K + 1):(2K), :] .= reshape(z[:, 2, :], K, B)
    end
    return out
end

function raw_flat_batch64(z::Array{Float32, 3})
    return Float64.(raw_flat_batch(z))
end

function split_dataset(samples::Array{Float32, 3}, train_fraction::Float64, rng::AbstractRNG)
    B = size(samples, 3)
    idx = randperm(rng, B)
    ntrain = clamp(floor(Int, train_fraction * B), 1, B - 1)
    train_raw = samples[:, :, idx[1:ntrain]]
    val_raw = samples[:, :, idx[(ntrain + 1):end]]
    stats = channel_shared_stats(train_raw)
    return NormalizedDataset(apply_stats(train_raw, stats), stats),
        NormalizedDataset(apply_stats(val_raw, stats), stats)
end

function density_laplacian!(lap::Vector{Float64}, rho::AbstractVector{<:Real}, phys::FHDPhys)
    N = length(rho)
    invdx2 = 1.0 / phys.dx^2
    @inbounds for i in 1:N
        lap[i] = (Float64(rho[periodic(i + 1, N)]) - 2.0 * Float64(rho[i]) +
            Float64(rho[periodic(i - 1, N)])) * invdx2
    end
    return lap
end

function analytic_score_raw(samples::Array{Float32, 3}, phys::FHDPhys)
    K, _, B = size(samples)
    score = Array{Float32}(undef, K, 2, B)
    lap = Vector{Float64}(undef, K)
    rho_score = Vector{Float64}(undef, K)
    m_score = Vector{Float64}(undef, K)
    @inbounds for b in 1:B
        rho = @view samples[:, 1, b]
        m = @view samples[:, 2, b]
        density_laplacian!(lap, rho, phys)
        for i in 1:K
            rhoi = max(Float64(rho[i]), 1.0e-14)
            ui = Float64(m[i]) / rhoi
            rho_score[i] = -phys.dx / phys.theta *
                (phys.cs^2 * log(rhoi / phys.rho0) - 0.5 * ui^2 - phys.kappa * lap[i])
            m_score[i] = -phys.dx / phys.theta * ui
        end
        mr = mean(rho_score)
        mm = mean(m_score)
        for i in 1:K
            score[i, 1, b] = Float32(rho_score[i] - mr)
            score[i, 2, b] = Float32(m_score[i] - mm)
        end
    end
    return score
end

function standardized_analytic_score(norm_samples::Array{Float32, 3}, stats::DataStats,
        phys::FHDPhys)
    raw = denormalize_tensor(norm_samples, stats)
    raw_score = analytic_score_raw(raw, phys)
    std_tensor = reshape(permutedims(stats.std, (2, 1)), size(raw_score, 1), size(raw_score, 2), 1)
    return raw_score .* std_tensor
end

function fill_projected_noise!(noise, rng::AbstractRNG)
    if noise isa CUDA.CuArray
        randn!(CUDA.default_rng(), noise)
    else
        randn!(rng, noise)
    end
    project_zero_modes!(noise)
    return noise
end

function score_from_model(model, batch, sigma::Float32)
    pred = model(batch)
    return project_zero_modes((-one(eltype(pred)) / sigma) .* pred)
end

function build_model(cfg::ScoreUNetConfig, normalization::Symbol)
    return ProjectedScoreUNet(build_unet(cfg; normalization=normalization))
end

function make_noisy_batch(clean, sigma::Float32, rng::AbstractRNG, dev)
    noise_cpu = Array{Float32}(undef, size(clean))
    fill_projected_noise!(noise_cpu, rng)
    batch = move_array(clean, dev)
    noise = move_array(noise_cpu, dev)
    noisy = project_zero_modes(batch .+ sigma .* noise)
    return noisy, noise
end

function validation_dsm_loss(model, data::Array{Float32, 3}, params::ScoreParams,
        rng::AbstractRNG, dev)
    Flux.testmode!(model)
    losses = Float64[]
    for part in Iterators.partition(1:size(data, 3), params.batch_size)
        batch = copy(data[:, :, collect(part)])
        noisy, noise = make_noisy_batch(batch, params.sigma, rng, dev)
        loss = Flux.Losses.mse(model(noisy), noise)
        push!(losses, Float64(to_host(loss)))
    end
    Flux.trainmode!(model)
    return mean(losses)
end

function train_score(train_ds::NormalizedDataset, val_ds::NormalizedDataset,
        params::ScoreParams, dev)
    Random.seed!(params.init_seed)
    model = build_model(params.model_config, params.model_normalization)
    Random.seed!()
    model = move_model(model, dev)
    opt_state = Flux.setup(Flux.Optimisers.Adam(params.learning_rate), model)
    rng = MersenneTwister(params.seed)
    noise_rng = MersenneTwister(params.seed + 1)
    val_rng = MersenneTwister(params.seed + 2)
    history = Dict{Symbol, Any}(
        :train_loss => Float64[],
        :val_loss => Float64[],
        :epoch_time => Float64[],
    )
    best_val_loss = Inf
    best_epoch = 0
    best_model = nothing
    ntrain = length(train_ds)
    for epoch in 1:params.epochs
        t0 = time_ns()
        idx = randperm(rng, ntrain)
        if params.epoch_subset_size > 0 && params.epoch_subset_size < ntrain
            idx = idx[1:params.epoch_subset_size]
        end
        losses = Float64[]
        Flux.trainmode!(model)
        for part in Iterators.partition(idx, params.batch_size)
            batch = copy(train_ds.data[:, :, collect(part)])
            noisy, noise = make_noisy_batch(batch, params.sigma, noise_rng, dev)
            loss_value, grads = Flux.withgradient(model) do m
                Flux.Losses.mse(m(noisy), noise)
            end
            opt_state, model = Flux.update!(opt_state, model, grads[1])
            push!(losses, Float64(to_host(loss_value)))
        end
        train_loss = mean(losses)
        val_loss = validation_dsm_loss(model, val_ds.data, params,
            MersenneTwister(params.seed + 2), dev)
        push!(history[:train_loss], train_loss)
        push!(history[:val_loss], val_loss)
        push!(history[:epoch_time], (time_ns() - t0) / 1e9)
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(cpu(model))
        end
        @printf("epoch %04d train_loss %.8e val_loss %.8e time %.2fs\n",
            epoch, train_loss, val_loss, history[:epoch_time][end])
        flush(stdout)
    end
    if best_model !== nothing
        model = move_model(best_model, dev)
        history[:best_epoch] = best_epoch
        history[:best_val_loss] = best_val_loss
        @printf("Restored best validation checkpoint from epoch %d with val_loss %.8e\n",
            best_epoch, best_val_loss)
    end
    Flux.testmode!(model)
    return model, history
end

function evaluate_score_norm(model, norm_samples::Array{Float32, 3}, sigma::Float32,
        batch_size::Int, dev)
    Flux.testmode!(model)
    K, C, B = size(norm_samples)
    out = Array{Float32}(undef, K, C, B)
    for lo in 1:batch_size:B
        hi = min(lo + batch_size - 1, B)
        batch = move_array(copy(@view norm_samples[:, :, lo:hi]), dev)
        out[:, :, lo:hi] .= to_host(score_from_model(model, batch, sigma))
    end
    return out
end

function score_metric_pair(pred, target)
    rel = sqrt(sum(abs2, pred .- target) / max(sum(abs2, target), eps(Float64)))
    cosv = sum(pred .* target) / max(sqrt(sum(abs2, pred) * sum(abs2, target)), eps(Float64))
    return Float64(rel), Float64(cosv)
end

function projection_matrix(K::Int)
    Psite = Matrix{Float64}(I, K, K) .- 1.0 / K
    P = zeros(Float64, 2K, 2K)
    P[1:K, 1:K] .= Psite
    P[(K + 1):(2K), (K + 1):(2K)] .= Psite
    return P
end

function stein_matrix(score_norm::Array{Float32, 3}, samples_norm::Array{Float32, 3})
    K, _, B = size(score_norm)
    s = raw_flat_batch64(score_norm)
    x = raw_flat_batch64(samples_norm)
    return -(s * transpose(x)) ./ B
end

function score_diagnostics(model, ds::NormalizedDataset, phys::FHDPhys, params::ScoreParams, dev)
    rng = MersenneTwister(params.seed + 300)
    n = min(params.exact_score_samples, length(ds))
    idx = randperm(rng, length(ds))[1:n]
    clean = copy(ds.data[:, :, idx])
    pred = evaluate_score_norm(model, clean, params.sigma, params.batch_size, dev)
    target = standardized_analytic_score(clean, ds.stats, phys)
    rel, cosv = score_metric_pair(pred, target)
    nstein = min(params.stein_samples, length(ds))
    idxs = randperm(rng, length(ds))[1:nstein]
    stein_samples = copy(ds.data[:, :, idxs])
    score_samples = evaluate_score_norm(model, stein_samples, params.sigma, params.batch_size, dev)
    S = stein_matrix(score_samples, stein_samples)
    P = projection_matrix(size(ds.data, 1))
    stein_rel = norm(S - P) / norm(P)
    diagnostics = Dict{Symbol, Any}(
        :analytic_rel_rmse => rel,
        :analytic_cosine => cosv,
        :stein_relative_error => stein_rel,
        :final_val_loss => params.train ? missing : missing,
    )
    return diagnostics, S, vec(Float64.(target)), vec(Float64.(pred))
end

function load_or_build_dataset(input_hdf5::AbstractString, params::ScoreParams,
        stats_payload=nothing)
    times, states = load_states(input_hdf5)
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    raw = collect_postburnin_samples(states, start_idx, params.max_samples, MersenneTwister(params.seed + 10))
    if stats_payload === nothing
        train_ds, val_ds = split_dataset(raw, params.train_fraction, MersenneTwister(params.seed + 11))
    else
        stats = stats_payload isa DataStats ? stats_payload :
            DataStats(Float32.(stats_payload[:mean]), Float32.(stats_payload[:std]))
        train_ds = NormalizedDataset(apply_stats(raw, stats), stats)
        val_ds = train_ds
    end
    return train_ds, val_ds, times, states, start_idx
end

function save_checkpoint(path::AbstractString, model, train_ds::NormalizedDataset,
        params::ScoreParams, phys::FHDPhys, history, diagnostics, stein)
    ensure_parent_dir(path)
    host_model = cpu(model)
    stats = Dict(:mean => train_ds.stats.mean, :std => train_ds.stats.std)
    metadata = Dict(
        :system => "FHDChainCapillaryN32",
        :score_type => "stationary_DSM_projected_periodic_UNet",
        :sigma => params.sigma,
        :analytic_score_used_for_training => false,
        :phi_training_uses_analytic => false,
        :normalization => String(params.model_normalization),
        :state_layout => "site x channel x batch, channels=(rho,m)",
        :required_gpu_name => params.required_gpu_name,
        :expected_smi_index => params.expected_smi_index,
    )
    BSON.@save path host_model stats params phys history diagnostics stein metadata
    @printf("Saved score checkpoint to %s\n", path)
    return nothing
end

function load_checkpoint(path::AbstractString, dev)
    blob = BSON.load(path)
    model = move_model(blob[:host_model], dev)
    Flux.testmode!(model)
    stats_payload = blob[:stats]
    stats = DataStats(Float32.(stats_payload[:mean]), Float32.(stats_payload[:std]))
    return model, stats, blob[:params], blob[:phys], blob[:history], blob[:diagnostics], blob[:stein]
end

function initialize_langevin(ds::NormalizedDataset, ntraj::Int, rng::AbstractRNG)
    K, C, B = size(ds.data)
    out = Array{Float32}(undef, K, C, ntraj)
    @inbounds for b in 1:ntraj
        out[:, :, b] .= ds.data[:, :, rand(rng, 1:B)]
    end
    project_zero_modes!(out)
    return out
end

function integrate_score_langevin(model, ds::NormalizedDataset, params::ScoreParams, dev)
    nsteps = ceil(Int, params.langevin_total_time / params.langevin_dt)
    burn_steps = floor(Int, params.langevin_burnin_time / params.langevin_dt)
    save_every = max(1, round(Int, params.langevin_save_dt / params.langevin_dt))
    actual_save_dt = save_every * params.langevin_dt
    nsaved = fld(max(nsteps - burn_steps, 0), save_every) + 1
    rng = MersenneTwister(params.seed + 1000)
    z = initialize_langevin(ds, params.langevin_ntraj, rng)
    zdev = move_array(z, dev)
    noise = similar(zdev)
    saved_norm = Array{Float32}(undef, nsaved, size(z, 1), size(z, 2), size(z, 3))
    times = Vector{Float64}(undef, nsaved)
    dt = Float32(params.langevin_dt)
    sqrtdt = Float32(sqrt(2.0 * params.langevin_dt))
    save_idx = 0
    stride = max(1, nsteps ÷ 20)
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            times[save_idx] = (step - burn_steps) * params.langevin_dt
            saved_norm[save_idx, :, :, :] .= to_host(zdev)
        end
        step == nsteps && break
        score = score_from_model(model, zdev, params.sigma)
        clamp!(score, -params.langevin_score_clip, params.langevin_score_clip)
        fill_projected_noise!(noise, rng)
        @. zdev = zdev + dt * score + sqrtdt * noise
        project_zero_modes!(zdev)
        if step > 0 && step % stride == 0
            @printf("score Langevin progress %.1f%%\n", 100.0 * step / nsteps)
            flush(stdout)
        end
    end
    raw = Array{Float32}(undef, nsaved, size(z, 1), size(z, 2), size(z, 3))
    for t in 1:nsaved
        raw[t, :, :, :] .= denormalize_tensor(saved_norm[t, :, :, :], ds.stats)
    end
    return times, raw, actual_save_dt
end

function draw_channel_values(states::Array{Float32, 4}, start_idx::Int, channel::Int,
        max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * K * ntraj
    n = min(max_samples, total)
    vals = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tmp = linear ÷ (nt - start_idx + 1)
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        vals[s] = Float64(states[t, i, channel, tr])
    end
    return vals
end

function draw_velocity_values(states::Array{Float32, 4}, start_idx::Int, phys::FHDPhys,
        max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * K * ntraj
    n = min(max_samples, total)
    vals = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tmp = linear ÷ (nt - start_idx + 1)
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        vals[s] = Float64(states[t, i, 2, tr]) /
            max(Float64(states[t, i, 1, tr]), phys.velocity_density_floor)
    end
    return vals
end

function sampled_covariance(states::Array{Float32, 4}, start_idx::Int, max_samples::Int,
        rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * ntraj
    n = min(max_samples, total)
    X = Matrix{Float64}(undef, 2K, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tr = (linear ÷ (nt - start_idx + 1)) + 1
        for i in 1:K
            X[i, s] = Float64(states[t, i, 1, tr])
            X[K + i, s] = Float64(states[t, i, 2, tr])
        end
    end
    mu = mean(X; dims=2)
    X .-= mu
    return (X * transpose(X)) ./ max(n - 1, 1)
end

function spatial_power_spectrum(states::Array{Float32, 4}, start_idx::Int, channel::Int)
    nt, K, _, ntraj = size(states)
    powers = zeros(Float64, K ÷ 2 + 1)
    count = 0
    @inbounds for tr in 1:ntraj, t in start_idx:nt
        vals = Float64.(states[t, :, channel, tr])
        vals .-= mean(vals)
        for k in 0:(K ÷ 2)
            re = 0.0
            im = 0.0
            for j in 1:K
                angle = -2.0 * pi * k * (j - 1) / K
                re += vals[j] * cos(angle)
                im += vals[j] * sin(angle)
            end
            powers[k + 1] += (re^2 + im^2) / K
        end
        count += 1
    end
    return collect(0:(K ÷ 2)), powers ./ max(count, 1)
end

function hist_density(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Float64, length(edges) - 1)
    lo = first(edges)
    hi = last(edges)
    width = (hi - lo) / length(counts)
    @inbounds for v in values
        if lo <= v <= hi
            idx = clamp(floor(Int, (v - lo) / width) + 1, 1, length(counts))
            counts[idx] += 1.0
        end
    end
    return 0.5 .* (edges[1:end-1] .+ edges[2:end]), counts ./ max(sum(counts) * width, eps(Float64))
end

function zero_mode_drift(states::Array{Float32, 4})
    nt, _, _, ntraj = size(states)
    mass0 = [sum(Float64, @view states[1, :, 1, tr]) for tr in 1:ntraj]
    mom0 = [sum(Float64, @view states[1, :, 2, tr]) for tr in 1:ntraj]
    max_mass = 0.0
    max_mom = 0.0
    @inbounds for tr in 1:ntraj, t in 1:nt
        max_mass = max(max_mass, abs(sum(Float64, @view states[t, :, 1, tr]) - mass0[tr]))
        max_mom = max(max_mom, abs(sum(Float64, @view states[t, :, 2, tr]) - mom0[tr]))
    end
    return max_mass, max_mom
end

function score_langevin_metrics(obs_states::Array{Float32, 4}, obs_start::Int,
        lang_states::Array{Float32, 4}, phys::FHDPhys, params::ScoreParams)
    rng = MersenneTwister(params.seed + 2000)
    maxs = params.langevin_max_pdf_samples
    obs_sets = [
        draw_channel_values(obs_states, obs_start, 1, maxs, rng),
        draw_channel_values(obs_states, obs_start, 2, maxs, rng),
        draw_velocity_values(obs_states, obs_start, phys, maxs, rng),
    ]
    lang_sets = [
        draw_channel_values(lang_states, 1, 1, maxs, rng),
        draw_channel_values(lang_states, 1, 2, maxs, rng),
        draw_velocity_values(lang_states, 1, phys, maxs, rng),
    ]
    rels = Float64[]
    for k in 1:3
        lo = quantile(obs_sets[k], 0.001)
        hi = quantile(obs_sets[k], 0.999)
        pad = 0.08 * max(hi - lo, 1.0e-8)
        edges = collect(range(lo - pad, hi + pad; length=141))
        _, po = hist_density(obs_sets[k], edges)
        _, pl = hist_density(lang_sets[k], edges)
        push!(rels, sqrt(sum(abs2, pl .- po) / max(sum(abs2, po), eps(Float64))))
    end
    cov_obs = sampled_covariance(obs_states, obs_start, 120000, rng)
    cov_l = sampled_covariance(lang_states, 1, 120000, rng)
    mass_drift, mom_drift = zero_mode_drift(lang_states)
    return Dict{Symbol, Float64}(
        :rho_pdf_rel_l2 => rels[1],
        :m_pdf_rel_l2 => rels[2],
        :u_pdf_rel_l2 => rels[3],
        :mean_pdf_rel_l2 => mean(rels),
        :covariance_rel_rmse => sqrt(sum(abs2, cov_l .- cov_obs) / max(sum(abs2, cov_obs), eps(Float64))),
        :max_mass_drift => mass_drift,
        :max_momentum_drift => mom_drift,
    )
end

function save_score_langevin(path::AbstractString, times::Vector{Float64},
        states::Array{Float32, 4}, save_dt::Float64, metrics::Dict)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time", times)
        write(h5, "/states", states)
        write(h5, "/metadata/save_dt", save_dt)
        for (k, v) in metrics
            write(h5, "/metrics/$(String(k))", v)
        end
    end
    @printf("Saved score-only Langevin trajectory to %s\n", path)
    return nothing
end

function load_score_langevin(path::AbstractString)
    h5open(path, "r") do h5
        states = Float32.(read(h5["/states"]))
        metrics = Dict{Symbol, Float64}()
        if haskey(h5, "metrics")
            for key in keys(h5["/metrics"])
                metrics[Symbol(key)] = Float64(read(h5["/metrics/$(key)"]))
            end
        end
        return states, metrics
    end
end

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
    if haskey(ENV, "DISPLAY") && !isempty(ENV["DISPLAY"]) && display_is_usable(ENV["DISPLAY"])
        return nothing
    end
    xvfb = Sys.which("Xvfb")
    xvfb === nothing && return nothing
    for display_id in 151:260
        display = ":" * string(display_id)
        display_is_usable(display) && (ENV["DISPLAY"] = display; return nothing)
    end
    for display_id in 151:260
        display = ":" * string(display_id)
        isfile("/tmp/.X$(display_id)-lock") && continue
        run(pipeline(`$xvfb $display -screen 0 1920x1200x24 -nolisten tcp`,
            stdout=devnull, stderr=devnull); wait=false)
        sleep(1.0)
        if display_is_usable(display)
            ENV["DISPLAY"] = display
            return nothing
        end
    end
    return nothing
end

function render_score_figure(path::AbstractString, params::ScoreParams, train_ds::NormalizedDataset,
        val_ds::NormalizedDataset, history, diagnostics, stein, scatter_target, scatter_pred,
        obs_states=nothing, obs_start::Int=1, lang_states=nothing, lang_metrics=nothing,
        phys=nothing)
    start_xvfb!()
    @eval using GLMakie
    include(STYLE_FILE)
    Base.invokelatest(GLMakie.activate!)
    Base.invokelatest(with_scaled_figure_style, function (_)
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("DSM sigma=%.3f, train=%d, val=%d, analytic rel=%.3e cos=%.5f",
            params.sigma, length(train_ds), length(val_ds),
            diagnostics[:analytic_rel_rmse], diagnostics[:analytic_cosine])
        figure_title!(fig, "Capillary FHD N32 stationary score"; subtitle=subtitle)
        epochs = collect(1:length(history[:train_loss]))
        ax1 = Axis(fig[1, 1]; title="DSM loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax1, epochs, history[:train_loss]; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="train")
        lines!(ax1, epochs, history[:val_loss]; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash, label="validation")
        axislegend(ax1; position=:rt)
        ax2 = Axis(fig[1, 2]; title="learned vs analytic score", xlabel="analytic", ylabel="learned")
        step = max(1, length(scatter_target) ÷ 25000)
        tx = scatter_target[1:step:end]
        py = scatter_pred[1:step:end]
        scatter!(ax2, tx, py; markersize=3, color=(STYLE_PRIMARY, 0.22))
        lim = maximum(abs, vcat(tx, py))
        lines!(ax2, [-lim, lim], [-lim, lim]; color=STYLE_REFERENCE, linestyle=:dash)
        P = projection_matrix(size(train_ds.data, 1))
        residual = stein - P
        clim = max(maximum(abs, residual), 1.0e-6)
        ax3 = Axis(fig[1, 3]; title="Stein residual", xlabel="column", ylabel="row")
        hm = heatmap!(ax3, residual; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
        Colorbar(fig[1, 3, Right()], hm)

        ax4 = Axis(fig[2, 1]; title="score error", xlabel="learned - analytic", ylabel="density")
        hist!(ax4, py .- tx; bins=100, normalization=:pdf, color=(STYLE_SECONDARY, 0.65))
        lines = String[
            @sprintf("final train loss = %.8e", history[:train_loss][end]),
            @sprintf("final val loss = %.8e", history[:val_loss][end]),
            @sprintf("Stein rel.error = %.8e", diagnostics[:stein_relative_error]),
            @sprintf("analytic rel.RMSE = %.8e", diagnostics[:analytic_rel_rmse]),
            @sprintf("analytic cosine = %.8e", diagnostics[:analytic_cosine]),
            "DSM labels used only projected Gaussian noise.",
            "Analytic score appears only in this ex-post panel.",
        ]
        if lang_metrics !== nothing
            append!(lines, [
                @sprintf("score-only rho PDF rel.L2 = %.8e", lang_metrics[:rho_pdf_rel_l2]),
                @sprintf("score-only m PDF rel.L2 = %.8e", lang_metrics[:m_pdf_rel_l2]),
                @sprintf("score-only covariance rel.RMSE = %.8e", lang_metrics[:covariance_rel_rmse]),
            ])
        end
        text_panel!(fig[2, 2:3], lines; title="Audit and metrics")

        if lang_states !== nothing && obs_states !== nothing && phys !== nothing
            rng = MersenneTwister(params.seed + 3000)
            specs = [
                (draw_channel_values(obs_states, obs_start, 1, params.langevin_max_pdf_samples, rng),
                    draw_channel_values(lang_states, 1, 1, params.langevin_max_pdf_samples, rng), "rho PDF", "rho"),
                (draw_channel_values(obs_states, obs_start, 2, params.langevin_max_pdf_samples, rng),
                    draw_channel_values(lang_states, 1, 2, params.langevin_max_pdf_samples, rng), "m PDF", "m"),
                (draw_velocity_values(obs_states, obs_start, phys, params.langevin_max_pdf_samples, rng),
                    draw_velocity_values(lang_states, 1, phys, params.langevin_max_pdf_samples, rng), "u PDF", "u"),
            ]
            for (idx, (obs, lang, ttl, xl)) in enumerate(specs)
                lo = quantile(obs, 0.001)
                hi = quantile(obs, 0.999)
                pad = 0.08 * max(hi - lo, 1.0e-8)
                edges = collect(range(lo - pad, hi + pad; length=141))
                centers, po = hist_density(obs, edges)
                _, pl = hist_density(lang, edges)
                ax = Axis(fig[3, idx]; title=ttl, xlabel=xl, ylabel="density")
                lines!(ax, centers, po; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="obs")
                lines!(ax, centers, pl; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dash, label="score Langevin")
                idx == 1 && axislegend(ax; position=:rt)
            end
        end
        save_figure_checked(path, fig)
    end, params.figure_width, params.figure_height)
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base, params.input_hdf5)
    output_bson = resolve_path(base, params.output_bson)
    output_png = resolve_path(base, params.output_png)
    require_condition(isfile(input_hdf5), "Input HDF5 not found: $(input_hdf5)")
    dev = activate_device!(params.device, params.expected_smi_index, params.visible_device_id,
        params.required_gpu_name)
    phys = load_phys(input_hdf5)

    model = nothing
    history = nothing
    diagnostics = nothing
    stein = nothing
    train_ds = nothing
    val_ds = nothing
    times, states = load_states(input_hdf5)
    obs_start = burnin_start_index(length(times), params.burnin_fraction)

    if params.train
        raw = collect_postburnin_samples(states, obs_start, params.max_samples,
            MersenneTwister(params.seed + 10))
        train_ds, val_ds = split_dataset(raw, params.train_fraction, MersenneTwister(params.seed + 11))
        @printf("Loaded score dataset: train=%d val=%d K=%d\n",
            length(train_ds), length(val_ds), size(train_ds.data, 1))
        model, history = train_score(train_ds, val_ds, params, dev)
        if params.evaluate
            diagnostics, stein, scatter_target, scatter_pred =
                score_diagnostics(model, val_ds, phys, params, dev)
            diagnostics[:final_val_loss] = history[:val_loss][end]
            save_checkpoint(output_bson, model, train_ds, params, phys, history, diagnostics, stein)
            @printf("Score diagnostics: val_loss=%.8e Stein=%.8e analytic_rel=%.8e cosine=%.8f\n",
                history[:val_loss][end], diagnostics[:stein_relative_error],
                diagnostics[:analytic_rel_rmse], diagnostics[:analytic_cosine])
        else
            diagnostics = Dict{Symbol, Any}(:final_val_loss => history[:val_loss][end])
            stein = zeros(Float64, 2 * phys.N, 2 * phys.N)
            save_checkpoint(output_bson, model, train_ds, params, phys, history, diagnostics, stein)
        end
    else
        require_condition(isfile(output_bson), "Existing checkpoint missing: $(output_bson)")
        model, stats, saved_params, phys_saved, history, diagnostics, stein = load_checkpoint(output_bson, dev)
        phys = phys_saved
        raw = collect_postburnin_samples(states, obs_start, params.max_samples,
            MersenneTwister(params.seed + 10))
        train_ds = NormalizedDataset(apply_stats(raw, stats), stats)
        val_ds = train_ds
        if params.evaluate
            diagnostics, stein, scatter_target, scatter_pred =
                score_diagnostics(model, val_ds, phys, params, dev)
            @printf("Checkpoint diagnostics: Stein=%.8e analytic_rel=%.8e cosine=%.8f\n",
                diagnostics[:stein_relative_error], diagnostics[:analytic_rel_rmse],
                diagnostics[:analytic_cosine])
        end
    end

    lang_states = nothing
    lang_metrics = nothing
    if params.langevin_validate
        langevin_times, lang_states, actual_save_dt = integrate_score_langevin(model, train_ds, params, dev)
        lang_metrics = score_langevin_metrics(states, obs_start, lang_states, phys, params)
        save_score_langevin(resolve_path(base, params.output_langevin_hdf5),
            langevin_times, lang_states, actual_save_dt, lang_metrics)
        @printf("Score-only Langevin metrics: mean_pdf=%.8e covariance=%.8e mass_drift=%.3e mom_drift=%.3e\n",
            lang_metrics[:mean_pdf_rel_l2], lang_metrics[:covariance_rel_rmse],
            lang_metrics[:max_mass_drift], lang_metrics[:max_momentum_drift])
    end

    if params.render
        cached_langevin = resolve_path(base, params.output_langevin_hdf5)
        if lang_states === nothing && isfile(cached_langevin)
            lang_states, lang_metrics = load_score_langevin(cached_langevin)
        end
        if !@isdefined(scatter_target)
            diagnostics, stein, scatter_target, scatter_pred =
                score_diagnostics(model, val_ds, phys, params, dev)
        end
        render_score_figure(output_png, params, train_ds, val_ds, history, diagnostics,
            stein, scatter_target, scatter_pred, states, obs_start, lang_states,
            lang_metrics, phys)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "score_gpu0.toml")) : abspath(ARGS[1])
    run_pipeline(param_file)
end
