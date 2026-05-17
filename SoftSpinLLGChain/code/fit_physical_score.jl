#!/usr/bin/env julia

include(joinpath(@__DIR__, "score.jl"))

using BSON
using Printf

Base.@kwdef struct PhysicalScoreFitConfig
    input_hdf5::String
    burnin_fraction::Float64 = 0.1
    max_samples::Int = 4_194_304
    spin_inversion_augment::Bool = true
    enforce_zero_mean::Bool = true
    sigma::Float32 = 0.05f0
    noise_repeats::Int = 4
    batch_size::Int = 65_536
    ridge::Float64 = 1.0e-8
    seed::Int = 20260513
    output_bson::String
    metrics_txt::String
end

function load_physical_score_config(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    train = raw["training"]
    out = raw["output"]
    return PhysicalScoreFitConfig(
        input_hdf5=String(data["input_hdf5"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        max_samples=Int(get(data, "max_samples", 4_194_304)),
        spin_inversion_augment=Bool(get(data, "spin_inversion_augment", true)),
        enforce_zero_mean=Bool(get(data, "enforce_zero_mean", true)),
        sigma=Float32(get(train, "sigma", 0.05)),
        noise_repeats=Int(get(train, "noise_repeats", 4)),
        batch_size=Int(get(train, "batch_size", 65_536)),
        ridge=Float64(get(train, "ridge", 1.0e-8)),
        seed=Int(get(train, "seed", 20260513)),
        output_bson=String(out["model_bson"]),
        metrics_txt=String(out["metrics_txt"]),
    )
end

function load_physical_score_dataset(path::AbstractString, cfg::PhysicalScoreFitConfig)
    times, states = load_spin_states(path)
    start = burnin_start_index(length(times), cfg.burnin_fraction)
    raw_n = cfg.spin_inversion_augment && cfg.max_samples > 0 ?
        max(1, ceil(Int, cfg.max_samples / 2)) : cfg.max_samples
    raw = sample_state_tensor(states, start, raw_n, MersenneTwister(cfg.seed))
    if cfg.spin_inversion_augment
        raw = cat(raw, -raw; dims=3)
        if cfg.max_samples > 0 && size(raw, 3) > cfg.max_samples
            raw = raw[:, :, 1:cfg.max_samples]
        end
    end
    stats = channel_shared_stats(raw)
    if cfg.enforce_zero_mean
        stats = DataStats(zeros(Float32, size(stats.mean)), stats.std)
    end
    normed = apply_stats_tensor(raw, stats)
    return NormalizedDataset(normed, stats), times, states, start
end

function accumulate_physical_score_normal_eq!(G, h, noisy_norm::Array{Float32, 3},
        target::Array{Float32, 3}, stats::DataStats)
    raw = Float64.(denormalize_tensor(noisy_norm, stats))
    targ = Float64.(target)
    N, _, B = size(raw)
    r2 = Array{Float64}(undef, N, B)
    @inbounds for b in 1:B, i in 1:N
        r2[i, b] = raw[i, 1, b]^2 + raw[i, 2, b]^2 + raw[i, 3, b]^2
    end
    lap = Array{Float64}(undef, N, SPIN_CHANNELS, B)
    @inbounds for b in 1:B, c in 1:SPIN_CHANNELS, i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        lap[i, c, b] = raw[im, c, b] + raw[ip, c, b] - 2.0 * raw[i, c, b]
    end
    @inbounds for c in 1:SPIN_CHANNELS
        for b in 1:B, i in 1:N
            sc = Float64(stats.std[c, i])
            rc = raw[i, c, b]
            rr = r2[i, b]
            feats = (
                sc * rc,
                sc * rr * rc,
                sc * lap[i, c, b],
                sc * rr * rr * rc,
            )
            y = targ[i, c, b]
            for a in 1:4
                h[c, a] += feats[a] * y
                for d in 1:4
                    G[c, a, d] += feats[a] * feats[d]
                end
            end
        end
    end
    return nothing
end

function fit_physical_score(cfg_path::AbstractString)
    base = dirname(cfg_path)
    cfg = load_physical_score_config(cfg_path)
    data_h5 = resolve_path(base, cfg.input_hdf5)
    dataset, _, _, _ = load_physical_score_dataset(data_h5, cfg)
    p = load_phys(data_h5)
    @printf("Loaded %d samples for physical-feature DSM score fit, sigma %.5g\n",
        length(dataset), cfg.sigma)
    G = zeros(Float64, SPIN_CHANNELS, 4, 4)
    h = zeros(Float64, SPIN_CHANNELS, 4)
    rng = MersenneTwister(cfg.seed + 17)
    n = length(dataset)
    progress = ProgressMeter.Progress(cfg.noise_repeats * cld(n, cfg.batch_size);
        desc="Fitting physical-feature score")
    for rep in 1:cfg.noise_repeats
        order = randperm(rng, n)
        for part in Iterators.partition(order, cfg.batch_size)
            idx = collect(part)
            batch = copy(dataset.data[:, :, idx])
            noise = randn(rng, Float32, size(batch))
            noisy = batch .+ cfg.sigma .* noise
            target = noise .* (-1f0 / cfg.sigma)
            accumulate_physical_score_normal_eq!(G, h, noisy, target, dataset.stats)
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
    coeff = zeros(Float32, SPIN_CHANNELS, 4)
    @inbounds for c in 1:SPIN_CHANNELS
        A = Matrix(G[c, :, :]) + cfg.ridge * I(4)
        b = Vector(h[c, :])
        coeff[c, :] .= Float32.(A \ b)
    end
    model = PhysicalFeatureScore(coeff, dataset.stats.mean, dataset.stats.std)
    trainer_cfg = Dict(:sigma => cfg.sigma, :output_mode => "score",
        :input_features => "physical_poly_r2_lap",
        :spin_inversion_augment => cfg.spin_inversion_augment,
        :enforce_zero_mean => cfg.enforce_zero_mean,
        :noise_repeats => cfg.noise_repeats,
        :max_samples => cfg.max_samples,
        :ridge => cfg.ridge,
        :fit_method => "streamed DSM normal equations")
    history = Dict(:coeff => coeff)
    metadata = Dict(:no_cheating_audit =>
        "Coefficients are fitted only from trajectory samples plus Gaussian DSM noise. Analytic score and true coefficients are excluded from the fit and target.")
    out = resolve_path(base, cfg.output_bson)
    ensure_parent_dir(out)
    BSON.bson(out, Dict(:host_model => model,
        :model_cfg => Dict(:kind => "PhysicalFeatureScore", :features => ["x", "r2*x", "lap", "r2^2*x"]),
        :stats => dataset.stats, :trainer_cfg => trainer_cfg, :history => history,
        :phys => p, :metadata => metadata))
    ensure_parent_dir(resolve_path(base, cfg.metrics_txt))
    open(resolve_path(base, cfg.metrics_txt), "w") do io
        println(io, "SoftSpinLLGChain physical-feature score fit")
        println(io, "No-cheating audit: DSM normal equations used trajectory samples plus Gaussian noise only.")
        println(io, @sprintf("sigma = %.8e", cfg.sigma))
        println(io, @sprintf("samples = %d", length(dataset)))
        println(io, @sprintf("noise_repeats = %d", cfg.noise_repeats))
        for c in 1:SPIN_CHANNELS
            println(io, @sprintf("coeff[%s] = %.10e %.10e %.10e %.10e",
                channel_names()[c], coeff[c, 1], coeff[c, 2], coeff[c, 3], coeff[c, 4]))
        end
    end
    @printf("Saved physical-feature score checkpoint to %s\n", out)
    return out
end

function main()
    length(ARGS) >= 1 || error("Usage: fit_physical_score.jl CONFIG")
    fit_physical_score(ARGS[1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
