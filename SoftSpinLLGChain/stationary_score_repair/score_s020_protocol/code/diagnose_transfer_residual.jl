#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "..", "code", "cond_score.jl"))

using BSON
using LinearAlgebra
using Printf
using Random
using Statistics

const DEFAULT_COND_CONFIG = normpath(joinpath(@__DIR__, "..", "..", "configs",
    "cond_repaired_score_gpu2_main.toml"))
const DEFAULT_COND_CHECKPOINT = normpath(joinpath(@__DIR__, "..", "..", "models",
    "cond_repaired_score_gpu2_main_epoch0720.bson"))

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("Unexpected argument $(a)")
        key = a[3:end]
        i == length(args) && error("Missing value for --$(key)")
        opts[key] = args[i + 1]
        i += 2
    end
    return opts
end

resolve_cli_path(path::AbstractString) = isabspath(path) ? path : normpath(joinpath(pwd(), path))

function load_frozen_conditional(cond_config_path::AbstractString,
        checkpoint_path::AbstractString, device::ExecutionDevice)
    cond_config_path = abspath(cond_config_path)
    params = load_config(cond_config_path)
    base = dirname(cond_config_path)
    old_score_path = resolve_path(base, params.score_bson)
    old_score_model, old_stats, old_sigma, _ = load_stationary_checkpoint(old_score_path, device)
    blob = BSON.load(abspath(checkpoint_path), @__MODULE__)
    cond_model = move_model(blob[:host_model], device)
    Flux.testmode!(cond_model)
    epoch = haskey(blob, :metadata) && haskey(blob[:metadata], :completed_epoch) ?
        Int(blob[:metadata][:completed_epoch]) :
        (haskey(blob, :completed_epoch) ? Int(blob[:completed_epoch]) : -1)
    return (; params, base, old_score_path, old_score_model, old_stats, old_sigma,
        cond_model, epoch)
end

function evaluate_stationary_raw(model, xraw::Array{Float32, 3}, stats::DataStats,
        sigma::Float32, device::ExecutionDevice; batch_size::Int)
    xnorm = apply_stats_tensor(xraw, stats)
    snorm = evaluate_score_norm(model, xnorm, sigma, device; batch_size=batch_size)
    return normalized_score_to_raw(snorm, stats)
end

function evaluate_transferred_raw(stack, new_score_model, new_stats::DataStats,
        new_sigma::Float32, x0::Array{Float32, 3}, xt::Array{Float32, 3},
        tau_norm::Vector{Float32}, device::ExecutionDevice; batch_size::Int)
    rnorm_old = evaluate_residual_norm(stack.cond_model, x0, xt, tau_norm,
        stack.old_stats, stack.params, device; batch_size=batch_size,
        score_model=stack.old_score_model, score_sigma=stack.old_sigma)
    rraw_old = normalized_residual_to_raw(rnorm_old, stack.old_stats)
    sold_raw = evaluate_stationary_raw(stack.old_score_model, x0, stack.old_stats,
        stack.old_sigma, device; batch_size=batch_size)
    snew_raw = evaluate_stationary_raw(new_score_model, x0, new_stats,
        new_sigma, device; batch_size=batch_size)
    rraw_transfer = rraw_old .+ sold_raw .- snew_raw
    return rraw_old, rraw_transfer, sold_raw, snew_raw
end

function relcorr(estimate::AbstractArray, reference::AbstractArray)
    e = vec(Float64.(estimate))
    r = vec(Float64.(reference))
    mask = isfinite.(e) .& isfinite.(r)
    e = e[mask]
    r = r[mask]
    rel = sqrt(mean(abs2, e .- r)) / max(sqrt(mean(abs2, r)), eps(Float64))
    centered_corr = length(r) > 2 ? cor(r, e) : NaN
    cosine_corr = dot(e, r) / max(norm(e) * norm(r), eps(Float64))
    return Dict(:relative_rmse => rel, :correlation => centered_corr,
        :cosine => cosine_corr, :estimate_rms => sqrt(mean(abs2, e)),
        :reference_rms => sqrt(mean(abs2, r)))
end

function observable_matrix(xt::Array{Float32, 3}, p::SpinParams,
        means::Vector{Float64}, expected_names::Vector{String})
    obs, names = observable_values_cond(xt, p)
    names == expected_names || error("Phi artifact observables $(expected_names) do not match supported $(names).")
    center_observables!(obs, means)
    return reshape(obs, size(obs, 1) * size(obs, 2), size(obs, 3))
end

function phi_operator_curve!(dest::Array{Float64, 4}, li::Int, obs_flat::Matrix{Float32},
        Phi::AbstractMatrix, rraw::Array{Float32, 3})
    B = size(obs_flat, 2)
    rflat = Matrix{Float64}(flatten_batch(rraw))
    action = transpose(Matrix{Float64}(Phi)) * rflat
    mat = -Matrix{Float64}(obs_flat) * transpose(action) / B
    dest[li, :, :, :] .= reshape(mat, size(dest, 2), size(dest, 3), size(dest, 4))
    return nothing
end

function residual_moment_metrics(rraw::Array{Float32, 3}, x0::Array{Float32, 3})
    rflat = Matrix{Float64}(flatten_batch(rraw))
    xflat = Matrix{Float64}(flatten_batch(x0))
    D, B = size(rflat)
    mean_norm = norm(vec(mean(rflat; dims=2))) / sqrt(D)
    stein_norm = norm((rflat * transpose(xflat)) ./ B) / sqrt(D)
    rms = sqrt(mean(abs2, rflat))
    return mean_norm, stein_norm, rms
end

function mean_dictionary(values::Vector{Dict{Symbol, Float64}})
    out = Dict{Symbol, Float64}()
    isempty(values) && return out
    for key in keys(first(values))
        out[key] = mean(v[key] for v in values)
    end
    return out
end

function run_transfer_diagnostic(; cond_config_path::String, checkpoint_path::String,
        new_score_path::String, phi_path::String, out_path::String, artifact_path::String,
        pairs_per_lag::Int, max_lags::Int, batch_size::Int, device_name::String,
        required_gpu_name::String, seed::Int)
    device = detect_spin_device(device_name, required_gpu_name)
    activate_and_describe_device!(device, device_name, required_gpu_name)

    stack = load_frozen_conditional(cond_config_path, checkpoint_path, device)
    data_h5 = resolve_path(stack.base, stack.params.input_hdf5)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, stack.params.burnin_fraction,
        stack.params.tau_max_decorrelation_multiples, stack.params.lag_stride)

    new_score_model, new_stats, new_sigma, new_blob =
        load_stationary_checkpoint(new_score_path, device)
    phi_blob = BSON.load(phi_path, @__MODULE__)
    names = Vector{String}(phi_blob[:observable_names])
    means = Vector{Float64}(phi_blob[:observable_means])
    Phi = haskey(phi_blob, :Phi_projected) ? Matrix{Float64}(phi_blob[:Phi_projected]) :
        Matrix{Float64}(phi_blob[:Phi])
    lags_all = Vector{Int}(phi_blob[:lags])
    nlag = min(max_lags, length(lags_all))
    lags = lags_all[1:nlag]
    Cdot_phi_ref = Array{Float64}(phi_blob[:Cdot_phi])[1:nlag, :, :, :]
    Cdot_data_ref = Array{Float64}(phi_blob[:Cdot_data])[1:nlag, :, :, :]

    Cold = similar(Cdot_phi_ref)
    Ctransfer = similar(Cdot_phi_ref)
    rng = MersenneTwister(seed)
    moment_old = Dict{Symbol, Float64}[]
    moment_transfer = Dict{Symbol, Float64}[]
    shift_ratios = Float64[]
    score_shift_rms = Float64[]
    old_score_rms = Float64[]
    new_score_rms = Float64[]

    @printf("Cheap transfer diagnostic with %d lags, %d pairs/lag\n", nlag, pairs_per_lag)
    @printf("Frozen conditional checkpoint: %s\n", checkpoint_path)
    @printf("Old stationary score: %s\n", stack.old_score_path)
    @printf("New stationary score: %s\n", new_score_path)
    @printf("Phi artifact: %s\n", phi_path)
    flush(stdout)

    for (li, lag) in enumerate(lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, pairs_per_lag, rng)
        rraw_old, rraw_transfer, sold_raw, snew_raw = evaluate_transferred_raw(stack,
            new_score_model, new_stats, new_sigma, x0, xt, tau_norm, device;
            batch_size=batch_size)
        obs_flat = observable_matrix(xt, p, means, names)
        phi_operator_curve!(Cold, li, obs_flat, Phi, rraw_old)
        phi_operator_curve!(Ctransfer, li, obs_flat, Phi, rraw_transfer)

        mean_old, stein_old, rms_old = residual_moment_metrics(rraw_old, x0)
        mean_transfer, stein_transfer, rms_transfer = residual_moment_metrics(rraw_transfer, x0)
        push!(moment_old, Dict(:mean_norm => mean_old, :stein_norm => stein_old, :rms => rms_old))
        push!(moment_transfer, Dict(:mean_norm => mean_transfer,
            :stein_norm => stein_transfer, :rms => rms_transfer))
        shift = Matrix{Float64}(flatten_batch(sold_raw .- snew_raw))
        sold = Matrix{Float64}(flatten_batch(sold_raw))
        snew = Matrix{Float64}(flatten_batch(snew_raw))
        shift_rms = sqrt(mean(abs2, shift))
        push!(score_shift_rms, shift_rms)
        push!(old_score_rms, sqrt(mean(abs2, sold)))
        push!(new_score_rms, sqrt(mean(abs2, snew)))
        push!(shift_ratios, shift_rms / max(rms_old, eps(Float64)))
        @printf("lag %.6g (%d/%d): old rms %.4e, transfer rms %.4e, score-shift/r_old %.4e\n",
            lag * sampler.save_dt, li, nlag, rms_old, rms_transfer, last(shift_ratios))
        flush(stdout)
        GC.gc()
    end

    metrics = Dict{Symbol, Any}(
        :old_vs_phi_gfdt => relcorr(Cold, Cdot_phi_ref),
        :transfer_vs_phi_gfdt => relcorr(Ctransfer, Cdot_phi_ref),
        :old_vs_data_cdot => relcorr(Cold, Cdot_data_ref),
        :transfer_vs_data_cdot => relcorr(Ctransfer, Cdot_data_ref),
        :transfer_minus_old_rel_to_phi_gfdt => sqrt(mean(abs2, Ctransfer .- Cold)) /
            max(sqrt(mean(abs2, Cdot_phi_ref)), eps(Float64)),
        :old_residual_moments => mean_dictionary(moment_old),
        :transfer_residual_moments => mean_dictionary(moment_transfer),
        :mean_score_shift_rms => mean(score_shift_rms),
        :mean_old_score_rms => mean(old_score_rms),
        :mean_new_score_rms => mean(new_score_rms),
        :mean_score_shift_over_old_residual_rms => mean(shift_ratios),
        :new_score_completed_epoch => haskey(new_blob, :metadata) &&
            haskey(new_blob[:metadata], :completed_epoch) ? new_blob[:metadata][:completed_epoch] : missing,
        :conditional_completed_epoch => stack.epoch,
        :lags => lags .* sampler.save_dt,
        :pairs_per_lag => pairs_per_lag,
        :no_cheating_audit => "Transfer diagnostic used clean trajectory pairs, learned old/new stationary scores, the frozen learned conditional residual, and data-only Phi/Cdot artifacts. No analytic score or true mobility entered any estimate.")

    ensure_parent_dir(out_path)
    open(out_path, "w") do io
        println(io, "SoftSpinLLGChain cheap transferred conditional residual diagnostic")
        println(io, "Frozen conditional checkpoint = $(checkpoint_path)")
        println(io, "Old stationary score = $(stack.old_score_path)")
        println(io, "New stationary score = $(new_score_path)")
        println(io, "Matched Phi artifact = $(phi_path)")
        println(io, "Device request = $(device_name), required GPU name substring = $(required_gpu_name)")
        println(io, "conditional completed epoch = $(stack.epoch)")
        println(io, "new score completed epoch = $(metrics[:new_score_completed_epoch])")
        println(io, "operator lags = $(nlag)")
        println(io, "pairs per lag = $(pairs_per_lag)")
        for key in (:old_vs_phi_gfdt, :transfer_vs_phi_gfdt,
                :old_vs_data_cdot, :transfer_vs_data_cdot)
            m = metrics[key]
            println(io, @sprintf("%s rel.RMSE = %.8e", String(key), m[:relative_rmse]))
            println(io, @sprintf("%s corr = %.8e", String(key), m[:correlation]))
            println(io, @sprintf("%s cosine = %.8e", String(key), m[:cosine]))
            println(io, @sprintf("%s estimate RMS = %.8e", String(key), m[:estimate_rms]))
            println(io, @sprintf("%s reference RMS = %.8e", String(key), m[:reference_rms]))
        end
        println(io, @sprintf("transfer-minus-old RMS / Phi-GFDT RMS = %.8e",
            metrics[:transfer_minus_old_rel_to_phi_gfdt]))
        for (prefix, m) in (("old residual", metrics[:old_residual_moments]),
                ("transfer residual", metrics[:transfer_residual_moments]))
            println(io, @sprintf("%s mean ||E[r]||/sqrt(D) = %.8e", prefix, m[:mean_norm]))
            println(io, @sprintf("%s mean ||E[r x0']||/sqrt(D) = %.8e", prefix, m[:stein_norm]))
            println(io, @sprintf("%s mean raw RMS = %.8e", prefix, m[:rms]))
        end
        println(io, @sprintf("mean old stationary score RMS = %.8e", metrics[:mean_old_score_rms]))
        println(io, @sprintf("mean new stationary score RMS = %.8e", metrics[:mean_new_score_rms]))
        println(io, @sprintf("mean score-shift RMS = %.8e", metrics[:mean_score_shift_rms]))
        println(io, @sprintf("mean score-shift RMS / old residual RMS = %.8e",
            metrics[:mean_score_shift_over_old_residual_rms]))
        println(io, "No-cheating audit: $(metrics[:no_cheating_audit])")
    end

    ensure_parent_dir(artifact_path)
    BSON.bson(artifact_path, Dict(
        :metrics => metrics,
        :operator_old_residual => Cold,
        :operator_transfer_residual => Ctransfer,
        :Cdot_phi_ref => Cdot_phi_ref,
        :Cdot_data_ref => Cdot_data_ref,
        :observable_names => names,
        :observable_means => means,
        :Phi => Phi,
        :lags => lags,
        :taus => lags .* sampler.save_dt,
        :cond_config_path => cond_config_path,
        :checkpoint_path => checkpoint_path,
        :old_score_path => stack.old_score_path,
        :new_score_path => new_score_path,
        :phi_path => phi_path,
        :pairs_per_lag => pairs_per_lag,
        :seed => seed,
        :no_cheating_audit => metrics[:no_cheating_audit]))
    @printf("Saved transfer metrics to %s\n", out_path)
    @printf("Saved transfer curves to %s\n", artifact_path)
    return metrics
end

function main(args=ARGS)
    opts = parse_args(args)
    cond_config_path = resolve_cli_path(get(opts, "cond-config", DEFAULT_COND_CONFIG))
    checkpoint_path = resolve_cli_path(get(opts, "cond-checkpoint", DEFAULT_COND_CHECKPOINT))
    haskey(opts, "new-score") || error("Pass --new-score PATH.")
    haskey(opts, "phi") || error("Pass --phi PATH.")
    haskey(opts, "out") || error("Pass --out PATH.")
    new_score_path = resolve_cli_path(opts["new-score"])
    phi_path = resolve_cli_path(opts["phi"])
    out_path = resolve_cli_path(opts["out"])
    artifact_path = resolve_cli_path(get(opts, "artifact",
        replace(out_path, r"_metrics\.txt$" => "_curves.bson")))
    pairs_per_lag = parse(Int, get(opts, "pairs-per-lag", "50000"))
    max_lags = parse(Int, get(opts, "max-lags", "24"))
    batch_size = parse(Int, get(opts, "batch-size", "4096"))
    device_name = get(opts, "device", "GPU:2")
    required_gpu_name = get(opts, "required-gpu-name", "5070")
    seed = parse(Int, get(opts, "seed", "2026060902"))
    run_transfer_diagnostic(; cond_config_path, checkpoint_path, new_score_path,
        phi_path, out_path, artifact_path, pairs_per_lag, max_lags, batch_size,
        device_name, required_gpu_name, seed)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
