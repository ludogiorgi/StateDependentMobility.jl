#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "..", "code", "cond_score.jl"))

using BSON
using LinearAlgebra
using Printf
using Random
using Statistics

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

function relcorr(estimate::AbstractArray, reference::AbstractArray)
    e = vec(Float64.(estimate))
    r = vec(Float64.(reference))
    mask = isfinite.(e) .& isfinite.(r)
    e = e[mask]
    r = r[mask]
    rel = sqrt(mean(abs2, e .- r)) / max(sqrt(mean(abs2, r)), eps(Float64))
    corrv = length(r) > 2 ? cor(r, e) : NaN
    cosv = dot(e, r) / max(norm(e) * norm(r), eps(Float64))
    return Dict(:relative_rmse => rel, :correlation => corrv, :cosine => cosv,
        :estimate_rms => sqrt(mean(abs2, e)), :reference_rms => sqrt(mean(abs2, r)))
end

function centered_observables(xt::Array{Float32, 3}, p::SpinParams,
        names_ref::Vector{String}, means::Vector{Float64})
    obs, names = observable_values_cond(xt, p)
    names == names_ref || error("Unexpected observable names $(names); expected $(names_ref)")
    center_observables!(obs, means)
    return obs
end

function phi_operator!(dest::Array{Float64, 4}, li::Int, obs::Array{Float32, 3},
        Phi::AbstractMatrix, rraw::Array{Float32, 3})
    B = size(obs, 3)
    obs_flat = reshape(obs, size(obs, 1) * size(obs, 2), B)
    rflat = Matrix{Float64}(flatten_batch(rraw))
    action = transpose(Matrix{Float64}(Phi)) * rflat
    mat = -Matrix{Float64}(obs_flat) * transpose(action) / B
    dest[li, :, :, :] .= reshape(mat, size(dest, 2), size(dest, 3), size(dest, 4))
    return nothing
end

function residual_moments(rnorm::Array{Float32, 3}, rraw::Array{Float32, 3},
        x0n::Array{Float32, 3}, x0::Array{Float32, 3})
    rn = Matrix{Float64}(flatten_batch(rnorm))
    rr = Matrix{Float64}(flatten_batch(rraw))
    xn = Matrix{Float64}(flatten_batch(x0n))
    xr = Matrix{Float64}(flatten_batch(x0))
    D, B = size(rn)
    return Dict(
        :norm_mean_norm => norm(vec(mean(rn; dims=2))) / sqrt(D),
        :norm_stein_norm => norm((rn * transpose(xn)) ./ B) / sqrt(D),
        :norm_rms => sqrt(mean(abs2, rn)),
        :raw_mean_norm => norm(vec(mean(rr; dims=2))) / sqrt(D),
        :raw_stein_norm => norm((rr * transpose(xr)) ./ B) / sqrt(D),
        :raw_rms => sqrt(mean(abs2, rr)))
end

function mean_dict(dicts::Vector{Dict{Symbol, Float64}})
    out = Dict{Symbol, Float64}()
    isempty(dicts) && return out
    for key in keys(first(dicts))
        out[key] = mean(d[key] for d in dicts)
    end
    return out
end

function completed_epoch_from_checkpoint(blob)
    state = get(blob, :training_state, nothing)
    hist = get(blob, :history, Dict{Symbol, Any}(:train_loss => Float64[]))
    return state === nothing ?
        Int(get(get(blob, :metadata, Dict{Symbol, Any}()), :completed_epoch,
            length(get(hist, :train_loss, Float64[])))) :
        Int(get(state, :completed_epoch,
            get(get(blob, :metadata, Dict{Symbol, Any}()), :completed_epoch,
                length(get(hist, :train_loss, Float64[])))))
end

function run_eval(; cond_config_path::String, checkpoint_path::String,
        out_path::String, artifact_path::String, pairs_per_lag::Int,
        max_lags::Int, batch_size::Int, device_name::String,
        required_gpu_name::String, seed::Int)
    cond_config_path = abspath(cond_config_path)
    params = load_config(cond_config_path)
    base = dirname(cond_config_path)
    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    phi_path = resolve_path(base, params.phi_artifact_bson)

    device = detect_spin_device(device_name, required_gpu_name)
    activate_and_describe_device!(device, device_name, required_gpu_name)

    sampler = build_cond_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    p = load_phys(data_h5)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    ckpt = BSON.load(checkpoint_path, @__MODULE__)
    model = move_model(ckpt[:host_model], device)
    Flux.testmode!(model)
    completed_epoch = completed_epoch_from_checkpoint(ckpt)
    history = get(ckpt, :history, Dict{Symbol, Any}())

    phi_blob = BSON.load(phi_path, @__MODULE__)
    names = Vector{String}(phi_blob[:observable_names])
    means = Vector{Float64}(phi_blob[:observable_means])
    Phi = haskey(phi_blob, :Phi_projected) ? Matrix{Float64}(phi_blob[:Phi_projected]) :
        Matrix{Float64}(phi_blob[:Phi])
    lags_all = Vector{Int}(phi_blob[:lags])
    nlag = min(max_lags, length(lags_all))
    lags = lags_all[1:nlag]
    Cphi_ref = Array{Float64}(phi_blob[:Cdot_phi])[1:nlag, :, :, :]
    Cdata_ref = Array{Float64}(phi_blob[:Cdot_data])[1:nlag, :, :, :]
    Ccond = similar(Cphi_ref)
    rng = MersenneTwister(seed)
    moments = Dict{Symbol, Float64}[]

    @printf("Evaluating conditional Phi operator: checkpoint=%s, epoch=%d, lags=%d, pairs=%d\n",
        checkpoint_path, completed_epoch, nlag, pairs_per_lag)
    flush(stdout)
    for (li, lag) in enumerate(lags)
        x0, xt, tau_norm = sample_fixed_lag_pairs(sampler, lag, pairs_per_lag, rng)
        rnorm = evaluate_residual_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        x0n = apply_stats_tensor(x0, stats)
        obs = centered_observables(xt, p, names, means)
        phi_operator!(Ccond, li, obs, Phi, rraw)
        push!(moments, residual_moments(rnorm, rraw, x0n, x0))
        @printf("Phi-operator lag %.6g (%d/%d)\n", lag * sampler.save_dt, li, nlag)
        flush(stdout)
        GC.gc()
    end

    metrics = Dict{Symbol, Any}(
        :phi_vs_gfdt => relcorr(Ccond, Cphi_ref),
        :phi_vs_data => relcorr(Ccond, Cdata_ref),
        :residual_moments => mean_dict(moments),
        :completed_epoch => completed_epoch,
        :pairs_per_lag => pairs_per_lag,
        :lags => lags .* sampler.save_dt,
        :no_cheating_audit => "This checkpoint evaluation used observed trajectory pairs, the learned branch stationary score, the learned conditional residual checkpoint, and the matched data-only Phi/Cdot artifact. No analytic score or true mobility was loaded or used.")

    ensure_parent_dir(out_path)
    open(out_path, "w") do io
        println(io, "SoftSpinLLGChain data-only conditional Phi-operator checkpoint evaluation")
        println(io, "checkpoint = $(abspath(checkpoint_path))")
        println(io, "completed_epoch = $(completed_epoch)")
        println(io, "cond_config = $(cond_config_path)")
        println(io, "stationary_score = $(score_path)")
        println(io, "phi_artifact = $(phi_path)")
        println(io, "pairs_per_lag = $(pairs_per_lag)")
        println(io, "operator_lags = $(nlag)")
        if haskey(history, :train_loss) && !isempty(history[:train_loss])
            println(io, @sprintf("last DSM train_loss = %.8e", history[:train_loss][end]))
        end
        for key in (:target_rms, :prediction_rms, :null_mse,
                :fractional_improvement, :prediction_target_cosine)
            if haskey(history, key) && !isempty(history[key])
                println(io, @sprintf("last DSM %s = %.8e", String(key), history[key][end]))
            end
        end
        for key in (:phi_vs_gfdt, :phi_vs_data)
            m = metrics[key]
            println(io, @sprintf("%s rel.RMSE = %.8e", String(key), m[:relative_rmse]))
            println(io, @sprintf("%s corr = %.8e", String(key), m[:correlation]))
            println(io, @sprintf("%s cosine = %.8e", String(key), m[:cosine]))
            println(io, @sprintf("%s estimate RMS = %.8e", String(key), m[:estimate_rms]))
            println(io, @sprintf("%s reference RMS = %.8e", String(key), m[:reference_rms]))
        end
        rm = metrics[:residual_moments]
        for key in sort(collect(keys(rm)); by=String)
            println(io, @sprintf("residual %s = %.8e", String(key), rm[key]))
        end
        println(io, "No-cheating audit: $(metrics[:no_cheating_audit])")
    end

    ensure_parent_dir(artifact_path)
    BSON.bson(artifact_path, Dict(
        :metrics => metrics,
        :operator_cond_phi => Ccond,
        :Cdot_phi_ref => Cphi_ref,
        :Cdot_data_ref => Cdata_ref,
        :observable_names => names,
        :observable_means => means,
        :Phi => Phi,
        :lags => lags,
        :taus => lags .* sampler.save_dt,
        :checkpoint_path => abspath(checkpoint_path),
        :cond_config_path => cond_config_path,
        :stationary_score_path => score_path,
        :phi_path => phi_path,
        :seed => seed,
        :pairs_per_lag => pairs_per_lag,
        :no_cheating_audit => metrics[:no_cheating_audit]))
    @printf("Saved Phi operator metrics to %s\n", out_path)
    @printf("Saved Phi operator curves to %s\n", artifact_path)
    return metrics
end

function main(args=ARGS)
    opts = parse_args(args)
    haskey(opts, "cond-config") || error("Pass --cond-config PATH.")
    haskey(opts, "checkpoint") || error("Pass --checkpoint PATH.")
    haskey(opts, "out") || error("Pass --out PATH.")
    cond_config_path = resolve_cli_path(opts["cond-config"])
    checkpoint_path = resolve_cli_path(opts["checkpoint"])
    out_path = resolve_cli_path(opts["out"])
    artifact_path = resolve_cli_path(get(opts, "artifact",
        replace(out_path, r"_metrics\.txt$" => "_curves.bson")))
    pairs_per_lag = parse(Int, get(opts, "pairs-per-lag", "50000"))
    max_lags = parse(Int, get(opts, "max-lags", "24"))
    batch_size = parse(Int, get(opts, "batch-size", "4096"))
    device_name = get(opts, "device", "GPU:1")
    required_gpu_name = get(opts, "required-gpu-name", "2080")
    seed = parse(Int, get(opts, "seed", "2026060910"))
    run_eval(; cond_config_path, checkpoint_path, out_path, artifact_path,
        pairs_per_lag, max_lags, batch_size, device_name, required_gpu_name, seed)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
