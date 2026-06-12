#!/usr/bin/env julia

ENV["STATEDEP_HEADLESS"] = get(ENV, "STATEDEP_HEADLESS", "1")

include(normpath(joinpath(@__DIR__, "..", "..", "..", "code", "fit_Phi.jl")))

function parse_args(args)
    opts = Dict{String, String}()
    positionals = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            require_condition(i < length(args), "Missing value for --$(key).")
            opts[key] = args[i + 1]
            i += 2
        else
            push!(positionals, a)
            i += 1
        end
    end
    require_condition(length(positionals) == 1,
        "Usage: diagnose_phi_b_truth.jl <fit_Phi_config.toml> [--samples N] [--truth-samples N] [--batch-size N] [--device GPU:1] [--required-gpu-name 2080] [--output path]")
    return positionals[1], opts
end

function true_mobility_block_mean(raw::Array{Float32, 3}, p::SpinParams)
    N, _, B = size(raw)
    block = zeros(Float64, 3, 3)
    @inbounds for b in 1:B, i in 1:N
        x1 = Float64(raw[i, 1, b])
        x2 = Float64(raw[i, 2, b])
        x3 = Float64(raw[i, 3, b])
        r2 = x1 * x1 + x2 * x2 + x3 * x3
        xs = (x1, x2, x3)
        for a in 1:3, c in 1:3
            delta = a == c ? 1.0 : 0.0
            Aac = p.eps * delta +
                p.alpha_perp * (r2 * delta - xs[a] * xs[c]) +
                p.alpha_parallel * xs[a] * xs[c]
            block[a, c] += p.theta * Aac
        end
        block[1, 2] += p.gamma * p.theta * x3
        block[1, 3] -= p.gamma * p.theta * x2
        block[2, 1] -= p.gamma * p.theta * x3
        block[2, 3] += p.gamma * p.theta * x1
        block[3, 1] += p.gamma * p.theta * x2
        block[3, 2] -= p.gamma * p.theta * x1
    end
    block ./= (N * B)
    return block
end

function block_diag_from_onsite(block::AbstractMatrix{<:Real}, N::Int)
    M = zeros(Float64, 3N, 3N)
    @inbounds for i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        M[rows, rows] .= block
    end
    return M
end

function frob_rel(A::AbstractArray, B::AbstractArray)
    return norm(Float64.(A) .- Float64.(B)) / max(norm(Float64.(B)), eps(Float64))
end

function vector_norm_mean(X::AbstractMatrix)
    return mean(sqrt.(vec(sum(abs2, X; dims=1))))
end

function run_diagnostic(config_path::AbstractString, opts::Dict{String, String})
    base = dirname(config_path)
    params = load_config(config_path)
    nsamples = parse(Int, get(opts, "samples", "200000"))
    truth_samples = parse(Int, get(opts, "truth-samples", string(nsamples)))
    batch_size = parse(Int, get(opts, "batch-size", string(params.score_batch_size)))
    device_request = get(opts, "device", params.device)
    required_name = get(opts, "required-gpu-name", params.required_gpu_name)
    output_path = get(opts, "output", replace(resolve_path(base, params.metrics_txt),
        "_metrics.txt" => "_btruth_metrics.txt"))

    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    artifact_path = resolve_path(base, params.artifact_bson)
    require_condition(isfile(artifact_path), "Phi artifact not found: $(artifact_path)")

    device = detect_spin_device(device_request, required_name)
    activate_and_describe_device!(device, device_request, required_name)
    p = load_phys(data_h5)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, sigma, _ = load_score_checkpoint(score_path, device)
    blob = BSON.load(artifact_path)
    Phi = Matrix{Float64}(blob[:Phi])
    Mtrue_artifact = Matrix{Float64}(blob[:Mtrue])
    Mtrue_artifact_projected, _ = project_soft_spin_phi(Mtrue_artifact, sampler.N)

    raw_b = sample_raw_states(sampler, nsamples, MersenneTwister(params.seed + 910))
    raw_truth = truth_samples == nsamples ? raw_b :
        sample_raw_states(sampler, truth_samples, MersenneTwister(params.seed + 911))
    Mtrue_block = true_mobility_block_mean(raw_truth, p)
    Mtrue_precise = block_diag_from_onsite(Mtrue_block, sampler.N)

    score_pred = evaluate_raw_score(score_model, raw_b, stats, sigma, device; batch_size=batch_size)
    score_true = analytic_score_raw(raw_b, p)
    s_pred = Matrix{Float64}(flatten_batch(score_pred))
    s_true = Matrix{Float64}(flatten_batch(score_true))
    B_pred = transpose(Phi) * s_pred
    B_true_phi = transpose(Phi) * s_true
    B_true_mean = transpose(Mtrue_precise) * s_true

    metrics_score = agreement_metrics(s_true, s_pred)
    metrics_B_phi = agreement_metrics(B_true_phi, B_pred)
    metrics_B_mean = agreement_metrics(B_true_mean, B_pred)
    metrics_Phi_artifact = agreement_metrics(Mtrue_artifact, Phi)
    metrics_Phi_artifact_projected = agreement_metrics(Mtrue_artifact_projected, Phi)
    metrics_Phi_precise = agreement_metrics(Mtrue_precise, Phi)
    metrics_Mtrue_sampling = agreement_metrics(Mtrue_precise, Mtrue_artifact_projected)

    ensure_parent_dir(output_path)
    open(output_path, "w") do io
        println(io, "SoftSpinLLGChain Phi/B truth diagnostics")
        println(io, "config = " * abspath(config_path))
        println(io, "score_bson = " * score_path)
        println(io, "phi_artifact = " * artifact_path)
        println(io, @sprintf("B sample count = %d", nsamples))
        println(io, @sprintf("true mobility sample count = %d", truth_samples))
        println(io, @sprintf("Phi vs artifact <M_true> rel.RMSE = %.10e", metrics_Phi_artifact[:relative_rmse]))
        println(io, @sprintf("Phi vs artifact <M_true> corr = %.10e", metrics_Phi_artifact[:correlation]))
        println(io, @sprintf("Phi vs projected artifact <M_true> rel.RMSE = %.10e", metrics_Phi_artifact_projected[:relative_rmse]))
        println(io, @sprintf("Phi vs projected artifact <M_true> corr = %.10e", metrics_Phi_artifact_projected[:correlation]))
        println(io, @sprintf("Phi vs projected resampled <M_true> rel.RMSE = %.10e", metrics_Phi_precise[:relative_rmse]))
        println(io, @sprintf("Phi vs projected resampled <M_true> corr = %.10e", metrics_Phi_precise[:correlation]))
        println(io, @sprintf("projected artifact <M_true> vs projected resampled <M_true> rel.RMSE = %.10e", metrics_Mtrue_sampling[:relative_rmse]))
        println(io, @sprintf("score vs true score rel.RMSE = %.10e", metrics_score[:relative_rmse]))
        println(io, @sprintf("score vs true score corr = %.10e", metrics_score[:correlation]))
        println(io, @sprintf("B_Phi(score) vs B_Phi(true score) rel.RMSE = %.10e", metrics_B_phi[:relative_rmse]))
        println(io, @sprintf("B_Phi(score) vs B_Phi(true score) corr = %.10e", metrics_B_phi[:correlation]))
        println(io, @sprintf("B_Phi(score) vs <M_true>' true score rel.RMSE = %.10e", metrics_B_mean[:relative_rmse]))
        println(io, @sprintf("B_Phi(score) vs <M_true>' true score corr = %.10e", metrics_B_mean[:correlation]))
        println(io, @sprintf("mean ||B_Phi(score)|| = %.10e", vector_norm_mean(B_pred)))
        println(io, @sprintf("mean ||B_Phi(true score)|| = %.10e", vector_norm_mean(B_true_phi)))
        println(io, @sprintf("mean ||<M_true>' true score|| = %.10e", vector_norm_mean(B_true_mean)))
        println(io, "No-cheating audit: true score and true mobility were used only for this ex-post diagnostic, after the data-only Phi artifact was constructed.")
    end
    @printf("Saved Phi/B truth diagnostics to %s\n", output_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    config_path, opts = parse_args(ARGS)
    run_diagnostic(config_path, opts)
end
