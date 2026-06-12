#!/usr/bin/env julia

include(joinpath(@__DIR__, "score.jl"))

using BSON
using HDF5
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using Statistics

const DEFAULT_DATA_H5 = normpath(joinpath(@__DIR__, "..", "data", "soft_spin_llg_chain.h5"))
const N_SITES = 12
const SPIN_DIM = 3
const D_STATE = N_SITES * SPIN_DIM

env_float(name::AbstractString, default::Real) =
    haskey(ENV, name) ? parse(Float64, ENV[name]) : Float64(default)

env_int(name::AbstractString, default::Integer) =
    haskey(ENV, name) ? parse(Int, ENV[name]) : Int(default)

const FIGURE_WIDTH = env_int("SOFTSPIN_PHITILDE_FIG_WIDTH", 2200)
const FIGURE_HEIGHT = env_int("SOFTSPIN_PHITILDE_FIG_HEIGHT", 2000)
const FIGURE_FONT_SCALE = env_float("SOFTSPIN_PHITILDE_FONT_SCALE", 2.35)

function usage()
    println("""
    Usage:
      julia render_phitilde_structure_diagnostic.jl SCORE_BSON OUT_PNG CACHE_BSON METRICS_TXT LABEL [DATA_H5] [PAIRS_PER_LAG] [DEVICE] [REQUIRED_GPU_NAME]

    Computes and renders the integrated constant-mobility structure diagnostic
    from Appendix app:phitilde_structure_diagnostic:

        C_xx(T)-C_xx(0) ~= (int_0^T <(x_t-mu) s_hat(x_0)^T> dt) * Phi_tilde

    stacked over horizons in T = {tau_k}.  The rendered heatmap is the normalized
    entry score rho_ij = |Phi_tilde_ij| / max_kl |Phi_tilde_kl|.
    """)
end

function load_stationary_checkpoint_local(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path, @__MODULE__)
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    stats_obj = blob[:stats]
    stats = stats_obj isa DataStats ? stats_obj :
        DataStats(Float32.(stats_obj[:mean]), Float32.(stats_obj[:std]))
    sigma = Float32(get(get(blob, :trainer_cfg, Dict{Symbol, Any}()), :sigma, 0.05f0))
    return model, stats, sigma, blob
end

function flat_to_site(xflat::AbstractMatrix{Float32})
    B = size(xflat, 2)
    out = Array{Float32}(undef, N_SITES, SPIN_DIM, B)
    @inbounds for b in 1:B, i in 1:N_SITES, c in 1:SPIN_DIM
        out[i, c, b] = xflat[(i - 1) * SPIN_DIM + c, b]
    end
    return out
end

function sample_flat_pairs(states::Array{Float32, 3}, start_idx::Int, lag::Int,
        npairs::Int, rng::AbstractRNG)
    nt, D, ntraj = size(states)
    upper = nt - lag
    upper >= start_idx || error("Lag $(lag) exceeds available post-burn trajectory window.")
    x0 = Matrix{Float32}(undef, D, npairs)
    xt = Matrix{Float32}(undef, D, npairs)
    @inbounds for b in 1:npairs
        tr = rand(rng, 1:ntraj)
        t = rand(rng, start_idx:upper)
        for d in 1:D
            x0[d, b] = states[t, d, tr]
            xt[d, b] = states[t + lag, d, tr]
        end
    end
    return x0, xt
end

function score_raw_flat(model, stats::DataStats, x0flat::Matrix{Float32},
        score_sigma::Float32, device::ExecutionDevice; batch_size::Int=4096)
    xsite = flat_to_site(x0flat)
    xnorm = apply_stats_tensor(xsite, stats)
    snorm = evaluate_score_norm(model, xnorm, score_sigma, device; batch_size)
    sraw = normalized_score_to_raw(snorm, stats)
    return flatten_batch(sraw)
end

function solve_multihorizon_phitilde(Gs::Array{Float64, 3}, Cs::Array{Float64, 3},
        taus::Vector{Float64}; fit_start::Int=4, ridge_rel::Float64=1.0e-8)
    K = length(taus)
    intG = zeros(Float64, K, D_STATE, D_STATE)
    for k in 2:K
        dt = taus[k] - taus[k - 1]
        intG[k, :, :] .= intG[k - 1, :, :] .+
            0.5 * dt .* (Gs[k, :, :] .+ Gs[k - 1, :, :])
    end
    rows = (K - fit_start + 1) * D_STATE
    A = Matrix{Float64}(undef, rows, D_STATE)
    Y = Matrix{Float64}(undef, rows, D_STATE)
    C0 = @view Cs[1, :, :]
    pos = 1
    for k in fit_start:K
        A[pos:(pos + D_STATE - 1), :] .= @view intG[k, :, :]
        Y[pos:(pos + D_STATE - 1), :] .= (@view Cs[k, :, :]) .- C0
        pos += D_STATE
    end
    AtA = transpose(A) * A
    ridge = ridge_rel * tr(AtA) / D_STATE
    Phi_tilde = (AtA + ridge * I) \ (transpose(A) * Y)
    rel = norm(A * Phi_tilde - Y) / max(norm(Y), eps(Float64))
    return Phi_tilde, intG, rel, ridge
end

function block_norm_matrix(A::AbstractMatrix{<:Real})
    out = zeros(Float64, N_SITES, N_SITES)
    @inbounds for i in 1:N_SITES, j in 1:N_SITES
        rows = ((i - 1) * SPIN_DIM + 1):(i * SPIN_DIM)
        cols = ((j - 1) * SPIN_DIM + 1):(j * SPIN_DIM)
        out[i, j] = norm(@view A[rows, cols])
    end
    return out
end

function separation_profile(blocknorm::AbstractMatrix{<:Real})
    sep_mean = zeros(Float64, div(N_SITES, 2) + 1)
    sep_max = similar(sep_mean)
    for d in 0:div(N_SITES, 2)
        vals = Float64[]
        for i in 1:N_SITES, j in 1:N_SITES
            sep = min(abs(i - j), N_SITES - abs(i - j))
            sep == d && push!(vals, Float64(blocknorm[i, j]))
        end
        sep_mean[d + 1] = mean(vals)
        sep_max[d + 1] = maximum(vals)
    end
    return sep_mean, sep_max
end

function structure_metrics(Phi_tilde::AbstractMatrix{<:Real})
    rho = abs.(Float64.(Phi_tilde))
    rho ./= max(maximum(rho), eps(Float64))
    blocknorm = block_norm_matrix(Phi_tilde)
    rho_block = blocknorm ./ max(maximum(blocknorm), eps(Float64))
    onsite = [blocknorm[i, i] for i in 1:N_SITES]
    offsite = [blocknorm[i, j] for i in 1:N_SITES for j in 1:N_SITES if i != j]
    sep_mean, sep_max = separation_profile(blocknorm)
    sep_mean_norm = sep_mean ./ max(maximum(sep_mean), eps(Float64))
    sep_max_norm = sep_max ./ max(maximum(sep_max), eps(Float64))
    return (; rho, blocknorm, rho_block, onsite, offsite, sep_mean, sep_max,
        sep_mean_norm, sep_max_norm,
        onsite_mean=mean(onsite),
        offsite_mean=mean(offsite),
        offsite_max=maximum(offsite),
        offsite_mean_over_onsite=mean(offsite) / max(mean(onsite), eps(Float64)),
        offsite_fro_fraction=sqrt(sum(abs2, offsite)) / max(norm(blocknorm), eps(Float64)))
end

function compute_diagnostic(score_path::AbstractString, data_h5::AbstractString,
        pairs_per_lag::Int, device_name::AbstractString, required_gpu_name::AbstractString)
    device = detect_spin_device(device_name, required_gpu_name)
    activate_and_describe_device!(device, device_name, required_gpu_name)

    model, stats, score_sigma, _ = load_stationary_checkpoint_local(score_path, device)
    times = h5read(data_h5, "/trajectories/time")
    states = Float32.(h5read(data_h5, "/trajectories/states_flat"))
    nt = size(states, 1)
    start_idx = max(1, floor(Int, 0.1 * nt) + 1)
    save_dt = Float64(times[2] - times[1])
    tD = Float64(h5read(data_h5, "/statistics/correlations/t_decorrelation"))
    max_lag = min(round(Int, tD / save_dt), nt - start_idx - 1)
    lags = collect(0:2:max_lag)
    taus = Float64.(lags) .* save_dt
    K = length(lags)
    mu = Float64.(mean_flat(stats))
    Cs = Array{Float64}(undef, K, D_STATE, D_STATE)
    Gs = similar(Cs)
    rng = MersenneTwister(2026060901)

    @printf("Estimating phitilde diagnostic: score=%s\n", score_path)
    @printf("  lags=%d, max tau=%.6g, pairs/lag=%d\n", K, taus[end], pairs_per_lag)
    for (k, lag) in enumerate(lags)
        x0, xt = sample_flat_pairs(states, start_idx, lag, pairs_per_lag, rng)
        s0 = Float64.(score_raw_flat(model, stats, x0, score_sigma, device; batch_size=4096))
        x0c = Float64.(x0)
        xtc = Float64.(xt)
        x0c .-= mu
        xtc .-= mu
        Cs[k, :, :] .= (xtc * transpose(x0c)) ./ pairs_per_lag
        Gs[k, :, :] .= (xtc * transpose(s0)) ./ pairs_per_lag
        @printf("  lag %4d/%4d  tau %.6f\n", lag, max_lag, taus[k])
        flush(stdout)
    end

    Phi_tilde, intG, rel_fit, ridge = solve_multihorizon_phitilde(Gs, Cs, taus)
    metrics = structure_metrics(Phi_tilde)
    return (; Phi_tilde, Cs, Gs, intG, taus, lags, rel_fit, ridge, score_sigma,
        score_path, data_h5, pairs_per_lag, metrics)
end

function load_or_compute(score_path::AbstractString, data_h5::AbstractString,
        cache_path::AbstractString, pairs_per_lag::Int, device_name::AbstractString,
        required_gpu_name::AbstractString; force::Bool=false)
    if isfile(cache_path) && !force
        @printf("Loading cached phitilde diagnostic from %s\n", cache_path)
        blob = BSON.load(cache_path)
        if haskey(blob, :result)
            return blob[:result]
        end
        return (; blob...)
    end
    result = compute_diagnostic(score_path, data_h5, pairs_per_lag, device_name,
        required_gpu_name)
    ensure_parent_dir(cache_path)
    BSON.@save cache_path result
    @printf("Saved phitilde cache to %s\n", cache_path)
    return result
end

function draw_block_grid!(ax, n::Int; step::Int=3, color=(:white, 0.38))
    for k in step:step:(n - step)
        vlines!(ax, [k + 0.5]; color, linewidth=1.4)
        hlines!(ax, [k + 0.5]; color, linewidth=1.4)
    end
    return nothing
end

function render_phitilde_figure(path::AbstractString, result, label::AbstractString)
    metrics = result.metrics
    rho = metrics.rho

    with_scaled_figure_style(FIGURE_WIDTH, FIGURE_HEIGHT; scale_override=FIGURE_FONT_SCALE) do _
        fig = Figure(; size=(FIGURE_WIDTH, FIGURE_HEIGHT), backgroundcolor=:white)

        ax1 = Axis(fig[1, 1];
            title=L"|\tilde{\Phi}_{ij}|/\max|\tilde{\Phi}|",
            xlabel=L"j", ylabel=L"i", aspect=DataAspect(),
            xticks=([1, 12, 24, 36], ["1", "12", "24", "36"]),
            yticks=([1, 12, 24, 36], ["1", "12", "24", "36"]),
            titlesize=current_figure_style().axis_titlesize,
            xlabelsize=current_figure_style().axis_labelsize,
            ylabelsize=current_figure_style().axis_labelsize,
            xticklabelsize=current_figure_style().axis_ticklabelsize,
            yticklabelsize=current_figure_style().axis_ticklabelsize)
        hm1 = heatmap!(ax1, 1:D_STATE, 1:D_STATE, rho; colormap=STYLE_SEQUENTIAL_BLUE,
            colorrange=(0, 1))
        draw_block_grid!(ax1, D_STATE)
        xlims!(ax1, 0.5, D_STATE + 0.5)
        ylims!(ax1, 0.5, D_STATE + 0.5)
        Colorbar(fig[1, 2], hm1;
            ticklabelsize=current_figure_style().colorbar_ticklabelsize)

        colgap!(fig.layout, 22)
        rowgap!(fig.layout, 12)
        save_figure_checked(path, fig)
    end
    @printf("Saved phitilde structure figure to %s\n", path)
end

function write_metrics(path::AbstractString, result, label::AbstractString)
    ensure_parent_dir(path)
    m = result.metrics
    open(path, "w") do io
        println(io, "Integrated constant-mobility structure diagnostic")
        println(io, "label = $(label)")
        println(io, "score_path = $(result.score_path)")
        println(io, "data_h5 = $(result.data_h5)")
        println(io, "pairs_per_lag = $(result.pairs_per_lag)")
        println(io, "lag_steps = $(first(result.lags)):2:$(last(result.lags))")
        @printf(io, "tau_max = %.10g\n", result.taus[end])
        @printf(io, "score_sigma = %.8g\n", result.score_sigma)
        @printf(io, "stacked_multihorizon_rel_residual = %.8e\n", result.rel_fit)
        @printf(io, "ridge_lambda = %.8e\n", result.ridge)
        @printf(io, "onsite_block_norm_mean = %.8e\n", m.onsite_mean)
        @printf(io, "offsite_block_norm_mean = %.8e\n", m.offsite_mean)
        @printf(io, "offsite_block_norm_max = %.8e\n", m.offsite_max)
        @printf(io, "offsite_mean_over_onsite_mean = %.8e\n", m.offsite_mean_over_onsite)
        @printf(io, "offsite_fro_fraction_blocknorm = %.8e\n", m.offsite_fro_fraction)
        println(io, "No-cheating audit: this diagnostic uses observed lagged coordinate pairs and the selected stationary score only. Analytic mobility, analytic score, and simulator coefficients are not used in the fit or support scores.")
    end
    @printf("Saved phitilde metrics to %s\n", path)
end

function main()
    if length(ARGS) < 5
        usage()
        exit(1)
    end
    score_path = abspath(ARGS[1])
    out_png = ARGS[2]
    cache_bson = ARGS[3]
    metrics_txt = ARGS[4]
    label = ARGS[5]
    data_h5 = length(ARGS) >= 6 ? ARGS[6] : DEFAULT_DATA_H5
    pairs_per_lag = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 80000
    device_name = length(ARGS) >= 8 ? ARGS[8] : "GPU:2"
    required_gpu_name = length(ARGS) >= 9 ? ARGS[9] : "5070"
    force = any(arg -> arg == "--force", ARGS)

    result = load_or_compute(score_path, data_h5, cache_bson, pairs_per_lag,
        device_name, required_gpu_name; force)
    render_phitilde_figure(out_png, result, label)
    write_metrics(metrics_txt, result, label)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
