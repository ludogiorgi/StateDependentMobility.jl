#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using BSON
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using Statistics

const PHYS_CFG = joinpath(@__DIR__, "..", "configs",
    "mobility11_analytic_fullcache_nosignal_gpu0.toml")
const STRICT_CFG = joinpath(@__DIR__, "..", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "configs",
    "fit_M_epoch240_transfer_warm_best_forward_gpu2.toml")
const TARGET = joinpath(@__DIR__, "..", "models", "mobility11", "analytic",
    "A_target.bson")
const OUT_PNG = joinpath(@__DIR__, "..", "figures", "manuscript_softspin_final",
    "spin_trueM_condscore_compare.png")
const OUT_CACHE = joinpath(@__DIR__, "..", "logs", "manuscript_softspin_final",
    "spin_trueM_condscore_compare.bson")

family_prefix(label::AbstractString) = join(split(split(label, " -> ")[1], "_")[1:2], "_")

function true_coefficients_for_action(p::SpinParams)
    return Float64[p.theta * p.eps, p.theta * p.alpha_perp,
        p.theta * p.alpha_parallel, -p.gamma * p.theta]
end

function onsite_transpose_action(raw::Array{Float32, 3}, rraw::Array{Float32, 3},
        coeff::AbstractVector{<:Real})
    N, _, B = size(raw)
    D = 3N
    out = Matrix{Float32}(undef, D, B)
    c0, cperp, cpar, cx = Float32.(coeff)
    @inbounds for b in 1:B, i in 1:N
        x1, x2, x3 = raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]
        r1, r2v, r3 = rraw[i, 1, b], rraw[i, 2, b], rraw[i, 3, b]
        rsq = x1 * x1 + x2 * x2 + x3 * x3
        dotxr = x1 * r1 + x2 * r2v + x3 * r3
        cross_t_1 = -(x2 * r3 - x3 * r2v)
        cross_t_2 = -(x3 * r1 - x1 * r3)
        cross_t_3 = -(x1 * r2v - x2 * r1)
        q = (i - 1) * 3
        out[q + 1, b] = c0 * r1 + cperp * (rsq * r1 - x1 * dotxr) +
            cpar * x1 * dotxr + cx * cross_t_1
        out[q + 2, b] = c0 * r2v + cperp * (rsq * r2v - x2 * dotxr) +
            cpar * x2 * dotxr + cx * cross_t_2
        out[q + 3, b] = c0 * r3 + cperp * (rsq * r3 - x3 * dotxr) +
            cpar * x3 * dotxr + cx * cross_t_3
    end
    return out
end

function selected_from_action(obs_flat::Matrix{Float32}, action_flat::Matrix{Float32},
        rraw::Array{Float32, 3}, target; operator_sign::Real=-1.0)
    B = size(obs_flat, 2)
    Phi = Matrix{Float32}(target[:Phi])
    rflat = reshape(permutedims(rraw, (2, 1, 3)), size(Phi, 1), B)
    phi_action = Phi' * rflat
    delta_action = action_flat .- phi_action
    mat = Float32(operator_sign) .* (obs_flat * transpose(delta_action)) ./ Float32(B)
    return Float32.(select_prediction_entries(mat, target[:selected_indices],
        target_group_index_matrix(target)))
end

function load_cond_stack(cond_cfg_path::AbstractString, device::ExecutionDevice)
    cond_cfg_path = abspath(cond_cfg_path)
    cond_cfg = load_config(cond_cfg_path)
    base = dirname(cond_cfg_path)
    score_model, stats, score_sigma, _ =
        load_stationary_checkpoint(resolve_path(base, cond_cfg.score_bson), device)
    cond_blob = BSON.load(resolve_path(base, cond_cfg.output_bson), @__MODULE__)
    cond_model = move_model(cond_blob[:host_model], device)
    Flux.testmode!(cond_model)
    return (; cond_cfg, score_model, stats, score_sigma, cond_model)
end

function transition_raw_for_stack(stack, x0, xt, tau_norm, device::ExecutionDevice)
    rnorm = evaluate_residual_norm(stack.cond_model, x0, xt, tau_norm, stack.stats,
        stack.cond_cfg, device; batch_size=min(stack.cond_cfg.batch_size, size(x0, 3)),
        score_model=stack.score_model, score_sigma=stack.score_sigma)
    return normalized_residual_to_raw(rnorm, stack.stats)
end

function physics_stack_from_dm_config(cfg_path::AbstractString, device::ExecutionDevice)
    cfg_path = abspath(cfg_path)
    cfg = load_dm_config(cfg_path)
    base = dirname(cfg_path)
    score_model, stats, score_sigma, _ =
        load_stationary_checkpoint(resolve_path(base, cfg.score_bson), device)
    cond_model, cond_cfg = load_transition_source(:conditional_residual, cfg_path,
        resolve_path(base, cfg.cond_score_bson), base, device)
    return (; cfg, cond_cfg, score_model, stats, score_sigma, cond_model)
end

function strict_cond_cfg_path()
    strict = abspath(STRICT_CFG)
    cfg = load_dm_config(strict)
    return resolve_path(dirname(strict), configured_cond_score_config(strict))
end

function compute_curves(; pairs_per_lag::Int=60000, device_name::String="GPU:2",
        required_gpu_name::String="5070")
    device = detect_spin_device(device_name, required_gpu_name)
    activate_and_describe_device!(device, device_name, required_gpu_name)

    phys_cfg = abspath(PHYS_CFG)
    cfg = load_dm_config(phys_cfg)
    base = dirname(phys_cfg)
    data_h5 = resolve_path(base, cfg.input_hdf5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    p = load_phys(data_h5)
    target = BSON.load(abspath(TARGET), @__MODULE__)

    labels = String.(target[:group_labels])
    ngroups = length(labels)
    active = configured_lag_indices(phys_cfg, length(Vector{Int}(target[:lags])))
    taus = Float64.(target[:taus])[active]
    lags = Vector{Int}(target[:lags])
    means = Vector{Float64}(target[:observable_means])
    lib = target_nonlinear_library(target)
    operator_sign = configured_operator_sign(phys_cfg)

    phys_stack = physics_stack_from_dm_config(phys_cfg, device)
    data_stack = load_cond_stack(strict_cond_cfg_path(), device)
    true_coeff = true_coefficients_for_action(p)

    pred_phys_cond = Matrix{Float32}(undef, length(active), ngroups)
    pred_data_cond = similar(pred_phys_cond)
    ref = Matrix{Float32}(undef, length(active), ngroups)
    rng = MersenneTwister(cfg.seed + 9917)

    for (ai, li) in enumerate(active)
        lag = lags[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, pairs_per_lag, rng)
        obs = nonlinear_observables(xt, p, lib; score_raw=nothing)
        center_observables!(obs, means)
        obs_flat = reshape(obs, size(obs, 1) * size(obs, 2), pairs_per_lag)

        r_phys = transition_raw_for_stack(phys_stack, x0, xt, tau_norm, device)
        pred_phys_cond[ai, :] .= selected_from_action(obs_flat,
            onsite_transpose_action(x0, r_phys, true_coeff), r_phys, target;
            operator_sign=operator_sign)

        r_data = transition_raw_for_stack(data_stack, x0, xt, tau_norm, device)
        pred_data_cond[ai, :] .= selected_from_action(obs_flat,
            onsite_transpose_action(x0, r_data, true_coeff), r_data, target;
            operator_sign=operator_sign)

        ref[ai, :] .= vec(Array{Float32}(target[:target_vec][li, :]))
        @printf("True-M cond-score comparison lag %.5g (%d/%d), pairs=%d\n",
            lags[li] * sampler.save_dt, ai, length(active), pairs_per_lag)
        flush(stdout)
        GC.gc()
    end

    relcorr(pred) = begin
        pvec = vec(Float64.(pred))
        rvec = vec(Float64.(ref))
        rel = sqrt(mean(abs2, pvec .- rvec)) / max(sqrt(mean(abs2, rvec)), eps(Float64))
        corr = dot(pvec, rvec) / max(norm(pvec) * norm(rvec), eps(Float64))
        pred_rms = sqrt(mean(abs2, pvec))
        target_rms = sqrt(mean(abs2, rvec))
        (; rel, corr, pred_rms, target_rms)
    end
    metrics = Dict(:trueM_phys_cond => relcorr(pred_phys_cond),
        :trueM_data_cond => relcorr(pred_data_cond))
    return (; taus, labels, ref, pred_phys_cond, pred_data_cond, metrics,
        active_lag_indices=active, pairs_per_lag,
        phys_cond_config=phys_cfg, data_cond_config=strict_cond_cfg_path())
end

function selected_columns(ref::AbstractMatrix, labels::Vector{String})
    families = unique(family_prefix.(labels))
    selected = Int[]
    for fam in families
        cols = findall(label -> family_prefix(label) == fam, labels)
        rms = [sqrt(mean(abs2, @view ref[:, c])) for c in cols]
        push!(selected, cols[argmax(rms)])
    end
    chosen = Set(selected)
    rrms = [sqrt(mean(abs2, @view ref[:, c])) for c in axes(ref, 2)]
    for c in sortperm(rrms; rev=true)
        if !(c in chosen)
            push!(selected, c)
            push!(families, "extra")
            break
        end
    end
    return families, selected
end

function render_compare(path::AbstractString, curves)
    labels = curves.labels
    families, cols = selected_columns(curves.ref, labels)
    with_scaled_figure_style(5200, 3900; scale_override=2.95) do _
        fig = Figure(; size=(5200, 3900), backgroundcolor=:white)
        for (idx, col) in enumerate(cols)
            row = 1 + (idx - 1) ÷ 4
            c = 1 + (idx - 1) % 4
            rr = Float64.(@view curves.ref[:, col])
            yy_phys = Float64.(@view curves.pred_phys_cond[:, col])
            yy_data = Float64.(@view curves.pred_data_cond[:, col])
            ax = Axis(fig[row, c];
                title="$(families[idx])",
                xlabel=row == 3 ? L"t" : "",
                ylabel=c == 1 ? L"A_{mn}(t)" : "",
                titlesize=current_figure_style().axis_titlesize,
                xlabelsize=current_figure_style().axis_labelsize,
                ylabelsize=current_figure_style().axis_labelsize,
                xticklabelsize=current_figure_style().axis_ticklabelsize,
                yticklabelsize=current_figure_style().axis_ticklabelsize)
            lines!(ax, curves.taus, rr; color=:black,
                linewidth=curve_linewidth(; emphasis=1.05))
            lines!(ax, curves.taus, yy_phys; color=STYLE_REFERENCE,
                linewidth=curve_linewidth(; emphasis=0.95), linestyle=:dot)
            lines!(ax, curves.taus, yy_data; color=STYLE_ACCENT,
                linewidth=curve_linewidth(; emphasis=0.95), linestyle=:dash)
            text!(ax, curves.taus[1], maximum(vcat(rr, yy_phys, yy_data));
                text=labels[col], fontsize=current_figure_style().text_fontsize * 0.72,
                color=STYLE_MUTED, align=(:left, :top))
        end
        elems = [
            LineElement(color=:black, linewidth=curve_linewidth()),
            LineElement(color=STYLE_REFERENCE, linewidth=curve_linewidth(), linestyle=:dot),
            LineElement(color=STYLE_ACCENT, linewidth=curve_linewidth(), linestyle=:dash),
        ]
        legend_labels = [
            "data target",
            "true M + physics cond. score",
            "true M + data-only cond. score",
        ]
        Legend(fig[4, 1:4], elems, legend_labels; orientation=:horizontal,
            framevisible=false, labelsize=current_figure_style().legend_labelsize)
        rowgap!(fig.layout, 20)
        colgap!(fig.layout, 22)
        save_figure_checked(path, fig)
    end
    @printf("Saved true-M conditional-score comparison figure to %s\n", path)
end

function write_metrics(path::AbstractString, curves)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain true-M A operator comparison across conditional scores")
        println(io, "pairs_per_lag = $(curves.pairs_per_lag)")
        println(io, "active_lag_indices = $(first(curves.active_lag_indices)):$(last(curves.active_lag_indices))")
        println(io, "physics_cond_config = $(curves.phys_cond_config)")
        println(io, "data_cond_config = $(curves.data_cond_config)")
        for key in (:trueM_phys_cond, :trueM_data_cond)
            m = curves.metrics[key]
            @printf(io, "%s_rel_rmse = %.8e\n", String(key), m.rel)
            @printf(io, "%s_corr = %.8e\n", String(key), m.corr)
            @printf(io, "%s_pred_rms = %.8e\n", String(key), m.pred_rms)
            @printf(io, "%s_target_rms = %.8e\n", String(key), m.target_rms)
        end
    end
end

function main()
    out = length(ARGS) >= 1 ? ARGS[1] : OUT_PNG
    cache = length(ARGS) >= 2 ? ARGS[2] : OUT_CACHE
    pairs = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 60000
    if isfile(cache)
        @printf("Loading cached curves from %s\n", cache)
        curves = BSON.load(cache, @__MODULE__)[:curves]
    else
        curves = compute_curves(; pairs_per_lag=pairs)
        ensure_parent_dir(cache)
        BSON.bson(cache, Dict(:curves => curves))
        @printf("Saved curve cache to %s\n", cache)
    end
    write_metrics(replace(cache, r"\.bson$" => ".txt"), curves)
    render_compare(out, curves)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
