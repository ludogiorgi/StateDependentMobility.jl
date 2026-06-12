#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using BSON
using LaTeXStrings
using LinearAlgebra
using Printf
using Statistics

const DEFAULT_CFG = joinpath(@__DIR__, "..", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "configs",
    "fit_M_epoch240_transfer_warm_best_forward_gpu2.toml")
const DEFAULT_MODEL = joinpath(@__DIR__, "..", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "models",
    "M_epoch240_transfer_warm_gpu1_best_snapshot_forward.bson")
const DEFAULT_TARGET = joinpath(@__DIR__, "..", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "models",
    "A_epoch240_noncheating_globalscale.bson")
const DEFAULT_PHYS = joinpath(@__DIR__, "..", "models",
    "dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.bson")
const DEFAULT_DATA_COND_CFG = joinpath(@__DIR__, "..", "configs",
    "cond_score_dataonly_seed602_scaled_gpu0.toml")
const DEFAULT_OUT = joinpath(@__DIR__, "..", "figures", "manuscript_softspin_final",
    "spin_residuals.png")
const DEFAULT_CACHE = joinpath(@__DIR__, "..", "logs", "manuscript_softspin_final",
    "spin_residual_fourway_trainingcond.bson")

env_float(name::AbstractString, default::Real) =
    haskey(ENV, name) ? parse(Float64, ENV[name]) : Float64(default)

env_int(name::AbstractString, default::Integer) =
    haskey(ENV, name) ? parse(Int, ENV[name]) : Int(default)

const FIGURE_WIDTH = env_int("SOFTSPIN_RESIDUAL_FIG_WIDTH", 5200)
const FIGURE_HEIGHT = env_int("SOFTSPIN_RESIDUAL_FIG_HEIGHT", 3900)
const FIGURE_FONT_SCALE = env_float("SOFTSPIN_RESIDUAL_FONT_SCALE", 2.95)
const FIGURE_LEGEND_FONT_SCALE = env_float("SOFTSPIN_RESIDUAL_LEGEND_FONT_SCALE", 1.55)

family_prefix(label::AbstractString) = join(split(split(label, " -> ")[1], "_")[1:2], "_")

component_latex(name::AbstractString) = begin
    s = strip(name)
    if s == "mx"
        "x"
    elseif s == "my"
        "y"
    elseif s == "mz"
        "z"
    else
        replace(s, "_" => "\\_")
    end
end

function observable_family_latex(name::AbstractString)
    s = strip(name)
    if startswith(s, "rb") && length(s) > 2
        return "b_{" * s[3:end] * "}"
    end
    table = Dict(
        "u" => "u",
        "Tu" => "T_u",
        "Pu" => "P_u",
        "Cu" => "C_u",
        "a" => "a",
        "Ta" => "T_a",
        "Pa" => "P_a",
        "Ca" => "C_a",
    )
    return get(table, s, "\\mathrm{" * replace(s, "_" => "\\_") * "}")
end

function latex_channel_title(label::AbstractString)
    pieces = split(label, " -> ")
    length(pieces) == 2 || return latexstring("\\mathrm{" * replace(label, "_" => "\\_") * "}")

    lhs = strip(pieces[1])
    rhs_sep = split(strip(pieces[2]), ", sep ")
    length(rhs_sep) == 2 || return latexstring("\\mathrm{" * replace(label, "_" => "\\_") * "}")

    rhs = component_latex(rhs_sep[1])
    sep = strip(rhs_sep[2])
    lhs_parts = split(lhs, "_")
    length(lhs_parts) >= 3 || return latexstring("\\mathrm{" * replace(label, "_" => "\\_") * "}")

    obs_comp = component_latex(lhs_parts[end])
    fam = observable_family_latex(join(lhs_parts[2:end-1], "_"))
    return latexstring("\\mathcal{A}_{\\phi_{" * fam * "," * obs_comp * "},\\,m_{" *
        rhs * "}}^{(" * sep * ")}(\\tau)")
end

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
        # [m]_x^T r = -m x r.
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
    return (; cond_kind=:conditional_residual, cond_cfg, score_model, stats,
        score_sigma, cond_model)
end

function load_dm_stack(cfg_path::AbstractString, cfg, base::AbstractString,
        device::ExecutionDevice)
    score_model, stats, score_sigma, _ =
        load_stationary_checkpoint(resolve_path(base, cfg.score_bson), device)
    cond_kind = configured_cond_score_kind(cfg_path)
    cond_model, cond_cfg = load_transition_source(cond_kind, cfg_path,
        resolve_path(base, cfg.cond_score_bson), base, device)
    if cond_kind == :transferred_residual
        source = cond_model::TransferredResidualSource
        @printf("Using transferred residual source from %s with old score %s\n",
            cfg_path, source.old_score_path)
    else
        @printf("Using transition source kind %s from %s\n", String(cond_kind), cfg_path)
    end
    return (; cond_kind, cond_cfg, score_model, stats, score_sigma, cond_model)
end

function looks_like_dm_config(path::AbstractString)
    raw = TOML.parsefile(path)
    haskey(raw, "data") || return false
    data = raw["data"]
    return haskey(data, "cond_score_bson") && haskey(raw, "targets")
end

function load_secondary_stack(path::AbstractString, primary_stack, device::ExecutionDevice)
    stripped = strip(path)
    if isempty(stripped) || lowercase(stripped) in ("same", "branch", "__branch__")
        return primary_stack
    end
    stack_path = abspath(stripped)
    if looks_like_dm_config(stack_path)
        cfg = load_dm_config(stack_path)
        return load_dm_stack(stack_path, cfg, dirname(stack_path), device)
    end
    return load_cond_stack(stack_path, device)
end

function transition_raw_for_stack(stack, x0, xt, tau_norm, device::ExecutionDevice)
    rnorm = evaluate_transition_norm(stack.cond_kind, stack.cond_model, x0, xt,
        tau_norm, stack.stats, stack.cond_cfg, device;
        batch_size=min(stack.cond_cfg.batch_size, size(x0, 3)),
        score_model=stack.score_model, score_sigma=stack.score_sigma)
    return normalized_residual_to_raw(rnorm, stack.stats)
end

function operator_curves(cfg_path::AbstractString, model_path::AbstractString,
        target_path::AbstractString, phys_path::AbstractString,
        data_cond_cfg_path::AbstractString; pairs_per_lag::Int=60000,
        device_name::String="GPU:2", required_gpu_name::String="5070")
    cfg_path = abspath(cfg_path)
    base = dirname(cfg_path)
    cfg = load_dm_config(cfg_path)
    device = detect_spin_device(device_name, required_gpu_name)
    activate_and_describe_device!(device, device_name, required_gpu_name)

    data_h5 = resolve_path(base, cfg.input_hdf5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    p = load_phys(data_h5)
    target = BSON.load(abspath(target_path), @__MODULE__)
    labels = String.(target[:group_labels])
    ngroups = length(labels)
    active = configured_lag_indices(cfg_path, length(Vector{Int}(target[:lags])))
    taus = Float64.(target[:taus])[active]

    # Primary transition stack from the mobility config.  For transferred-residual
    # branches this is the same r_old + s_old - s_new source used in M training.
    phys_stack = load_dm_stack(cfg_path, cfg, base, device)

    # Optional secondary transition stack.  This may be a plain conditional-score
    # config for legacy figures, or another mobility config for transferred
    # residual branches.
    data_stack = load_secondary_stack(data_cond_cfg_path, phys_stack, device)

    model_blob = BSON.load(abspath(model_path), @__MODULE__)
    mnn = move_model(model_blob[:host_model], device)
    Flux.testmode!(mnn)
    nn_stats = phys_stack.stats

    phys_blob = BSON.load(abspath(phys_path), @__MODULE__)
    phys_coeff = Vector{Float64}(phys_blob[:coefficients])
    true_coeff = true_coefficients_for_action(p)

    lib = target_nonlinear_library(target)
    means = Vector{Float64}(target[:observable_means])
    rng = MersenneTwister(cfg.seed + 9900)
    lags = Vector{Int}(target[:lags])
    operator_sign = configured_operator_sign(cfg_path)
    pred_true = Matrix{Float32}(undef, length(active), ngroups)
    pred_phys = similar(pred_true)
    pred_nn = similar(pred_true)
    pred_nn_data = similar(pred_true)
    ref = Matrix{Float32}(undef, length(active), ngroups)

    for (ai, li) in enumerate(active)
        lag = lags[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, pairs_per_lag, rng)
        obs = nonlinear_observables(xt, p, lib; score_raw=nothing)
        center_observables!(obs, means)
        obs_flat = reshape(obs, size(obs, 1) * size(obs, 2), pairs_per_lag)

        r_phys = transition_raw_for_stack(phys_stack, x0, xt, tau_norm, device)
        true_action = onsite_transpose_action(x0, r_phys, true_coeff)
        phys_action = onsite_transpose_action(x0, r_phys, phys_coeff)
        pred_true[ai, :] .= selected_from_action(obs_flat, true_action, r_phys, target;
            operator_sign=operator_sign)
        pred_phys[ai, :] .= selected_from_action(obs_flat, phys_action, r_phys, target;
            operator_sign=operator_sign)

        x0n_nn = move_array(apply_stats_tensor(x0, nn_stats), device)
        r_phys_dev = move_array(r_phys, device)
        nn_action_phys = mobility_action(mnn, x0n_nn, r_phys_dev)
        nn_action_phys_flat = to_host(reshape(permutedims(nn_action_phys, (2, 1, 3)),
            size(target[:Phi], 1), pairs_per_lag))
        pred_nn[ai, :] .= selected_from_action(obs_flat, Float32.(nn_action_phys_flat), r_phys,
            target; operator_sign=operator_sign)

        if data_stack === phys_stack
            pred_nn_data[ai, :] .= pred_nn[ai, :]
        else
            r_data = transition_raw_for_stack(data_stack, x0, xt, tau_norm, device)
            r_data_dev = move_array(r_data, device)
            nn_action = mobility_action(mnn, x0n_nn, r_data_dev)
            nn_action_flat = to_host(reshape(permutedims(nn_action, (2, 1, 3)),
                size(target[:Phi], 1), pairs_per_lag))
            pred_nn_data[ai, :] .= selected_from_action(obs_flat,
                Float32.(nn_action_flat), r_data, target; operator_sign=operator_sign)
        end

        ref[ai, :] .= vec(Array{Float32}(target[:target_vec][li, :]))
        @printf("Four-way A curves lag %.5g (%d/%d), pairs=%d\n",
            lags[li] * sampler.save_dt, ai, length(active), pairs_per_lag)
        flush(stdout)
        GC.gc()
    end
    relcorr(pred) = begin
        pvec = vec(Float64.(pred))
        rvec = vec(Float64.(ref))
        rel = sqrt(mean(abs2, pvec .- rvec)) / max(sqrt(mean(abs2, rvec)), eps(Float64))
        corr = dot(pvec, rvec) / max(norm(pvec) * norm(rvec), eps(Float64))
        (; rel, corr)
    end
    metrics = Dict(:true_m => relcorr(pred_true), :phys => relcorr(pred_phys),
        :nn => relcorr(pred_nn), :nn_data => relcorr(pred_nn_data))
    return (; taus, labels, ref, pred_true, pred_phys, pred_nn, pred_nn_data, metrics,
        active_lag_indices=active, pairs_per_lag)
end

function selected_columns(ref::AbstractMatrix, labels::Vector{String})
    families = unique(family_prefix.(labels))
    selected = Int[]
    for fam in families
        cols = findall(label -> family_prefix(label) == fam, labels)
        rms = [sqrt(mean(abs2, @view ref[:, c])) for c in cols]
        push!(selected, cols[argmax(rms)])
    end
    # Twelfth panel: hardest high-signal channel not already selected.
    chosen = Set(selected)
    rrms = [sqrt(mean(abs2, @view ref[:, c])) for c in axes(ref, 2)]
    candidates = sortperm(rrms; rev=true)
    for c in candidates
        if !(c in chosen)
            push!(selected, c)
            push!(families, "extra")
            break
        end
    end
    return families, selected
end

function render_fourway(path::AbstractString, curves)
    labels = curves.labels
    _, cols = selected_columns(curves.ref, labels)
    with_scaled_figure_style(FIGURE_WIDTH, FIGURE_HEIGHT; scale_override=FIGURE_FONT_SCALE) do _
        fig = Figure(; size=(FIGURE_WIDTH, FIGURE_HEIGHT), backgroundcolor=:white)
        for (idx, col) in enumerate(cols)
            row = 1 + (idx - 1) ÷ 4
            c = 1 + (idx - 1) % 4
            rr = Float64.(@view curves.ref[:, col])
            yy_true = Float64.(@view curves.pred_true[:, col])
            yy_phys = Float64.(@view curves.pred_phys[:, col])
            yy_nn = Float64.(@view curves.pred_nn[:, col])
            ax = Axis(fig[row, c];
                title=latex_channel_title(labels[col]),
                xlabel=row == 3 ? L"\tau" : "",
                ylabel=c == 1 ? L"\mathcal{A}_{mn}(\tau)" : "",
                titlesize=current_figure_style().axis_titlesize,
                xlabelsize=current_figure_style().axis_labelsize,
                ylabelsize=current_figure_style().axis_labelsize,
                xticklabelsize=current_figure_style().axis_ticklabelsize,
                yticklabelsize=current_figure_style().axis_ticklabelsize)
            lines!(ax, curves.taus, rr; color=:black,
                linewidth=curve_linewidth(; emphasis=1.05))
            lines!(ax, curves.taus, yy_true; color=STYLE_REFERENCE,
                linewidth=curve_linewidth(; emphasis=0.85), linestyle=:dot)
            lines!(ax, curves.taus, yy_phys; color=STYLE_HIGHLIGHT,
                linewidth=curve_linewidth(; emphasis=0.9), linestyle=:dashdot)
            lines!(ax, curves.taus, yy_nn; color=STYLE_ACCENT,
                linewidth=curve_linewidth(; emphasis=0.9), linestyle=:dash)
        end
        elems = [
            LineElement(color=:black, linewidth=curve_linewidth()),
            LineElement(color=STYLE_REFERENCE, linewidth=curve_linewidth(), linestyle=:dot),
            LineElement(color=STYLE_HIGHLIGHT, linewidth=curve_linewidth(), linestyle=:dashdot),
            LineElement(color=STYLE_ACCENT, linewidth=curve_linewidth(), linestyle=:dash),
        ]
        legend_labels = [
            latexstring("\\mathcal{A}_{mn}^{\\mathrm{data}}"),
            latexstring("\\mathcal{A}_{mn}[M_{\\mathrm{true}}]"),
            latexstring("\\mathcal{A}_{mn}[M_{\\mathrm{phys}}]"),
            latexstring("\\mathcal{A}_{mn}[M_{\\mathrm{NN}}]"),
        ]
        Legend(fig[4, 1:4], elems, legend_labels; orientation=:horizontal,
            framevisible=false,
            labelsize=current_figure_style().legend_labelsize * FIGURE_LEGEND_FONT_SCALE)
        rowgap!(fig.layout, 20)
        colgap!(fig.layout, 22)
        save_figure_checked(path, fig)
    end
    @printf("Saved four-way residual figure to %s\n", path)
end

function write_metrics(path::AbstractString, curves)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain manuscript four-way 11-family A curves")
        println(io, "pairs_per_lag = $(curves.pairs_per_lag)")
        println(io, "active_lag_indices = $(first(curves.active_lag_indices)):$(last(curves.active_lag_indices))")
        true_key = haskey(curves.metrics, :true_m) ? :true_m : true
        keys_to_write = haskey(curves.metrics, :nn_data) ?
            (true_key, :phys, :nn, :nn_data) : (true_key, :phys, :nn)
        for key in keys_to_write
            m = curves.metrics[key]
            name = key === true ? "true_m" : String(key)
            @printf(io, "%s_rel_rmse = %.8e\n", name, m.rel)
            @printf(io, "%s_corr = %.8e\n", name, m.corr)
        end
    end
end

function main()
    cfg = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CFG
    model = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_MODEL
    target = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_TARGET
    phys = length(ARGS) >= 4 ? ARGS[4] : DEFAULT_PHYS
    data_cond_cfg = length(ARGS) >= 5 ? ARGS[5] : DEFAULT_DATA_COND_CFG
    out = length(ARGS) >= 6 ? ARGS[6] : DEFAULT_OUT
    cache = length(ARGS) >= 7 ? ARGS[7] : DEFAULT_CACHE
    pairs = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 60000
    device_name = length(ARGS) >= 9 ? ARGS[9] : "GPU:2"
    required_gpu_name = length(ARGS) >= 10 ? ARGS[10] : "5070"
    if isfile(cache)
        @printf("Loading cached four-way curves from %s\n", cache)
        curves = BSON.load(cache, @__MODULE__)[:curves]
        write_metrics(replace(cache, r"\.bson$" => ".txt"), curves)
    else
        curves = operator_curves(cfg, model, target, phys, data_cond_cfg;
            pairs_per_lag=pairs, device_name=device_name,
            required_gpu_name=required_gpu_name)
        ensure_parent_dir(cache)
        BSON.bson(cache, Dict(:curves => curves))
        write_metrics(replace(cache, r"\.bson$" => ".txt"), curves)
        @printf("Saved four-way curve cache to %s\n", cache)
    end
    render_fourway(out, curves)
end

main()
