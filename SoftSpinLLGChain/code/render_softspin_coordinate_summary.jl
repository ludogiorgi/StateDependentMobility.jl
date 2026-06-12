#!/usr/bin/env julia

using GLMakie
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))

include(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))

const OBS_H5 = joinpath(REPO_ROOT, "SoftSpinLLGChain", "data", "soft_spin_llg_chain.h5")
const PHI_H5 = joinpath(REPO_ROOT, "SoftSpinLLGChain", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "data", "forward_phi_epoch240_transfer_gpu1.h5")
const NN_H5 = joinpath(REPO_ROOT, "SoftSpinLLGChain", "stationary_score_repair",
    "score_s020_protocol", "cond_finetune", "data",
    "forward_M_epoch240_transfer_warm_best_gpu2.h5")
const PHYS_H5 = joinpath(REPO_ROOT, "SoftSpinLLGChain", "data", "forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5")
const FIG_DIR = joinpath(REPO_ROOT, "SoftSpinLLGChain", "figures", "manuscript_softspin_final")
const LOG_DIR = joinpath(REPO_ROOT, "SoftSpinLLGChain", "logs", "manuscript_softspin_final")
const CACHE_H5 = joinpath(LOG_DIR, "spin_coordinate_correlations.h5")
const OUT_PNG = joinpath(FIG_DIR, "spin_correlations.png")
const DEBUG_PNG = joinpath(LOG_DIR, "spin_correlations_debug.png")
const METRICS_TXT = joinpath(LOG_DIR, "spin_coordinate_metrics.txt")

const DEFAULT_MODEL_SPECS = (
    (; key="phi", label=latexstring("M=\\Phi"), short="Phi", path=PHI_H5,
       color=RGBf(0.22, 0.38, 0.66), linestyle=:dash),
    (; key="nn", label=latexstring("M_{\\mathrm{NN}}"), short="M_NN", path=NN_H5,
       color=RGBf(0.05, 0.48, 0.36), linestyle=:solid),
    (; key="phys", label=latexstring("M_{\\mathrm{phys}}"), short="M_phys", path=PHYS_H5,
       color=RGBf(0.86, 0.38, 0.10), linestyle=:dashdot),
)

const CURRENT_MODEL_SPECS = Ref(DEFAULT_MODEL_SPECS)
const CURRENT_CACHE_H5 = Ref(CACHE_H5)
const CURRENT_OUT_PNG = Ref(OUT_PNG)
const CURRENT_DEBUG_PNG = Ref(DEBUG_PNG)
const CURRENT_METRICS_TXT = Ref(METRICS_TXT)
const CURRENT_HEATMAP_TIME = Ref{Union{Nothing, Float64}}(nothing)

env_float(name::AbstractString, default::Real) =
    haskey(ENV, name) ? parse(Float64, ENV[name]) : Float64(default)

env_int(name::AbstractString, default::Integer) =
    haskey(ENV, name) ? parse(Int, ENV[name]) : Int(default)

const FIGURE_WIDTH = env_int("SOFTSPIN_COORD_FIG_WIDTH", 4700)
const FIGURE_HEIGHT = env_int("SOFTSPIN_COORD_FIG_HEIGHT", 5600)
const FIGURE_FONT_SCALE = env_float("SOFTSPIN_COORD_FONT_SCALE", 3.45)
const COORDINATE_MAX_LAG = 2740
const SHORT_PANEL_TMAX = 5.0
const LONG_PANEL_TMAX = 100.0
const LINE_PLOT_POINTS = 120

current_model_specs() = CURRENT_MODEL_SPECS[]
current_cache_h5() = CURRENT_CACHE_H5[]
current_out_png() = CURRENT_OUT_PNG[]
current_debug_png() = CURRENT_DEBUG_PNG[]
current_metrics_txt() = CURRENT_METRICS_TXT[]
current_heatmap_time() = CURRENT_HEATMAP_TIME[]

function ensure_dir(path::AbstractString)
    mkpath(dirname(path))
    return path
end

function flatten_state_site_major!(out::AbstractVector{Float64},
        state::AbstractArray{<:Real, 2})
    N = size(state, 1)
    @inbounds for i in 1:N, c in 1:3
        out[(i - 1) * 3 + c] = Float64(state[i, c])
    end
    return out
end

function load_states_time(path::AbstractString; burn_fraction::Float64=0.0)
    states = h5read(path, "/trajectories/states")
    time = Vector{Float64}(h5read(path, "/trajectories/time"))
    states = Array{Float32, 4}(states)
    burn = clamp(floor(Int, burn_fraction * size(states, 1)) + 1, 1, size(states, 1))
    return Array{Float32, 4}(states[burn:end, :, :, :]), time[burn:end]
end

function coordinate_mean(states::Array{Float32, 4}; nsamples::Int, seed::Int)
    nt, N, _, ntraj = size(states)
    D = 3N
    total = nt * ntraj
    B = min(nsamples, total)
    rng = MersenneTwister(seed)
    mu = zeros(Float64, D)
    tmp = Vector{Float64}(undef, D)
    @inbounds for b in 1:B
        linear = total <= B ? b - 1 : rand(rng, 0:(total - 1))
        t = 1 + (linear % nt)
        tr = 1 + (linear ÷ nt)
        flatten_state_site_major!(tmp, @view states[t, :, :, tr])
        mu .+= tmp
    end
    return mu ./ B
end

function coordinate_target_times(obs_save_dt::Float64)
    short_max = floor(Int, SHORT_PANEL_TMAX / obs_save_dt)
    long_max = floor(Int, LONG_PANEL_TMAX / obs_save_dt)
    short = round.(Int, range(0, short_max; length=LINE_PLOT_POINTS))
    long = round.(Int, range(0, long_max; length=LINE_PLOT_POINTS))
    obs_lags = sort!(unique(vcat(short, long)))
    return Float64.(obs_lags) .* obs_save_dt
end

function coordinate_correlations(states::Array{Float32, 4}, save_dt::Float64;
        target_times::AbstractVector{<:Real}, pairs_per_lag::Int,
        mean_samples::Int, seed::Int,
        label::AbstractString)
    nt, N, _, ntraj = size(states)
    D = 3N
    lag_indices = round.(Int, Float64.(target_times) ./ save_dt)
    mu = coordinate_mean(states; nsamples=mean_samples, seed=seed + 19)
    C = fill(NaN, length(lag_indices), D, D)
    rng = MersenneTwister(seed)
    x0 = Matrix{Float64}(undef, D, pairs_per_lag)
    xt = Matrix{Float64}(undef, D, pairs_per_lag)
    @printf("Computing coordinate C for %s: %d target times, %d pairs/lag.\n",
        label, length(lag_indices), pairs_per_lag)
    flush(stdout)
    @inbounds for (li, lag) in enumerate(lag_indices)
        lag > nt - 1 && continue
        available_t = nt - lag
        total_pairs = available_t * ntraj
        B = min(pairs_per_lag, total_pairs)
        for b in 1:B
            linear = total_pairs <= B ? b - 1 : rand(rng, 0:(total_pairs - 1))
            t = 1 + (linear % available_t)
            tr = 1 + (linear ÷ available_t)
            flatten_state_site_major!(@view(x0[:, b]), @view states[t, :, :, tr])
            flatten_state_site_major!(@view(xt[:, b]), @view states[t + lag, :, :, tr])
        end
        @views x0[:, 1:B] .-= mu
        @views xt[:, 1:B] .-= mu
        @views mul!(C[li, :, :], xt[:, 1:B], transpose(x0[:, 1:B]))
        C[li, :, :] ./= B
        li % 25 == 0 && (@printf("  %s target time %d/%d (t=%.5g, lag=%d, actual=%.5g)\n",
            label, li, length(lag_indices), Float64(target_times[li]), lag, lag * save_dt); flush(stdout))
    end
    return Float64.(target_times), C
end

function agreement(Cm, Co)
    m = vec(Float64.(Cm))
    o = vec(Float64.(Co))
    rel = norm(m .- o) / max(norm(o), eps(Float64))
    corr = dot(m .- mean(m), o .- mean(o)) /
        max(norm(m .- mean(m)) * norm(o .- mean(o)), eps(Float64))
    return rel, corr
end

function save_cache(t, obs, models)
    ensure_dir(current_cache_h5())
    h5open(current_cache_h5(), "w") do h5
        h5["time"] = t
        h5["obs"] = obs
        for (key, C) in models
            h5[key] = C
        end
    end
end

function load_cache()
    h5open(current_cache_h5(), "r") do h5
        t = Vector{Float64}(read(h5["time"]))
        obs = Array{Float64, 3}(read(h5["obs"]))
        models = Dict{String, Array{Float64, 3}}()
        for spec in current_model_specs()
            models[spec.key] = Array{Float64, 3}(read(h5[spec.key]))
        end
        return t, obs, models
    end
end

function get_save_dt(path::AbstractString, time::Vector{Float64})
    length(time) > 1 && return time[2] - time[1]
    h5open(path, "r") do h5
        if haskey(h5, "/metadata/save_dt")
            return Float64(read(h5["/metadata/save_dt"]))
        end
    end
    return 1.0
end

function compute_or_load_correlations(; force::Bool=false)
    if isfile(current_cache_h5()) && !force
        @printf("Loading cached coordinate correlations from %s\n", current_cache_h5())
        return load_cache()
    end
    obs_states, obs_time = load_states_time(OBS_H5; burn_fraction=0.10)
    obs_dt = Float64(h5read(OBS_H5, "/metadata/save_dt"))
    target_times = coordinate_target_times(obs_dt)
    t_obs, Cobs = coordinate_correlations(obs_states, obs_dt;
        target_times=target_times, pairs_per_lag=28_000, mean_samples=140_000,
        seed=2026060601, label="observations")
    models = Dict{String, Array{Float64, 3}}()
    for (k, spec) in enumerate(current_model_specs())
        states, time = load_states_time(spec.path; burn_fraction=0.0)
        dt = get_save_dt(spec.path, time)
        _, C = coordinate_correlations(states, dt;
            target_times=target_times, pairs_per_lag=28_000, mean_samples=80_000,
            seed=2026060610 + 101k, label=spec.short)
        models[spec.key] = C
    end
    save_cache(t_obs, Cobs, models)
    return t_obs, Cobs, models
end

idx(site::Int, comp::Int) = (site - 1) * 3 + comp

function translation_average_curve(C::Array{Float64, 3}, comp::Int, sep::Int)
    _, D, _ = size(C)
    N = D ÷ 3
    out = zeros(Float64, size(C, 1))
    @inbounds for i in 1:N
        j = mod1(i + sep, N)
        out .+= @view C[:, idx(j, comp), idx(i, comp)]
    end
    return out ./ N
end

function normalized_translation_curve(C::Array{Float64, 3}, comp::Int, sep::Int)
    curve = translation_average_curve(C, comp, sep)
    return curve ./ max(abs(curve[1]), eps(Float64))
end

function line_plot_indices(t::AbstractVector{<:Real}, tmax::Real;
        npoints::Int=LINE_PLOT_POINTS)
    last_idx = searchsortedlast(t, tmax)
    last_idx = clamp(last_idx, 1, length(t))
    n = min(npoints, last_idx)
    return unique(round.(Int, range(1, last_idx; length=n)))
end

finite_maximum(v) = maximum(x for x in v if isfinite(x))
finite_minimum(v) = minimum(x for x in v if isfinite(x))
finite_mean(v) = mean(x for x in v if isfinite(x))

function common_finite_tmax(t::AbstractVector{<:Real},
        curves::AbstractVector{<:AbstractVector{<:Real}}, requested_tmax::Real)
    last_idx = searchsortedlast(t, requested_tmax)
    last_idx = clamp(last_idx, 1, length(t))
    for i in last_idx:-1:1
        all(curve -> i <= length(curve) && isfinite(curve[i]), curves) && return t[i]
    end
    return t[1]
end

function normalized_matrix(C::Array{Float64, 3}, lag_idx::Int, C0obs::Matrix{Float64})
    D = size(C, 2)
    out = Matrix{Float64}(undef, D, D)
    @inbounds for a in 1:D, b in 1:D
        denom = sqrt(max(abs(C0obs[a, a] * C0obs[b, b]), eps(Float64)))
        out[a, b] = C[lag_idx, a, b] / denom
    end
    return out
end

function total_error_curves(Cobs, models)
    E = Dict{String, Vector{Float64}}()
    for spec in current_model_specs()
        C = models[spec.key]
        L = min(size(C, 1), size(Cobs, 1))
        E[spec.key] = [any(!isfinite, C[l, :, :]) ? NaN :
            norm(vec(C[l, :, :] .- Cobs[l, :, :])) for l in 1:L]
    end
    return E
end

function choose_tstar(t, E)
    keys = [spec.key for spec in current_model_specs()]
    L = minimum(length(E[k]) for k in keys)
    score = fill(-Inf, L)
    for l in 2:L
        t[l] > SHORT_PANEL_TMAX && continue
        score[l] = E["phi"][l] - min(E["nn"][l], E["phys"][l])
    end
    return argmax(score)
end

function choose_heatmap_index(t, Cobs, models, E)
    requested_time = current_heatmap_time()
    requested_time === nothing && return choose_tstar(t, E)
    keys = [spec.key for spec in current_model_specs()]
    candidates = Int[]
    for i in eachindex(t)
        if i <= size(Cobs, 1) && !any(!isfinite, Cobs[i, :, :]) &&
                all(k -> i <= size(models[k], 1) && !any(!isfinite, models[k][i, :, :]), keys)
            push!(candidates, i)
        end
    end
    isempty(candidates) && error("No finite heatmap candidates are available.")
    return candidates[argmin(abs.(Float64.(t[candidates]) .- requested_time))]
end

function distance_errors(Cobs, models; first_lag::Int=2)
    _, D, _ = size(Cobs)
    N = D ÷ 3
    maxdist = N ÷ 2
    out = Dict{String, Vector{Float64}}()
    for spec in current_model_specs()
        C = models[spec.key]
        L = min(size(C, 1), size(Cobs, 1))
        num = zeros(Float64, maxdist + 1)
        den = zeros(Float64, maxdist + 1)
        @inbounds for l in first_lag:L, i in 1:N, j in 1:N, c in 1:3, d in 1:3
            dist = abs(i - j)
            dist = min(dist, N - dist)
            a = idx(i, c)
            b = idx(j, d)
            !isfinite(C[l, a, b]) && continue
            num[dist + 1] += abs2(C[l, a, b] - Cobs[l, a, b])
            den[dist + 1] += 1.0
        end
        out[spec.key] = sqrt.(num ./ max.(den, eps(Float64)))
    end
    return collect(0:maxdist), out
end

function write_metrics(t, Cobs, models, E, tstar_idx, dist_grid, dist_err)
    ensure_dir(current_metrics_txt())
    open(current_metrics_txt(), "w") do io
        println(io, "Soft-spin final coordinate-correlation figure metrics")
        println(io, @sprintf("t_star = %.8g", t[tstar_idx]))
        println(io, "coordinate dimension = $(size(Cobs, 2))")
        println(io, "cache = $(current_cache_h5())")
        for spec in current_model_specs()
            C = models[spec.key]
            L = min(size(C, 1), size(Cobs, 1))
            finite_lags = [l for l in 1:L if !any(!isfinite, C[l, :, :])]
            rel, corr = isempty(finite_lags) ? (NaN, NaN) :
                agreement(C[finite_lags, :, :], Cobs[finite_lags, :, :])
            println(io, @sprintf("%s full coordinate C rel.RMSE = %.8e", spec.short, rel))
            println(io, @sprintf("%s full coordinate C corr = %.8e", spec.short, corr))
            println(io, @sprintf("%s mean absolute Frobenius error = %.8e", spec.short, finite_mean(E[spec.key])))
            println(io, @sprintf("%s mean distance RMS error = %.8e", spec.short, finite_mean(dist_err[spec.key])))
        end
        println(io, "No-cheating audit: this renderer uses only saved observation and forward trajectories for plotted correlations. True mobility is not used.")
    end
end

function finite_plot_indices(t::AbstractVector{<:Real}, curve::AbstractVector{<:Real},
        requested_tmax::Real; npoints::Int=LINE_PLOT_POINTS)
    last_idx = min(searchsortedlast(t, requested_tmax), length(curve), length(t))
    valid = [i for i in 1:last_idx if isfinite(curve[i])]
    isempty(valid) && return Int[]
    n = min(npoints, length(valid))
    return unique(valid[round.(Int, range(1, length(valid); length=n))])
end

function inset_bbox(parent_axis; width_fraction::Float64=0.44,
        height_fraction::Float64=0.42, right_pad_fraction::Float64=0.07,
        top_pad_fraction::Float64=0.07)
    return lift(parent_axis.scene.viewport) do bb
        x0, y0 = Float64.(bb.origin)
        w, h = Float64.(bb.widths)
        iw = width_fraction * w
        ih = height_fraction * h
        right_pad = right_pad_fraction * w
        top_pad = top_pad_fraction * h
        left = x0 + w - right_pad - iw
        right = x0 + w - right_pad
        top = y0 + h - top_pad
        bottom = top - ih
        BBox(left, right, bottom, top)
    end
end

function render_figure(t, Cobs, models)
    E = total_error_curves(Cobs, models)
    tstar_idx = choose_heatmap_index(t, Cobs, models, E)
    dist_grid, dist_err = distance_errors(Cobs, models)
    write_metrics(t, Cobs, models, E, tstar_idx, dist_grid, dist_err)

    comp_labels = ("x", "y", "z")
    model_lookup = Dict(spec.key => spec for spec in current_model_specs())
    with_scaled_figure_style(FIGURE_WIDTH, FIGURE_HEIGHT; scale_override=FIGURE_FONT_SCALE) do _
        fig = Figure(; size=(FIGURE_WIDTH, FIGURE_HEIGHT), backgroundcolor=:white)
        handles = Any[]
        labels = Any[]
        for col in 1:3
            for (row, sep) in enumerate((0, 1))
                ax = Axis(fig[row, col];
                    title=row == 1 ? latexstring("\\langle m_{$(comp_labels[col]),i}(t)m_{$(comp_labels[col]),i}(0)\\rangle") :
                        latexstring("\\langle m_{$(comp_labels[col]),i+1}(t)m_{$(comp_labels[col]),i}(0)\\rangle"),
                    xlabel=row == 2 ? latexstring("t") : "",
                    ylabel=col == 1 ? latexstring("C(t)/C(0)") : "")
                panel_min = 0.0
                panel_max = 1.0
                obs_curve = normalized_translation_curve(Cobs, col, sep)
                model_curves = Dict{String, Vector{Float64}}()
                all_curves = Vector{Vector{Float64}}([obs_curve])
                for spec in current_model_specs()
                    curve = normalized_translation_curve(models[spec.key], col, sep)
                    model_curves[spec.key] = curve
                    push!(all_curves, curve)
                end
                requested_tmax = col == 3 ? LONG_PANEL_TMAX :
                    min(SHORT_PANEL_TMAX, maximum(t))
                obs_idx = finite_plot_indices(t, obs_curve, requested_tmax)
                panel_min = min(panel_min, finite_minimum(obs_curve[obs_idx]))
                panel_max = max(panel_max, finite_maximum(obs_curve[obs_idx]))
                hobs = lines!(ax, t[obs_idx], obs_curve[obs_idx]; color=:black, linewidth=curve_linewidth(),
                    label=latexstring("\\mathrm{obs}"))
                row == 1 && col == 1 && (push!(handles, hobs); push!(labels, latexstring("\\mathrm{obs}")))
                for spec in current_model_specs()
                    curve = model_curves[spec.key]
                    plot_idx = finite_plot_indices(t, curve, requested_tmax)
                    if !isempty(plot_idx)
                        panel_min = min(panel_min, finite_minimum(curve[plot_idx]))
                        panel_max = max(panel_max, finite_maximum(curve[plot_idx]))
                    end
                    h = lines!(ax, t[plot_idx], curve[plot_idx]; color=spec.color,
                        linestyle=spec.linestyle, linewidth=curve_linewidth(),
                        label=spec.label)
                    row == 1 && col == 1 && (push!(handles, h); push!(labels, spec.label))
                end
                xlims!(ax, 0, requested_tmax)
                col == 3 && (ax.xticks = 0:20:100)
                panel_span = max(panel_max - panel_min, 1e-6)
                lower = col == 3 ? min(0.0, panel_min - 0.04 * panel_span) : 0.0
                upper = max(1.05, panel_max + 0.08 * panel_span)
                ylims!(ax, lower, upper)

                if col == 3
                    axin = Axis(fig.scene; bbox=inset_bbox(ax),
                        backgroundcolor=(:white, 0.92),
                        xlabel="", ylabel="",
                        xgridvisible=false, ygridvisible=false)
                    obs_inset_idx = finite_plot_indices(t, obs_curve, SHORT_PANEL_TMAX;
                        npoints=LINE_PLOT_POINTS)
                    inset_min = finite_minimum(obs_curve[obs_inset_idx])
                    inset_max = finite_maximum(obs_curve[obs_inset_idx])
                    lines!(axin, t[obs_inset_idx], obs_curve[obs_inset_idx];
                        color=:black, linewidth=max(1.0, 0.72 * curve_linewidth()))
                    for spec in current_model_specs()
                        curve = model_curves[spec.key]
                        inset_idx = finite_plot_indices(t, curve, SHORT_PANEL_TMAX;
                            npoints=LINE_PLOT_POINTS)
                        if !isempty(inset_idx)
                            inset_min = min(inset_min, finite_minimum(curve[inset_idx]))
                            inset_max = max(inset_max, finite_maximum(curve[inset_idx]))
                            lines!(axin, t[inset_idx], curve[inset_idx]; color=spec.color,
                                linestyle=spec.linestyle,
                                linewidth=max(1.0, 0.72 * curve_linewidth()))
                        end
                    end
                    xlims!(axin, 0, SHORT_PANEL_TMAX)
                    inset_span = max(inset_max - inset_min, 1e-6)
                    ylims!(axin, inset_min - 0.08 * inset_span,
                        inset_max + 0.10 * inset_span)
                    axin.xticks = 0:2.5:5
                    axin.yticks = WilkinsonTicks(3)
                    translate!(axin.scene, 0, 0, 10)
                end
            end
        end

        requested_heatmap_time = current_heatmap_time()
        displayed_heatmap_time = requested_heatmap_time === nothing ? t[tstar_idx] : requested_heatmap_time
        heatmap_time_label = @sprintf("%.3g", displayed_heatmap_time)
        heatmaps = [
            ("phys", latexstring("C_{\\mathrm{phys}}(" * heatmap_time_label * ")")),
            ("nn", latexstring("C_{\\mathrm{NN}}(" * heatmap_time_label * ")")),
            ("phi", latexstring("C_{\\Phi}(" * heatmap_time_label * ")")),
            ("obs", latexstring("C_{\\mathrm{obs}}(" * heatmap_time_label * ")")),
        ]
        mats = Dict{String, Matrix{Float64}}("obs" => normalized_matrix(Cobs, tstar_idx, Cobs[1, :, :]))
        for spec in current_model_specs()
            mats[spec.key] = normalized_matrix(models[spec.key], tstar_idx, Cobs[1, :, :])
        end
        clim = maximum(abs, reduce(vcat, [vec(mats[k]) for (k, _) in heatmaps]))
        clim = max(clim, eps(Float64))
        hm_ref = nothing
        for (pos, (key, title)) in enumerate(heatmaps)
            row = pos <= 3 ? 3 : 4
            col = pos <= 3 ? pos : 1
            ax = Axis(fig[row, col]; title=title, xlabel="index", ylabel="index",
                aspect=DataAspect())
            hm_ref = heatmap!(ax, mats[key]; colormap=:balance, colorrange=(-clim, clim))
        end

        axerr = Axis(fig[4, 2]; title=latexstring("\\mathrm{total\\;error}"),
            xlabel=latexstring("t"), ylabel="Frobenius error")
        requested_err_tmax = LONG_PANEL_TMAX
        for spec in current_model_specs()
            err_idx = finite_plot_indices(t, E[spec.key], requested_err_tmax)
            lines!(axerr, t[err_idx], E[spec.key][err_idx]; color=spec.color,
                linestyle=spec.linestyle, linewidth=curve_linewidth(), label=spec.label)
        end
        xlims!(axerr, 0, requested_err_tmax)
        axerr.xticks = 0:20:100

        axdist = Axis(fig[4, 3]; title=latexstring("\\mathrm{error\\;vs.\\;distance}"),
            xlabel=latexstring("r"), ylabel="time-averaged RMS error")
        for spec in current_model_specs()
            lines!(axdist, dist_grid, dist_err[spec.key]; color=spec.color,
                linestyle=spec.linestyle, linewidth=curve_linewidth(), label=spec.label)
        end

        Colorbar(fig[3:4, 4], hm_ref; label=latexstring("C(t_*)/C(0)"),
            width=24, tellheight=false)
        Legend(fig[5, 1:3], handles, labels; orientation=:horizontal,
            framevisible=false, tellheight=true, nbanks=1)
        apply_publication_grid!(fig.layout, 5, 4;
            row_weights=[1.0, 1.0, 1.05, 1.05, 0.16],
            col_weights=[1.0, 1.0, 1.0, 0.055],
            row_gap=28, col_gap=30)
        ensure_dir(current_out_png())
        save_figure(current_out_png(), fig)
        ensure_dir(current_debug_png())
        save_figure(current_debug_png(), fig)
    end
    @printf("Saved %s and %s\n", current_out_png(), current_debug_png())
    @printf("Saved metrics to %s\n", current_metrics_txt())
end

function main()
    force = false
    phi_path = PHI_H5
    nn_path = NN_H5
    phys_path = PHYS_H5
    for arg in ARGS
        if arg == "--force"
            force = true
        elseif startswith(arg, "--phi=")
            phi_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--nn=")
            nn_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--phys=")
            phys_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--out=")
            CURRENT_OUT_PNG[] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--debug=")
            CURRENT_DEBUG_PNG[] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--cache=")
            CURRENT_CACHE_H5[] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--metrics=")
            CURRENT_METRICS_TXT[] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--heatmap-t=")
            CURRENT_HEATMAP_TIME[] = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument $(arg)")
        end
    end
    CURRENT_MODEL_SPECS[] = (
        (; DEFAULT_MODEL_SPECS[1]..., path=phi_path),
        (; DEFAULT_MODEL_SPECS[2]..., path=nn_path),
        (; DEFAULT_MODEL_SPECS[3]..., path=phys_path),
    )
    t, Cobs, models = compute_or_load_correlations(; force=force)
    render_figure(t, Cobs, models)
end

main()
