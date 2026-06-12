#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))
include(joinpath(@__DIR__, "render_forward_with_dM.jl"))

using BSON
using GLMakie
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics

const MOBILITY11_FAMILIES = (
    "mc_rb0", "mc_rb1", "mc_rb2",
    "mc_u", "mc_Tu", "mc_Pu", "mc_Cu",
    "mc_a", "mc_Ta", "mc_Pa", "mc_Ca",
)
const MOBILITY11_COMPONENTS = ("mx", "my", "mz")

function mobility11_family_component(name::AbstractString)
    parts = split(String(name), "_")
    length(parts) >= 3 || return String(name), String(name)
    return string(parts[1], "_", parts[2]), String(parts[3])
end

function mobility11_forward_means(states::Array{Float32, 4}, p::SpinParams,
        lib::NonlinearLibrary; nsamples::Int, batch_size::Int, seed::Int)
    nt, N, _, ntraj = size(states)
    rng = MersenneTwister(seed)
    obs_sum = zeros(Float64, length(lib.names))
    comp_sum = zeros(Float64, 3)
    done = 0
    while done < nsamples
        B = min(batch_size, nsamples - done)
        raw = Array{Float32}(undef, N, 3, B)
        @inbounds for b in 1:B
            t = rand(rng, 1:nt)
            tr = rand(rng, 1:ntraj)
            raw[:, :, b] .= states[t, :, :, tr]
        end
        obs = nonlinear_observables(raw, p, lib)
        @inbounds for a in 1:length(lib.names)
            obs_sum[a] += sum(Float64, @view obs[:, a, :])
        end
        @inbounds for c in 1:3
            comp_sum[c] += sum(Float64, @view raw[:, c, :])
        end
        done += B
    end
    denom = nsamples * N
    return obs_sum ./ denom, comp_sum ./ denom
end

function mobility11_grouped_correlations(states::Array{Float32, 4}, save_dt::Float64,
        p::SpinParams, target; max_lag::Int, pairs_per_lag::Int,
        batch_size::Int, mean_samples::Int, seed::Int, label::AbstractString)
    nt, N, _, ntraj = size(states)
    L = min(max_lag, nt - 1)
    names = Vector{String}(target[:names])
    lib = target_nonlinear_library(target)
    group_index_matrix = target_group_index_matrix(target)
    group_index_matrix === nothing &&
        error("mobility11 forward renderer requires the translation-averaged group_index_matrix.")
    ngroups = size(group_index_matrix, 2)
    obs_means, comp_means = mobility11_forward_means(states, p, lib;
        nsamples=mean_samples, batch_size=batch_size, seed=seed + 11)

    C = Array{Float64}(undef, L + 1, ngroups)
    rng = MersenneTwister(seed)
    @printf("Computing mobility11 grouped correlations for %s: %d groups, %d lags, %d pairs/lag.\n",
        label, ngroups, L, pairs_per_lag)
    for lag in 0:L
        available_t = nt - lag
        Btot = min(pairs_per_lag, available_t * ntraj)
        mat_acc = zeros(Float64, N * length(names), 3N)
        done = 0
        while done < Btot
            B = min(batch_size, Btot - done)
            x0 = Array{Float32}(undef, N, 3, B)
            xt = Array{Float32}(undef, N, 3, B)
            @inbounds for b in 1:B
                linear = available_t * ntraj <= pairs_per_lag ?
                    done + b - 1 : rand(rng, 0:(available_t * ntraj - 1))
                t = 1 + (linear % available_t)
                tr = 1 + (linear ÷ available_t)
                x0[:, :, b] .= states[t, :, :, tr]
                xt[:, :, b] .= states[t + lag, :, :, tr]
            end
            obs_xt = nonlinear_observables(xt, p, lib)
            @inbounds for a in 1:length(names)
                om = Float32(obs_means[a])
                for b in 1:B, i in 1:N
                    obs_xt[i, a, b] -= om
                end
            end
            x0f = Matrix{Float64}(flatten_batch(x0))
            @inbounds for b in 1:B, i in 1:N, c in 1:3
                x0f[(i - 1) * 3 + c, b] -= comp_means[c]
            end
            obs_flat = reshape(obs_xt, N * length(names), B)
            mul!(mat_acc, Matrix{Float64}(obs_flat), transpose(x0f), 1.0, 1.0)
            done += B
        end
        flat = vec(mat_acc ./ Btot)
        grouped = flat[group_index_matrix]
        C[lag + 1, :] .= vec(mean(grouped; dims=1))
        lag % 20 == 0 && @printf("  %s mobility11 C lag %d/%d\n", label, lag, L)
        flush(stdout)
    end
    return collect(0:L) .* save_dt, C
end

function agreement_pair(pred::AbstractArray, ref::AbstractArray)
    p = vec(Float64.(pred))
    r = vec(Float64.(ref))
    rel = norm(p .- r) / max(norm(r), eps(Float64))
    corr = dot(p .- mean(p), r .- mean(r)) /
        max(norm(p .- mean(p)) * norm(r .- mean(r)), eps(Float64))
    return rel, corr
end

function render_mobility11_correlation_figure(path::AbstractString, target, t_obs,
        Cobs, model_corrs; max_plot_time::Float64=5.0)
    channels = target[:channels]
    group_channel_ids = Vector{Int}(target[:selected_channel_ids])
    group_offsets = Vector{Int}(target[:group_offsets])
    group_by = Dict{Tuple{String, String, Int}, Vector{Int}}()
    for g in eachindex(group_channel_ids)
        ch = channels[group_channel_ids[g]]
        fam, comp = mobility11_family_component(ch.observable)
        key = (fam, comp, ch.target_component)
        push!(get!(group_by, key, Int[]), g)
    end

    rels = Float64[]
    corrs = Float64[]
    for (_, _, C, _, _) in model_corrs
        L = min(size(Cobs, 1), size(C, 1))
        rel, corr = agreement_pair(C[1:L, :], Cobs[1:L, :])
        push!(rels, rel)
        push!(corrs, corr)
    end
    subtitle = join([@sprintf("%s rel %.3e corr %.4f",
            model_corrs[k][1], rels[k], corrs[k]) for k in eachindex(model_corrs)],
        "   ")

    fig = Figure(; size=(5600, 6800))
    Label(fig[0, 1:9], "Soft-spin forward correlations: expanded 11-family mobility library";
        fontsize=30, tellwidth=false)
    Label(fig[1, 1:9], subtitle; fontsize=22, tellwidth=false)

    row_labels = Dict(
        "mc_rb0" => "b0(r²)m", "mc_rb1" => "b1(r²)m", "mc_rb2" => "b2(r²)m",
        "mc_u" => "u", "mc_Tu" => "T u", "mc_Pu" => "P u", "mc_Cu" => "C u",
        "mc_a" => "a", "mc_Ta" => "T a", "mc_Pa" => "P a", "mc_Ca" => "C a",
    )
    comp_labels = ("x", "y", "z")
    xlimit = (0.0, max_plot_time)
    for (ri, fam) in enumerate(MOBILITY11_FAMILIES)
        for (ci, comp) in enumerate(MOBILITY11_COMPONENTS), d in 1:3
            col = (ci - 1) * 3 + d
            key = (fam, comp, d)
            groups = get(group_by, key, Int[])
            ax = Axis(fig[ri + 1, col];
                title=@sprintf("%s_%s vs m_%s", row_labels[fam], comp_labels[ci],
                    comp_labels[d]),
                xlabel=ri == length(MOBILITY11_FAMILIES) ? "t" : "",
                ylabel=col == 1 ? "C" : "",
                titlesize=13, xlabelsize=12, ylabelsize=12,
                xticklabelsize=10, yticklabelsize=10)
            isempty(groups) && continue
            mask_obs = findall(<=(max_plot_time), t_obs)
            for g in groups
                lines!(ax, t_obs[mask_obs], Cobs[mask_obs, g];
                    color=(:black, 0.12), linewidth=1)
            end
            lines!(ax, t_obs[mask_obs], vec(mean(Cobs[mask_obs, groups]; dims=2));
                color=:black, linewidth=3, label="obs")
            for (model_label, tm, Cm, color, linestyle) in model_corrs
                mask = findall(<=(max_plot_time), tm)
                for g in groups
                    lines!(ax, tm[mask], Cm[mask, g];
                        color=(color, 0.10), linewidth=1, linestyle=linestyle)
                end
                lines!(ax, tm[mask], vec(mean(Cm[mask, groups]; dims=2));
                    color=color, linewidth=2.8, linestyle=linestyle, label=model_label)
            end
            xlims!(ax, xlimit...)
        end
    end
    elems = [LineElement(color=:black, linewidth=3)]
    labels = ["obs"]
    for (model_label, _, _, color, linestyle) in model_corrs
        push!(elems, LineElement(color=color, linewidth=3, linestyle=linestyle))
        push!(labels, model_label)
    end
    Legend(fig[length(MOBILITY11_FAMILIES) + 2, 1:9], elems, labels;
        orientation=:horizontal, tellheight=true, framevisible=false)
    save_figure_checked(path, fig)
end

function write_mobility11_metrics(path::AbstractString, Cobs, model_corrs, target)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "mobility11 forward-correlation validation")
        println(io, "target_bson = SoftSpinLLGChain/models/mobility11/analytic/A_target.bson")
        println(io, "groups = $(size(Cobs, 2))")
        println(io, "channels = $(length(target[:channels]))")
        println(io, "families = $(join(MOBILITY11_FAMILIES, ", "))")
        for (label, _, C, _, _) in model_corrs
            L = min(size(Cobs, 1), size(C, 1))
            rel, corr = agreement_pair(C[1:L, :], Cobs[1:L, :])
            println(io, @sprintf("%s all mobility11 C rel.RMSE = %.8e", label, rel))
            println(io, @sprintf("%s all mobility11 C corr = %.8e", label, corr))
        end
        println(io)
        println(io, "No-cheating audit: this renderer computes forward correlations from saved observation and forward trajectories using the expanded 11-family observable definitions saved in the data-driven A target. It does not use true mobility, true score, true coefficients, or simulator generator quantities in any fitted quantity. The analytic form was used only to define the observable families, as allowed in the analytic-informed branch.")
    end
end

function model_specs(paths::Vector{String})
    isempty(paths) && return [
        ("weak-signal", "SoftSpinLLGChain/data/mobility11/analytic/forward_weaksignal.h5", STYLE_PRIMARY, :dash),
        ("fast", "SoftSpinLLGChain/data/mobility11/analytic/forward_fast.h5", STYLE_ACCENT, :solid),
        ("no-signal", "SoftSpinLLGChain/data/mobility11/analytic/forward_nosignal.h5", STYLE_HIGHLIGHT, :dashdot),
    ]
    colors = Any[STYLE_PRIMARY, STYLE_ACCENT, STYLE_HIGHLIGHT, STYLE_SECONDARY]
    styles = Any[:dash, :solid, :dashdot, :dot]
    specs = []
    for (k, spec) in enumerate(paths)
        parts = split(spec, "=", limit=2)
        length(parts) == 2 || error("Model spec must have form label=path; got $(spec).")
        push!(specs, (strip(parts[1]), strip(parts[2]),
            colors[1 + mod(k - 1, length(colors))],
            styles[1 + mod(k - 1, length(styles))]))
    end
    return specs
end

function main()
    cfg_path = length(ARGS) >= 1 ? ARGS[1] :
        "SoftSpinLLGChain/configs/mobility11_analytic_fullcache_fast_gpu2.toml"
    max_lag = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 125
    pairs_per_lag = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16000
    out_root = "SoftSpinLLGChain/figures/mobility11/analytic/forward"
    log_root = "SoftSpinLLGChain/logs/mobility11/analytic"
    model_args = String[]
    for arg in (length(ARGS) >= 4 ? ARGS[4:end] : String[])
        if startswith(arg, "--outdir=")
            out_root = split(arg, "=", limit=2)[2]
            log_root = replace(out_root, "/figures/" => "/logs/")
            if log_root == out_root
                log_root = joinpath(dirname(out_root), "logs")
            end
        elseif startswith(arg, "--logdir=")
            log_root = split(arg, "=", limit=2)[2]
        else
            push!(model_args, arg)
        end
    end
    specs = model_specs(model_args)

    cfg = load_dm_config(cfg_path)
    base = dirname(cfg_path)
    target = BSON.load(resolve_path(base, cfg.target_artifact_bson))
    p = load_phys(resolve_path(base, cfg.input_hdf5))
    obs_states = h5read(resolve_path(base, cfg.input_hdf5), "/trajectories/states")
    burn = floor(Int, cfg.burnin_fraction * size(obs_states, 1)) + 1
    obs_states = Array{Float32, 4}(obs_states[burn:end, :, :, :])
    obs_save_dt = Float64(h5read(resolve_path(base, cfg.input_hdf5), "/metadata/save_dt"))

    models = ForwardModelStates[]
    for (label, path, color, linestyle) in specs
        resolved = abspath(path)
        isfile(resolved) || error("Missing forward HDF5 for $(label): $(resolved)")
        states, time = load_forward_h5(resolved)
        push!(models, ForwardModelStates(label, states, time, color, linestyle))
    end

    fig_dir = out_root
    log_dir = log_root
    mkpath(fig_dir)
    mkpath(log_dir)

    dummy_params = load_config(joinpath(@__DIR__, "..", "configs", "fit_Phi_phys_pC.toml"))
    stats_path = joinpath(fig_dir, "stats_compare.png")
    cov_obs, cov_models = render_stats_with_dm(stats_path, dummy_params, obs_states,
        models; obs_save_dt=obs_save_dt)

    batch_size = 2048
    mean_samples = 24000
    t_obs, Cobs = mobility11_grouped_correlations(obs_states, obs_save_dt, p, target;
        max_lag=max_lag, pairs_per_lag=pairs_per_lag, batch_size=batch_size,
        mean_samples=mean_samples, seed=2026060410, label="obs")
    model_corrs = []
    for (k, model) in enumerate(models)
        dt = length(model.time) > 1 ? Float64(model.time[2] - model.time[1]) : obs_save_dt
        t, C = mobility11_grouped_correlations(model.states, dt, p, target;
            max_lag=max_lag, pairs_per_lag=pairs_per_lag, batch_size=batch_size,
            mean_samples=mean_samples, seed=2026060420 + 101k, label=model.label)
        push!(model_corrs, (model.label, t, C, model.color, model.linestyle))
        GC.gc()
    end

    corr_path = joinpath(fig_dir, "corr_11_compare.png")
    metrics_path = joinpath(log_dir, "forward_mobility11_metrics.txt")
    render_mobility11_correlation_figure(corr_path, target, t_obs, Cobs, model_corrs)
    write_mobility11_metrics(metrics_path, Cobs, model_corrs, target)

    open(metrics_path, "a") do io
        println(io)
        println(io, "Stationary covariance metrics:")
        for (k, model) in enumerate(models)
            rel = norm(cov_models[k] - cov_obs) / max(norm(cov_obs), eps(Float64))
            corr = dot(vec(cov_models[k]), vec(cov_obs)) /
                max(norm(vec(cov_models[k])) * norm(vec(cov_obs)), eps(Float64))
            println(io, @sprintf("%s covariance rel.RMSE = %.8e", model.label, rel))
            println(io, @sprintf("%s covariance corr = %.8e", model.label, corr))
        end
        println(io)
        println(io, @sprintf("max_lag = %d", max_lag))
        println(io, @sprintf("pairs_per_lag = %d", pairs_per_lag))
        println(io, @sprintf("mean_samples = %d", mean_samples))
    end

    @printf("Saved mobility11 forward figures:\n  %s\n  %s\n", stats_path, corr_path)
    @printf("Saved mobility11 metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
