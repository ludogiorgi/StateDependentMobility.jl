#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi_gaussian_score.jl"))
include(joinpath(@__DIR__, "fit_Phi_dataonly.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct PhiDtSource
    label::String
    hdf5::String
    stride::Int
    candidate_lags::Vector{Int}
    polynomial_degree::Int
end

Base.@kwdef struct PhiDtSweepParams
    input_hdf5::String
    covariance_hdf5::String
    score_bson::String
    reference_forward_hdf5::String
    burnin_fraction::Float64
    covariance_burnin_fraction::Float64
    ridge_relative::Float64
    sources::Vector{PhiDtSource}
    true_mobility_samples::Int
    forward_save_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_ntraj::Int
    seed::Int
    output_hdf5::String
    metrics_txt::String
    figure_png::String
end

function load_phi_dt_sweep_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    score = get(raw, "score", Dict{String, Any}())
    phi = raw["phi"]
    forward = raw["forward"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    sources = PhiDtSource[]
    for item in phi["sources"]
        push!(sources, PhiDtSource(
            label=String(item["label"]),
            hdf5=String(item["hdf5"]),
            stride=Int(get(item, "stride", 1)),
            candidate_lags=Int.(get(item, "candidate_lags", [1, 2, 3, 4, 5, 8, 10])),
            polynomial_degree=Int(get(item, "polynomial_degree", 2)),
        ))
    end
    return PhiDtSweepParams(
        input_hdf5=String(data["input_hdf5"]),
        covariance_hdf5=String(data["covariance_hdf5"]),
        score_bson=String(data["score_bson"]),
        reference_forward_hdf5=String(get(data, "reference_forward_hdf5", "")),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        covariance_burnin_fraction=Float64(get(data, "covariance_burnin_fraction", 0.0)),
        ridge_relative=Float64(get(score, "ridge_relative", 1.0e-8)),
        sources=sources,
        true_mobility_samples=Int(get(phi, "true_mobility_samples", 12000)),
        forward_save_dt=Float64(forward["save_dt"]),
        forward_total_time=Float64(forward["total_time"]),
        forward_burnin_time=Float64(forward["burnin_time"]),
        forward_ntraj=Int(forward["ntrajectories"]),
        seed=Int(get(run, "seed", 20260830)),
        output_hdf5=String(output["hdf5_file"]),
        metrics_txt=String(output["metrics_txt"]),
        figure_png=String(output["figure_png"]),
    )
end

function subsample_time(states::Array{Float32, 4}, times::Vector{Float64}, stride::Int)
    idx = collect(1:stride:length(times))
    return times[idx], states[idx, :, :, :]
end

function estimate_phi_for_source(src::PhiDtSource, base::AbstractString, stats::DataStats,
        phys::FHDPhys, Mtrue::AbstractMatrix{<:Real}, seed::Int)
    path = resolve_path(base, src.hdf5)
    require_condition(isfile(path), "Missing source HDF5: $(path)")
    times, states = load_states(path)
    times, states = subsample_time(states, times, src.stride)
    save_dt = times[2] - times[1]
    norm_states = normalize_states(states, stats)
    rows, taus, C = estimator_table(norm_states, save_dt, src.candidate_lags,
        src.polynomial_degree, phys.N)
    for row in rows
        rel, corr = agreement_metrics(Mtrue, row[:Phi])
        row[:true_rel] = rel
        row[:true_corr] = corr
    end
    best = rows[1]
    best_expost = rows[argmin([r[:true_rel] for r in rows])]
    @printf("%s: save_dt=%.8g accepted=%s score=%.6e ex-post rel=%.6e corr=%.6f best-expost=%s rel=%.6e\n",
        src.label, save_dt, best[:name], best[:score], best[:true_rel], best[:true_corr],
        best_expost[:name], best_expost[:true_rel])
    return Dict{Symbol, Any}(
        :source => src,
        :path => path,
        :save_dt => save_dt,
        :times => taus,
        :C => C,
        :rows => rows,
        :accepted => best,
        :best_expost => best_expost,
    )
end

function lightweight_forward_params(params::PhiDtSweepParams)
    return GaussianScoreForwardParams(
        input_hdf5=params.input_hdf5,
        covariance_hdf5=params.covariance_hdf5,
        score_bson=params.score_bson,
        phi_hdf5="",
        reference_forward_hdf5=params.reference_forward_hdf5,
        burnin_fraction=params.burnin_fraction,
        covariance_burnin_fraction=params.covariance_burnin_fraction,
        ridge_relative=params.ridge_relative,
        forward_dt=params.forward_save_dt,
        forward_total_time=params.forward_total_time,
        forward_burnin_time=params.forward_burnin_time,
        forward_save_dt=params.forward_save_dt,
        forward_ntraj=params.forward_ntraj,
        seed=params.seed,
        metrics_txt=params.metrics_txt,
        figure_png=params.figure_png,
        forward_hdf5=params.output_hdf5,
    )
end

function run_forward_for_phi(Phi::Matrix{Float64}, label::AbstractString,
        G::Matrix{Float64}, mu::Vector{Float64}, Cscore::Matrix{Float64},
        norm_obs::Array{Float32, 4}, obs_states::Array{Float32, 4}, obs_start::Int,
        true_states::Array{Float32, 4}, stats::DataStats, phys::FHDPhys,
        params::PhiDtSweepParams, seed_offset::Int)
    fparams = lightweight_forward_params(params)
    fparams = GaussianScoreForwardParams(
        input_hdf5=fparams.input_hdf5,
        covariance_hdf5=fparams.covariance_hdf5,
        score_bson=fparams.score_bson,
        phi_hdf5=fparams.phi_hdf5,
        reference_forward_hdf5=fparams.reference_forward_hdf5,
        burnin_fraction=fparams.burnin_fraction,
        covariance_burnin_fraction=fparams.covariance_burnin_fraction,
        ridge_relative=fparams.ridge_relative,
        forward_dt=fparams.forward_dt,
        forward_total_time=fparams.forward_total_time,
        forward_burnin_time=fparams.forward_burnin_time,
        forward_save_dt=fparams.forward_save_dt,
        forward_ntraj=fparams.forward_ntraj,
        seed=params.seed + seed_offset,
        metrics_txt=fparams.metrics_txt,
        figure_png=fparams.figure_png,
        forward_hdf5=fparams.forward_hdf5,
    )
    times, states, save_dt, min_eig = integrate_gaussian_phi_exact(Phi, G, mu, Cscore,
        norm_obs, obs_start, stats, phys, fparams)
    obs_save_dt = 0.5
    metrics = forward_metrics(obs_states, true_states, states, obs_start, phys, obs_save_dt)
    rho = metrics[:rho_learned_acf_rel_l2]
    mom = metrics[:m_learned_acf_rel_l2]
    metrics[:combined_acf_rel_l2] = sqrt(0.5 * (rho^2 + mom^2))
    metrics[:forward_min_eig] = min_eig
    @printf("%s forward: rho_acf=%.6e m_acf=%.6e combined=%.6e cov=%.6e\n",
        label, metrics[:rho_learned_acf_rel_l2], metrics[:m_learned_acf_rel_l2],
        metrics[:combined_acf_rel_l2], metrics[:learned_covariance_rel_rmse])
    return times, states, metrics, save_dt
end

function render_phi_dt_sweep(path, results)
    labels = [String(r[:label]) for r in results]
    dts = [Float64(r[:phi_dt]) for r in results]
    rho = [Float64(r[:metrics][:rho_learned_acf_rel_l2]) for r in results]
    mom = [Float64(r[:metrics][:m_learned_acf_rel_l2]) for r in results]
    combined = [Float64(r[:metrics][:combined_acf_rel_l2]) for r in results]
    phi_rel = [Float64(r[:phi_true_rel]) for r in results]
    cov = [Float64(r[:metrics][:learned_covariance_rel_rmse]) for r in results]
    with_scaled_figure_style(2600, 1700) do _
        fig = Figure(; size=(2600, 1700))
        figure_title!(fig, "Gaussian-score forward validation across Phi estimation cadences";
            subtitle="Phi candidates selected data-only at each cadence")
        ax1 = Axis(fig[1, 1]; title="Observed time-correlation recovery", xlabel="Phi estimation cadence", ylabel="relative RMSE")
        xs = collect(1:length(results))
        lines!(ax1, xs, rho; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="rho ACF")
        scatter!(ax1, xs, rho; color=STYLE_PRIMARY, markersize=16)
        lines!(ax1, xs, mom; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label="m ACF")
        scatter!(ax1, xs, mom; color=STYLE_SECONDARY, markersize=16)
        lines!(ax1, xs, combined; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="combined")
        scatter!(ax1, xs, combined; color=STYLE_REFERENCE, markersize=16)
        ax1.xticks = (xs, labels)
        axislegend(ax1; position=:lt)
        ax2 = Axis(fig[1, 2]; title="Phi and covariance diagnostics", xlabel="Phi estimation cadence", ylabel="relative error")
        lines!(ax2, xs, phi_rel; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="Phi vs <M> ex-post")
        scatter!(ax2, xs, phi_rel; color=STYLE_PRIMARY, markersize=16)
        lines!(ax2, xs, cov; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash, label="forward covariance")
        scatter!(ax2, xs, cov; color=STYLE_SECONDARY, markersize=16)
        ax2.xticks = (xs, labels)
        axislegend(ax2; position=:lt)
        lines = String[]
        for r in results
            push!(lines, @sprintf("%s: dt=%.4g, Phi rel=%.4g, rho ACF=%.4g, m ACF=%.4g, combined=%.4g",
                r[:label], r[:phi_dt], r[:phi_true_rel],
                r[:metrics][:rho_learned_acf_rel_l2], r[:metrics][:m_learned_acf_rel_l2],
                r[:metrics][:combined_acf_rel_l2]))
        end
        text_panel!(fig[2, 1:2], lines; title="Summary")
        save_figure_checked(path, fig)
    end
end

function run_phi_dt_sweep(config_path::AbstractString)
    params = load_phi_dt_sweep_params(config_path)
    base = dirname(abspath(config_path))
    input_h5 = resolve_path(base, params.input_hdf5)
    cov_h5 = resolve_path(base, params.covariance_hdf5)
    score_bson = resolve_path(base, params.score_bson)
    reference_h5 = isempty(strip(params.reference_forward_hdf5)) ? "" :
        resolve_path(base, params.reference_forward_hdf5)
    out_h5 = resolve_path(base, params.output_hdf5)
    metrics_txt = resolve_path(base, params.metrics_txt)
    figure_png = resolve_path(base, params.figure_png)
    for p in (input_h5, cov_h5, score_bson)
        require_condition(isfile(p), "Missing required input: $(p)")
    end

    _, stats, _, phys, _, _, _ = load_checkpoint(score_bson, :cpu)
    obs_times, obs_states = load_states(input_h5)
    cov_times, cov_states = load_states(cov_h5)
    norm_obs = normalize_states(obs_states, stats)
    norm_cov = normalize_states(cov_states, stats)
    obs_start = burnin_start_index(length(obs_times), params.burnin_fraction)
    cov_start = burnin_start_index(length(cov_times), params.covariance_burnin_fraction)
    G, mu, Cscore, cov_eigs, ridge = empirical_gaussian_score_matrix(norm_cov, cov_start,
        phys.N, params.ridge_relative)
    raw_samples = collect_postburnin_samples(obs_states, obs_start, 0, MersenneTwister(params.seed + 1))
    Mtrue_phys = estimate_mean_true_mobility(raw_samples, phys, params.true_mobility_samples,
        MersenneTwister(params.seed + 2))
    Mtrue = transform_mobility_to_norm(Mtrue_phys, stats)
    Mtrue = projection_matrix(phys.N) * project_block_circulant_matrix(Mtrue, phys.N) * projection_matrix(phys.N)
    true_states = if !isempty(reference_h5) && isfile(reference_h5)
        h5open(reference_h5, "r") do h5
            Float32.(read(h5["/true_phi/states"]))
        end
    else
        obs_states
    end

    results = Vector{Dict{Symbol, Any}}()
    for (idx, src) in enumerate(params.sources)
        est = estimate_phi_for_source(src, base, stats, phys, Mtrue, params.seed + 10idx)
        Phi = projection_matrix(phys.N) * Matrix(est[:accepted][:Phi]) * projection_matrix(phys.N)
        ftime, fstates, metrics, fsave = run_forward_for_phi(Phi, src.label, G, mu, Cscore,
            norm_obs, obs_states, obs_start, true_states, stats, phys, params, 1000idx)
        push!(results, Dict{Symbol, Any}(
            :label => src.label,
            :phi_dt => est[:save_dt],
            :accepted_name => String(est[:accepted][:name]),
            :selection_score => Float64(est[:accepted][:score]),
            :split_rel => Float64(est[:accepted][:split_rel]),
            :small_lag_residual => Float64(est[:accepted][:small_lag_residual]),
            :phi_true_rel => Float64(est[:accepted][:true_rel]),
            :phi_true_corr => Float64(est[:accepted][:true_corr]),
            :best_expost_name => String(est[:best_expost][:name]),
            :best_expost_rel => Float64(est[:best_expost][:true_rel]),
            :times => ftime,
            :states => fstates,
            :metrics => metrics,
            :forward_save_dt => fsave,
            :Phi => Phi,
            :candidate_rows => est[:rows],
        ))
    end

    ensure_parent_dir(metrics_txt)
    open(metrics_txt, "w") do io
        println(io, "FHDChainCapillaryN32 Gaussian-score forward comparison across Phi estimation cadences")
        println(io, "Phi selection at each cadence is data-only; true mobility is ex-post.")
        println(io, "Gaussian score covariance ridge = $(ridge)")
        println(io, "label\tphi_dt\taccepted\tdata_score\tsplit_rel\tself_residual\tphi_true_rel\tphi_true_corr\tbest_expost\tbest_expost_rel\trho_acf_rmse\tm_acf_rmse\tcombined_acf_rmse\tcov_rmse")
        for r in results
            m = r[:metrics]
            println(io, @sprintf("%s\t%.10e\t%s\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%s\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e",
                r[:label], r[:phi_dt], r[:accepted_name], r[:selection_score], r[:split_rel],
                r[:small_lag_residual], r[:phi_true_rel], r[:phi_true_corr], r[:best_expost_name],
                r[:best_expost_rel], m[:rho_learned_acf_rel_l2], m[:m_learned_acf_rel_l2],
                m[:combined_acf_rel_l2], m[:learned_covariance_rel_rmse]))
        end
    end
    render_phi_dt_sweep(figure_png, results)
    ensure_parent_dir(out_h5)
    h5open(out_h5, "w") do h5
        write(h5, "/score/G", G)
        write(h5, "/score/mu", mu)
        write(h5, "/score/C", Cscore)
        write(h5, "/score/ridge", ridge)
        for r in results
            g = create_group(h5, "/cases/" * r[:label])
            write(g, "phi_dt", r[:phi_dt])
            write(g, "accepted_name", r[:accepted_name])
            write(g, "Phi", r[:Phi])
            write(g, "time", r[:times])
            write(g, "states", r[:states])
            for (k, v) in r[:metrics]
                v isa Number && write(g, "metrics/" * String(k), Float64(v))
            end
            write(g, "metrics/phi_true_rel", r[:phi_true_rel])
            write(g, "metrics/phi_true_corr", r[:phi_true_corr])
        end
    end
    @printf("Saved Phi dt sweep outputs to %s\n", out_h5)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi_dt_sweep_gaussian.toml")) : abspath(ARGS[1])
    run_phi_dt_sweep(cfg)
end
