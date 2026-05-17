#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct DataOnlyPhiParams
    input_hdf5::String
    phi_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    candidate_lags::Vector{Int}
    polynomial_degree::Int
    true_mobility_samples::Int
    seed::Int
    output_hdf5::String
    metrics_txt::String
    figure_png::String
end

function load_dataonly_phi_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    phi = raw["phi"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    return DataOnlyPhiParams(
        input_hdf5=String(data["input_hdf5"]),
        phi_hdf5=String(data["phi_hdf5"]),
        score_bson=String(data["score_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        candidate_lags=Int.(get(phi, "candidate_lags", [1, 2, 3, 4, 5, 8, 12, 16, 20])),
        polynomial_degree=Int(get(phi, "polynomial_degree", 2)),
        true_mobility_samples=Int(get(phi, "true_mobility_samples", 12000)),
        seed=Int(get(run, "seed", 20260710)),
        output_hdf5=String(output["hdf5_file"]),
        metrics_txt=String(output["metrics_txt"]),
        figure_png=String(output["figure_png"]),
    )
end

function flat_norm_trajectory(norm_states::Array{Float32, 4}, trajs::Vector{Int})
    nt, K, _, _ = size(norm_states)
    D = 2K
    Z = Array{Float64}(undef, nt, D, length(trajs))
    @inbounds for (jj, tr) in enumerate(trajs), t in 1:nt
        for i in 1:K
            Z[t, i, jj] = norm_states[t, i, 1, tr]
            Z[t, K + i, jj] = norm_states[t, i, 2, tr]
        end
    end
    return Z
end

function global_mean_flat(Z::Array{Float64, 3})
    nt, D, ntraj = size(Z)
    mu = zeros(Float64, D)
    @inbounds for tr in 1:ntraj, t in 1:nt
        mu .+= @view Z[t, :, tr]
    end
    return mu ./ (nt * ntraj)
end

function lag_covariances_global(norm_states::Array{Float32, 4}, save_dt::Float64,
        max_lag::Int, trajs::Vector{Int})
    Z = flat_norm_trajectory(norm_states, trajs)
    nt, D, ntraj = size(Z)
    L = min(max_lag, nt - 1)
    mu = global_mean_flat(Z)
    C = Array{Float64}(undef, L + 1, D, D)
    taus = collect(0:L) .* save_dt
    @inbounds for lag in 0:L
        count = (nt - lag) * ntraj
        Cacc = zeros(Float64, D, D)
        for tr in 1:ntraj, t in 1:(nt - lag)
            x0 = @view Z[t, :, tr]
            xt = @view Z[t + lag, :, tr]
            for i in 1:D
                xi = xt[i] - mu[i]
                for j in 1:D
                    Cacc[i, j] += xi * (x0[j] - mu[j])
                end
            end
        end
        C[lag + 1, :, :] .= Cacc ./ count
    end
    return taus, C
end

function qv_symmetric(norm_states::Array{Float32, 4}, save_dt::Float64, trajs::Vector{Int})
    Z = flat_norm_trajectory(norm_states, trajs)
    nt, D, ntraj = size(Z)
    S = zeros(Float64, D, D)
    count = (nt - 1) * ntraj
    @inbounds for tr in 1:ntraj, t in 1:(nt - 1)
        for i in 1:D
            di = Z[t + 1, i, tr] - Z[t, i, tr]
            for j in 1:D
                S[i, j] += di * (Z[t + 1, j, tr] - Z[t, j, tr])
            end
        end
    end
    S ./= (2.0 * save_dt * count)
    return sympart(S)
end

function finite_difference_phi(taus::Vector{Float64}, C::Array{Float64, 3}, L::Int, degree::Int)
    return phi_from_covariance_derivative(taus[1:(L + 1)], C[1:(L + 1), :, :], degree)[2]
end

function stabilized(A::AbstractMatrix{<:Real}, K::Int)
    return project_tangent_symmetric_psd(project_block_circulant_matrix(A, K), K)
end

function small_lag_residual(Phi::AbstractMatrix{<:Real}, taus::Vector{Float64},
        C::Array{Float64, 3}, L::Int, K::Int)
    C0 = Matrix(C[1, :, :])
    denom = 0.0
    numer = 0.0
    for ell in 2:(L + 1)
        pred = C0 .- taus[ell] .* Matrix(Phi)
        obs = Matrix(C[ell, :, :])
        bp = project_block_circulant_matrix(pred, K)
        bo = project_block_circulant_matrix(obs, K)
        numer += sum(abs2, bp .- bo)
        denom += sum(abs2, bo .- project_block_circulant_matrix(C0, K))
    end
    return sqrt(numer / max(denom, eps(Float64)))
end

function estimator_table(norm_states::Array{Float32, 4}, save_dt::Float64,
        lags::Vector{Int}, degree::Int, K::Int)
    ntraj = size(norm_states, 4)
    all_trajs = collect(1:ntraj)
    odd_trajs = collect(1:2:ntraj)
    even_trajs = collect(2:2:ntraj)
    maxlag = maximum(lags)
    taus_all, C_all = lag_covariances_global(norm_states, save_dt, maxlag, all_trajs)
    taus_odd, C_odd = lag_covariances_global(norm_states, save_dt, maxlag, odd_trajs)
    taus_even, C_even = lag_covariances_global(norm_states, save_dt, maxlag, even_trajs)
    S_all = stabilized(qv_symmetric(norm_states, save_dt, all_trajs), K)
    S_odd = stabilized(qv_symmetric(norm_states, save_dt, odd_trajs), K)
    S_even = stabilized(qv_symmetric(norm_states, save_dt, even_trajs), K)
    rows = Vector{Dict{Symbol, Any}}()
    for L in lags
        _, raw_log, imag_rel = phi_from_covariance_log(taus_all[1:(L + 1)], C_all[1:(L + 1), :, :], K)
        _, odd_log, _ = phi_from_covariance_log(taus_odd[1:(L + 1)], C_odd[1:(L + 1), :, :], K)
        _, even_log, _ = phi_from_covariance_log(taus_even[1:(L + 1)], C_even[1:(L + 1), :, :], K)
        raw_fd = finite_difference_phi(taus_all, C_all, L, degree)
        odd_fd = finite_difference_phi(taus_odd, C_odd, L, degree)
        even_fd = finite_difference_phi(taus_even, C_even, L, degree)

        candidates = [
            ("logcov_L$(L)", stabilized(raw_log, K), stabilized(odd_log, K), stabilized(even_log, K), imag_rel),
            ("poly_L$(L)", stabilized(raw_fd, K), stabilized(odd_fd, K), stabilized(even_fd, K), 0.0),
            ("qv_plus_skew_logcov_L$(L)", stabilized(S_all + skewpart(raw_log), K),
                stabilized(S_odd + skewpart(odd_log), K), stabilized(S_even + skewpart(even_log), K), imag_rel),
            ("qv_plus_skew_poly_L$(L)", stabilized(S_all + skewpart(raw_fd), K),
                stabilized(S_odd + skewpart(odd_fd), K), stabilized(S_even + skewpart(even_fd), K), 0.0),
        ]
        for (name, Phi, Phi_odd, Phi_even, imag) in candidates
            split_rel = norm(Phi_odd - Phi_even) / max(norm(0.5 .* (Phi_odd .+ Phi_even)), eps(Float64))
            self = small_lag_residual(Phi, taus_all, C_all, min(L, 4), K)
            eigs = tangent_eigs(Phi, K)
            push!(rows, Dict{Symbol, Any}(
                :name => name,
                :lag_count => L,
                :Phi => Phi,
                :split_rel => split_rel,
                :small_lag_residual => self,
                :score => split_rel + 0.25 * self,
                :imag_rel => imag,
                :eig_min => minimum(eigs),
                :eig_max => maximum(eigs),
            ))
        end
    end
    sort!(rows; by=r -> (r[:score], r[:split_rel], r[:small_lag_residual]))
    return rows, taus_all, C_all
end

function render_dataonly_phi(path, rows, Mtrue, K)
    best = rows[1]
    with_scaled_figure_style(3200, 2200) do _
        fig = Figure(; size=(3200, 2200))
        figure_title!(fig, "Capillary FHD N32 data-only Phi estimator";
            subtitle=string("accepted by split stability: ", best[:name]))
        mats = [best[:Phi], Mtrue, best[:Phi] .- Mtrue]
        titles = ["accepted data-only Phi", "<M true> ex-post", "Phi - <M true>"]
        for i in 1:3
            ax = Axis(fig[1, i]; title=titles[i], xlabel="column", ylabel="row")
            mat = mats[i]
            if i == 3
                lim = max(maximum(abs, mat), 1.0e-8)
                hm = heatmap!(ax, mat; colormap=STYLE_DIVERGING_SOFT, colorrange=(-lim, lim))
            else
                hm = heatmap!(ax, mat; colormap=:viridis)
            end
            Colorbar(fig[1, i, Right()], hm)
        end
        topn = min(14, length(rows))
        names = [String(rows[i][:name]) for i in 1:topn]
        split = [Float64(rows[i][:split_rel]) for i in 1:topn]
        self = [Float64(rows[i][:small_lag_residual]) for i in 1:topn]
        reltrue = [Float64(rows[i][:true_rel]) for i in 1:topn]
        ax2 = Axis(fig[2, 1:2]; title="Data-only selection diagnostics", xlabel="candidate rank", ylabel="relative value")
        lines!(ax2, 1:topn, split; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="split half rel.diff")
        lines!(ax2, 1:topn, self; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash, label="small-lag residual")
        lines!(ax2, 1:topn, reltrue; color=STYLE_REFERENCE, linewidth=curve_linewidth(), linestyle=:dot, label="ex-post true rel.RMSE")
        axislegend(ax2; position=:rt)
        ax3 = Axis(fig[2, 3]; title="candidate names", xlabel="", ylabel="")
        hidedecorations!(ax3)
        hidespines!(ax3)
        text!(ax3, 0.0, 1.0; text=join(["$(i). $(names[i])" for i in 1:topn], "\n"),
            align=(:left, :top), fontsize=18)
        lines = [
            @sprintf("accepted = %s", best[:name]),
            @sprintf("data-only score = %.8e", best[:score]),
            @sprintf("split-half rel.diff = %.8e", best[:split_rel]),
            @sprintf("small-lag self residual = %.8e", best[:small_lag_residual]),
            @sprintf("ex-post Phi vs <M_true> rel.RMSE = %.8e", best[:true_rel]),
            @sprintf("ex-post Phi vs <M_true> corr = %.8e", best[:true_corr]),
            @sprintf("sym(Phi) tangent eig min/max = %.8e / %.8e", best[:eig_min], best[:eig_max]),
            "True mobility is shown only after data-only estimator ranking.",
        ]
        text_panel!(fig[3, 1:3], lines; title="No-cheating audit")
        save_figure_checked(path, fig)
    end
    return nothing
end

function run_dataonly_phi(config_path::AbstractString)
    params = load_dataonly_phi_params(config_path)
    base = dirname(abspath(config_path))
    input_h5 = resolve_path(base, params.input_hdf5)
    phi_h5 = resolve_path(base, params.phi_hdf5)
    score_bson = resolve_path(base, params.score_bson)
    out_h5 = resolve_path(base, params.output_hdf5)
    metrics_txt = resolve_path(base, params.metrics_txt)
    fig_png = resolve_path(base, params.figure_png)
    require_condition(isfile(input_h5), "Missing input_hdf5: $(input_h5)")
    require_condition(isfile(phi_h5), "Missing phi_hdf5: $(phi_h5)")
    require_condition(isfile(score_bson), "Missing score_bson: $(score_bson)")

    _, stats, _, phys, _, _, _ = load_checkpoint(score_bson, :cpu)
    phi_times, phi_states = load_states(phi_h5)
    save_dt = phi_times[2] - phi_times[1]
    norm_phi_states = normalize_states(phi_states, stats)
    rows, taus, C = estimator_table(norm_phi_states, save_dt, params.candidate_lags,
        params.polynomial_degree, phys.N)

    raw_samples = collect_postburnin_samples(phi_states, 1, 0, MersenneTwister(params.seed + 1))
    Mtrue_phys = estimate_mean_true_mobility(raw_samples, phys, params.true_mobility_samples,
        MersenneTwister(params.seed + 2))
    Mtrue = transform_mobility_to_norm(Mtrue_phys, stats)
    Mtrue = project_block_circulant_matrix(Mtrue, phys.N)
    Mtrue = projection_matrix(phys.N) * Mtrue * projection_matrix(phys.N)
    for row in rows
        rel, corr = agreement_metrics(Mtrue, row[:Phi])
        row[:true_rel] = rel
        row[:true_corr] = corr
    end

    ensure_parent_dir(metrics_txt)
    open(metrics_txt, "w") do io
        println(io, "FHDChainCapillaryN32 data-only Phi estimator sweep")
        println(io, "Selection score uses only split-trajectory stability and observed short-lag covariance residuals.")
        println(io, "True mobility appears only in ex-post columns below.")
        println(io, "rank\tname\tdata_score\tsplit_rel\tself_residual\texpost_true_rel\texpost_true_corr\teig_min\teig_max")
        for (rank, row) in enumerate(rows)
            println(io, @sprintf("%d\t%s\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e",
                rank, row[:name], row[:score], row[:split_rel], row[:small_lag_residual],
                row[:true_rel], row[:true_corr], row[:eig_min], row[:eig_max]))
        end
    end
    @printf("Accepted data-only estimator: %s split=%.6e self=%.6e ex-post true rel=%.6e corr=%.6f\n",
        rows[1][:name], rows[1][:split_rel], rows[1][:small_lag_residual],
        rows[1][:true_rel], rows[1][:true_corr])

    ensure_parent_dir(out_h5)
    h5open(out_h5, "w") do h5
        write(h5, "/Phi", rows[1][:Phi])
        write(h5, "/Mtrue_norm_expost", Mtrue)
        write(h5, "/lag_covariance/taus", taus)
        write(h5, "/lag_covariance/C", C)
        for (rank, row) in enumerate(rows)
            g = create_group(h5, "/candidates/rank_$(rank)")
            write(g, "name", String(row[:name]))
            write(g, "Phi", row[:Phi])
            write(g, "data_score", Float64(row[:score]))
            write(g, "split_rel", Float64(row[:split_rel]))
            write(g, "small_lag_residual", Float64(row[:small_lag_residual]))
            write(g, "expost_true_rel", Float64(row[:true_rel]))
            write(g, "expost_true_corr", Float64(row[:true_corr]))
        end
    end
    render_dataonly_phi(fig_png, rows, Mtrue, phys.N)
    @printf("Saved data-only Phi sweep to %s\n", out_h5)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi_dataonly.toml")) : abspath(ARGS[1])
    run_dataonly_phi(cfg)
end
