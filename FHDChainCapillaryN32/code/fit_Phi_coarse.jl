#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi_dataonly.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct CoarsePhiParams
    input_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    fit_lags::Vector{Int}
    validation_lags::Vector{Int}
    polynomial_degrees::Vector{Int}
    true_mobility_samples::Int
    seed::Int
    output_hdf5::String
    metrics_txt::String
    figure_png::String
end

function load_coarse_phi_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    phi = raw["phi"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    return CoarsePhiParams(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        fit_lags=Int.(get(phi, "fit_lags", [1, 2, 3, 4, 5, 8, 10, 12, 16, 20])),
        validation_lags=Int.(get(phi, "validation_lags", [1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 30, 40])),
        polynomial_degrees=Int.(get(phi, "polynomial_degrees", [1, 2, 3])),
        true_mobility_samples=Int(get(phi, "true_mobility_samples", 12000)),
        seed=Int(get(run, "seed", 20260770)),
        output_hdf5=String(output["hdf5_file"]),
        metrics_txt=String(output["metrics_txt"]),
        figure_png=String(output["figure_png"]),
    )
end

function postburnin_norm_states(states::Array{Float32, 4}, stats::DataStats,
        burnin_fraction::Float64)
    start_idx = burnin_start_index(size(states, 1), burnin_fraction)
    norm_states = normalize_states(states[start_idx:end, :, :, :], stats)
    return norm_states, start_idx
end

function projected_lag_covariances(norm_states::Array{Float32, 4}, save_dt::Float64,
        max_lag::Int, trajs::Vector{Int}, K::Int)
    taus, C = lag_covariances_global(norm_states, save_dt, max_lag, trajs)
    P = projection_matrix(K)
    Cp = similar(C)
    @inbounds for ell in axes(C, 1)
        Cp[ell, :, :] .= project_block_circulant_matrix(P * Matrix(C[ell, :, :]) * P, K)
    end
    return taus, Cp
end

function reduced_covariances(C::Array{Float64, 3}, Q::AbstractMatrix{<:Real})
    L = size(C, 1)
    d = size(Q, 2)
    out = Array{Float64}(undef, L, d, d)
    @inbounds for ell in 1:L
        out[ell, :, :] .= Q' * Matrix(C[ell, :, :]) * Q
    end
    return out
end

function weighted_log_generator(taus::Vector{Float64}, Cred::Array{Float64, 3},
        L::Int, weight_power::Float64)
    C0 = Matrix(Cred[1, :, :])
    C0inv = pinv(C0; rtol=1.0e-8)
    A = zeros(Float64, size(C0))
    imag_max = 0.0
    wsum = 0.0
    for lag in 1:L
        tau = taus[lag + 1]
        F = Matrix(Cred[lag + 1, :, :]) * C0inv
        LF, imag_rel = real_matrix_log(F)
        w = tau^(-weight_power)
        A .+= w .* (LF ./ tau)
        wsum += w
        imag_max = max(imag_max, imag_rel)
    end
    return A ./ wsum, imag_max
end

function var1_generator(taus::Vector{Float64}, Cred::Array{Float64, 3}, L::Int)
    d = size(Cred, 2)
    left = zeros(Float64, d, d)
    right = zeros(Float64, d, d)
    for lag in 0:(L - 1)
        Cnow = Matrix(Cred[lag + 1, :, :])
        Cnext = Matrix(Cred[lag + 2, :, :])
        left .+= Cnext * transpose(Cnow)
        right .+= Cnow * transpose(Cnow)
    end
    F = left * pinv(right; rtol=1.0e-8)
    LF, imag_rel = real_matrix_log(F)
    return LF ./ (taus[2] - taus[1]), imag_rel
end

function fourier_blocks(A::AbstractMatrix{<:Real}, K::Int)
    prof = block_profile(A, K)
    blocks = Array{ComplexF64}(undef, K, 2, 2)
    @inbounds for q in 0:(K - 1), a in 1:2, b in 1:2
        acc = 0.0 + 0.0im
        for r in 0:(K - 1)
            acc += prof[r + 1, a, b] * cis(-2π * q * r / K)
        end
        blocks[q + 1, a, b] = acc
    end
    return blocks
end

function matrix_from_fourier_blocks(blocks::Array{ComplexF64, 3})
    K = size(blocks, 1)
    prof = zeros(Float64, K, 2, 2)
    @inbounds for r in 0:(K - 1), a in 1:2, b in 1:2
        acc = 0.0 + 0.0im
        for q in 0:(K - 1)
            acc += blocks[q + 1, a, b] * cis(2π * q * r / K)
        end
        prof[r + 1, a, b] = real(acc / K)
    end
    return matrix_from_block_profile(prof)
end

function unwrap_phases(phases::Vector{Float64})
    out = copy(phases)
    for i in 2:length(out)
        while out[i] - out[i - 1] > π
            out[i] -= 2π
        end
        while out[i] - out[i - 1] < -π
            out[i] += 2π
        end
    end
    return out
end

function var1_propagator_matrix(C::Array{Float64, 3}, L::Int)
    D = size(C, 2)
    left = zeros(Float64, D, D)
    right = zeros(Float64, D, D)
    for lag in 0:(L - 1)
        Cnow = Matrix(C[lag + 1, :, :])
        Cnext = Matrix(C[lag + 2, :, :])
        left .+= Cnext * transpose(Cnow)
        right .+= Cnow * transpose(Cnow)
    end
    return project_block_circulant_matrix(left * pinv(right; rtol=1.0e-8), size(C, 2) ÷ 2)
end

function unwrapped_fourier_generator(C::Array{Float64, 3}, taus::Vector{Float64}, L::Int,
        K::Int; source::Symbol=:var1)
    dt = taus[2] - taus[1]
    C0_blocks = fourier_blocks(Matrix(C[1, :, :]), K)
    Fmat = if source == :var1
        var1_propagator_matrix(C, L)
    elseif source == :lag1
        Matrix(C[2, :, :]) * pinv(Matrix(C[1, :, :]); rtol=1.0e-8)
    else
        error("Unsupported Fourier propagator source $(source)")
    end
    F_blocks = fourier_blocks(project_block_circulant_matrix(Fmat, K), K)
    qmax = K ÷ 2
    plus_phase = zeros(Float64, qmax)
    minus_phase = zeros(Float64, qmax)
    plus_eval = Vector{ComplexF64}(undef, qmax)
    minus_eval = Vector{ComplexF64}(undef, qmax)
    plus_vec = Vector{Matrix{ComplexF64}}(undef, qmax)
    minus_order = zeros(Int, qmax)
    @inbounds for q in 1:qmax
        Fq = Matrix(F_blocks[q + 1, :, :])
        E = eigen(Fq)
        vals = E.values
        order = sortperm(imag.(log.(vals)); rev=true)
        plus_idx = order[1]
        minus_idx = order[end]
        plus_eval[q] = vals[plus_idx]
        minus_eval[q] = vals[minus_idx]
        plus_phase[q] = angle(vals[plus_idx])
        minus_phase[q] = angle(vals[minus_idx])
        plus_vec[q] = E.vectors
        minus_order[q] = minus_idx == 1 ? 2 : 1
    end
    plus_unwrapped = unwrap_phases(plus_phase)
    minus_unwrapped = unwrap_phases(minus_phase)
    A_blocks = zeros(ComplexF64, K, 2, 2)
    roughness = 0.0
    if length(plus_unwrapped) >= 3
        roughness += sum(abs2, diff(diff(plus_unwrapped)))
        roughness += sum(abs2, diff(diff(minus_unwrapped)))
    end
    imag_max = 0.0
    @inbounds for q in 1:qmax
        Fq = Matrix(F_blocks[q + 1, :, :])
        E = eigen(Fq)
        vals = E.values
        order = sortperm(imag.(log.(vals)); rev=true)
        lambda = similar(vals)
        lambda[order[1]] = (log(abs(vals[order[1]])) + im * plus_unwrapped[q]) / dt
        lambda[order[end]] = (log(abs(vals[order[end]])) + im * minus_unwrapped[q]) / dt
        Aq = E.vectors * Diagonal(lambda) * inv(E.vectors)
        A_blocks[q + 1, :, :] .= Aq
        qmirror = K - q
        if qmirror != q && qmirror != K
            A_blocks[qmirror + 1, :, :] .= conj.(Aq)
        end
        imag_max = max(imag_max, norm(imag.(Aq)) / max(norm(real.(Aq)), eps(Float64)))
    end
    Phi_blocks = zeros(ComplexF64, K, 2, 2)
    @inbounds for q in 1:K
        Phi_blocks[q, :, :] .= -Matrix(A_blocks[q, :, :]) * Matrix(C0_blocks[q, :, :])
    end
    Phi = stabilized(matrix_from_fourier_blocks(Phi_blocks), K)
    return Phi, roughness / max(qmax, 1), imag_max
end

function euler_generator(taus::Vector{Float64}, Cred::Array{Float64, 3}, L::Int)
    d = size(Cred, 2)
    left = zeros(Float64, d, d)
    right = zeros(Float64, d, d)
    dt = taus[2] - taus[1]
    for lag in 0:(L - 1)
        Cnow = Matrix(Cred[lag + 1, :, :])
        dC = (Matrix(Cred[lag + 2, :, :]) - Cnow) ./ dt
        left .+= dC * transpose(Cnow)
        right .+= Cnow * transpose(Cnow)
    end
    return left * pinv(right; rtol=1.0e-8), 0.0
end

function generator_to_phi(Ared::AbstractMatrix{<:Real}, C0red::AbstractMatrix{<:Real},
        Q::AbstractMatrix{<:Real}, K::Int)
    Cdot = Q * (Matrix(Ared) * Matrix(C0red)) * Q'
    return stabilized(-Cdot, K)
end

function poly_phi_from_projected(taus::Vector{Float64}, C::Array{Float64, 3},
        L::Int, degree::Int, K::Int)
    _, raw = phi_from_covariance_derivative(taus[1:(L + 1)], C[1:(L + 1), :, :], degree)
    return stabilized(raw, K)
end

function prediction_residual(Phi::AbstractMatrix{<:Real}, C::Array{Float64, 3},
        taus::Vector{Float64}, validation_lags::Vector{Int}, K::Int)
    Q = nonzero_mode_basis(K)
    Cred = reduced_covariances(C, Q)
    C0 = Matrix(Cred[1, :, :])
    A = -(Q' * Matrix(Phi) * Q) * pinv(C0; rtol=1.0e-8)
    denom = 0.0
    numer = 0.0
    for lag in validation_lags
        lag + 1 <= size(Cred, 1) || continue
        pred = exp(taus[lag + 1] .* A) * C0
        obs = Matrix(Cred[lag + 1, :, :])
        numer += sum(abs2, pred .- obs)
        denom += sum(abs2, obs)
    end
    return sqrt(numer / max(denom, eps(Float64)))
end

function one_step_residual(Phi::AbstractMatrix{<:Real}, C::Array{Float64, 3}, K::Int)
    return prediction_residual(Phi, C, [0.0, 1.0], [1], K)
end

function build_coarse_candidates(C_all::Array{Float64, 3}, C_odd::Array{Float64, 3},
        C_even::Array{Float64, 3}, taus::Vector{Float64}, fit_lags::Vector{Int},
        validation_lags::Vector{Int}, degrees::Vector{Int}, K::Int,
        S_all::AbstractMatrix{<:Real}, S_odd::AbstractMatrix{<:Real},
        S_even::AbstractMatrix{<:Real})
    Q = nonzero_mode_basis(K)
    Cred_all = reduced_covariances(C_all, Q)
    Cred_odd = reduced_covariances(C_odd, Q)
    Cred_even = reduced_covariances(C_even, Q)
    C0red = Matrix(Cred_all[1, :, :])
    C0red_odd = Matrix(Cred_odd[1, :, :])
    C0red_even = Matrix(Cred_even[1, :, :])
    rows = Vector{Dict{Symbol, Any}}()
    for L in fit_lags
        L + 1 <= length(taus) || continue
        for degree in degrees
            Phi = poly_phi_from_projected(taus, C_all, L, degree, K)
            Phi_odd = poly_phi_from_projected(taus, C_odd, L, degree, K)
            Phi_even = poly_phi_from_projected(taus, C_even, L, degree, K)
            push!(rows, Dict{Symbol, Any}(
                :name => "projected_poly_d$(degree)_L$(L)",
                :Phi => Phi,
                :Phi_odd => Phi_odd,
                :Phi_even => Phi_even,
                :imag_rel => 0.0,
                :roughness => 0.0,
            ))
            push!(rows, Dict{Symbol, Any}(
                :name => "coarse_qv_plus_poly_skew_d$(degree)_L$(L)",
                :Phi => stabilized(S_all + skewpart(Phi), K),
                :Phi_odd => stabilized(S_odd + skewpart(Phi_odd), K),
                :Phi_even => stabilized(S_even + skewpart(Phi_even), K),
                :imag_rel => 0.0,
                :roughness => 0.0,
            ))
        end
        for power in (0.0, 1.0, 2.0)
            A, imag = weighted_log_generator(taus, Cred_all, L, power)
            Aodd, _ = weighted_log_generator(taus, Cred_odd, L, power)
            Aeven, _ = weighted_log_generator(taus, Cred_even, L, power)
            Phi = generator_to_phi(A, C0red, Q, K)
            Phi_odd = generator_to_phi(Aodd, C0red_odd, Q, K)
            Phi_even = generator_to_phi(Aeven, C0red_even, Q, K)
            push!(rows, Dict{Symbol, Any}(
                :name => @sprintf("projected_weighted_log_p%.0f_L%d", power, L),
                :Phi => Phi,
                :Phi_odd => Phi_odd,
                :Phi_even => Phi_even,
                :imag_rel => imag,
                :roughness => 0.0,
            ))
            push!(rows, Dict{Symbol, Any}(
                :name => @sprintf("coarse_qv_plus_log_skew_p%.0f_L%d", power, L),
                :Phi => stabilized(S_all + skewpart(Phi), K),
                :Phi_odd => stabilized(S_odd + skewpart(Phi_odd), K),
                :Phi_even => stabilized(S_even + skewpart(Phi_even), K),
                :imag_rel => imag,
                :roughness => 0.0,
            ))
        end
        Avar, imag_var = var1_generator(taus, Cred_all, L)
        Avar_odd, _ = var1_generator(taus, Cred_odd, L)
        Avar_even, _ = var1_generator(taus, Cred_even, L)
        Phi_var = generator_to_phi(Avar, C0red, Q, K)
        Phi_var_odd = generator_to_phi(Avar_odd, C0red_odd, Q, K)
        Phi_var_even = generator_to_phi(Avar_even, C0red_even, Q, K)
        push!(rows, Dict{Symbol, Any}(
            :name => "projected_varlog_L$(L)",
            :Phi => Phi_var,
            :Phi_odd => Phi_var_odd,
            :Phi_even => Phi_var_even,
            :imag_rel => imag_var,
            :roughness => 0.0,
        ))
        push!(rows, Dict{Symbol, Any}(
            :name => "coarse_qv_plus_varlog_skew_L$(L)",
            :Phi => stabilized(S_all + skewpart(Phi_var), K),
            :Phi_odd => stabilized(S_odd + skewpart(Phi_var_odd), K),
            :Phi_even => stabilized(S_even + skewpart(Phi_var_even), K),
            :imag_rel => imag_var,
            :roughness => 0.0,
        ))
        Ae, imag_e = euler_generator(taus, Cred_all, L)
        Ae_odd, _ = euler_generator(taus, Cred_odd, L)
        Ae_even, _ = euler_generator(taus, Cred_even, L)
        Phi_euler = generator_to_phi(Ae, C0red, Q, K)
        Phi_euler_odd = generator_to_phi(Ae_odd, C0red_odd, Q, K)
        Phi_euler_even = generator_to_phi(Ae_even, C0red_even, Q, K)
        push!(rows, Dict{Symbol, Any}(
            :name => "projected_euler_ls_L$(L)",
            :Phi => Phi_euler,
            :Phi_odd => Phi_euler_odd,
            :Phi_even => Phi_euler_even,
            :imag_rel => imag_e,
            :roughness => 0.0,
        ))
        push!(rows, Dict{Symbol, Any}(
            :name => "coarse_qv_plus_euler_skew_L$(L)",
            :Phi => stabilized(S_all + skewpart(Phi_euler), K),
            :Phi_odd => stabilized(S_odd + skewpart(Phi_euler_odd), K),
            :Phi_even => stabilized(S_even + skewpart(Phi_euler_even), K),
            :imag_rel => imag_e,
            :roughness => 0.0,
        ))
        for source in (:var1, :lag1)
            Phi, rough, imag = unwrapped_fourier_generator(C_all, taus, L, K; source=source)
            Phi_odd, _, _ = unwrapped_fourier_generator(C_odd, taus, L, K; source=source)
            Phi_even, _, _ = unwrapped_fourier_generator(C_even, taus, L, K; source=source)
            push!(rows, Dict{Symbol, Any}(
                :name => "fourier_unwrap_$(String(source))_L$(L)",
                :Phi => Phi,
                :Phi_odd => Phi_odd,
                :Phi_even => Phi_even,
                :imag_rel => imag,
                :roughness => rough,
            ))
        end
    end
    for row in rows
        Phi_odd = row[:Phi_odd]
        Phi_even = row[:Phi_even]
        split_rel = norm(Phi_odd - Phi_even) / max(norm(0.5 .* (Phi_odd .+ Phi_even)), eps(Float64))
        pred = prediction_residual(row[:Phi], C_all, taus, validation_lags, K)
        step = prediction_residual(row[:Phi], C_all, taus, [1], K)
        eigs = tangent_eigs(row[:Phi], K)
        row[:split_rel] = split_rel
        row[:prediction_residual] = pred
        row[:one_step_residual] = step
        row[:eig_min] = minimum(eigs)
        row[:eig_max] = maximum(eigs)
        rough = Float64(get(row, :roughness, 0.0))
        row[:roughness] = rough
        row[:score] = split_rel + pred + 0.25 * step + 0.02 * rough +
            min(10.0, 10.0 * row[:imag_rel])
    end
    sort!(rows; by=r -> (r[:score], r[:prediction_residual], r[:split_rel]))
    return rows
end

function render_coarse_phi(path, rows, Mtrue, K)
    best = rows[1]
    best_true = rows[argmin([r[:true_rel] for r in rows])]
    with_scaled_figure_style(3200, 2300) do _
        fig = Figure(; size=(3200, 2300))
        figure_title!(fig, "Coarse-snapshot Phi estimator";
            subtitle="100 snapshots per decorrelation time; true mobility is ex-post only")
        mats = [best[:Phi], best_true[:Phi], Mtrue, best[:Phi] .- Mtrue]
        titles = ["accepted by data-only score", "best ex-post candidate", "<M true> ex-post", "accepted - <M true>"]
        for i in 1:4
            ax = Axis(fig[1, i]; title=titles[i], xlabel="column", ylabel="row")
            mat = mats[i]
            if i == 4
                lim = max(maximum(abs, mat), 1.0e-8)
                hm = heatmap!(ax, mat; colormap=STYLE_DIVERGING_SOFT, colorrange=(-lim, lim))
            else
                hm = heatmap!(ax, mat; colormap=:viridis)
            end
            Colorbar(fig[1, i, Right()], hm)
        end
        topn = min(18, length(rows))
        rank = collect(1:topn)
        ax2 = Axis(fig[2, 1:3]; title="Candidate diagnostics", xlabel="data-only rank", ylabel="relative value")
        lines!(ax2, rank, [rows[i][:split_rel] for i in rank]; color=STYLE_PRIMARY,
            linewidth=curve_linewidth(), label="split-half")
        lines!(ax2, rank, [rows[i][:prediction_residual] for i in rank]; color=STYLE_SECONDARY,
            linewidth=curve_linewidth(), linestyle=:dash, label="held-out lag prediction")
        lines!(ax2, rank, [rows[i][:true_rel] for i in rank]; color=STYLE_REFERENCE,
            linewidth=curve_linewidth(), linestyle=:dot, label="ex-post true RMSE")
        axislegend(ax2; position=:rt)
        names = join(["$(i). $(rows[i][:name])" for i in rank], "\n")
        ax3 = Axis(fig[2, 4]; title="top candidates", xlabel="", ylabel="")
        hidedecorations!(ax3)
        hidespines!(ax3)
        text!(ax3, 0.0, 1.0; text=names, align=(:left, :top), fontsize=15)
        lines = [
            @sprintf("accepted estimator = %s", best[:name]),
            @sprintf("data-only score = %.8e", best[:score]),
            @sprintf("split-half rel.diff = %.8e", best[:split_rel]),
            @sprintf("held-out lag prediction residual = %.8e", best[:prediction_residual]),
            @sprintf("ex-post accepted rel.RMSE = %.8e", best[:true_rel]),
            @sprintf("ex-post accepted corr = %.8e", best[:true_corr]),
            @sprintf("best ex-post row = %s", best_true[:name]),
            @sprintf("best ex-post rel.RMSE = %.8e", best_true[:true_rel]),
            "The accepted row is chosen before reading true-mobility diagnostics.",
        ]
        text_panel!(fig[3, 1:4], lines; title="No-cheating audit")
        save_figure_checked(path, fig)
    end
    return nothing
end

function run_coarse_phi(config_path::AbstractString)
    params = load_coarse_phi_params(config_path)
    base = dirname(abspath(config_path))
    input_h5 = resolve_path(base, params.input_hdf5)
    score_bson = resolve_path(base, params.score_bson)
    out_h5 = resolve_path(base, params.output_hdf5)
    metrics_txt = resolve_path(base, params.metrics_txt)
    fig_png = resolve_path(base, params.figure_png)
    require_condition(isfile(input_h5), "Missing input_hdf5: $(input_h5)")
    require_condition(isfile(score_bson), "Missing score_bson: $(score_bson)")

    _, stats, _, phys, _, _, _ = load_checkpoint(score_bson, :cpu)
    times, states = load_states(input_h5)
    save_dt = times[2] - times[1]
    norm_states, start_idx = postburnin_norm_states(states, stats, params.burnin_fraction)
    maxlag = maximum(union(params.fit_lags, params.validation_lags))
    ntraj = size(norm_states, 4)
    odd_trajs = collect(1:2:ntraj)
    even_trajs = collect(2:2:ntraj)
    @printf("Coarse Phi input %s\n", input_h5)
    @printf("Using post-burnin index %d, save_dt %.8g, normalized shape %s\n",
        start_idx, save_dt, string(size(norm_states)))
    taus, C_all = projected_lag_covariances(norm_states, save_dt, maxlag, collect(1:ntraj), phys.N)
    _, C_odd = projected_lag_covariances(norm_states, save_dt, maxlag, odd_trajs, phys.N)
    _, C_even = projected_lag_covariances(norm_states, save_dt, maxlag, even_trajs, phys.N)
    S_all = stabilized(qv_symmetric(norm_states, save_dt, collect(1:ntraj)), phys.N)
    S_odd = stabilized(qv_symmetric(norm_states, save_dt, odd_trajs), phys.N)
    S_even = stabilized(qv_symmetric(norm_states, save_dt, even_trajs), phys.N)
    rows = build_coarse_candidates(C_all, C_odd, C_even, taus, params.fit_lags,
        params.validation_lags, params.polynomial_degrees, phys.N, S_all, S_odd, S_even)

    raw_samples = collect_postburnin_samples(states, start_idx, 0, MersenneTwister(params.seed + 1))
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
        println(io, "FHDChainCapillaryN32 coarse-snapshot Phi estimator sweep")
        println(io, "Input data are the original short trajectory only: save_dt=$(save_dt), postburnin states=$(size(norm_states)).")
        println(io, "Selection score uses only split-trajectory stability and observed lag-covariance prediction residuals.")
        println(io, "True mobility appears only in ex-post columns below.")
        println(io, "rank\tname\tdata_score\tsplit_rel\tprediction_residual\tone_step_residual\troughness\timag_rel\texpost_true_rel\texpost_true_corr\teig_min\teig_max")
        for (rank, row) in enumerate(rows)
            println(io, @sprintf("%d\t%s\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e",
                rank, row[:name], row[:score], row[:split_rel], row[:prediction_residual],
                row[:one_step_residual], row[:roughness], row[:imag_rel], row[:true_rel], row[:true_corr],
                row[:eig_min], row[:eig_max]))
        end
    end

    ensure_parent_dir(out_h5)
    h5open(out_h5, "w") do h5
        write(h5, "/Phi", rows[1][:Phi])
        write(h5, "/Mtrue_norm_expost", Mtrue)
        write(h5, "/lag_covariance/taus", taus)
        write(h5, "/lag_covariance/C_projected", C_all)
        write(h5, "/metadata/save_dt", save_dt)
        write(h5, "/metadata/postburnin_start_index", start_idx)
        write(h5, "/metadata/selection_rule", "split_rel + prediction_residual + 0.25*one_step_residual + 0.02*roughness + imag_penalty")
        for (rank, row) in enumerate(rows)
            g = create_group(h5, "/candidates/rank_$(rank)")
            write(g, "name", String(row[:name]))
            write(g, "Phi", row[:Phi])
            write(g, "data_score", Float64(row[:score]))
            write(g, "split_rel", Float64(row[:split_rel]))
            write(g, "prediction_residual", Float64(row[:prediction_residual]))
            write(g, "one_step_residual", Float64(row[:one_step_residual]))
            write(g, "roughness", Float64(row[:roughness]))
            write(g, "imag_rel", Float64(row[:imag_rel]))
            write(g, "expost_true_rel", Float64(row[:true_rel]))
            write(g, "expost_true_corr", Float64(row[:true_corr]))
        end
    end
    render_coarse_phi(fig_png, rows, Mtrue, phys.N)
    @printf("Accepted coarse estimator: %s score=%.6e split=%.6e pred=%.6e ex-post rel=%.6e corr=%.6f\n",
        rows[1][:name], rows[1][:score], rows[1][:split_rel], rows[1][:prediction_residual],
        rows[1][:true_rel], rows[1][:true_corr])
    @printf("Saved coarse Phi sweep to %s\n", out_h5)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi_coarse.toml")) : abspath(ARGS[1])
    run_coarse_phi(cfg)
end
