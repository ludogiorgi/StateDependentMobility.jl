#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct GaussianScoreForwardParams
    input_hdf5::String
    covariance_hdf5::String
    score_bson::String
    phi_hdf5::String
    reference_forward_hdf5::String
    burnin_fraction::Float64
    covariance_burnin_fraction::Float64
    ridge_relative::Float64
    forward_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_save_dt::Float64
    forward_ntraj::Int
    seed::Int
    metrics_txt::String
    figure_png::String
    forward_hdf5::String
end

function load_gaussian_forward_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    score = get(raw, "score", Dict{String, Any}())
    forward = raw["forward"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    return GaussianScoreForwardParams(
        input_hdf5=String(data["input_hdf5"]),
        covariance_hdf5=String(data["covariance_hdf5"]),
        score_bson=String(data["score_bson"]),
        phi_hdf5=String(data["phi_hdf5"]),
        reference_forward_hdf5=String(get(data, "reference_forward_hdf5", "")),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        covariance_burnin_fraction=Float64(get(data, "covariance_burnin_fraction", 0.0)),
        ridge_relative=Float64(get(score, "ridge_relative", 1.0e-8)),
        forward_dt=Float64(forward["dt"]),
        forward_total_time=Float64(forward["total_time"]),
        forward_burnin_time=Float64(forward["burnin_time"]),
        forward_save_dt=Float64(forward["save_dt"]),
        forward_ntraj=Int(forward["ntrajectories"]),
        seed=Int(get(run, "seed", 20260820)),
        metrics_txt=String(output["metrics_txt"]),
        figure_png=String(output["figure_png"]),
        forward_hdf5=String(output["forward_hdf5"]),
    )
end

function flat_norm_with_mean(norm_states::Array{Float32, 4}, start_idx::Int)
    nt, K, _, ntraj = size(norm_states)
    B = (nt - start_idx + 1) * ntraj
    Z = Matrix{Float64}(undef, 2K, B)
    col = 0
    @inbounds for tr in 1:ntraj, t in start_idx:nt
        col += 1
        for i in 1:K
            Z[i, col] = norm_states[t, i, 1, tr]
            Z[K + i, col] = norm_states[t, i, 2, tr]
        end
    end
    mu = vec(mean(Z; dims=2))
    Z .-= mu
    return Z, mu
end

function empirical_gaussian_score_matrix(norm_states::Array{Float32, 4},
        start_idx::Int, K::Int, ridge_relative::Float64)
    Z, mu = flat_norm_with_mean(norm_states, start_idx)
    Q = nonzero_mode_basis(K)
    Zred = transpose(Q) * Z
    Cred = (Zred * transpose(Zred)) ./ max(size(Zred, 2) - 1, 1)
    ridge = ridge_relative * mean(diag(Cred))
    E = eigen(Symmetric(Cred + ridge * I))
    invC = E.vectors * Diagonal(1.0 ./ E.values) * transpose(E.vectors)
    Cfull = Q * Cred * transpose(Q)
    G = -Q * invC * transpose(Q)
    eigs = eigen(Symmetric(Cred)).values
    return G, projection_matrix(K) * mu, Cfull, eigs, ridge
end

function sample_initial_norm_from_states(norm_states::Array{Float32, 4},
        start_idx::Int, ntraj::Int, K::Int, rng::AbstractRNG)
    nt, _, _, nobs = size(norm_states)
    Z = Matrix{Float64}(undef, 2K, ntraj)
    @inbounds for b in 1:ntraj
        t = rand(rng, start_idx:nt)
        tr = rand(rng, 1:nobs)
        for i in 1:K
            Z[i, b] = norm_states[t, i, 1, tr]
            Z[K + i, b] = norm_states[t, i, 2, tr]
        end
    end
    project_flat_zero_modes!(Z, K)
    return Z
end

function integrate_gaussian_phi(Phi::Matrix{Float64}, G::Matrix{Float64}, mu::Vector{Float64},
        norm_states::Array{Float32, 4}, start_idx::Int, stats::DataStats, phys::FHDPhys,
        params::GaussianScoreForwardParams)
    K = phys.N
    nsteps = ceil(Int, params.forward_total_time / params.forward_dt)
    burn_steps = floor(Int, params.forward_burnin_time / params.forward_dt)
    save_every = max(1, round(Int, params.forward_save_dt / params.forward_dt))
    actual_save_dt = save_every * params.forward_dt
    nsaved = fld(max(nsteps - burn_steps, 0), save_every) + 1
    rng = MersenneTwister(params.seed + 100)
    Z = sample_initial_norm_from_states(norm_states, start_idx, params.forward_ntraj, K, rng)
    sqrtPhi, min_eig = tangent_sqrt(Phi)
    noise_basis = Matrix{Float64}(undef, size(sqrtPhi, 2), params.forward_ntraj)
    noise = Matrix{Float64}(undef, 2K, params.forward_ntraj)
    saved = Array{Float32}(undef, nsaved, K, 2, params.forward_ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    stride = max(1, nsteps ÷ 20)
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            times[save_idx] = (step - burn_steps) * params.forward_dt
            saved[save_idx, :, :, :] .= denormalize_flat_states(Z, stats)
        end
        step == nsteps && break
        score = G * (Z .- mu)
        drift1 = Phi * score
        Zmid = Z .+ 0.5 * params.forward_dt .* drift1
        project_flat_zero_modes!(Zmid, K)
        score_mid = G * (Zmid .- mu)
        drift = Phi * score_mid
        randn!(rng, noise_basis)
        mul!(noise, sqrtPhi, noise_basis)
        @. Z = Z + params.forward_dt * drift + sqrt(2.0 * params.forward_dt) * noise
        project_flat_zero_modes!(Z, K)
        all(isfinite, Z) || error("Non-finite Gaussian-score state at step $(step).")
        if step > 0 && step % stride == 0
            @printf("Gaussian-score Phi forward %.1f%%\n", 100.0 * step / nsteps)
            flush(stdout)
        end
    end
    return times, saved, actual_save_dt, min_eig
end

function psd_sqrt(A::AbstractMatrix{<:Real}; tol::Float64=1.0e-12)
    E = eigen(Symmetric(sympart(A)))
    keep = findall(>(tol), E.values)
    return E.vectors[:, keep] * Diagonal(sqrt.(E.values[keep])), minimum(E.values)
end

function integrate_gaussian_phi_exact(Phi::Matrix{Float64}, G::Matrix{Float64},
        mu::Vector{Float64}, C::Matrix{Float64}, norm_states::Array{Float32, 4},
        start_idx::Int, stats::DataStats, phys::FHDPhys, params::GaussianScoreForwardParams)
    K = phys.N
    save_dt = params.forward_save_dt
    burn_steps = floor(Int, params.forward_burnin_time / save_dt)
    nsteps = ceil(Int, params.forward_total_time / save_dt)
    nsaved = max(nsteps - burn_steps, 0) + 1
    A = Phi * G
    E = exp(A * save_dt)
    noise_cov = sympart(C .- E * C * transpose(E))
    sqrtQ, min_noise_eig = psd_sqrt(noise_cov)
    _, min_phi_eig = tangent_sqrt(Phi)
    rng = MersenneTwister(params.seed + 200)
    Z = sample_initial_norm_from_states(norm_states, start_idx, params.forward_ntraj, K, rng)
    noise_basis = Matrix{Float64}(undef, size(sqrtQ, 2), params.forward_ntraj)
    noise = Matrix{Float64}(undef, 2K, params.forward_ntraj)
    saved = Array{Float32}(undef, nsaved, K, 2, params.forward_ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    stride = max(1, nsteps ÷ 20)
    @inbounds for step in 0:nsteps
        if step >= burn_steps
            save_idx += 1
            times[save_idx] = (step - burn_steps) * save_dt
            saved[save_idx, :, :, :] .= denormalize_flat_states(Z, stats)
        end
        step == nsteps && break
        randn!(rng, noise_basis)
        mul!(noise, sqrtQ, noise_basis)
        Z .= mu .+ E * (Z .- mu) .+ noise
        project_flat_zero_modes!(Z, K)
        all(isfinite, Z) || error("Non-finite exact Gaussian-score state at step $(step).")
        if step > 0 && step % stride == 0
            @printf("Exact Gaussian-score Phi forward %.1f%%\n", 100.0 * step / nsteps)
            flush(stdout)
        end
    end
    return times, saved, save_dt, min(min_phi_eig, min_noise_eig)
end

function render_gaussian_forward_figure(path, obs_states, true_states, gauss_states,
        obs_start, phys, save_dt, metrics)
    rng = MersenneTwister(20260821)
    with_scaled_figure_style(3600, 2700) do _
        fig = Figure(; size=(3600, 2700))
        figure_title!(fig, "Capillary FHD N32 improved Phi with data Gaussian score";
            subtitle=@sprintf("Gaussian-score covariance rel.RMSE %.3e", metrics[:gaussian_covariance_rel_rmse]))
        specs = [
            (:rho, "rho PDF", "rho", (s, st) -> draw_channel_values(s, st, 1, 250000, rng)),
            (:m, "m PDF", "m", (s, st) -> draw_channel_values(s, st, 2, 250000, rng)),
            (:u, "u PDF", "u", (s, st) -> draw_velocity_values(s, st, phys, 250000, rng)),
        ]
        for (idx, (_, ttl, xl, getter)) in enumerate(specs)
            obs = getter(obs_states, obs_start)
            tru = getter(true_states, 1)
            gau = getter(gauss_states, 1)
            lo = quantile(obs, 0.001)
            hi = quantile(obs, 0.999)
            pad = 0.08 * max(hi - lo, 1.0e-8)
            edges = collect(range(lo - pad, hi + pad; length=151))
            centers, po = hist_density(obs, edges)
            _, pt = hist_density(tru, edges)
            _, pg = hist_density(gau, edges)
            ax = Axis(fig[1, idx]; title=ttl, xlabel=xl, ylabel="density")
            lines!(ax, centers, po; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="obs")
            lines!(ax, centers, pt; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash, label="true score + <M>")
            lines!(ax, centers, pg; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot, label="data Gaussian score + data Phi")
            idx == 1 && axislegend(ax; position=:rt)
        end
        for (idx, (ch, ttl)) in enumerate([(1, "rho ACF"), (2, "m ACF")])
            lags, co = channel_acf(obs_states, obs_start, ch, round(Int, 20.0 / save_dt))
            _, ct = channel_acf(true_states, 1, ch, min(length(lags) - 1, size(true_states, 1) - 1))
            _, cg = channel_acf(gauss_states, 1, ch, min(length(lags) - 1, size(gauss_states, 1) - 1))
            n = min(length(co), length(ct), length(cg))
            ax = Axis(fig[2, idx]; title=ttl, xlabel="tau", ylabel="C/C(0)")
            lines!(ax, lags[1:n] .* save_dt, co[1:n]; color=STYLE_REFERENCE, linewidth=curve_linewidth())
            lines!(ax, lags[1:n] .* save_dt, ct[1:n]; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash)
            lines!(ax, lags[1:n] .* save_dt, cg[1:n]; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot)
        end
        for (idx, ch) in enumerate([1, 2])
            ko, po = spatial_power_spectrum(obs_states, obs_start, ch)
            kt, pt = spatial_power_spectrum(true_states, 1, ch)
            kg, pg = spatial_power_spectrum(gauss_states, 1, ch)
            ax = Axis(fig[2, idx + 2]; title=ch == 1 ? "rho spectrum" : "m spectrum", xlabel="mode", ylabel="power")
            lines!(ax, ko, po; color=STYLE_REFERENCE, linewidth=curve_linewidth())
            lines!(ax, kt, pt; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash)
            lines!(ax, kg, pg; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot)
        end
        cov_obs = sampled_covariance(obs_states, obs_start, 120000, rng)
        cov_true = sampled_covariance(true_states, 1, 120000, rng)
        cov_gauss = sampled_covariance(gauss_states, 1, 120000, rng)
        for (idx, mat, ttl) in [(1, cov_true .- cov_obs, "true Phi covariance error"),
                (2, cov_gauss .- cov_obs, "Gaussian-score covariance error"),
                (3, cov_gauss .- cov_true, "Gaussian-score - true covariance")]
            clim = max(maximum(abs, mat), 1.0e-8)
            ax = Axis(fig[3, idx]; title=ttl, xlabel="column", ylabel="row")
            hm = heatmap!(ax, mat; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
            Colorbar(fig[3, idx, Right()], hm)
        end
        lines = [@sprintf("%s = %.8e", String(k), Float64(metrics[k]))
            for k in sort([kk for kk in keys(metrics) if metrics[kk] isa Number]; by=String)]
        text_panel!(fig[3, 4], lines; title="Agreement metrics")
        save_figure_checked(path, fig)
    end
    return nothing
end

function run_gaussian_forward(config_path::AbstractString)
    params = load_gaussian_forward_params(config_path)
    base = dirname(abspath(config_path))
    input_h5 = resolve_path(base, params.input_hdf5)
    cov_h5 = resolve_path(base, params.covariance_hdf5)
    score_bson = resolve_path(base, params.score_bson)
    phi_h5 = resolve_path(base, params.phi_hdf5)
    reference_h5 = isempty(strip(params.reference_forward_hdf5)) ? "" :
        resolve_path(base, params.reference_forward_hdf5)
    metrics_txt = resolve_path(base, params.metrics_txt)
    figure_png = resolve_path(base, params.figure_png)
    forward_h5 = resolve_path(base, params.forward_hdf5)
    for p in (input_h5, cov_h5, score_bson, phi_h5)
        require_condition(isfile(p), "Missing required input: $(p)")
    end
    model, stats, _, phys, _, _, _ = load_checkpoint(score_bson, :cpu)
    model = nothing
    times, obs_states = load_states(input_h5)
    cov_times, cov_states = load_states(cov_h5)
    norm_obs = normalize_states(obs_states, stats)
    norm_cov = normalize_states(cov_states, stats)
    obs_start = burnin_start_index(length(times), params.burnin_fraction)
    cov_start = burnin_start_index(length(cov_times), params.covariance_burnin_fraction)
    Phi, _ = load_external_phi(phi_h5)
    P = projection_matrix(phys.N)
    Phi = P * Phi * P
    G, mu, Cscore, cov_eigs, ridge = empirical_gaussian_score_matrix(norm_cov, cov_start, phys.N,
        params.ridge_relative)
    @printf("Empirical Gaussian score: covariance samples=%d ridge=%.6e min/max eig=%.6e/%.6e\n",
        (size(norm_cov, 1) - cov_start + 1) * size(norm_cov, 4),
        ridge, minimum(cov_eigs), maximum(cov_eigs))
    gauss_times, gauss_states, gauss_save_dt, min_eig = integrate_gaussian_phi_exact(Phi, G, mu,
        Cscore, norm_obs, obs_start, stats, phys, params)
    true_states = if !isempty(reference_h5) && isfile(reference_h5)
        h5open(reference_h5, "r") do h5
            Float32.(read(h5["/true_phi/states"]))
        end
    else
        gauss_states
    end
    metrics = forward_metrics(obs_states, true_states, gauss_states, obs_start, phys,
        length(times) > 1 ? times[2] - times[1] : params.forward_save_dt)
    metrics[:gaussian_forward_min_sym_eig] = min_eig
    metrics[:gaussian_covariance_rel_rmse] = metrics[:learned_covariance_rel_rmse]
    metrics[:gaussian_rho_pdf_rel_l2] = metrics[:rho_learned_pdf_rel_l2]
    metrics[:gaussian_m_pdf_rel_l2] = metrics[:m_learned_pdf_rel_l2]
    metrics[:gaussian_u_pdf_rel_l2] = metrics[:u_learned_pdf_rel_l2]
    metrics[:gaussian_rho_acf_rel_l2] = metrics[:rho_learned_acf_rel_l2]
    metrics[:gaussian_m_acf_rel_l2] = metrics[:m_learned_acf_rel_l2]
    metrics[:score_covariance_ridge] = ridge
    metrics[:score_covariance_min_eig] = minimum(cov_eigs)
    metrics[:score_covariance_max_eig] = maximum(cov_eigs)
    ensure_parent_dir(metrics_txt)
    open(metrics_txt, "w") do io
        println(io, "FHDChainCapillaryN32 improved Phi + empirical Gaussian data score metrics")
        println(io, "Score matrix is estimated only from observed covariance in normalized constrained coordinates.")
        for key in sort(collect(keys(metrics)); by=String)
            val = metrics[key]
            val isa Number && println(io, @sprintf("%s = %.10e", String(key), Float64(val)))
        end
    end
    render_gaussian_forward_figure(figure_png, obs_states, true_states, gauss_states, obs_start,
        phys, length(times) > 1 ? times[2] - times[1] : params.forward_save_dt, metrics)
    ensure_parent_dir(forward_h5)
    h5open(forward_h5, "w") do h5
        write(h5, "/gaussian_phi/time", gauss_times)
        write(h5, "/gaussian_phi/states", gauss_states)
        write(h5, "/metadata/save_dt", gauss_save_dt)
        write(h5, "/score/G", G)
        write(h5, "/score/mu", mu)
        write(h5, "/Phi", Phi)
    end
    @printf("Saved Gaussian-score improved-Phi forward outputs to %s\n", forward_h5)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi_gaussian_score.toml")) : abspath(ARGS[1])
    run_gaussian_forward(cfg)
end
