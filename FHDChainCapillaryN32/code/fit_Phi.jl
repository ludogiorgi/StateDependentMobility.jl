#!/usr/bin/env julia

include(joinpath(@__DIR__, "score.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

start_xvfb!()
using GLMakie
include(STYLE_FILE)
GLMakie.activate!()

function save_figure_checked(path::AbstractString, fig)
    ensure_parent_dir(path)
    save(path, fig)
    @printf("Saved figure to %s\n", path)
    return nothing
end

Base.@kwdef struct FitPhiParams
    input_hdf5::String
    phi_hdf5::String
    external_phi_hdf5::String
    score_bson::String
    burnin_fraction::Float64
    phi_fit_max_lag::Int
    phi_fit_degree::Int
    phi_estimator::String
    use_stein_correction::Bool
    stein_correction_orientation::String
    project_block_circulant::Bool
    project_symmetric_psd::Bool
    reuse_artifact::Bool
    stein_samples::Int
    true_mobility_samples::Int
    score_calibration::String
    score_rho_scale::Float64
    score_m_scale::Float64
    forward_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_save_dt::Float64
    forward_ntraj::Int
    forward_score_clip::Float64
    seed::Int
    device::String
    expected_smi_index::Int
    visible_device_id::String
    required_gpu_name::String
    artifact_hdf5::String
    metrics_txt::String
    phi_png::String
    forward_png::String
    forward_hdf5::String
    figure_width::Int
    figure_height::Int
end

function load_fit_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    phi = raw["phi"]
    forward = raw["forward"]
    run = get(raw, "run", Dict{String, Any}())
    output = raw["output"]
    figure = get(raw, "figure", Dict{String, Any}())
    return FitPhiParams(
        input_hdf5=String(data["input_hdf5"]),
        phi_hdf5=String(get(data, "phi_hdf5", data["input_hdf5"])),
        external_phi_hdf5=String(get(data, "external_phi_hdf5", "")),
        score_bson=String(data["score_bson"]),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        phi_fit_max_lag=Int(get(phi, "fit_max_lag", 8)),
        phi_fit_degree=Int(get(phi, "fit_degree", 2)),
        phi_estimator=String(get(phi, "estimator", "logcov")),
        use_stein_correction=Bool(get(phi, "use_stein_correction", true)),
        stein_correction_orientation=String(get(phi, "stein_correction_orientation", "right")),
        project_block_circulant=Bool(get(phi, "project_block_circulant", true)),
        project_symmetric_psd=Bool(get(phi, "project_symmetric_psd", true)),
        reuse_artifact=Bool(get(phi, "reuse_artifact", false)),
        stein_samples=Int(get(phi, "stein_samples", 0)),
        true_mobility_samples=Int(get(phi, "true_mobility_samples", 5000)),
        score_calibration=String(get(phi, "score_calibration", "none")),
        score_rho_scale=Float64(get(phi, "score_rho_scale", 1.0)),
        score_m_scale=Float64(get(phi, "score_m_scale", 1.0)),
        forward_dt=Float64(get(forward, "dt", 0.005)),
        forward_total_time=Float64(get(forward, "total_time", 160.0)),
        forward_burnin_time=Float64(get(forward, "burnin_time", 40.0)),
        forward_save_dt=Float64(get(forward, "save_dt", 0.5)),
        forward_ntraj=Int(get(forward, "ntrajectories", 32)),
        forward_score_clip=Float64(get(forward, "score_clip", 120.0)),
        seed=Int(get(run, "seed", 20260630)),
        device=String(get(run, "device", "gpu")),
        expected_smi_index=Int(get(run, "expected_smi_index", 2)),
        visible_device_id=String(get(run, "visible_device_id", "0")),
        required_gpu_name=String(get(run, "required_gpu_name", "RTX 5070")),
        artifact_hdf5=String(output["artifact_hdf5"]),
        metrics_txt=String(output["metrics_txt"]),
        phi_png=String(output["phi_figure_png"]),
        forward_png=String(output["forward_figure_png"]),
        forward_hdf5=String(output["forward_hdf5"]),
        figure_width=Int(get(figure, "width", 3600)),
        figure_height=Int(get(figure, "height", 2700)),
    )
end

function apply_stein_correction(Phi_raw::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real},
        orientation::AbstractString)
    mode = lowercase(strip(orientation))
    if mode == "right"
        return Matrix(Phi_raw) * pinv(Matrix(V); rtol=1.0e-5)
    elseif mode == "left"
        # GFDT coordinate convention: Cdot_xx(t) = <x_t s(x_0)^T> Phi.
        # With V = -<s x^T>, the t=0 learned-score relation is
        # -Phi_raw ~= -V' Phi, so this correction acts from the left.
        return pinv(transpose(Matrix(V)); rtol=1.0e-5) * Matrix(Phi_raw)
    else
        error("Unsupported phi.stein_correction_orientation=$(orientation); use right or left.")
    end
end

function tensor_from_flat(Z::AbstractMatrix{<:Real}, K::Int)
    B = size(Z, 2)
    out = Array{Float32}(undef, K, 2, B)
    @views begin
        out[:, 1, :] .= reshape(Float32.(Z[1:K, :]), K, B)
        out[:, 2, :] .= reshape(Float32.(Z[(K + 1):(2K), :]), K, B)
    end
    return out
end

function flat_from_norm_states(norm_states::Array{Float32, 4}, start_idx::Int)
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
    Z .-= mean(Z; dims=2)
    return Z
end

function normalize_states(states::Array{Float32, 4}, stats::DataStats)
    nt, K, C, ntraj = size(states)
    out = Array{Float32}(undef, nt, K, C, ntraj)
    for t in 1:nt
        out[t, :, :, :] .= apply_stats(states[t, :, :, :], stats)
    end
    return out
end

function polynomial_derivative_at(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
        x0::Real, degree::Int)
    deg = min(degree, length(xs) - 1)
    X = Matrix{Float64}(undef, length(xs), deg + 1)
    @inbounds for i in eachindex(xs)
        dx = Float64(xs[i]) - Float64(x0)
        X[i, 1] = 1.0
        for p in 1:deg
            X[i, p + 1] = X[i, p] * dx
        end
    end
    coeff = X \ Float64.(ys)
    return deg >= 1 ? coeff[2] : 0.0
end

function lag_covariances(norm_states::Array{Float32, 4}, start_idx::Int, save_dt::Float64,
        max_lag::Int)
    nt, K, _, ntraj = size(norm_states)
    D = 2K
    L = min(max_lag, nt - start_idx)
    C = Array{Float64}(undef, L + 1, D, D)
    taus = collect(0:L) .* save_dt
    for lag in 0:L
        count = (nt - start_idx + 1 - lag) * ntraj
        X0 = Matrix{Float64}(undef, D, count)
        Xt = Matrix{Float64}(undef, D, count)
        col = 0
        @inbounds for tr in 1:ntraj, t in start_idx:(nt - lag)
            col += 1
            for i in 1:K
                X0[i, col] = norm_states[t, i, 1, tr]
                X0[K + i, col] = norm_states[t, i, 2, tr]
                Xt[i, col] = norm_states[t + lag, i, 1, tr]
                Xt[K + i, col] = norm_states[t + lag, i, 2, tr]
            end
        end
        X0 .-= mean(X0; dims=2)
        Xt .-= mean(Xt; dims=2)
        C[lag + 1, :, :] .= (Xt * transpose(X0)) ./ count
        @printf("lag covariance %d / %d tau %.4g count %d\n", lag, L, taus[lag + 1], count)
    end
    return taus, C
end

function phi_from_covariance_derivative(taus::Vector{Float64}, C::Array{Float64, 3}, degree::Int)
    D = size(C, 2)
    Cdot0 = zeros(Float64, D, D)
    @inbounds for i in 1:D, j in 1:D
        Cdot0[i, j] = polynomial_derivative_at(taus, C[:, i, j], 0.0, degree)
    end
    return Cdot0, -Cdot0
end

function real_matrix_log(A::AbstractMatrix{<:Real})
    L = log(complex.(Matrix(A)))
    imag_rel = norm(imag.(L)) / max(norm(real.(L)), eps(Float64))
    return real.(L), imag_rel
end

function phi_from_covariance_log(taus::Vector{Float64}, C::Array{Float64, 3}, K::Int)
    Q = nonzero_mode_basis(K)
    C0 = Q' * Matrix(C[1, :, :]) * Q
    C0inv = pinv(C0; rtol=1.0e-8)
    Cdot_red_acc = zeros(Float64, size(C0))
    imag_max = 0.0
    used = 0
    for ell in 2:length(taus)
        tau = taus[ell]
        tau <= 0.0 && continue
        Ct = Q' * Matrix(C[ell, :, :]) * Q
        propagator = Ct * C0inv
        Ared, imag_rel = real_matrix_log(propagator)
        imag_max = max(imag_max, imag_rel)
        Cdot_red_acc .+= (Ared / tau) * C0
        used += 1
    end
    require_condition(used > 0, "Need at least one positive lag for logcov Phi.")
    Cdot_red = Cdot_red_acc ./ used
    Cdot0 = Q * Cdot_red * Q'
    return Cdot0, -Cdot0, imag_max
end

function nonzero_mode_basis(K::Int)
    D = 2K
    A = zeros(Float64, D, D - 2)
    col = 1
    for i in 1:(K - 1)
        A[i, col] = 1.0
        A[K, col] = -1.0
        col += 1
    end
    for i in 1:(K - 1)
        A[K + i, col] = 1.0
        A[2K, col] = -1.0
        col += 1
    end
    return Matrix(qr(A).Q)[:, 1:(D - 2)]
end

sympart(A) = 0.5 .* (Matrix(A) .+ transpose(Matrix(A)))
skewpart(A) = 0.5 .* (Matrix(A) .- transpose(Matrix(A)))

function block_profile(A::AbstractMatrix{<:Real}, K::Int)
    prof = zeros(Float64, K, 2, 2)
    counts = zeros(Int, K, 2, 2)
    @inbounds for i in 1:K, r in 0:(K - 1), a in 1:2, b in 1:2
        j = periodic(i + r, K)
        row = a == 1 ? i : K + i
        col = b == 1 ? j : K + j
        prof[r + 1, a, b] += Float64(A[row, col])
        counts[r + 1, a, b] += 1
    end
    return prof ./ counts
end

function matrix_from_block_profile(prof::Array{Float64, 3})
    K = size(prof, 1)
    A = zeros(Float64, 2K, 2K)
    @inbounds for i in 1:K, r in 0:(K - 1), a in 1:2, b in 1:2
        j = periodic(i + r, K)
        row = a == 1 ? i : K + i
        col = b == 1 ? j : K + j
        A[row, col] = prof[r + 1, a, b]
    end
    return A
end

project_block_circulant_matrix(A::AbstractMatrix{<:Real}, K::Int) =
    matrix_from_block_profile(block_profile(A, K))

function project_tangent_symmetric_psd(A::AbstractMatrix{<:Real}, K::Int; floor::Float64=0.0)
    P = projection_matrix(K)
    Ap = P * Matrix(A) * P
    Q = nonzero_mode_basis(K)
    Sred = Q' * sympart(Ap) * Q
    E = eigen(Symmetric(Sred))
    Spsd = Q * E.vectors * Diagonal(max.(E.values, floor)) * transpose(E.vectors) * Q'
    return P * (Spsd + skewpart(Ap)) * P
end

function tangent_eigs(A::AbstractMatrix{<:Real}, K::Int)
    Q = nonzero_mode_basis(K)
    return eigen(Symmetric(Q' * sympart(A) * Q)).values
end

function agreement_metrics(reference, estimate)
    r = vec(Float64.(reference))
    e = vec(Float64.(estimate))
    rel = sqrt(sum(abs2, e .- r) / max(sum(abs2, r), eps(Float64)))
    corr = cor(r, e)
    return rel, corr
end

function model_stein_matrix(model, stats::DataStats, norm_samples::Array{Float32, 3},
        sigma::Float32, batch_size::Int, dev)
    scores = evaluate_score_norm(model, norm_samples, sigma, batch_size, dev)
    return stein_matrix(scores, norm_samples)
end

function edge_viscosity(rho::AbstractVector{<:Real}, phys::FHDPhys)
    eta = Vector{Float64}(undef, length(rho))
    @inbounds for i in eachindex(rho)
        ip = periodic(i + 1, length(rho))
        edge = max(0.5 * (Float64(rho[i]) + Float64(rho[ip])), 1.0e-14)
        eta[i] = phys.eta0 * (edge / phys.rho0)^phys.zeta
    end
    return eta
end

function true_D_action_capillary!(dest_m::Vector{Float64}, rho::AbstractVector{<:Real},
        v_m::AbstractVector{<:Real}, phys::FHDPhys)
    N = length(rho)
    eta = edge_viscosity(rho, phys)
    fill!(dest_m, 0.0)
    pref = phys.theta / phys.dx^3
    @inbounds for i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        dest_m[i] = pref * ((eta[i] + eta[im]) * Float64(v_m[i]) -
            eta[i] * Float64(v_m[ip]) - eta[im] * Float64(v_m[im]))
    end
    return nothing
end

function Lh_action_capillary!(out_rho::Vector{Float64}, out_m::Vector{Float64},
        rho::AbstractVector{<:Real}, m::AbstractVector{<:Real},
        v_rho::AbstractVector{<:Real}, v_m::AbstractVector{<:Real}, phys::FHDPhys)
    N = length(rho)
    h = phys.dx
    @inbounds for i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        beta_im = Float64(v_m[im])
        beta_i = Float64(v_m[i])
        beta_ip = Float64(v_m[ip])
        alpha_im = Float64(v_rho[im])
        alpha_ip = Float64(v_rho[ip])
        rho_i = Float64(rho[i])
        rho_im = Float64(rho[im])
        rho_ip = Float64(rho[ip])
        m_i = Float64(m[i])
        m_im = Float64(m[im])
        m_ip = Float64(m[ip])
        out_rho[i] = -((rho_i * beta_i + rho_ip * beta_ip) -
            (rho_im * beta_im + rho_i * beta_i)) / (2h)
        grad_alpha = (alpha_ip - alpha_im) / (2h)
        grad_beta = (beta_ip - beta_im) / (2h)
        flux_right = 0.5 * (m_i * beta_i + m_ip * beta_ip)
        flux_left = 0.5 * (m_im * beta_im + m_i * beta_i)
        out_m[i] = -rho_i * grad_alpha - m_i * grad_beta - (flux_right - flux_left) / h
    end
    return nothing
end

function true_mobility_transpose_action(z::AbstractMatrix{<:Real}, v::AbstractVector{<:Real},
        phys::FHDPhys)
    N = size(z, 1)
    rho = @view z[:, 1]
    m = @view z[:, 2]
    vr = @view v[1:N]
    vm = @view v[(N + 1):(2N)]
    d_m = zeros(Float64, N)
    l_r = zeros(Float64, N)
    l_m = zeros(Float64, N)
    true_D_action_capillary!(d_m, rho, vm, phys)
    Lh_action_capillary!(l_r, l_m, rho, m, vr, vm, phys)
    out = Vector{Float64}(undef, 2N)
    @inbounds for i in 1:N
        out[i] = phys.theta / phys.dx * l_r[i]
        out[N + i] = d_m[i] + phys.theta / phys.dx * l_m[i]
    end
    return out
end

function true_mobility_matrix(z::AbstractMatrix{<:Real}, phys::FHDPhys)
    D = 2size(z, 1)
    M = Matrix{Float64}(undef, D, D)
    for j in 1:D
        e = zeros(Float64, D)
        e[j] = 1.0
        M[j, :] .= true_mobility_transpose_action(z, e, phys)
    end
    return M
end

function estimate_mean_true_mobility(raw_samples::Array{Float32, 3}, phys::FHDPhys,
        nsamples::Int, rng::AbstractRNG)
    B = size(raw_samples, 3)
    n = nsamples <= 0 ? B : min(nsamples, B)
    idx = randperm(rng, B)[1:n]
    D = 2size(raw_samples, 1)
    M = zeros(Float64, D, D)
    for (count, b) in enumerate(idx)
        M .+= true_mobility_matrix(@view(raw_samples[:, :, b]), phys)
        if count % max(1, n ÷ 10) == 0
            @printf("true <M> samples %d / %d\n", count, n)
            flush(stdout)
        end
    end
    return M ./ n
end

function transform_mobility_to_norm(Mphys::AbstractMatrix{<:Real}, stats::DataStats)
    invstd = 1.0 ./ std_flat(stats)
    return Diagonal(invstd) * Matrix(Mphys) * Diagonal(invstd)
end

function tangent_sqrt(A::AbstractMatrix{<:Real}; tol::Float64=1.0e-10)
    E = eigen(Symmetric(sympart(A)))
    keep = findall(>(tol), E.values)
    return E.vectors[:, keep] * Diagonal(sqrt.(E.values[keep])), minimum(E.values)
end

function project_flat_zero_modes!(Z::Matrix{Float64}, K::Int)
    @inbounds for b in axes(Z, 2)
        mr = mean(@view Z[1:K, b])
        mm = mean(@view Z[(K + 1):(2K), b])
        Z[1:K, b] .-= mr
        Z[(K + 1):(2K), b] .-= mm
    end
    return Z
end

function score_batch_norm(mode::Symbol, Z::Matrix{Float64}, K::Int, stats::DataStats,
        phys::FHDPhys, model, sigma::Float32, batch_size::Int, dev)
    tensor = tensor_from_flat(Z, K)
    if mode == :analytic
        s = standardized_analytic_score(tensor, stats, phys)
    elseif mode == :unet
        s = evaluate_score_norm(model, tensor, sigma, batch_size, dev)
    else
        error("Unknown score mode $(mode)")
    end
    return raw_flat_batch64(s)
end

function score_calibration_matrix(mode::AbstractString, V::AbstractMatrix{<:Real}, K::Int)
    name = lowercase(strip(mode))
    P = projection_matrix(K)
    if name == "none"
        return nothing
    elseif name == "stein"
        return P * pinv(Matrix(V); rtol=1.0e-5) * P
    else
        error("Unsupported phi.score_calibration=$(mode); use none or stein.")
    end
end

function sample_initial_norm(norm_states::Array{Float32, 4}, start_idx::Int, ntraj::Int,
        rng::AbstractRNG)
    nt, K, _, nobs = size(norm_states)
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

function denormalize_flat_states(Z::Matrix{Float64}, stats::DataStats)
    tensor = tensor_from_flat(Z, size(stats.mean, 2))
    return denormalize_tensor(tensor, stats)
end

function integrate_constant_phi(mode::Symbol, Phi::Matrix{Float64},
        norm_states::Array{Float32, 4}, start_idx::Int, stats::DataStats, phys::FHDPhys,
        model, sigma::Float32, params::FitPhiParams, dev; score_transform=nothing)
    K = phys.N
    nsteps = ceil(Int, params.forward_total_time / params.forward_dt)
    burn_steps = floor(Int, params.forward_burnin_time / params.forward_dt)
    save_every = max(1, round(Int, params.forward_save_dt / params.forward_dt))
    actual_save_dt = save_every * params.forward_dt
    nsaved = fld(max(nsteps - burn_steps, 0), save_every) + 1
    rng = MersenneTwister(params.seed + (mode == :analytic ? 100 : 200))
    Z = sample_initial_norm(norm_states, start_idx, params.forward_ntraj, rng)
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
        score = score_batch_norm(mode == :analytic ? :analytic : :unet, Z, K, stats, phys,
            model, sigma, 4096, dev)
        if mode == :unet && score_transform !== nothing
            score = score_transform * score
        end
        if mode == :unet
            @views score[1:K, :] .*= params.score_rho_scale
            @views score[(K + 1):(2K), :] .*= params.score_m_scale
        end
        clamp!(score, -params.forward_score_clip, params.forward_score_clip)
        drift1 = Phi * score
        Zmid = Z .+ 0.5 * params.forward_dt .* drift1
        project_flat_zero_modes!(Zmid, K)
        score_mid = score_batch_norm(mode == :analytic ? :analytic : :unet, Zmid, K, stats, phys,
            model, sigma, 4096, dev)
        if mode == :unet && score_transform !== nothing
            score_mid = score_transform * score_mid
        end
        if mode == :unet
            @views score_mid[1:K, :] .*= params.score_rho_scale
            @views score_mid[(K + 1):(2K), :] .*= params.score_m_scale
        end
        clamp!(score_mid, -params.forward_score_clip, params.forward_score_clip)
        drift = Phi * score_mid
        randn!(rng, noise_basis)
        mul!(noise, sqrtPhi, noise_basis)
        @. Z = Z + params.forward_dt * drift + sqrt(2.0 * params.forward_dt) * noise
        project_flat_zero_modes!(Z, K)
        all(isfinite, Z) || error("Non-finite state in $(String(mode)) constant-Phi forward at step $(step).")
        if step > 0 && step % stride == 0
            @printf("%s Phi forward %.1f%%\n", String(mode), 100.0 * step / nsteps)
            flush(stdout)
        end
    end
    return times, saved, actual_save_dt, min_eig
end

function channel_acf(states::Array{Float32, 4}, start_idx::Int, channel::Int, max_lag::Int)
    nt, K, _, ntraj = size(states)
    L = min(max_lag, nt - start_idx)
    vals = Vector{Float64}(undef, L + 1)
    data = @view states[start_idx:end, :, channel, :]
    mu = mean(Float64, data)
    var = mean(x -> abs2(Float64(x) - mu), data)
    @inbounds for lag in 0:L
        s = 0.0
        count = 0
        for tr in 1:ntraj, t in start_idx:(nt - lag), i in 1:K
            s += (Float64(states[t + lag, i, channel, tr]) - mu) *
                (Float64(states[t, i, channel, tr]) - mu)
            count += 1
        end
        vals[lag + 1] = s / max(count * var, eps(Float64))
    end
    return collect(0:L), vals
end

function pdf_rel(obs_vals, model_vals)
    lo = quantile(obs_vals, 0.001)
    hi = quantile(obs_vals, 0.999)
    pad = 0.08 * max(hi - lo, 1.0e-8)
    edges = collect(range(lo - pad, hi + pad; length=151))
    centers, po = hist_density(obs_vals, edges)
    _, pm = hist_density(model_vals, edges)
    return sqrt(sum(abs2, pm .- po) / max(sum(abs2, po), eps(Float64))), centers, po, pm
end

function forward_metrics(obs_states, true_states, learned_states, obs_start, phys, save_dt)
    rng = MersenneTwister(20260701)
    metrics = Dict{Symbol, Float64}()
    for (name, getter) in [
            (:rho, (s, st) -> draw_channel_values(s, st, 1, 250000, rng)),
            (:m, (s, st) -> draw_channel_values(s, st, 2, 250000, rng)),
            (:u, (s, st) -> draw_velocity_values(s, st, phys, 250000, rng))]
        obs = getter(obs_states, obs_start)
        tru = getter(true_states, 1)
        lea = getter(learned_states, 1)
        metrics[Symbol(name, "_true_pdf_rel_l2")] = pdf_rel(obs, tru)[1]
        metrics[Symbol(name, "_learned_pdf_rel_l2")] = pdf_rel(obs, lea)[1]
        metrics[Symbol(name, "_learned_vs_true_pdf_rel_l2")] = pdf_rel(tru, lea)[1]
    end
    cov_obs = sampled_covariance(obs_states, obs_start, 120000, rng)
    cov_true = sampled_covariance(true_states, 1, 120000, rng)
    cov_learned = sampled_covariance(learned_states, 1, 120000, rng)
    metrics[:true_covariance_rel_rmse] = agreement_metrics(cov_obs, cov_true)[1]
    metrics[:learned_covariance_rel_rmse] = agreement_metrics(cov_obs, cov_learned)[1]
    metrics[:learned_vs_true_covariance_rel_rmse] = agreement_metrics(cov_true, cov_learned)[1]
    for ch in 1:2
        lags, co = channel_acf(obs_states, obs_start, ch, round(Int, 20.0 / save_dt))
        _, ct = channel_acf(true_states, 1, ch, min(length(lags) - 1, size(true_states, 1) - 1))
        _, cl = channel_acf(learned_states, 1, ch, min(length(lags) - 1, size(learned_states, 1) - 1))
        n = min(length(co), length(ct), length(cl))
        cname = ch == 1 ? "rho" : "m"
        metrics[Symbol(cname, "_true_acf_rel_l2")] = agreement_metrics(co[1:n], ct[1:n])[1]
        metrics[Symbol(cname, "_learned_acf_rel_l2")] = agreement_metrics(co[1:n], cl[1:n])[1]
        metrics[Symbol(cname, "_learned_vs_true_acf_rel_l2")] = agreement_metrics(ct[1:n], cl[1:n])[1]
    end
    return metrics
end

function render_phi_figure(path, Phi_raw, V, Phi, Mtrue, K, metrics, eigs)
    with_scaled_figure_style(3200, 2200) do _
        fig = Figure(; size=(3200, 2200))
        figure_title!(fig, "Capillary FHD N32 data-driven Phi";
            subtitle=@sprintf("ex-post <M_true> rel.RMSE %.3e corr %.4f", metrics[:phi_true_rel], metrics[:phi_true_corr]))
        mats = [Mtrue, Phi_raw, V, Phi, Phi .- Mtrue]
        external = haskey(metrics, :external_phi_hdf5) && !isempty(String(metrics[:external_phi_hdf5]))
        titles = external ?
            ["<M true> diagnostic", "external data-only Phi", "score Stein/identity diagnostic", "forward data Phi", "data Phi - <M true>"] :
            ["<M true> diagnostic", "raw -Cdot(0+)", "learned Stein V", "data Phi", "data Phi - <M true>"]
        for idx in 1:5
            ax = Axis(fig[1 + (idx - 1) ÷ 3, 1 + (idx - 1) % 3]; title=titles[idx], xlabel="column", ylabel="row")
            mat = mats[idx]
            clim = idx == 5 ? max(maximum(abs, mat), 1.0e-8) : nothing
            hm = clim === nothing ? heatmap!(ax, mat; colormap=:viridis) :
                heatmap!(ax, mat; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
            Colorbar(fig[1 + (idx - 1) ÷ 3, 1 + (idx - 1) % 3, Right()], hm)
        end
        ax = Axis(fig[2, 3]; title="sym(Phi) tangent eigenvalues", xlabel="index", ylabel="eigenvalue")
        lines!(ax, 1:length(eigs), eigs; color=STYLE_PRIMARY, linewidth=curve_linewidth())
        text_panel!(fig[3, 1:3], [
            @sprintf("Phi vs <M_true> rel.RMSE = %.8e", metrics[:phi_true_rel]),
            @sprintf("Phi vs <M_true> corr = %.8e", metrics[:phi_true_corr]),
            @sprintf("raw Phi vs <M_true> rel.RMSE = %.8e", metrics[:phi_raw_true_rel]),
            @sprintf("Stein correction used = %s", string(metrics[:stein_correction])),
            @sprintf("min/max tangent eig sym(Phi) = %.8e / %.8e", minimum(eigs), maximum(eigs)),
            "The true mobility panel is an ex-post diagnostic only.",
        ]; title="No-cheating audit")
        save_figure_checked(path, fig)
    end
    return nothing
end

function render_forward_figure(path, obs_states, true_states, learned_states, obs_start,
        phys, save_dt, metrics)
    rng = MersenneTwister(20260702)
    with_scaled_figure_style(3600, 2700) do _
        fig = Figure(; size=(3600, 2700))
        figure_title!(fig, "Capillary FHD N32 Phi forward validation";
            subtitle=@sprintf("learned-vs-true covariance rel.RMSE %.3e", metrics[:learned_vs_true_covariance_rel_rmse]))
        specs = [
            (:rho, "rho PDF", "rho", (s, st) -> draw_channel_values(s, st, 1, 250000, rng)),
            (:m, "m PDF", "m", (s, st) -> draw_channel_values(s, st, 2, 250000, rng)),
            (:u, "u PDF", "u", (s, st) -> draw_velocity_values(s, st, phys, 250000, rng)),
        ]
        for (idx, (_, ttl, xl, getter)) in enumerate(specs)
            obs = getter(obs_states, obs_start)
            tru = getter(true_states, 1)
            lea = getter(learned_states, 1)
            lo = quantile(obs, 0.001)
            hi = quantile(obs, 0.999)
            pad = 0.08 * max(hi - lo, 1.0e-8)
            edges = collect(range(lo - pad, hi + pad; length=151))
            centers, po = hist_density(obs, edges)
            _, pt = hist_density(tru, edges)
            _, pl = hist_density(lea, edges)
            ax = Axis(fig[1, idx]; title=ttl, xlabel=xl, ylabel="density")
            lines!(ax, centers, po; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="obs")
            lines!(ax, centers, pt; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash, label="true score + <M>")
            lines!(ax, centers, pl; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot, label="U-Net score + data Phi")
            idx == 1 && axislegend(ax; position=:rt)
        end
        for (idx, (ch, ttl)) in enumerate([(1, "rho ACF"), (2, "m ACF")])
            lags, co = channel_acf(obs_states, obs_start, ch, round(Int, 20.0 / save_dt))
            _, ct = channel_acf(true_states, 1, ch, min(length(lags) - 1, size(true_states, 1) - 1))
            _, cl = channel_acf(learned_states, 1, ch, min(length(lags) - 1, size(learned_states, 1) - 1))
            n = min(length(co), length(ct), length(cl))
            ax = Axis(fig[2, idx]; title=ttl, xlabel="tau", ylabel="C/C(0)")
            lines!(ax, lags[1:n] .* save_dt, co[1:n]; color=STYLE_REFERENCE, linewidth=curve_linewidth())
            lines!(ax, lags[1:n] .* save_dt, ct[1:n]; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash)
            lines!(ax, lags[1:n] .* save_dt, cl[1:n]; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot)
        end
        for (idx, ch) in enumerate([1, 2])
            ko, po = spatial_power_spectrum(obs_states, obs_start, ch)
            kt, pt = spatial_power_spectrum(true_states, 1, ch)
            kl, pl = spatial_power_spectrum(learned_states, 1, ch)
            ax = Axis(fig[2, idx + 2]; title=ch == 1 ? "rho spectrum" : "m spectrum", xlabel="mode", ylabel="power")
            lines!(ax, ko, po; color=STYLE_REFERENCE, linewidth=curve_linewidth())
            lines!(ax, kt, pt; color=STYLE_SECONDARY, linewidth=curve_linewidth(), linestyle=:dash)
            lines!(ax, kl, pl; color=STYLE_PRIMARY, linewidth=curve_linewidth(), linestyle=:dot)
        end
        cov_obs = sampled_covariance(obs_states, obs_start, 120000, rng)
        cov_true = sampled_covariance(true_states, 1, 120000, rng)
        cov_learned = sampled_covariance(learned_states, 1, 120000, rng)
        for (idx, mat, ttl) in [(1, cov_true .- cov_obs, "true Phi covariance error"),
                (2, cov_learned .- cov_obs, "learned Phi covariance error"),
                (3, cov_learned .- cov_true, "learned - true covariance")]
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

function write_metrics(path, metrics::Dict)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "FHDChainCapillaryN32 Step 2 Phi metrics")
        println(io, "No analytic score or true mobility was used to train the score model or construct data-driven Phi.")
        for key in sort(collect(keys(metrics)); by=String)
            val = metrics[key]
            if val isa Number
                println(io, @sprintf("%s = %.10e", String(key), Float64(val)))
            else
                println(io, "$(String(key)) = $(val)")
            end
        end
    end
    @printf("Saved metrics to %s\n", path)
    return nothing
end

function load_external_phi(path::AbstractString)
    Phi = nothing
    Mtrue = nothing
    h5open(path, "r") do h5
        require_condition(haskey(h5, "Phi"), "external_phi_hdf5 must contain /Phi.")
        Phi = Matrix{Float64}(read(h5["/Phi"]))
        if haskey(h5, "Mtrue_norm_expost")
            Mtrue = Matrix{Float64}(read(h5["/Mtrue_norm_expost"]))
        elseif haskey(h5, "Mtrue_norm")
            Mtrue = Matrix{Float64}(read(h5["/Mtrue_norm"]))
        end
    end
    return Phi, Mtrue
end

function run_pipeline(param_file::AbstractString)
    params = load_fit_params(param_file)
    base = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base, params.input_hdf5)
    phi_hdf5 = resolve_path(base, params.phi_hdf5)
    external_phi_h5 = isempty(strip(params.external_phi_hdf5)) ? "" :
        resolve_path(base, params.external_phi_hdf5)
    score_bson = resolve_path(base, params.score_bson)
    artifact_h5 = resolve_path(base, params.artifact_hdf5)
    metrics_txt = resolve_path(base, params.metrics_txt)
    phi_png = resolve_path(base, params.phi_png)
    forward_png = resolve_path(base, params.forward_png)
    forward_h5 = resolve_path(base, params.forward_hdf5)
    for path in (input_hdf5, phi_hdf5, score_bson)
        require_condition(isfile(path), "Required input missing: $(path)")
    end
    !isempty(external_phi_h5) && require_condition(isfile(external_phi_h5),
        "Required external_phi_hdf5 missing: $(external_phi_h5)")
    dev = activate_device!(params.device, params.expected_smi_index, params.visible_device_id,
        params.required_gpu_name)
    model, stats, score_params, phys, _, _, _ = load_checkpoint(score_bson, dev)
    sigma = score_params.sigma
    times, states = load_states(input_hdf5)
    phi_times, phi_states = load_states(phi_hdf5)
    save_dt = length(times) > 1 ? times[2] - times[1] : params.forward_save_dt
    phi_save_dt = length(phi_times) > 1 ? phi_times[2] - phi_times[1] : save_dt
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    phi_start_idx = burnin_start_index(length(phi_times), 0.0)
    norm_states = normalize_states(states, stats)
    norm_phi_states = normalize_states(phi_states, stats)
    raw_samples = collect_postburnin_samples(phi_states, phi_start_idx, 0, MersenneTwister(params.seed + 1))
    norm_samples = apply_stats(raw_samples, stats)
    @printf("Phi fit dataset: obs=%s phi=%s norm_samples=%s obs_save_dt=%.5g phi_save_dt=%.5g\n",
        string(size(states)), string(size(phi_states)), string(size(norm_samples)), save_dt, phi_save_dt)

    Phi_raw = nothing
    V = nothing
    Phi = nothing
    Mtrue = nothing
    Cdot0 = nothing
    C = nothing
    taus = nothing
    log_imag_rel = 0.0
    if !isempty(external_phi_h5)
        @printf("Using external data-only Phi from %s\n", external_phi_h5)
        Phi_loaded, Mtrue_loaded = load_external_phi(external_phi_h5)
        P = projection_matrix(phys.N)
        Phi_raw = P * Phi_loaded * P
        Cdot0 = -Phi_raw
        D = size(Phi_raw, 1)
        taus = [0.0]
        C = zeros(Float64, 1, D, D)
        if params.use_stein_correction || params.stein_samples > 0 ||
                lowercase(strip(params.score_calibration)) != "none"
            nstein = params.stein_samples <= 0 ? size(norm_samples, 3) : min(params.stein_samples, size(norm_samples, 3))
            V = model_stein_matrix(model, stats, norm_samples[:, :, 1:nstein], sigma, 4096, dev)
        else
            V = Matrix{Float64}(I, D, D)
        end
        Phi_full = params.use_stein_correction ?
            apply_stein_correction(Phi_raw, V, params.stein_correction_orientation) : Phi_raw
        Phi_profile = params.project_block_circulant ? project_block_circulant_matrix(Phi_full, phys.N) : Phi_full
        Phi = params.project_symmetric_psd ? project_tangent_symmetric_psd(Phi_profile, phys.N) : P * Phi_profile * P
        Cdot0 = -Phi_raw
        if Mtrue_loaded === nothing
            Mtrue_phys = estimate_mean_true_mobility(raw_samples, phys, params.true_mobility_samples,
                MersenneTwister(params.seed + 2))
            Mtrue = transform_mobility_to_norm(Mtrue_phys, stats)
            Mtrue = project_block_circulant_matrix(Mtrue, phys.N)
            Mtrue = P * Mtrue * P
        else
            Mtrue = P * Mtrue_loaded * P
        end
    elseif params.reuse_artifact && isfile(artifact_h5)
        @printf("Reusing Phi artifact from %s\n", artifact_h5)
        h5open(artifact_h5, "r") do h5
            Phi_raw = read(h5["/Phi_raw"])
            Mtrue = read(h5["/Mtrue_norm"])
            V = read(h5["/Stein_V"])
        end
        if params.use_stein_correction && lowercase(strip(params.stein_correction_orientation)) == "right"
            h5open(artifact_h5, "r") do h5
                Phi = read(h5["/Phi"])
            end
        elseif params.use_stein_correction
            Phi_full = apply_stein_correction(Phi_raw, V, params.stein_correction_orientation)
            Phi_profile = params.project_block_circulant ? project_block_circulant_matrix(Phi_full, phys.N) : Phi_full
            Phi = params.project_symmetric_psd ? project_tangent_symmetric_psd(Phi_profile, phys.N) : projection_matrix(phys.N) * Phi_profile * projection_matrix(phys.N)
        else
            Phi_profile = params.project_block_circulant ? project_block_circulant_matrix(Phi_raw, phys.N) : Phi_raw
            Phi = params.project_symmetric_psd ? project_tangent_symmetric_psd(Phi_profile, phys.N) : projection_matrix(phys.N) * Phi_profile * projection_matrix(phys.N)
        end
    else
        taus, C = lag_covariances(norm_phi_states, phi_start_idx, phi_save_dt, params.phi_fit_max_lag)
        Cdot0, Phi_raw, log_imag_rel = if lowercase(params.phi_estimator) == "logcov"
            phi_from_covariance_log(taus, C, phys.N)
        elseif lowercase(params.phi_estimator) == "polynomial"
            cdot, phi_raw = phi_from_covariance_derivative(taus, C, params.phi_fit_degree)
            cdot, phi_raw, 0.0
        else
            error("Unknown phi.estimator=$(params.phi_estimator)")
        end
        nstein = params.stein_samples <= 0 ? size(norm_samples, 3) : min(params.stein_samples, size(norm_samples, 3))
        V = model_stein_matrix(model, stats, norm_samples[:, :, 1:nstein], sigma, 4096, dev)
        Phi_full = params.use_stein_correction ?
            apply_stein_correction(Phi_raw, V, params.stein_correction_orientation) : Phi_raw
        Phi_profile = params.project_block_circulant ? project_block_circulant_matrix(Phi_full, phys.N) : Phi_full
        Phi = params.project_symmetric_psd ? project_tangent_symmetric_psd(Phi_profile, phys.N) : projection_matrix(phys.N) * Phi_profile * projection_matrix(phys.N)

        Mtrue_phys = estimate_mean_true_mobility(raw_samples, phys, params.true_mobility_samples,
            MersenneTwister(params.seed + 2))
        Mtrue = transform_mobility_to_norm(Mtrue_phys, stats)
        Mtrue = project_block_circulant_matrix(Mtrue, phys.N)
        Mtrue = projection_matrix(phys.N) * Mtrue * projection_matrix(phys.N)
    end
    phi_true_rel, phi_true_corr = agreement_metrics(Mtrue, Phi)
    phi_raw_true_rel, phi_raw_true_corr = agreement_metrics(Mtrue, Phi_raw)
    eigs = tangent_eigs(Phi, phys.N)
    @printf("Data Phi vs <M_true> ex-post: rel=%.6e corr=%.6f eig_min=%.6e\n",
        phi_true_rel, phi_true_corr, minimum(eigs))

    metrics = Dict{Symbol, Any}(
        :phi_true_rel => phi_true_rel,
        :phi_true_corr => phi_true_corr,
        :phi_raw_true_rel => phi_raw_true_rel,
        :phi_raw_true_corr => phi_raw_true_corr,
        :stein_correction => params.use_stein_correction,
        :stein_correction_orientation => params.stein_correction_orientation,
        :score_calibration => params.score_calibration,
        :score_rho_scale => params.score_rho_scale,
        :score_m_scale => params.score_m_scale,
        :phi_estimator => params.phi_estimator,
        :external_phi_hdf5 => external_phi_h5,
        :logcov_imaginary_relative_norm => log_imag_rel,
        :phi_min_tangent_eig => minimum(eigs),
        :phi_max_tangent_eig => maximum(eigs),
    )
    render_phi_figure(phi_png, Phi_raw, V, Phi, Mtrue, phys.N, metrics, eigs)

    true_times, true_states, true_save_dt, min_true_eig = integrate_constant_phi(:analytic,
        Mtrue, norm_states, start_idx, stats, phys, model, sigma, params, dev)
    score_transform = score_calibration_matrix(params.score_calibration, V, phys.N)
    learned_times, learned_states, learned_save_dt, min_learned_eig = integrate_constant_phi(:unet,
        Phi, norm_states, start_idx, stats, phys, model, sigma, params, dev;
        score_transform=score_transform)
    metrics[:true_forward_min_sym_eig] = min_true_eig
    metrics[:learned_forward_min_sym_eig] = min_learned_eig
    fmetrics = forward_metrics(states, true_states, learned_states, start_idx, phys, save_dt)
    merge!(metrics, fmetrics)
    write_metrics(metrics_txt, metrics)
    render_forward_figure(forward_png, states, true_states, learned_states, start_idx,
        phys, save_dt, metrics)

    ensure_parent_dir(forward_h5)
    h5open(forward_h5, "w") do h5
        write(h5, "/true_phi/time", true_times)
        write(h5, "/true_phi/states", true_states)
        write(h5, "/learned_phi/time", learned_times)
        write(h5, "/learned_phi/states", learned_states)
        write(h5, "/metadata/save_dt", true_save_dt)
        write(h5, "/metadata/learned_save_dt", learned_save_dt)
    end
    if !(params.reuse_artifact && isfile(artifact_h5))
        ensure_parent_dir(artifact_h5)
        h5open(artifact_h5, "w") do h5
            write(h5, "/Phi_raw", Phi_raw)
            write(h5, "/Phi", Phi)
            write(h5, "/Mtrue_norm", Mtrue)
            write(h5, "/Stein_V", V)
            write(h5, "/Cdot0_raw", Cdot0)
            write(h5, "/lag_covariance/taus", taus)
            write(h5, "/lag_covariance/C", C)
            for (k, v) in metrics
                v isa Number && write(h5, "/metrics/$(String(k))", Float64(v))
            end
        end
    end
    @printf("Saved Phi artifact to %s and forward trajectories to %s\n", artifact_h5, forward_h5)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi.toml")) : abspath(ARGS[1])
    run_pipeline(param_file)
end
