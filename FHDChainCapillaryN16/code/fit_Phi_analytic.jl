#!/usr/bin/env julia

include(joinpath(@__DIR__, "sim.jl"))

using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using GLMakie

const DEFAULT_ANALYTIC_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "sim.toml"))

function raw_flat(z::AbstractArray{<:Real, 3})
    K = size(z, 1)
    B = size(z, 3)
    out = Matrix{Float64}(undef, 2K, B)
    @inbounds for b in 1:B, i in 1:K
        out[i, b] = z[i, 1, b]
        out[K + i, b] = z[i, 2, b]
    end
    return out
end

function project_zero_modes!(z::Array{Float64, 3}, params::SimParams)
    @inbounds for b in axes(z, 3)
        rho_shift = mean(@view z[:, 1, b]) - params.rho0
        m_shift = mean(@view z[:, 2, b])
        z[:, 1, b] .-= rho_shift
        z[:, 2, b] .-= m_shift
    end
    return z
end

function analytic_score_flat(z::Array{Float64, 3}, params::SimParams)
    K, _, B = size(z)
    out = Matrix{Float64}(undef, 2K, B)
    laprho = Vector{Float64}(undef, K)
    score = Matrix{Float64}(undef, K, 2)
    @inbounds for b in 1:B
        zb = @view z[:, :, b]
        density_laplacian!(laprho, zb, params)
        analytic_score!(score, laprho, zb, params)
        for i in 1:K
            out[i, b] = score[i, 1]
            out[K + i, b] = score[i, 2]
        end
    end
    return out
end

function true_D_action_capillary!(dest_m::Vector{Float64}, rho::AbstractVector{<:Real},
                                  v_m::AbstractVector{<:Real}, params::SimParams)
    N = length(rho)
    h = dx(params)
    eta = Vector{Float64}(undef, N)
    edge_viscosity!(eta, hcat(Float64.(rho), zeros(Float64, N)), params)
    fill!(dest_m, 0.0)
    @inbounds for i in 1:N
        im = periodic_index(i - 1, N)
        ip = periodic_index(i + 1, N)
        dest_m[i] = params.theta / h^3 *
            ((eta[i] + eta[im]) * Float64(v_m[i]) -
             eta[i] * Float64(v_m[ip]) -
             eta[im] * Float64(v_m[im]))
    end
    return nothing
end

function Lh_action_capillary!(out_rho::Vector{Float64}, out_m::Vector{Float64},
                              rho::AbstractVector{<:Real}, m::AbstractVector{<:Real},
                              v_rho::AbstractVector{<:Real}, v_m::AbstractVector{<:Real},
                              params::SimParams)
    N = length(rho)
    h = dx(params)
    @inbounds for i in 1:N
        im = periodic_index(i - 1, N)
        ip = periodic_index(i + 1, N)
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
                                        params::SimParams)
    N = size(z, 1)
    rho = @view z[:, 1]
    m = @view z[:, 2]
    v_rho = @view v[1:N]
    v_m = @view v[(N + 1):(2N)]
    d_m = zeros(Float64, N)
    l_r = zeros(Float64, N)
    l_m = zeros(Float64, N)
    true_D_action_capillary!(d_m, rho, v_m, params)
    Lh_action_capillary!(l_r, l_m, rho, m, v_rho, v_m, params)
    out = Vector{Float64}(undef, 2N)
    h = dx(params)
    @inbounds for i in 1:N
        out[i] = params.theta / h * l_r[i]
        out[N + i] = d_m[i] + params.theta / h * l_m[i]
    end
    return out
end

function true_mobility_matrix(z::AbstractMatrix{<:Real}, params::SimParams)
    D = 2size(z, 1)
    M = Matrix{Float64}(undef, D, D)
    for j in 1:D
        e = zeros(Float64, D)
        e[j] = 1.0
        M[j, :] .= true_mobility_transpose_action(z, e, params)
    end
    return M
end

function estimate_mean_true_mobility(states::Array{Float32, 4}, start_idx::Int,
                                     params::SimParams; nsamples::Int=80_000, seed::Int=1)
    rng = MersenneTwister(seed)
    nt, K, _, ntraj = size(states)
    D = 2K
    M = zeros(Float64, D, D)
    npost = nt - start_idx + 1
    for _ in 1:nsamples
        t = start_idx + rand(rng, 0:(npost - 1))
        tr = rand(rng, 1:ntraj)
        M .+= true_mobility_matrix(@view(states[t, :, :, tr]), params)
    end
    return M ./ nsamples
end

function tangent_sqrt(A::AbstractMatrix{<:Real}; tol::Float64=1.0e-10)
    S = Symmetric(0.5 .* (Matrix(A) .+ transpose(Matrix(A))))
    eig = eigen(S)
    vals = eig.values
    keep = findall(>(tol), vals)
    sqrtA = eig.vectors[:, keep] * Diagonal(sqrt.(vals[keep]))
    return sqrtA, minimum(vals)
end

function sample_initial_states(states::Array{Float32, 4}, start_idx::Int, ntraj::Int,
                               params::SimParams, rng::AbstractRNG)
    nt, K, _, nobs = size(states)
    z = Array{Float64}(undef, K, 2, ntraj)
    npost = nt - start_idx + 1
    @inbounds for b in 1:ntraj
        t = start_idx + rand(rng, 0:(npost - 1))
        tr = rand(rng, 1:nobs)
        z[:, :, b] .= Float64.(@view states[t, :, :, tr])
    end
    project_zero_modes!(z, params)
    return z
end

function integrate_phi_langevin(Phi::Matrix{Float64}, states::Array{Float32, 4},
                                start_idx::Int, params::SimParams;
                                dt::Float64, total_time::Float64, burnin_time::Float64,
                                save_dt::Float64, ntraj::Int, seed::Int)
    rng = MersenneTwister(seed)
    K = params.N
    D = 2K
    nsteps = round(Int, total_time / dt)
    burn_steps = round(Int, burnin_time / dt)
    save_every = max(1, round(Int, save_dt / dt))
    nsaved = fld(nsteps - burn_steps, save_every) + 1
    z = sample_initial_states(states, start_idx, ntraj, params, rng)
    sqrtPhi, min_eig = tangent_sqrt(Phi)
    noise_basis = Matrix{Float64}(undef, size(sqrtPhi, 2), ntraj)
    noise = Matrix{Float64}(undef, D, ntraj)
    flat = raw_flat(z)
    saved = Array{Float32}(undef, nsaved, K, 2, ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    heartbeat_stride = max(1, nsteps ÷ 20)
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            times[save_idx] = (step - burn_steps) * dt
            saved[save_idx, :, :, :] .= Float32.(z)
        end
        step == nsteps && break
        if step > 0 && step % heartbeat_stride == 0
            @printf("Phi forward progress %.1f%% at step %d / %d\n",
                100.0 * step / nsteps, step, nsteps)
            flush(stdout)
        end
        score = analytic_score_flat(z, params)
        drift = Phi * score
        randn!(rng, noise_basis)
        mul!(noise, sqrtPhi, noise_basis)
        @. flat = flat + dt * drift + sqrt(2.0 * dt) * noise
        @inbounds for b in 1:ntraj, i in 1:K
            z[i, 1, b] = flat[i, b]
            z[i, 2, b] = flat[K + i, b]
        end
        if minimum(@view z[:, 1, :]) <= 0.0 || !all(isfinite, z)
            error(@sprintf("Phi Langevin left physical domain at step %d: min rho %.6e",
                step, minimum(@view z[:, 1, :])))
        end
        project_zero_modes!(z, params)
        flat .= raw_flat(z)
    end
    return times, saved, min_eig
end

function draw_values(states::Array{Float32, 4}, params::SimParams, quantity::Symbol,
                     max_samples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = nt * K * ntraj
    n = min(max_samples, total)
    vals = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = (linear % nt) + 1
        tmp = linear ÷ nt
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        if quantity == :rho
            vals[s] = states[t, i, 1, tr]
        elseif quantity == :m
            vals[s] = states[t, i, 2, tr]
        elseif quantity == :u
            vals[s] = states[t, i, 2, tr] / max(states[t, i, 1, tr], params.velocity_density_floor)
        elseif quantity == :eta
            ip = periodic_index(i + 1, K)
            rho_edge = 0.5 * (states[t, i, 1, tr] + states[t, ip, 1, tr])
            vals[s] = params.eta0 * (rho_edge / params.rho0)^params.zeta
        end
    end
    return vals
end

function histogram_density(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Float64, length(edges) - 1)
    lo = first(edges)
    hi = last(edges)
    width = (hi - lo) / length(counts)
    @inbounds for v in values
        if lo <= v <= hi
            idx = clamp(floor(Int, (v - lo) / width) + 1, 1, length(counts))
            counts[idx] += 1.0
        end
    end
    return 0.5 .* (edges[1:end-1] .+ edges[2:end]), counts ./ max(sum(counts) * width, eps(Float64))
end

function rel_l2(a, b)
    return norm(a .- b) / max(norm(a), eps(Float64))
end

function covariance_flat(states::Array{Float32, 4})
    nt, K, _, ntraj = size(states)
    D = 2K
    ns = nt * ntraj
    X = Matrix{Float64}(undef, D, ns)
    col = 0
    @inbounds for tr in 1:ntraj, t in 1:nt
        col += 1
        for i in 1:K
            X[i, col] = states[t, i, 1, tr]
            X[K + i, col] = states[t, i, 2, tr]
        end
    end
    mu = mean(X; dims=2)
    X .-= mu
    return (X * transpose(X)) ./ max(ns - 1, 1)
end

function channel_acf(states::Array{Float32, 4}, channel::Int, max_lag::Int, stride::Int)
    nt, K, _, ntraj = size(states)
    lags = collect(0:stride:max_lag)
    mu = mean(Float64, states[:, :, channel, :])
    var = mean(x -> (Float64(x) - mu)^2, states[:, :, channel, :])
    vals = Float64[]
    for lag in lags
        s = 0.0
        count = 0
        @inbounds for tr in 1:ntraj, t in 1:(nt - lag), i in 1:K
            s += (Float64(states[t + lag, i, channel, tr]) - mu) *
                (Float64(states[t, i, channel, tr]) - mu)
            count += 1
        end
        push!(vals, s / max(count * var, eps(Float64)))
    end
    return lags, vals
end

function observable_values_capillary(z::Array{Float32, 3}, params::SimParams)
    K, _, B = size(z)
    obs = Array{Float32}(undef, K, 14, B)
    h = Float32(dx(params))
    @inbounds for b in 1:B, i in 1:K
        im = periodic_index(i - 1, K)
        ip = periodic_index(i + 1, K)
        rho_i = Float64(z[i, 1, b])
        m_i = Float64(z[i, 2, b])
        u_i = m_i / max(rho_i, params.velocity_density_floor)
        rho_edge_r = 0.5 * (Float64(z[i, 1, b]) + Float64(z[ip, 1, b]))
        rho_edge_l = 0.5 * (Float64(z[im, 1, b]) + Float64(z[i, 1, b]))
        eta_cell = 0.5 * params.eta0 * ((rho_edge_r / params.rho0)^params.zeta +
            (rho_edge_l / params.rho0)^params.zeta)
        obs[i, 1, b] = Float32(rho_i - params.rho0)
        obs[i, 2, b] = Float32(m_i)
        obs[i, 3, b] = Float32(u_i)
        obs[i, 4, b] = (z[ip, 1, b] - z[im, 1, b]) / (2h)
        obs[i, 5, b] = (z[ip, 2, b] - z[im, 2, b]) / (2h)
        obs[i, 6, b] = (z[ip, 1, b] - 2f0 * z[i, 1, b] + z[im, 1, b]) / (h * h)
        obs[i, 7, b] = (z[ip, 2, b] - 2f0 * z[i, 2, b] + z[im, 2, b]) / (h * h)
        obs[i, 8, b] = Float32(eta_cell)
        obs[i, 9, b] = Float32((rho_i - params.rho0) * m_i)
        obs[i, 10, b] = Float32((rho_i - params.rho0)^2)
        obs[i, 11, b] = Float32(m_i^2)
        obs[i, 12, b] = Float32((rho_i - params.rho0) * u_i)
        obs[i, 13, b] = Float32(m_i * u_i + params.cs^2 * rho_i)
        obs[i, 14, b] = Float32(eta_cell * (Float64(z[ip, 2, b]) / max(Float64(z[ip, 1, b]), params.velocity_density_floor) -
            Float64(z[im, 2, b]) / max(Float64(z[im, 1, b]), params.velocity_density_floor)) / (2.0 * dx(params)))
    end
    return obs
end

function observable_names_capillary()
    return ["rho-rho0", "m", "u", "grad rho", "grad m", "lap rho", "lap m",
        "eta cell", "(rho-rho0)m", "(rho-rho0)^2", "m^2", "(rho-rho0)u",
        "momentum flux", "eta grad u"]
end

function observable_means(states::Array{Float32, 4}, start_idx::Int, params::SimParams;
                          nsamples::Int=100_000, seed::Int=1)
    rng = MersenneTwister(seed)
    nt, K, _, ntraj = size(states)
    sums = zeros(Float64, 14)
    npost = nt - start_idx + 1
    x = Array{Float32}(undef, K, 2, 1)
    for _ in 1:nsamples
        t = start_idx + rand(rng, 0:(npost - 1))
        tr = rand(rng, 1:ntraj)
        x[:, :, 1] .= states[t, :, :, tr]
        obs = observable_values_capillary(x, params)
        for a in 1:14
            sums[a] += mean(Float64, @view obs[:, a, 1])
        end
    end
    return sums ./ nsamples
end

function correlation_profiles(states::Array{Float32, 4}, save_dt::Float64, taus::Vector{Float64},
                              means::Vector{Float64}, coord_mean::Vector{Float64},
                              params::SimParams; pairs_per_lag::Int=30_000, seed::Int=1)
    rng = MersenneTwister(seed)
    nt, K, _, ntraj = size(states)
    C = Array{Float64}(undef, length(taus), 14, 2, K)
    x0 = Array{Float32}(undef, K, 2, 1)
    xt = similar(x0)
    for (li, tau) in enumerate(taus)
        lag = clamp(round(Int, tau / save_dt), 0, nt - 1)
        n = min(pairs_per_lag, max(1, (nt - lag) * ntraj))
        sums = zeros(Float64, K, 14, 2, K)
        for _ in 1:n
            tr = rand(rng, 1:ntraj)
            t = rand(rng, 1:(nt - lag))
            x0[:, :, 1] .= states[t, :, :, tr]
            xt[:, :, 1] .= states[t + lag, :, :, tr]
            obs = observable_values_capillary(xt, params)
            @inbounds for i in 1:K, a in 1:14
                oval = Float64(obs[i, a, 1]) - means[a]
                for j in 1:K
                    offset = mod(i - j, K) + 1
                    sums[offset, a, 1, 1] += 0.0
                    sums[offset, a, 1, j] += oval * (Float64(x0[j, 1, 1]) - coord_mean[j])
                    sums[offset, a, 2, j] += oval * (Float64(x0[j, 2, 1]) - coord_mean[K + j])
                end
            end
        end
        prof = zeros(Float64, 14, 2, K)
        @inbounds for a in 1:14, comp in 1:2, r in 1:K
            # Average over origins with the same periodic offset.
            total = 0.0
            for j in 1:K
                total += sums[r, a, comp, j]
            end
            prof[a, comp, r] = total / (n * K)
        end
        C[li, :, :, :] .= prof
    end
    return C
end

function agreement_metrics(a, b)
    av = vec(Float64.(a))
    bv = vec(Float64.(b))
    rel = norm(bv .- av) / max(norm(av), eps(Float64))
    corr = cor(av, bv)
    return rel, corr
end

function select_profile_channels(C; nshow::Int=16)
    scored = Tuple{Float64, Int, Int, Int}[]
    for a in axes(C, 2), comp in axes(C, 3), r in axes(C, 4)
        push!(scored, (maximum(abs, @view C[:, a, comp, r]), a, comp, r))
    end
    sort!(scored; by=x -> x[1], rev=true)
    return scored[1:min(nshow, length(scored))]
end

function render_stats(path, obs_states, phi_states, params, save_dt)
    rng = MersenneTwister(params.seed + 91)
    specs = [(:rho, "rho PDF", "rho"), (:m, "m PDF", "m"),
        (:u, "velocity PDF", "u"), (:eta, "edge viscosity PDF", "eta")]
    metrics = Dict{String, Float64}()
    fig = Figure(; size=(3600, 2600))
    figure_title!(fig, "Capillary FHD analytic Phi forward statistics";
        subtitle="observations vs CPU analytic M=Phi Langevin")
    for (idx, (q, title, xlabel)) in enumerate(specs)
        obs = draw_values(obs_states, params, q, 300_000, rng)
        phi = draw_values(phi_states, params, q, 300_000, rng)
        lo = quantile(obs, 0.001)
        hi = quantile(obs, 0.999)
        pad = 0.08 * max(hi - lo, 1.0e-8)
        edges = collect(range(lo - pad, hi + pad; length=161))
        centers, po = histogram_density(obs, edges)
        _, pp = histogram_density(phi, edges)
        metrics[string(q, "_pdf_rel_l2")] = rel_l2(po, pp)
        ax = Axis(fig[1, idx]; title=title, xlabel=xlabel, ylabel="density")
        lines!(ax, centers, po; color=STYLE_REFERENCE, linewidth=3, label="observed")
        lines!(ax, centers, pp; color=STYLE_SECONDARY, linewidth=3, linestyle=:dash, label="M=Phi analytic")
        idx == 1 && axislegend(ax; position=:rt)
    end
    max_lag = min(size(obs_states, 1), size(phi_states, 1)) - 1
    max_lag = min(max_lag, round(Int, 8.0 / save_dt))
    stride = max(1, round(Int, 0.08 / save_dt))
    for (idx, (channel, title)) in enumerate([(1, "rho autocorrelation"), (2, "m autocorrelation")])
        lags, co = channel_acf(obs_states, channel, max_lag, stride)
        _, cp = channel_acf(phi_states, channel, max_lag, stride)
        metrics[string(channel == 1 ? "rho" : "m", "_acf_rel_l2")] = rel_l2(co, cp)
        ax = Axis(fig[2, idx]; title=title, xlabel="tau", ylabel="C/C(0)")
        lines!(ax, lags .* save_dt, co; color=STYLE_REFERENCE, linewidth=3)
        lines!(ax, lags .* save_dt, cp; color=STYLE_SECONDARY, linewidth=3, linestyle=:dash)
    end
    cov_obs = covariance_flat(obs_states)
    cov_phi = covariance_flat(phi_states)
    metrics["covariance_rel_rmse"] = agreement_metrics(cov_obs, cov_phi)[1]
    metrics["covariance_corr"] = agreement_metrics(cov_obs, cov_phi)[2]
    for (idx, mat, title) in [(3, cov_obs, "observed covariance"), (4, cov_phi .- cov_obs, "Phi covariance error")]
        ax = Axis(fig[2, idx]; title=title, xlabel="column", ylabel="row")
        clim = max(maximum(abs, mat), 1.0e-8)
        hm = idx == 3 ? heatmap!(ax, mat; colormap=:viridis) :
            heatmap!(ax, mat; colormap=STYLE_DIVERGING, colorrange=(-clim, clim))
        Colorbar(fig[2, idx, Right()], hm)
    end
    text_panel!(fig[3, 1:4], [@sprintf("%s = %.6e", k, metrics[k]) for k in sort(collect(keys(metrics)))] ;
        title="Agreement metrics")
    save_figure(path, fig)
    return metrics
end

function render_cmn(path, taus, tD, names, Cobs, Cphi, selected)
    rel, corr = agreement_metrics(Cobs, Cphi)
    fig = Figure(; size=(3600, 3000))
    figure_title!(fig, "Capillary FHD analytic Phi forward Cmn(t)";
        subtitle=@sprintf("Phi rel.RMSE=%.3e corr=%.4f", rel, corr))
    for (idx, (_, a, comp, r)) in enumerate(selected)
        row = 1 + (idx - 1) ÷ 4
        col = 1 + (idx - 1) % 4
        cname = comp == 1 ? "rho0" : "m0"
        ax = Axis(fig[row, col];
            title=@sprintf("%s vs %s, offset %d", names[a], cname, r - 1),
            xlabel="tau / t_D", ylabel="C")
        lines!(ax, taus ./ tD, Cobs[:, a, comp, r]; color=STYLE_REFERENCE, linewidth=3, label="observed")
        lines!(ax, taus ./ tD, Cphi[:, a, comp, r]; color=STYLE_SECONDARY, linewidth=3,
            linestyle=:dash, label="M=Phi analytic")
        idx == 1 && axislegend(ax; position=:rt)
    end
    save_figure(path, fig)
    return rel, corr
end

function run_pipeline(param_file::AbstractString=DEFAULT_ANALYTIC_PARAM_FILE)
    base_dir = dirname(abspath(param_file))
    params = load_params(param_file)
    h5_path = resolve_path(base_dir, params.output_hdf5)
    require_condition(isfile(h5_path), "Simulation HDF5 is missing: $(h5_path)")
    states = h5open(h5_path, "r") do h5
        Array{Float32}(read(h5["/trajectories/states"]))
    end
    save_dt = h5open(h5_path, "r") do h5
        Float64(read(h5["/metadata/save_dt"]))
    end
    tD = h5open(h5_path, "r") do h5
        Float64(read(h5["/statistics/correlations/t_decorrelation"]))
    end
    start_idx = burnin_start_index(size(states, 1), params.burnin_fraction)
    @printf("Loaded capillary observations: %s, start_idx=%d, save_dt=%.6g, tD=%.6g\n",
        string(size(states)), start_idx, save_dt, tD)

    Phi = estimate_mean_true_mobility(states, start_idx, params; nsamples=80_000, seed=params.seed + 401)
    sqrtPhi, min_eig = tangent_sqrt(Phi)
    @printf("Estimated Phi=<M_true>: min sym eig %.6e, sqrt columns %d\n", min_eig, size(sqrtPhi, 2))
    forward_dt = min(0.0025, save_dt / 40)
    forward_total = 200.0 * tD
    forward_burnin = 20.0 * tD
    times, phi_states, min_phi_eig = integrate_phi_langevin(Phi, states, start_idx, params;
        dt=forward_dt, total_time=forward_total, burnin_time=forward_burnin,
        save_dt=save_dt, ntraj=size(states, 4), seed=params.seed + 501)
    @printf("Forward Phi complete: states=%s, min eig %.6e\n", string(size(phi_states)), min_phi_eig)

    out_h5 = resolve_path(base_dir, "../data/phi_analytic_forward_langevin.h5")
    h5open(out_h5, "w") do h5
        h5["/time"] = times
        h5["/states"] = phi_states
        h5["/Phi"] = Phi
        h5["/min_phi_diffusion_eig"] = min_phi_eig
    end

    obs_stop = min(size(states, 1), start_idx + size(phi_states, 1) - 1)
    obs_states = states[start_idx:obs_stop, :, :, :]
    if size(obs_states, 1) < size(phi_states, 1)
        phi_states = phi_states[1:size(obs_states, 1), :, :, :]
        times = times[1:size(obs_states, 1)]
    end
    stats_png = resolve_path(base_dir, "../figures/phi_analytic_forward_stats.png")
    cmn_png = resolve_path(base_dir, "../figures/phi_analytic_forward_cmn.png")
    metrics = render_stats(stats_png, obs_states, phi_states, params, save_dt)

    names = observable_names_capillary()
    means = observable_means(states, start_idx, params; nsamples=120_000, seed=params.seed + 601)
    coord_mean = vcat(fill(params.rho0, params.N), zeros(params.N))
    taus = collect(range(save_dt, min(0.67 * tD, 40save_dt); length=40))
    Cobs = correlation_profiles(obs_states, save_dt, taus, means, coord_mean, params;
        pairs_per_lag=30_000, seed=params.seed + 701)
    Cphi = correlation_profiles(phi_states, save_dt, taus, means, coord_mean, params;
        pairs_per_lag=30_000, seed=params.seed + 702)
    selected = select_profile_channels(Cobs; nshow=16)
    cmn_rel, cmn_corr = render_cmn(cmn_png, taus, tD, names, Cobs, Cphi, selected)

    metrics_txt = resolve_path(base_dir, "../logs/fit_Phi_analytic_metrics.txt")
    open(metrics_txt, "w") do io
        println(io, "Capillary FHD analytic Phi validation")
        println(io, "CPU-only run; no NN, no DSM score, no GPU.")
        println(io, "Analytic score uses s_rho=-dx/Theta*(cs^2 log(rho/rho0)-0.5u^2-kappa Delta_h rho), projected by channel means.")
        println(io, "True mobility uses D_mm=Theta/dx^3 B diag(eta) B' and R=-(Theta/dx)L_h.")
        println(io, @sprintf("Phi samples = %d", 80_000))
        println(io, @sprintf("Forward dt = %.8e", forward_dt))
        println(io, @sprintf("Forward total/burnin/save = %.8e %.8e %.8e", forward_total, forward_burnin, save_dt))
        println(io, @sprintf("min sym(Phi) eig = %.8e", min_phi_eig))
        for k in sort(collect(keys(metrics)))
            println(io, @sprintf("%s = %.8e", k, metrics[k]))
        end
        println(io, @sprintf("Cmn rel.RMSE = %.8e", cmn_rel))
        println(io, @sprintf("Cmn corr = %.8e", cmn_corr))
        println(io, "Forward HDF5 = " * out_h5)
        println(io, "Stats figure = " * stats_png)
        println(io, "Cmn figure = " * cmn_png)
    end
    @printf("Saved analytic Phi metrics to %s\n", metrics_txt)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(isempty(ARGS) ? DEFAULT_ANALYTIC_PARAM_FILE : abspath(ARGS[1]))
end
