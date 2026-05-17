#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi.jl"))

using FFTW
using HDF5
using KernelDensity
using Printf
using Statistics

const DEFAULT_PHI_CONFIG = joinpath(@__DIR__, "..", "configs", "fit_Phi.toml")
const DEFAULT_PDF_SAMPLE_COUNT = 1_000_000
const DEFAULT_COV_SAMPLE_COUNT = 200_000
const DEFAULT_FORWARD_CMN_CHANNELS_TOML =
    joinpath(@__DIR__, "..", "configs", "nonlinear_observable_retained_channels_compact.toml")
const DEFAULT_FORWARD_OBSERVABLE_CORR_PAIRS = 30_000

const FORWARD_SUPPORTED_OBSERVABLES = Set([
    "mz_gradm_my", "mz_lap_my", "cross_m_x", "cross_p_x", "mx_r4",
    "mz_gradp_my", "mx_r2", "mx_mx2", "mx_neighbor_r2sum",
    "mx_r2_nnavg", "mx_r2_mperp2", "mx_mperp2",
    "my_r2_mperp2", "my_mperp2", "my_r4", "my_r2", "my_twist2",
    "my_mperp4", "my_my2", "my_grad2", "my_mx2", "my_lap2",
    "my_neighbor_r2sum", "my_r2_nnavg", "my_mx_nnavg",
    "cross_p_z", "cross_m_z", "mz_lap2", "mx_cross_m_my",
    "mz_grad2", "mx_my_nnavg", "mx_lap_my", "mx_gradm_my",
    "my_cross_p_x", "mx_cross_p_y", "mx_my_p",
])

struct ForwardModelStates
    label::String
    states::Array{Float32, 4}
    time::Vector{Float64}
    color
    linestyle
end

struct ForwardCmnChannel
    observable::String
    target_component::Int
end

struct ForwardNonlinearLibrary
    names::Vector{String}
    index::Dict{String, Int}
end

function ForwardNonlinearLibrary(names::Vector{String})
    unsupported = setdiff(names, collect(FORWARD_SUPPORTED_OBSERVABLES))
    require_condition(isempty(unsupported),
        "Unsupported forward Cmn observables: $(join(unsupported, ", ")).")
    return ForwardNonlinearLibrary(names, Dict(name => i for (i, name) in enumerate(names)))
end

function load_forward_cmn_channels(path::AbstractString)
    parsed = TOML.parsefile(path)
    channels = ForwardCmnChannel[]
    for ch in get(parsed, "channels", Any[])
        target = String(ch["target_component"])
        c = findfirst(==(target), ["mx", "my", "mz"])
        c === nothing && error("Unknown target component $(target) in $(path).")
        observable = String(ch["observable"])
        observable in FORWARD_SUPPORTED_OBSERVABLES ||
            error("Unsupported forward Cmn observable $(observable).")
        push!(channels, ForwardCmnChannel(observable, c))
    end
    require_condition(!isempty(channels), "No forward Cmn channels found in $(path).")
    names = unique([ch.observable for ch in channels])
    return channels, ForwardNonlinearLibrary(names)
end

channel_label(ch::ForwardCmnChannel) =
    string(ch.observable, " -> ", ("mx", "my", "mz")[ch.target_component])

function load_forward_h5(path::AbstractString)
    states = h5read(path, "/trajectories/states")
    time = h5read(path, "/trajectories/time")
    return Array{Float32, 4}(states), Vector{Float64}(time)
end

function sampled_state_values(states::AbstractArray{<:Real, 4}, j::Int, max_values::Int,
        rng::AbstractRNG)
    nt, N, _, ntraj = size(states)
    total = nt * N * ntraj
    n = min(max_values, total)
    out = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = total <= max_values ? s - 1 : rand(rng, 0:(total - 1))
        t = 1 + (linear % nt)
        rem = linear ÷ nt
        i = 1 + (rem % N)
        tr = 1 + (rem ÷ N)
        if j <= 3
            out[s] = Float64(states[t, i, j, tr])
        else
            out[s] = sqrt(sum(abs2, @view states[t, i, :, tr]))
        end
    end
    return out
end

function flat_samples_from_any_states(states::AbstractArray{<:Real, 4}, max_samples::Int,
        rng::AbstractRNG)
    nt, N, _, ntraj = size(states)
    total = nt * ntraj
    n = min(max_samples, total)
    out = Matrix{Float64}(undef, 3N, n)
    @inbounds for s in 1:n
        linear = total <= max_samples ? s - 1 : rand(rng, 0:(total - 1))
        t = 1 + (linear % nt)
        tr = 1 + (linear ÷ nt)
        out[:, s] .= flatten_state(@view states[t, :, :, tr])
    end
    return out
end

function covariance_from_states(states::AbstractArray{<:Real, 4}, seed::Int)
    flat = flat_samples_from_any_states(states, DEFAULT_COV_SAMPLE_COUNT, MersenneTwister(seed))
    return cov(permutedims(flat))
end

function global_component_series(states::AbstractArray{<:Real, 4})
    nt, N, _, ntraj = size(states)
    series = Array{Float32}(undef, nt, 3, ntraj)
    invN = Float32(1 / N)
    @inbounds for tr in 1:ntraj, t in 1:nt
        for c in 1:3
            acc = 0f0
            for i in 1:N
                acc += Float32(states[t, i, c, tr])
            end
            series[t, c, tr] = acc * invN
        end
    end
    return series
end

function exact_global_component_correlations(states::AbstractArray{<:Real, 4},
        save_dt::Float64, max_lags::Int)
    series = global_component_series(states)
    nt, _, ntraj = size(series)
    L = min(max_lags, nt - 1)
    nfft = 2 ^ ceil(Int, log2(2nt - 1))
    means = zeros(Float64, 3)
    @inbounds for c in 1:3
        means[c] = sum(Float64, @view series[:, c, :]) / (nt * ntraj)
    end

    C = zeros(Float64, L + 1, 3, 3)
    spectra = Vector{Vector{ComplexF64}}(undef, 3)
    padded = Vector{Float64}(undef, nfft)
    @inbounds for tr in 1:ntraj
        for c in 1:3
            fill!(padded, 0.0)
            for t in 1:nt
                padded[t] = Float64(series[t, c, tr]) - means[c]
            end
            spectra[c] = fft(padded)
        end
        for c in 1:3, d in 1:3
            corr = real.(ifft(spectra[c] .* conj.(spectra[d])))
            C[:, c, d] .+= @view corr[1:(L + 1)]
        end
    end
    @inbounds for lag in 0:L
        C[lag + 1, :, :] ./= (nt - lag) * ntraj
    end

    R = similar(C)
    @inbounds for c in 1:3, d in 1:3
        denom = sqrt(max(abs(C[1, c, c] * C[1, d, d]), eps(Float64)))
        R[:, c, d] .= C[:, c, d] ./ denom
    end
    GC.gc()
    return collect(0:L) .* save_dt, C, R
end

function transverse_autocorr(C::Array{Float64, 3})
    denom = C[1, 1, 1] + C[1, 2, 2]
    return (C[:, 1, 1] .+ C[:, 2, 2]) ./ (abs(denom) > eps(Float64) ? denom : eps(Float64))
end

function render_component_corr_panel!(fig, row::Int, col::Int, title::AbstractString,
        ylabel::AbstractString, obs_corr, model_corrs, curve_fn)
    ax = Axis(fig[row, col]; title=title, xlabel="t", ylabel=ylabel,
        titlesize=18, xlabelsize=16, ylabelsize=16,
        xticklabelsize=13, yticklabelsize=13)
    lines!(ax, obs_corr[1], curve_fn(obs_corr[2], obs_corr[3]); color=:black,
        linewidth=2.6, label="obs")
    for (model, corr) in model_corrs
        lines!(ax, corr[1], curve_fn(corr[2], corr[3]); color=model.color,
            linewidth=2, linestyle=model.linestyle, label=model.label)
    end
    return ax
end

function render_stats_with_dm(path, params, obs_states::AbstractArray{<:Real, 4},
        models::Vector{ForwardModelStates}; obs_save_dt::Float64=params.forward_save_dt,
        corr_lags::Int=DEFAULT_FORWARD_CORR_MAX_LAGS)
    labels = ["mx", "my", "mz", "|m|"]
    fig = Figure(; size=(max(params.figure_width, 4600), max(params.figure_height, 3800)))
    figure_title!(fig, "Soft-spin LLG forward statistics";
        subtitle="Stationary PDFs/covariance plus exact global-component ACFs and cross-correlations")
    first_pdf_axis = nothing
    for j in 1:4
        ax = Axis(fig[1, j]; title="PDF $(labels[j])", xlabel=labels[j], ylabel="density")
        vo = sampled_state_values(obs_states, j, DEFAULT_PDF_SAMPLE_COUNT,
            MersenneTwister(100 + j))
        ko = kde(vo)
        lines!(ax, ko.x, ko.density; color=:black, linewidth=2.6, label="obs")
        for (k, model) in enumerate(models)
            vm = sampled_state_values(model.states, j, DEFAULT_PDF_SAMPLE_COUNT,
                MersenneTwister(1000 + 31j + 101k))
            km = kde(vm)
            lines!(ax, km.x, km.density; color=model.color, linewidth=2,
                linestyle=model.linestyle, label=model.label)
        end
        j == 1 && (first_pdf_axis = ax)
    end
    first_pdf_axis !== nothing &&
        Legend(fig[1, 5:6], first_pdf_axis; framevisible=false, tellheight=false)

    cov_obs = covariance_from_states(obs_states, 1)
    cov_models = [covariance_from_states(m.states, 10 + k) for (k, m) in enumerate(models)]
    cov_lim = max(maximum(abs, cov_obs), eps(Float64))
    err_lim = maximum(abs, reduce(vcat, [vec(c - cov_obs) for c in cov_models]))
    err_lim = max(err_lim, eps(Float64))
    map_models = models[1:min(length(models), 3)]
    ax_obs = Axis(fig[2, 1]; title="Observed covariance", xlabel="flat index",
        ylabel="flat index", aspect=DataAspect())
    hm_obs = heatmap!(ax_obs, cov_obs; colormap=:balance, colorrange=(-cov_lim, cov_lim))
    hm_err = nothing
    for (k, model) in enumerate(map_models)
        ax = Axis(fig[2, k + 1]; title="$(model.label) covariance error",
            xlabel="flat index", ylabel="flat index")
        hm_err = heatmap!(ax, cov_models[k] - cov_obs; colormap=:balance,
            colorrange=(-err_lim, err_lim))
    end
    Colorbar(fig[2, 5], hm_obs; label="cov", labelsize=15, ticklabelsize=12)
    hm_err !== nothing &&
        Colorbar(fig[2, 6], hm_err; label="model - obs", labelsize=15, ticklabelsize=12)
    colsize!(fig.layout, 5, Fixed(58))
    colsize!(fig.layout, 6, Fixed(58))

    corr_lags = min(corr_lags, size(obs_states, 1) - 1,
        minimum(size(model.states, 1) - 1 for model in models))
    @printf("Computing exact forward_stats global-component correlations (%d lags, FFT).\n",
        corr_lags)
    obs_corr = exact_global_component_correlations(obs_states, obs_save_dt, corr_lags)
    model_corrs = []
    for (k, model) in enumerate(models)
        dt = length(model.time) > 1 ? model.time[2] - model.time[1] : params.forward_save_dt
        corr = exact_global_component_correlations(model.states, dt, corr_lags)
        push!(model_corrs, (model, corr))
    end

    corr_specs = [
        ("ACF mx", "R", (C, R) -> R[:, 1, 1], (-0.15, 1.05)),
        ("ACF my", "R", (C, R) -> R[:, 2, 2], (-0.15, 1.05)),
        ("ACF mz", "R", (C, R) -> R[:, 3, 3], (0.65, 1.03)),
        ("ACF mperp", "R", (C, R) -> transverse_autocorr(C), (-0.15, 1.05)),
        ("Cross mx(t), my(0)", "R", (C, R) -> R[:, 1, 2], nothing),
        ("Cross my(t), mx(0)", "R", (C, R) -> R[:, 2, 1], nothing),
        ("Cross mx(t), mz(0)", "R", (C, R) -> R[:, 1, 3], nothing),
        ("Cross mz(t), mx(0)", "R", (C, R) -> R[:, 3, 1], nothing),
    ]
    for (idx, (title, ylabel, fn, ylimits)) in enumerate(corr_specs)
        row = 3 + (idx - 1) ÷ 4
        col = 1 + (idx - 1) % 4
        ax = render_component_corr_panel!(fig, row, col, title, ylabel,
            obs_corr, model_corrs, fn)
        ylimits !== nothing && ylims!(ax, ylimits...)
    end

    save_figure_checked(path, fig)
    return cov_obs, cov_models
end

function nonlinear_observables_forward(raw::Array{Float32, 3}, p::SpinParams,
        lib::ForwardNonlinearLibrary)
    N, _, B = size(raw)
    obs = Array{Float32}(undef, N, length(lib.names), B)
    fill!(obs, 0f0)
    idx = lib.index
    i_mz_gradm_my = get(idx, "mz_gradm_my", 0)
    i_mz_lap_my = get(idx, "mz_lap_my", 0)
    i_cross_m_x = get(idx, "cross_m_x", 0)
    i_cross_p_x = get(idx, "cross_p_x", 0)
    i_mx_r4 = get(idx, "mx_r4", 0)
    i_mz_gradp_my = get(idx, "mz_gradp_my", 0)
    i_mx_r2 = get(idx, "mx_r2", 0)
    i_mx_mx2 = get(idx, "mx_mx2", 0)
    i_mx_neighbor_r2sum = get(idx, "mx_neighbor_r2sum", 0)
    i_mx_r2_nnavg = get(idx, "mx_r2_nnavg", 0)
    i_mx_r2_mperp2 = get(idx, "mx_r2_mperp2", 0)
    i_mx_mperp2 = get(idx, "mx_mperp2", 0)
    i_my_r2_mperp2 = get(idx, "my_r2_mperp2", 0)
    i_my_mperp2 = get(idx, "my_mperp2", 0)
    i_my_r4 = get(idx, "my_r4", 0)
    i_my_r2 = get(idx, "my_r2", 0)
    i_my_twist2 = get(idx, "my_twist2", 0)
    i_my_mperp4 = get(idx, "my_mperp4", 0)
    i_my_my2 = get(idx, "my_my2", 0)
    i_my_grad2 = get(idx, "my_grad2", 0)
    i_my_mx2 = get(idx, "my_mx2", 0)
    i_my_lap2 = get(idx, "my_lap2", 0)
    i_my_neighbor_r2sum = get(idx, "my_neighbor_r2sum", 0)
    i_my_r2_nnavg = get(idx, "my_r2_nnavg", 0)
    i_my_mx_nnavg = get(idx, "my_mx_nnavg", 0)
    i_cross_p_z = get(idx, "cross_p_z", 0)
    i_cross_m_z = get(idx, "cross_m_z", 0)
    i_mz_lap2 = get(idx, "mz_lap2", 0)
    i_mx_cross_m_my = get(idx, "mx_cross_m_my", 0)
    i_mz_grad2 = get(idx, "mz_grad2", 0)
    i_mx_my_nnavg = get(idx, "mx_my_nnavg", 0)
    i_mx_lap_my = get(idx, "mx_lap_my", 0)
    i_mx_gradm_my = get(idx, "mx_gradm_my", 0)
    i_my_cross_p_x = get(idx, "my_cross_p_x", 0)
    i_mx_cross_p_y = get(idx, "mx_cross_p_y", 0)
    i_mx_my_p = get(idx, "mx_my_p", 0)

    @inbounds for b in 1:B, i in 1:N
        im = periodic(i - 1, N)
        ip = periodic(i + 1, N)
        x1 = Float64(raw[i, 1, b])
        x2 = Float64(raw[i, 2, b])
        x3 = Float64(raw[i, 3, b])
        xm1 = Float64(raw[im, 1, b])
        xm2 = Float64(raw[im, 2, b])
        xm3 = Float64(raw[im, 3, b])
        xp1 = Float64(raw[ip, 1, b])
        xp2 = Float64(raw[ip, 2, b])
        xp3 = Float64(raw[ip, 3, b])
        r2 = x1 * x1 + x2 * x2 + x3 * x3
        r2p = xp1 * xp1 + xp2 * xp2 + xp3 * xp3
        r2m = xm1 * xm1 + xm2 * xm2 + xm3 * xm3
        mperp2 = x1 * x1 + x2 * x2
        diffp1 = xp1 - x1
        diffp2 = xp2 - x2
        diffp3 = xp3 - x3
        diffm1 = x1 - xm1
        diffm2 = x2 - xm2
        diffm3 = x3 - xm3
        lap1 = xp1 + xm1 - 2.0 * x1
        lap2 = xp2 + xm2 - 2.0 * x2
        lap3 = xp3 + xm3 - 2.0 * x3
        grad2 = diffp1^2 + diffp2^2 + diffp3^2 + diffm1^2 + diffm2^2 + diffm3^2
        lap_norm2 = lap1^2 + lap2^2 + lap3^2
        crossp = cross3(x1, x2, x3, xp1, xp2, xp3)
        crossm = cross3(x1, x2, x3, xm1, xm2, xm3)
        twist2 = crossp[1]^2 + crossp[2]^2 + crossp[3]^2 +
            crossm[1]^2 + crossm[2]^2 + crossm[3]^2
        r4 = r2 * r2
        mperp4 = mperp2 * mperp2

        i_mz_gradm_my > 0 && (obs[i, i_mz_gradm_my, b] = Float32(x3 * diffm2))
        i_mz_lap_my > 0 && (obs[i, i_mz_lap_my, b] = Float32(x3 * lap2))
        i_cross_m_x > 0 && (obs[i, i_cross_m_x, b] = Float32(crossm[1]))
        i_cross_p_x > 0 && (obs[i, i_cross_p_x, b] = Float32(crossp[1]))
        i_mx_r4 > 0 && (obs[i, i_mx_r4, b] = Float32(x1 * r4))
        i_mz_gradp_my > 0 && (obs[i, i_mz_gradp_my, b] = Float32(x3 * diffp2))
        i_mx_r2 > 0 && (obs[i, i_mx_r2, b] = Float32(x1 * r2))
        i_mx_mx2 > 0 && (obs[i, i_mx_mx2, b] = Float32(x1^3))
        i_mx_neighbor_r2sum > 0 && (obs[i, i_mx_neighbor_r2sum, b] = Float32(x1 * (r2p + r2m)))
        i_mx_r2_nnavg > 0 && (obs[i, i_mx_r2_nnavg, b] = Float32(0.5 * x1 * (r2p + r2m)))
        i_mx_r2_mperp2 > 0 && (obs[i, i_mx_r2_mperp2, b] = Float32(x1 * r2 * mperp2))
        i_mx_mperp2 > 0 && (obs[i, i_mx_mperp2, b] = Float32(x1 * mperp2))
        i_my_r2_mperp2 > 0 && (obs[i, i_my_r2_mperp2, b] = Float32(x2 * r2 * mperp2))
        i_my_mperp2 > 0 && (obs[i, i_my_mperp2, b] = Float32(x2 * mperp2))
        i_my_r4 > 0 && (obs[i, i_my_r4, b] = Float32(x2 * r4))
        i_my_r2 > 0 && (obs[i, i_my_r2, b] = Float32(x2 * r2))
        i_my_twist2 > 0 && (obs[i, i_my_twist2, b] = Float32(x2 * twist2))
        i_my_mperp4 > 0 && (obs[i, i_my_mperp4, b] = Float32(x2 * mperp4))
        i_my_my2 > 0 && (obs[i, i_my_my2, b] = Float32(x2^3))
        i_my_grad2 > 0 && (obs[i, i_my_grad2, b] = Float32(x2 * grad2))
        i_my_mx2 > 0 && (obs[i, i_my_mx2, b] = Float32(x2 * x1 * x1))
        i_my_lap2 > 0 && (obs[i, i_my_lap2, b] = Float32(x2 * lap_norm2))
        i_my_neighbor_r2sum > 0 && (obs[i, i_my_neighbor_r2sum, b] = Float32(x2 * (r2p + r2m)))
        i_my_r2_nnavg > 0 && (obs[i, i_my_r2_nnavg, b] = Float32(0.5 * x2 * (r2p + r2m)))
        i_my_mx_nnavg > 0 && (obs[i, i_my_mx_nnavg, b] = Float32(0.5 * x2 * (xp1 + xm1)))
        i_cross_p_z > 0 && (obs[i, i_cross_p_z, b] = Float32(crossp[3]))
        i_cross_m_z > 0 && (obs[i, i_cross_m_z, b] = Float32(crossm[3]))
        i_mz_lap2 > 0 && (obs[i, i_mz_lap2, b] = Float32(x3 * lap_norm2))
        i_mx_cross_m_my > 0 && (obs[i, i_mx_cross_m_my, b] = Float32(x1 * crossm[2]))
        i_mz_grad2 > 0 && (obs[i, i_mz_grad2, b] = Float32(x3 * grad2))
        i_mx_my_nnavg > 0 && (obs[i, i_mx_my_nnavg, b] = Float32(0.5 * x1 * (xp2 + xm2)))
        i_mx_lap_my > 0 && (obs[i, i_mx_lap_my, b] = Float32(x1 * lap2))
        i_mx_gradm_my > 0 && (obs[i, i_mx_gradm_my, b] = Float32(x1 * diffm2))
        i_my_cross_p_x > 0 && (obs[i, i_my_cross_p_x, b] = Float32(x2 * crossp[1]))
        i_mx_cross_p_y > 0 && (obs[i, i_mx_cross_p_y, b] = Float32(x1 * crossp[2]))
        i_mx_my_p > 0 && (obs[i, i_mx_my_p, b] = Float32(x1 * xp2))
    end
    return obs
end

function precompute_forward_observables(states::Array{Float32, 4}, p::SpinParams,
        lib::ForwardNonlinearLibrary)
    nt, N, _, ntraj = size(states)
    obs = Array{Float32}(undef, N, length(lib.names), nt, ntraj)
    raw = Array{Float32}(undef, N, 3, nt)
    @inbounds for tr in 1:ntraj
        for t in 1:nt
            raw[:, :, t] .= states[t, :, :, tr]
        end
        obs[:, :, :, tr] .= nonlinear_observables_forward(raw, p, lib)
    end
    return obs
end

function forward_observable_means(obs::Array{Float32, 4})
    _, nobs, _, _ = size(obs)
    means = zeros(Float64, nobs)
    denom = size(obs, 1) * size(obs, 3) * size(obs, 4)
    @inbounds for a in 1:nobs
        means[a] = sum(Float64, @view obs[:, a, :, :]) / denom
    end
    return means
end

function forward_component_means(states::Array{Float32, 4})
    means = zeros(Float64, 3)
    denom = size(states, 1) * size(states, 2) * size(states, 4)
    @inbounds for c in 1:3
        means[c] = sum(Float64, @view states[:, :, c, :]) / denom
    end
    return means
end

function retained_channel_correlations(states::Array{Float32, 4}, save_dt::Float64,
        p::SpinParams, channels::Vector{ForwardCmnChannel},
        lib::ForwardNonlinearLibrary, max_lags::Int; seed::Int,
        pairs_per_lag::Int=DEFAULT_FORWARD_OBSERVABLE_CORR_PAIRS,
        label::AbstractString="trajectory")
    nt, N, _, ntraj = size(states)
    L = min(max_lags, nt - 1)
    @printf("Precomputing retained observables for %s (%d channels, %d lags).\n",
        label, length(channels), L)
    obs = precompute_forward_observables(states, p, lib)
    obs_means = forward_observable_means(obs)
    comp_means = forward_component_means(states)
    obs_indices = [lib.index[ch.observable] for ch in channels]
    comp_indices = [ch.target_component for ch in channels]
    C = Array{Float64}(undef, L + 1, length(channels))
    rng = MersenneTwister(seed)
    times = Vector{Int}(undef, pairs_per_lag)
    trajs = Vector{Int}(undef, pairs_per_lag)
    @inbounds for lag in 0:L
        available_t = nt - lag
        available = available_t * ntraj
        B = min(pairs_per_lag, available)
        if available <= pairs_per_lag
            for b in 1:B
                linear = b - 1
                times[b] = 1 + (linear % available_t)
                trajs[b] = 1 + (linear ÷ available_t)
            end
        else
            for b in 1:B
                linear = rand(rng, 0:(available - 1))
                times[b] = 1 + (linear % available_t)
                trajs[b] = 1 + (linear ÷ available_t)
            end
        end
        for k in eachindex(channels)
            a = obs_indices[k]
            c = comp_indices[k]
            om = obs_means[a]
            xm = comp_means[c]
            acc = 0.0
            for b in 1:B
                t = times[b]
                tr = trajs[b]
                for i in 1:N
                    acc += (Float64(obs[i, a, t + lag, tr]) - om) *
                        (Float64(states[t, i, c, tr]) - xm)
                end
            end
            C[lag + 1, k] = acc / (B * N)
        end
        lag % 50 == 0 && @printf("  %s retained C lag %d/%d\n", label, lag, L)
    end
    GC.gc()
    return collect(0:L) .* save_dt, C
end

function render_cmn_with_dm(path, params, obs_t, Cobs, model_corrs;
        channel_labels=nothing,
        title::AbstractString="Soft-spin LLG forward retained-observable correlations",
        correlation_kind::AbstractString="retained observable C")
    panels = size(Cobs, 2)
    ncols = min(6, panels)
    nrows = cld(panels, ncols)
    fig_width = max(params.figure_width, 4200)
    fig_height = max(params.figure_height, 520 * nrows + 260)
    fig = Figure(; size=(fig_width, fig_height))
    Lphi = min(size(Cobs, 1), size(model_corrs[1][3], 1))
    phi_metrics = agreement_metrics(Cobs[1:Lphi, :], model_corrs[1][3][1:Lphi, :])
    figure_title!(fig, title;
        subtitle=@sprintf("All %d %s channels; Phi rel.RMSE %.4e, corr %.5f",
            panels, correlation_kind, phi_metrics[:relative_rmse], phi_metrics[:correlation]))
    for j in 1:panels
        row = 1 + (j - 1) ÷ ncols
        col = 1 + (j - 1) % ncols
        panel_title = channel_labels === nothing ? "C$(j)(t)" :
            replace(channel_labels[j], " -> " => "\n")
        ax = Axis(fig[row, col]; title=panel_title,
            xlabel=row == nrows ? "t" : "",
            ylabel=col == 1 ? "C" : "",
            titlesize=14, xlabelsize=13, ylabelsize=13,
            xticklabelsize=10, yticklabelsize=10)
        lines!(ax, obs_t, Cobs[:, j]; color=:black, linewidth=2.4, label="obs")
        for (label, t, C, color, linestyle) in model_corrs
            lines!(ax, t, C[:, j]; color=color, linewidth=2,
                linestyle=linestyle, label=label)
        end
        j == 1 && axislegend(ax; position=:rt, framevisible=false)
    end
    save_figure_checked(path, fig)
end

function write_forward_metrics(path, cov_obs, cov_models, model_labels, Cobs, model_corrs;
        correlation_kind::AbstractString="coordinate C")
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain learned-M forward diagnostics")
        println(io, "No-cheating audit: forward trajectories use the learned stationary score with the labeled constant, neural, or declared parametric mobility model. Any true-M curve included in this comparison is an explicitly labeled ex-post diagnostic and was not used for training.")
        for (k, label) in enumerate(model_labels)
            cm = agreement_metrics(cov_obs, cov_models[k])
            println(io, @sprintf("%s covariance rel.RMSE = %.8e", label, cm[:relative_rmse]))
            println(io, @sprintf("%s covariance corr = %.8e", label, cm[:correlation]))
        end
        for (label, _t, C, _color, _style) in model_corrs
            L = min(size(Cobs, 1), size(C, 1))
            mm = agreement_metrics(Cobs[1:L, :], C[1:L, :])
            println(io, @sprintf("%s %s rel.RMSE = %.8e", label, correlation_kind, mm[:relative_rmse]))
            println(io, @sprintf("%s %s corr = %.8e", label, correlation_kind, mm[:correlation]))
        end
    end
end

function run_render(phi_config::AbstractString)
    base = dirname(phi_config)
    params = load_config(phi_config)
    data_h5 = resolve_path(base, params.input_hdf5)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    phi_states, phi_time = load_forward_h5(resolve_path(base, params.forward_hdf5))
    dm_paths = [
        ("M_NN C", resolve_path(base, "../data/forward_dM_gpu2_vC.h5"), STYLE_HIGHLIGHT, :solid),
        ("M_NN A", resolve_path(base, "../data/forward_dM_gpu0_vA.h5"), STYLE_ACCENT, :dashdot),
        ("M_NN B", resolve_path(base, "../data/forward_dM_gpu1_vB.h5"), STYLE_SECONDARY, :dot),
    ]
    models = ForwardModelStates[ForwardModelStates("Phi", phi_states, phi_time, STYLE_PRIMARY, :dash)]
    for (label, path, color, linestyle) in dm_paths
        s, t = load_forward_h5(path)
        push!(models, ForwardModelStates(label, s, t, color, linestyle))
    end
    stats_path = resolve_path(base, "../figures/forward_stats_with_dM.png")
    obs_stats = @view sampler.states[sampler.start_idx:end, :, :, :]
    cov_obs, cov_models = render_stats_with_dm(stats_path, params, obs_stats, models;
        obs_save_dt=sampler.save_dt)

    obs = sampler.states[sampler.start_idx:end, :, :, :]
    aligned = models
    cmn_path = resolve_path(base, "../figures/forward_cmn_with_dM.png")
    obs_nt = size(obs, 1)
    corr_lags = min(DEFAULT_FORWARD_CORR_MAX_LAGS, obs_nt - 1,
        minimum(size(model.states, 1) - 1 for model in aligned))
    p = load_phys(data_h5)
    channel_path = resolve_path(base, DEFAULT_FORWARD_CMN_CHANNELS_TOML)
    channels, lib = load_forward_cmn_channels(channel_path)
    channel_labels = channel_label.(channels)
    obs_t, Cobs = retained_channel_correlations(obs, sampler.save_dt, p,
        channels, lib, corr_lags; seed=params.seed + 9000, label="obs")
    model_corrs = []
    for (model_index, model) in enumerate(aligned)
        dt = length(model.time) > 1 ? model.time[2] - model.time[1] : params.forward_save_dt
        t, C = retained_channel_correlations(model.states, dt, p, channels, lib,
            min(DEFAULT_FORWARD_CORR_MAX_LAGS, size(model.states, 1) - 1);
            seed=params.seed + 10_000 + 101 * model_index, label=model.label)
        push!(model_corrs, (model.label, t, C, model.color, model.linestyle))
        GC.gc()
    end
    render_cmn_with_dm(cmn_path, params, obs_t, Cobs, model_corrs;
        channel_labels=channel_labels,
        correlation_kind="retained observable C")
    metrics_path = resolve_path(base, "../logs/forward_with_dM_metrics.txt")
    write_forward_metrics(metrics_path, cov_obs, cov_models, [m.label for m in aligned],
        Cobs, model_corrs; correlation_kind="retained observable C")
    open(metrics_path, "a") do io
        println(io)
        println(io, "Retained observable correlations:")
        channel_relpath = relpath(channel_path, normpath(joinpath(@__DIR__, "..")))
        println(io, "channels_toml = $(channel_relpath)")
        println(io, @sprintf("channels = %d", length(channels)))
        println(io, @sprintf("pairs_per_lag = %d time-trajectory pairs, all lattice sites", DEFAULT_FORWARD_OBSERVABLE_CORR_PAIRS))
    end
    @printf("Saved learned-M forward figures:\n  %s\n  %s\n", stats_path, cmn_path)
    @printf("Saved metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_render(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PHI_CONFIG)
end
