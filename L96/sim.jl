#!/usr/bin/env julia

# Implementation summary:
# - Simulates the stochastic Lorenz--96 system from the high-dimensional Results subsection of main.tex.
# - The SDE is
#       dx_i = [x_{i-1}(x_{i+1}-x_{i-2}) - x_i + F] dt
#              + sqrt(2) * sum_j Qsqrt[i,j] dW_j,
#   where Q is a translation-invariant circulant covariance defined spectrally.
# - Writes the raw trajectories and observational diagnostics to HDF5.
# - Renders a multipanel diagnostic figure with the averaged univariate PDF, averaged ACF,
#   averaged cross-correlations, averaged bivariate PDFs for selected mode separations,
#   the diffusion matrix Q, and the noise spectrum.

import Pkg

function ensure_packages(packages::Vector{String})
    missing = String[]
    for pkg in packages
        if Base.find_package(pkg) === nothing
            push!(missing, pkg)
        end
    end
    if !isempty(missing)
        @info "Installing missing Julia packages" missing
        Pkg.add(missing)
    end
    return nothing
end

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ensure_packages(["KernelDensity", "HDF5", "GLMakie"])

using HDF5
using KernelDensity
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

# Use the nearest available style file, but keep this script self-contained.
const STYLE_FILE_CANDIDATES = (
    joinpath(@__DIR__, "src", "figure_style.jl"),
    normpath(joinpath(@__DIR__, "..", "2D", "src", "figure_style.jl")),
)
for style_file in STYLE_FILE_CANDIDATES
    if isfile(style_file)
        include(style_file)
        break
    end
end

using GLMakie

if !isdefined(@__MODULE__, :STYLE_PRIMARY)
    const STYLE_PRIMARY = :dodgerblue4
end
if !isdefined(@__MODULE__, :STYLE_SECONDARY)
    const STYLE_SECONDARY = :darkorange2
end
if !isdefined(@__MODULE__, :STYLE_ACCENT)
    const STYLE_ACCENT = :seagreen4
end
if !isdefined(@__MODULE__, :STYLE_HIGHLIGHT)
    const STYLE_HIGHLIGHT = :firebrick3
end
if !isdefined(@__MODULE__, :STYLE_VIOLET)
    const STYLE_VIOLET = :mediumpurple4
end
if !isdefined(@__MODULE__, :STYLE_ZERO)
    const STYLE_ZERO = :gray40
end
if !isdefined(@__MODULE__, :STYLE_SEQUENTIAL_BLUE)
    const STYLE_SEQUENTIAL_BLUE = :viridis
end
if !isdefined(@__MODULE__, :STYLE_DIVERGING)
    const STYLE_DIVERGING = :balance
end
if !isdefined(@__MODULE__, :figure_title!)
    function figure_title!(fig::Figure, title::AbstractString; subtitle::AbstractString="")
        Label(fig[0, :], isempty(subtitle) ? title : string(title, "\n", subtitle);
              fontsize=24, font=:bold, tellwidth=false)
        return nothing
    end
end
if !isdefined(@__MODULE__, :save_figure)
    function save_figure(path::AbstractString, fig::Figure)
        save(path, fig)
        return nothing
    end
end

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "sim.toml")

struct SimL96Params
    K::Int
    F::Float64
    sigma::Float64
    rho::Float64
    ell::Float64
    p::Float64
    t0::Float64
    t1::Float64
    dt::Float64
    save_dt::Float64
    ntrajectories::Int
    requested_threads::Int
    seed::Int
    initial_mean::Float64
    perturb_index::Int
    perturbation::Float64
    burnin_fraction::Float64
    histogram_bins::Int
    correlation_stride::Int
    decorrelation_threshold::Float64
    max_decorrelation_time::Float64
    cross_offsets::Vector{Int}
    bivariate_offsets::Vector{Int}
    max_pdf_samples::Int
    figure_width::Int
    figure_height::Int
    output_hdf5::String
    output_png::String
end

struct HistogramResult
    centers::Vector{Float64}
    density::Vector{Float64}
end

struct PairPdfResult
    offset::Int
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct CorrelationResult
    lags::Vector{Float64}
    acf_mean::Vector{Float64}
    cross_offsets::Vector{Int}
    cross_mean::Matrix{Float64}
    t_decorrelation::Float64
    mean_value::Float64
    variance_value::Float64
end

function require(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    sim = raw["simulation"]
    ic = raw["initial_condition"]
    stats = raw["statistics"]
    fig = raw["figure"]
    out = raw["output"]

    params = SimL96Params(
        Int(sim["K"]),
        Float64(sim["F"]),
        Float64(sim["sigma"]),
        Float64(sim["rho"]),
        Float64(sim["ell"]),
        Float64(sim["p"]),
        Float64(sim["t0"]),
        Float64(sim["t1"]),
        Float64(sim["dt"]),
        Float64(sim["save_dt"]),
        Int(sim["ntrajectories"]),
        Int(sim["requested_threads"]),
        Int(sim["seed"]),
        Float64(ic["mean"]),
        Int(ic["perturb_index"]),
        Float64(ic["perturbation"]),
        Float64(stats["burnin_fraction"]),
        Int(stats["histogram_bins"]),
        Int(stats["correlation_stride"]),
        Float64(stats["decorrelation_threshold"]),
        Float64(stats["max_decorrelation_time"]),
        Int.(stats["cross_offsets"]),
        Int.(stats["bivariate_offsets"]),
        Int(stats["max_pdf_samples"]),
        Int(fig["width"]),
        Int(fig["height"]),
        String(out["hdf5_file"]),
        String(out["figure_png"]),
    )

    require(params.K >= 4, "K must be at least 4 for Lorenz--96 periodic indexing.")
    require(params.F > 0.0, "F must be positive.")
    require(params.sigma > 0.0, "sigma must be positive.")
    require(0.0 <= params.rho < 1.0, "rho must satisfy 0 <= rho < 1, so that Q has a nonzero spectral floor.")
    require(params.ell >= 0.0, "ell must be nonnegative.")
    require(params.p > 0.0, "p must be positive.")
    require(params.t1 > params.t0, "t1 must be larger than t0.")
    require(params.dt > 0.0, "dt must be positive.")
    require(params.save_dt > 0.0, "save_dt must be positive.")
    require(params.save_dt >= params.dt, "save_dt must be at least dt.")
    require(isapprox(params.save_dt / params.dt, round(params.save_dt / params.dt); atol=1e-12),
            "save_dt must be an integer multiple of dt.")
    nsteps_float = (params.t1 - params.t0) / params.dt
    require(isapprox(nsteps_float, round(nsteps_float); atol=1e-10), "t1 - t0 must be an integer multiple of dt.")
    nsteps = round(Int, nsteps_float)
    save_every = round(Int, params.save_dt / params.dt)
    require(nsteps % save_every == 0, "The number of time steps must be divisible by save_every = save_dt / dt.")
    require(params.ntrajectories >= 1, "ntrajectories must be positive.")
    require(params.requested_threads >= 0, "requested_threads must be nonnegative. Use 0 to disable the check.")
    require(1 <= params.perturb_index <= params.K, "perturb_index must be between 1 and K.")
    require(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require(params.histogram_bins >= 16, "Use at least 16 histogram bins.")
    require(params.correlation_stride >= 1, "correlation_stride must be positive.")
    require(0.0 < params.decorrelation_threshold < 1.0, "decorrelation_threshold must lie in (0, 1).")
    require(params.max_decorrelation_time > 0.0, "max_decorrelation_time must be positive.")
    require(!isempty(params.cross_offsets), "cross_offsets must not be empty.")
    require(!isempty(params.bivariate_offsets), "bivariate_offsets must not be empty.")
    require(all(1 .<= params.cross_offsets) && all(params.cross_offsets .<= params.K - 1), "Every cross offset must be in 1:(K-1).")
    require(all(1 .<= params.bivariate_offsets) && all(params.bivariate_offsets .<= params.K - 1), "Every bivariate offset must be in 1:(K-1).")
    require(params.max_pdf_samples >= 10_000, "max_pdf_samples should be at least 10000.")
    require(params.figure_width >= 1600 && params.figure_height >= 1200, "Figure dimensions are too small.")
    return params
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? path : normpath(joinpath(base_dir, path))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function ensure_thread_count(params::SimL96Params)
    if params.requested_threads > 0
        actual = Threads.nthreads()
        require(actual == params.requested_threads,
            @sprintf("Expected %d Julia threads, found %d. Run with `julia --threads %d %s` or set requested_threads = 0.",
                     params.requested_threads, actual, params.requested_threads, basename(@__FILE__)))
    end
    return nothing
end

function periodic_index(i::Int, K::Int)
    return mod1(i, K)
end

function l96_drift!(drift::Vector{Float64}, x::Vector{Float64}, F::Float64)
    K = length(x)
    @inbounds for i in 1:K
        im2 = periodic_index(i - 2, K)
        im1 = periodic_index(i - 1, K)
        ip1 = periodic_index(i + 1, K)
        drift[i] = x[im1] * (x[ip1] - x[im2]) - x[i] + F
    end
    return nothing
end

function noise_spectrum(params::SimL96Params)
    K = params.K
    lambdas = [4.0 * sin(pi * m / K)^2 for m in 0:(K - 1)]
    envelope = [(1.0 + params.ell^2 * λ)^(-params.p) for λ in lambdas]
    envelope_mean = mean(envelope)
    q = params.sigma^2 .* ((1.0 - params.rho) .+ params.rho .* (envelope ./ envelope_mean))
    require(all(q .> 0.0), "Noise spectrum must be strictly positive.")
    return lambdas, q
end

function circulant_matrix_from_spectrum(spectrum::Vector{Float64})
    K = length(spectrum)
    mat = Matrix{Float64}(undef, K, K)
    @inbounds for i in 1:K
        for j in 1:K
            d = (i - 1) - (j - 1)
            value = 0.0
            for m in 0:(K - 1)
                value += spectrum[m + 1] * cos(2.0 * pi * m * d / K)
            end
            mat[i, j] = value / K
        end
    end
    return Symmetric(0.5 .* (mat .+ mat')) |> Matrix
end

function build_diffusion_matrices(params::SimL96Params)
    lambdas, q = noise_spectrum(params)
    Q = circulant_matrix_from_spectrum(q)
    Qsqrt = circulant_matrix_from_spectrum(sqrt.(q))
    residual = norm(Qsqrt * Qsqrt' - Q) / max(norm(Q), eps(Float64))
    require(residual < 1e-10, @sprintf("Qsqrt validation failed: relative residual %.3e", residual))
    return lambdas, q, Q, Qsqrt
end

function initial_condition(params::SimL96Params)
    x0 = fill(params.initial_mean, params.K)
    x0[params.perturb_index] += params.perturbation
    return x0
end

function integrate_l96_ensemble(params::SimL96Params, Qsqrt::Matrix{Float64})
    ensure_thread_count(params)

    nsteps = round(Int, (params.t1 - params.t0) / params.dt)
    save_every = round(Int, params.save_dt / params.dt)
    nsaved = nsteps ÷ save_every + 1
    times = collect(range(params.t0, step=params.save_dt, length=nsaved))
    states = Array{Float64}(undef, nsaved, params.K, params.ntrajectories)
    sqrt_2dt = sqrt(2.0 * params.dt)
    x_initial = initial_condition(params)

    @printf("Integrating stochastic Lorenz--96: K=%d, trajectories=%d, dt=%.5g, save_dt=%.5g, threads=%d\n",
            params.K, params.ntrajectories, params.dt, params.save_dt, Threads.nthreads())

    Threads.@threads for traj_idx in 1:params.ntrajectories
        rng = MersenneTwister(params.seed + traj_idx)
        x = copy(x_initial)
        drift = similar(x)
        ξ = similar(x)
        correlated_noise = similar(x)
        states[1, :, traj_idx] .= x
        save_idx = 2

        @inbounds for step in 1:nsteps
            l96_drift!(drift, x, params.F)
            randn!(rng, ξ)
            mul!(correlated_noise, Qsqrt, ξ)
            @. x = x + params.dt * drift + sqrt_2dt * correlated_noise

            if !all(isfinite, x)
                error(@sprintf("Non-finite state encountered in trajectory %d at step %d. Try reducing dt.", traj_idx, step))
            end

            if step % save_every == 0
                states[save_idx, :, traj_idx] .= x
                save_idx += 1
            end
        end
    end

    return times, states
end

function burnin_start_index(nsaved::Int, burnin_fraction::Float64)
    return clamp(1 + floor(Int, burnin_fraction * (nsaved - 1)), 1, nsaved)
end

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.05 * span, 1e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function draw_univariate_samples(states::Array{Float64, 3}, start_idx::Int, max_samples::Int, rng::AbstractRNG)
    nt, K, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    values = Vector{Float64}(undef, nsamples)

    if nsamples == total
        cursor = 1
        @inbounds for traj_idx in 1:ntraj
            for mode_idx in 1:K
                for time_idx in start_idx:nt
                    values[cursor] = states[time_idx, mode_idx, traj_idx]
                    cursor += 1
                end
            end
        end
    else
        @inbounds for sample_idx in 1:nsamples
            linear = rand(rng, 0:(total - 1))
            time_local = linear % npost
            tmp = linear ÷ npost
            mode_idx = (tmp % K) + 1
            traj_idx = (tmp ÷ K) + 1
            values[sample_idx] = states[start_idx + time_local, mode_idx, traj_idx]
        end
    end

    return values
end

function draw_pair_samples(states::Array{Float64, 3}, start_idx::Int, offset::Int, max_samples::Int, rng::AbstractRNG)
    nt, K, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    x_values = Vector{Float64}(undef, nsamples)
    y_values = Vector{Float64}(undef, nsamples)

    if nsamples == total
        cursor = 1
        @inbounds for traj_idx in 1:ntraj
            for mode_idx in 1:K
                paired_idx = periodic_index(mode_idx + offset, K)
                for time_idx in start_idx:nt
                    x_values[cursor] = states[time_idx, mode_idx, traj_idx]
                    y_values[cursor] = states[time_idx, paired_idx, traj_idx]
                    cursor += 1
                end
            end
        end
    else
        @inbounds for sample_idx in 1:nsamples
            linear = rand(rng, 0:(total - 1))
            time_local = linear % npost
            tmp = linear ÷ npost
            mode_idx = (tmp % K) + 1
            traj_idx = (tmp ÷ K) + 1
            paired_idx = periodic_index(mode_idx + offset, K)
            time_idx = start_idx + time_local
            x_values[sample_idx] = states[time_idx, mode_idx, traj_idx]
            y_values[sample_idx] = states[time_idx, paired_idx, traj_idx]
        end
    end

    return x_values, y_values
end

function compute_univariate_pdf(states::Array{Float64, 3}, start_idx::Int, bins::Int, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed + 10_001)
    samples = draw_univariate_samples(states, start_idx, max_samples, rng)
    boundary = kde_range(samples)
    kde_result = kde(samples; npoints=bins, boundary=boundary)
    return HistogramResult(collect(kde_result.x), collect(kde_result.density))
end

function compute_pair_pdf(states::Array{Float64, 3}, start_idx::Int, offset::Int, bins::Int, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed + 20_003 + offset)
    x_values, y_values = draw_pair_samples(states, start_idx, offset, max_samples, rng)
    x_boundary = kde_range(x_values)
    y_boundary = kde_range(y_values)
    kde_result = kde((x_values, y_values); npoints=(bins, bins), boundary=(x_boundary, y_boundary))
    return PairPdfResult(offset, collect(kde_result.x), collect(kde_result.y), Array(kde_result.density))
end

function tensor_mean_and_variance(data::Array{Float64, 3})
    total = 0.0
    count = 0
    @inbounds for traj_idx in axes(data, 3)
        for mode_idx in axes(data, 2)
            for time_idx in axes(data, 1)
                total += data[time_idx, mode_idx, traj_idx]
                count += 1
            end
        end
    end
    mean_value = total / count

    sumsq = 0.0
    @inbounds for traj_idx in axes(data, 3)
        for mode_idx in axes(data, 2)
            for time_idx in axes(data, 1)
                δ = data[time_idx, mode_idx, traj_idx] - mean_value
                sumsq += δ * δ
            end
        end
    end
    return mean_value, sumsq / count
end

function estimate_decorrelation_time(lags::Vector{Float64}, acf::Vector{Float64}, threshold::Float64)
    n = length(lags)
    envelope = abs.(acf)
    running_max = similar(envelope)
    running_max[end] = envelope[end]
    @inbounds for i in (n - 1):-1:1
        running_max[i] = max(envelope[i], running_max[i + 1])
    end
    for i in 2:n
        if running_max[i] <= threshold
            return lags[i]
        end
    end
    return lags[end]
end

function downsample_states(states::Array{Float64, 3}, start_idx::Int, stride::Int)
    time_indices = collect(start_idx:stride:size(states, 1))
    return Array(@view states[time_indices, :, :])
end

function compute_lattice_correlations(data::Array{Float64, 3}, dt_corr::Float64,
                                      max_time::Float64, threshold::Float64,
                                      cross_offsets::Vector{Int})
    ntime, K, ntraj = size(data)
    mean_value, variance_value = tensor_mean_and_variance(data)
    require(variance_value > 0.0, "Cannot normalize correlations because the empirical variance is zero.")

    max_lag = min(ntime - 1, floor(Int, max_time / dt_corr))
    require(max_lag >= 1, "The correlation window is empty. Increase max_decorrelation_time or reduce correlation_stride.")
    lags = collect(0:max_lag) .* dt_corr
    acf_mean = zeros(Float64, max_lag + 1)
    cross_mean = zeros(Float64, max_lag + 1, length(cross_offsets))

    Threads.@threads for lag in 0:max_lag
        sum_acf = 0.0
        sum_cross = zeros(Float64, length(cross_offsets))
        count = 0

        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for time_idx in 1:upper
                future_time_idx = time_idx + lag
                for mode_idx in 1:K
                    x0 = data[time_idx, mode_idx, traj_idx] - mean_value
                    x_same = data[future_time_idx, mode_idx, traj_idx] - mean_value
                    sum_acf += x_same * x0
                    for (offset_idx, offset) in enumerate(cross_offsets)
                        paired_idx = periodic_index(mode_idx + offset, K)
                        x_pair = data[future_time_idx, paired_idx, traj_idx] - mean_value
                        sum_cross[offset_idx] += x_pair * x0
                    end
                    count += 1
                end
            end
        end

        acf_mean[lag + 1] = sum_acf / (count * variance_value)
        for offset_idx in eachindex(cross_offsets)
            cross_mean[lag + 1, offset_idx] = sum_cross[offset_idx] / (count * variance_value)
        end
    end

    t_decorrelation = estimate_decorrelation_time(lags, acf_mean, threshold)
    return CorrelationResult(lags, acf_mean, copy(cross_offsets), cross_mean,
                             t_decorrelation, mean_value, variance_value)
end

function truncate_series(xs::Vector{Float64}, ys::Vector{Float64}, xmax::Float64)
    if isempty(xs)
        return xs, ys
    end
    last_idx = searchsortedlast(xs, xmax)
    last_idx = max(last_idx, 2)
    last_idx = min(last_idx, length(xs))
    return xs[1:last_idx], ys[1:last_idx]
end

function save_hdf5(path::AbstractString, params::SimL96Params, times::Vector{Float64}, states::Array{Float64, 3},
                   lambdas::Vector{Float64}, q::Vector{Float64}, Q::Matrix{Float64}, Qsqrt::Matrix{Float64},
                   hist::HistogramResult, pair_pdfs::Vector{PairPdfResult}, corr::CorrelationResult)
    h5open(path, "w") do file
        file["/trajectories/time"] = times
        file["/trajectories/states"] = states

        file["/statistics/pdf/univariate_centers"] = hist.centers
        file["/statistics/pdf/univariate_density"] = hist.density
        file["/statistics/pdf/bivariate_offsets"] = [pdf.offset for pdf in pair_pdfs]
        for pdf in pair_pdfs
            base = @sprintf("/statistics/pdf/bivariate/offset_%d", pdf.offset)
            file[string(base, "/x_grid")] = pdf.x_grid
            file[string(base, "/y_grid")] = pdf.y_grid
            file[string(base, "/density")] = pdf.density
        end

        file["/statistics/correlations/lags"] = corr.lags
        file["/statistics/correlations/acf_mean"] = corr.acf_mean
        file["/statistics/correlations/cross_offsets"] = corr.cross_offsets
        file["/statistics/correlations/cross_mean"] = corr.cross_mean
        file["/statistics/correlations/t_decorrelation"] = corr.t_decorrelation
        file["/statistics/correlations/global_mean"] = corr.mean_value
        file["/statistics/correlations/global_variance"] = corr.variance_value

        file["/diffusion/laplacian_eigenvalues"] = lambdas
        file["/diffusion/noise_spectrum"] = q
        file["/diffusion/Q"] = Q
        file["/diffusion/Qsqrt"] = Qsqrt

        file["/metadata/model_name"] = "stochastic_lorenz96_additive_translation_invariant_correlated_noise"
        file["/metadata/K"] = params.K
        file["/metadata/F"] = params.F
        file["/metadata/sigma"] = params.sigma
        file["/metadata/rho"] = params.rho
        file["/metadata/ell"] = params.ell
        file["/metadata/p"] = params.p
        file["/metadata/dt"] = params.dt
        file["/metadata/save_dt"] = params.save_dt
        file["/metadata/ntrajectories"] = params.ntrajectories
        file["/metadata/burnin_fraction"] = params.burnin_fraction
        file["/metadata/histogram_bins"] = params.histogram_bins
        file["/metadata/correlation_stride"] = params.correlation_stride
        file["/metadata/max_pdf_samples"] = params.max_pdf_samples
        file["/metadata/sde_noise_prefactor"] = "sqrt(2) * Qsqrt"
    end
    return nothing
end

function heatmap_panel!(parent, x, y, z; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString,
                        colorbar_label::AbstractString="", colormap=STYLE_SEQUENTIAL_BLUE)
    layout = GridLayout(parent)
    ax = Axis(layout[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)
    hm = heatmap!(ax, x, y, z; colormap=colormap)
    Colorbar(layout[1, 2], hm; label=colorbar_label)
    return ax
end

function render_summary_figure(path::AbstractString, params::SimL96Params,
                               q::Vector{Float64}, Q::Matrix{Float64},
                               hist::HistogramResult, pair_pdfs::Vector{PairPdfResult}, corr::CorrelationResult)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    subtitle = @sprintf("K=%d   F=%.2f   σ=%.2f   ρ=%.2f   ℓ=%.2f   p=%.2f   dt=%.4f   save_dt=%.4f   t_decorr=%.2f   N_traj=%d",
                        params.K, params.F, params.sigma, params.rho, params.ell, params.p,
                        params.dt, params.save_dt, corr.t_decorrelation, params.ntrajectories)
    figure_title!(fig, "Stochastic Lorenz--96 — observational summary"; subtitle=subtitle)

    ax_pdf = Axis(fig[1, 1]; title="Translation-averaged univariate PDF", xlabel="x_i", ylabel="density")
    lines!(ax_pdf, hist.centers, hist.density; color=STYLE_PRIMARY, linewidth=3)

    corr_t_max = min(max(10.0 * corr.t_decorrelation, params.save_dt), corr.lags[end])
    lags_acf, acf_mean = truncate_series(corr.lags, corr.acf_mean, corr_t_max)
    ax_acf = Axis(fig[1, 2]; title="Translation-averaged autocorrelation", xlabel="lag τ", ylabel="C_0(τ)")
    hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_acf, lags_acf, acf_mean; color=STYLE_ACCENT, linewidth=3)
    xlims!(ax_acf, 0.0, corr_t_max)

    ax_cross = Axis(fig[1, 3]; title="Average shifted cross-correlations", xlabel="lag τ", ylabel="C_r(τ)")
    hlines!(ax_cross, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    for (offset_idx, offset) in enumerate(corr.cross_offsets)
        lags_cross, cross_values = truncate_series(corr.lags, corr.cross_mean[:, offset_idx], corr_t_max)
        lines!(ax_cross, lags_cross, cross_values; linewidth=2, label=@sprintf("r=%d", offset))
    end
    xlims!(ax_cross, 0.0, corr_t_max)
    axislegend(ax_cross; position=:rt, nbanks=2)

    heatmap_panel!(fig[2, 1], collect(1:params.K), collect(1:params.K), Q;
                   title="Diffusion tensor Q", xlabel="j", ylabel="i",
                   colorbar_label="Qᵢⱼ", colormap=STYLE_DIVERGING)

    ax_spec = Axis(fig[2, 2]; title="Noise spectrum", xlabel="Fourier mode m", ylabel="q_m")
    lines!(ax_spec, collect(0:(params.K - 1)), q; color=STYLE_SECONDARY, linewidth=3)
    scatter!(ax_spec, collect(0:(params.K - 1)), q; color=STYLE_SECONDARY, markersize=6)

    max_pair_panels = min(length(pair_pdfs), 4)
    panel_positions = [(2, 3), (3, 1), (3, 2), (3, 3)]
    for local_idx in 1:max_pair_panels
        pdf = pair_pdfs[local_idx]
        row, col = panel_positions[local_idx]
        heatmap_panel!(fig[row, col], pdf.x_grid, pdf.y_grid, pdf.density;
                       title=@sprintf("Avg. bivariate PDF: (x_i, x_{i+%d})", pdf.offset),
                       xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pdf.offset),
                       colorbar_label="density", colormap=STYLE_SEQUENTIAL_BLUE)
    end

    save_figure(path, fig)
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    output_hdf5 = resolve_path(base_dir, params.output_hdf5)
    output_png = resolve_path(base_dir, params.output_png)
    ensure_parent_dir(output_hdf5)
    ensure_parent_dir(output_png)

    lambdas, q, Q, Qsqrt = build_diffusion_matrices(params)
    @printf("Noise normalization: tr(Q)/K = %.8f, target sigma^2 = %.8f\n", tr(Q) / params.K, params.sigma^2)

    times, states = integrate_l96_ensemble(params, Qsqrt)
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    @printf("Using saved index %d/%d as the beginning of the stationary diagnostic window.\n", start_idx, length(times))

    hist = compute_univariate_pdf(states, start_idx, params.histogram_bins, params.max_pdf_samples, params.seed)
    pair_pdfs = [compute_pair_pdf(states, start_idx, offset, params.histogram_bins,
                                  params.max_pdf_samples, params.seed) for offset in params.bivariate_offsets]

    data_corr = downsample_states(states, start_idx, params.correlation_stride)
    dt_corr = params.save_dt * params.correlation_stride
    corr = compute_lattice_correlations(data_corr, dt_corr, params.max_decorrelation_time,
                                        params.decorrelation_threshold, params.cross_offsets)

    @printf("Writing HDF5 output to %s\n", output_hdf5)
    save_hdf5(output_hdf5, params, times, states, lambdas, q, Q, Qsqrt, hist, pair_pdfs, corr)

    @printf("Rendering summary figure to %s\n", output_png)
    render_summary_figure(output_png, params, q, Q, hist, pair_pdfs, corr)

    @printf("Completed. Estimated decorrelation time tD = %.6f\n", corr.t_decorrelation)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
