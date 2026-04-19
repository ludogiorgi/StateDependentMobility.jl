#!/usr/bin/env julia

# Agent summary:
# - Simulates the two-dimensional nonreversible overdamped Langevin benchmark with affine multiplicative noise and writes the HDF5 dataset used by downstream score-training scripts.
# - Also computes baseline observational diagnostics stored alongside the trajectories.

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
using Printf
using Random
using Statistics
using TOML

include(joinpath(@__DIR__, "src", "figure_style.jl"))

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "sim.toml")

struct SimParams
    omega::Float64
    B0::Matrix{Float64}
    B1::Matrix{Float64}
    B2::Matrix{Float64}
    t0::Float64
    t1::Float64
    dt::Float64
    save_dt::Float64
    ntrajectories::Int
    requested_threads::Int
    seed::Int
    x0::Float64
    y0::Float64
    burnin_fraction::Float64
    histogram_bins::Int
    correlation_stride::Int
    decorrelation_threshold::Float64
    max_decorrelation_time::Float64
    figure_width::Int
    figure_height::Int
    output_hdf5::String
    output_png::String
end

struct HistogramResult
    centers::Vector{Float64}
    density::Vector{Float64}
end

struct BivariateHistogramResult
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    density::Matrix{Float64}
end

struct CorrelationResult
    lags::Vector{Float64}
    acf_x::Vector{Float64}
    acf_y::Vector{Float64}
    cross_xy::Vector{Float64}
    cross_yx::Vector{Float64}
    t_decorrelation::Float64
    mean_x::Float64
    mean_y::Float64
    var_x::Float64
    var_y::Float64
end

function require(condition::Bool, message::String)
    condition || error(message)
    return nothing
end

function parse_matrix_2x2(raw, label::String)
    require(length(raw) == 2, "$(label) must have 2 rows.")
    row1 = Float64.(raw[1])
    row2 = Float64.(raw[2])
    require(length(row1) == 2 && length(row2) == 2, "$(label) must be 2x2.")
    return [row1[1] row1[2]; row2[1] row2[2]]
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    sim = raw["simulation"]
    stats = raw["statistics"]
    fig = raw["figure"]
    out = raw["output"]

    params = SimParams(
        Float64(sim["omega"]),
        parse_matrix_2x2(sim["B0"], "B0"),
        parse_matrix_2x2(sim["B1"], "B1"),
        parse_matrix_2x2(sim["B2"], "B2"),
        Float64(sim["t0"]),
        Float64(sim["t1"]),
        Float64(sim["dt"]),
        Float64(sim["save_dt"]),
        Int(sim["ntrajectories"]),
        Int(sim["requested_threads"]),
        Int(sim["seed"]),
        Float64(sim["x0"]),
        Float64(sim["y0"]),
        Float64(stats["burnin_fraction"]),
        Int(stats["histogram_bins"]),
        Int(stats["correlation_stride"]),
        Float64(stats["decorrelation_threshold"]),
        Float64(stats["max_decorrelation_time"]),
        Int(fig["width"]),
        Int(fig["height"]),
        String(out["hdf5_file"]),
        String(out["figure_png"]),
    )

    require(params.t1 > params.t0, "t1 must be larger than t0.")
    require(params.omega > 0.0, "omega must be positive.")
    require(params.dt > 0.0, "dt must be positive.")
    require(params.save_dt > 0.0, "save_dt must be positive.")
    require(params.save_dt >= params.dt, "save_dt must be at least dt.")
    require(isapprox(params.save_dt / params.dt, round(params.save_dt / params.dt); atol=1e-12), "save_dt must be an integer multiple of dt.")
    require(params.ntrajectories == 36, "This script is configured for a 36-member ensemble as requested.")
    require(params.requested_threads == 36, "requested_threads must be 36.")
    require(0.0 <= params.burnin_fraction < 1.0, "burnin_fraction must be in [0, 1).")
    require(params.histogram_bins >= 16, "Use at least 16 histogram bins.")
    require(params.correlation_stride >= 1, "correlation_stride must be positive.")
    require(params.max_decorrelation_time > 0.0, "max_decorrelation_time must be positive.")
    require(params.figure_width >= 1200 && params.figure_height >= 800, "Figure dimensions are too small.")
    return params
end

function resolve_path(base_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? path : normpath(joinpath(base_dir, path))
end

function ensure_parent_dir(path::AbstractString)
    mkpath(dirname(path))
    return nothing
end

function ensure_thread_count(params::SimParams)
    actual = Threads.nthreads()
    require(actual == params.requested_threads,
        @sprintf("Expected %d Julia threads, found %d. Run with `julia --threads 36 ...`.", params.requested_threads, actual))
    return nothing
end

function affine_noise_matrix(x::Float64, y::Float64, params::SimParams)
    return params.B0 .+ x .* params.B1 .+ y .* params.B2
end

function affine_langevin_drift(x::Float64, y::Float64, omega::Float64)
    fx = -x^3 - x * y^2 - x - omega * y
    fy = -y^3 - x^2 * y - y + omega * x
    return fx, fy
end

function integrate_affine_langevin_ensemble(params::SimParams)
    ensure_thread_count(params)

    nsteps = round(Int, (params.t1 - params.t0) / params.dt)
    save_every = round(Int, params.save_dt / params.dt)
    nsaved = nsteps ÷ save_every + 1
    times = collect(range(params.t0, step=params.save_dt, length=nsaved))
    trajectories = Array{Float64}(undef, nsaved, 2, params.ntrajectories)
    sqrt_dt = sqrt(params.dt)

    @printf("Integrating %d trajectories on %d threads...\n", params.ntrajectories, Threads.nthreads())

    Threads.@threads for traj_idx in 1:params.ntrajectories
        rng = MersenneTwister(params.seed + traj_idx)
        x = params.x0
        y = params.y0
        trajectories[1, 1, traj_idx] = x
        trajectories[1, 2, traj_idx] = y
        save_idx = 2

        @inbounds for step in 1:nsteps
            fx, fy = affine_langevin_drift(x, y, params.omega)
            bmat = affine_noise_matrix(x, y, params)
            ξ1 = randn(rng)
            ξ2 = randn(rng)
            x += fx * params.dt + sqrt_dt * (bmat[1, 1] * ξ1 + bmat[1, 2] * ξ2)
            y += fy * params.dt + sqrt_dt * (bmat[2, 1] * ξ1 + bmat[2, 2] * ξ2)

            if step % save_every == 0
                trajectories[save_idx, 1, traj_idx] = x
                trajectories[save_idx, 2, traj_idx] = y
                save_idx += 1
            end
        end
    end

    return times, trajectories
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

function collect_pdf_samples(data::Array{Float64, 3}, start_idx::Int)
    nsamples = (size(data, 1) - start_idx + 1) * size(data, 3)
    x = Vector{Float64}(undef, nsamples)
    y = Vector{Float64}(undef, nsamples)
    cursor = 1

    @inbounds for traj_idx in axes(data, 3)
        for time_idx in start_idx:size(data, 1)
            x[cursor] = data[time_idx, 1, traj_idx]
            y[cursor] = data[time_idx, 2, traj_idx]
            cursor += 1
        end
    end

    return x, y
end

function compute_histogram(data::Array{Float64, 3}, coord::Int, start_idx::Int, bins::Int)
    samples, other = collect_pdf_samples(data, start_idx)
    values = coord == 1 ? samples : other
    boundary = kde_range(values)
    kde_result = kde(values; npoints=bins, boundary=boundary)
    return HistogramResult(collect(kde_result.x), collect(kde_result.density))
end

function compute_bivariate_histogram(data::Array{Float64, 3}, start_idx::Int, bins::Int)
    x, y = collect_pdf_samples(data, start_idx)
    x_boundary = kde_range(x)
    y_boundary = kde_range(y)
    kde_result = kde((x, y); npoints=(bins, bins), boundary=(x_boundary, y_boundary))
    return BivariateHistogramResult(collect(kde_result.x), collect(kde_result.y), Array(kde_result.density))
end

function downsample_coordinates(data::Array{Float64, 3}, start_idx::Int, stride::Int)
    time_indices = collect(start_idx:stride:size(data, 1))
    ntime = length(time_indices)
    ntraj = size(data, 3)
    x = Array{Float64}(undef, ntime, ntraj)
    y = Array{Float64}(undef, ntime, ntraj)
    @inbounds for traj_idx in 1:ntraj
        for (local_idx, global_idx) in enumerate(time_indices)
            x[local_idx, traj_idx] = data[global_idx, 1, traj_idx]
            y[local_idx, traj_idx] = data[global_idx, 2, traj_idx]
        end
    end
    return time_indices, x, y
end

function matrix_mean_and_variance(data::Matrix{Float64})
    total = 0.0
    count = 0
    @inbounds for j in axes(data, 2)
        for i in axes(data, 1)
            total += data[i, j]
            count += 1
        end
    end
    mean_value = total / count

    sumsq = 0.0
    @inbounds for j in axes(data, 2)
        for i in axes(data, 1)
            delta = data[i, j] - mean_value
            sumsq += delta * delta
        end
    end
    return mean_value, sumsq / count
end

function estimate_decorrelation_time(lags::Vector{Float64}, acf_x::Vector{Float64}, acf_y::Vector{Float64}, threshold::Float64)
    n = length(lags)
    envelope = Vector{Float64}(undef, n)
    running_max = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        envelope[i] = max(abs(acf_x[i]), abs(acf_y[i]))
    end

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

function compute_correlations(x::Matrix{Float64}, y::Matrix{Float64}, dt_corr::Float64, max_time::Float64, threshold::Float64)
    ntime, ntraj = size(x)
    mean_x, var_x = matrix_mean_and_variance(x)
    mean_y, var_y = matrix_mean_and_variance(y)
    denom_xy = sqrt(var_x * var_y)
    max_lag = min(ntime - 1, floor(Int, max_time / dt_corr))

    acf_x = zeros(Float64, max_lag + 1)
    acf_y = zeros(Float64, max_lag + 1)
    cross_xy = zeros(Float64, max_lag + 1)
    cross_yx = zeros(Float64, max_lag + 1)

    Threads.@threads for lag in 0:max_lag
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0
        sum_yx = 0.0
        count = 0

        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for i in 1:upper
                x0 = x[i, traj_idx] - mean_x
                y0 = y[i, traj_idx] - mean_y
                x1 = x[i + lag, traj_idx] - mean_x
                y1 = y[i + lag, traj_idx] - mean_y
                sum_xx += x1 * x0
                sum_yy += y1 * y0
                sum_xy += x1 * y0
                sum_yx += y1 * x0
            end
            count += upper
        end

        acf_x[lag + 1] = sum_xx / (count * var_x)
        acf_y[lag + 1] = sum_yy / (count * var_y)
        cross_xy[lag + 1] = sum_xy / (count * denom_xy)
        cross_yx[lag + 1] = sum_yx / (count * denom_xy)
    end

    lags = collect(0:max_lag) .* dt_corr
    t_decorrelation = estimate_decorrelation_time(lags, acf_x, acf_y, threshold)
    return CorrelationResult(lags, acf_x, acf_y, cross_xy, cross_yx, t_decorrelation, mean_x, mean_y, var_x, var_y)
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

function save_hdf5(path::AbstractString, params::SimParams, times::Vector{Float64}, trajectories::Array{Float64, 3},
                   hist_x::HistogramResult, hist_y::HistogramResult, hist_xy::BivariateHistogramResult, corr::CorrelationResult)
    h5open(path, "w") do file
        file["/trajectories/time"] = times
        file["/trajectories/states"] = trajectories

        file["/statistics/pdf/x_centers"] = hist_x.centers
        file["/statistics/pdf/x_density"] = hist_x.density
        file["/statistics/pdf/y_centers"] = hist_y.centers
        file["/statistics/pdf/y_density"] = hist_y.density
        file["/statistics/pdf/xy_x_grid"] = hist_xy.x_grid
        file["/statistics/pdf/xy_y_grid"] = hist_xy.y_grid
        file["/statistics/pdf/xy_density"] = hist_xy.density

        file["/statistics/correlations/lags"] = corr.lags
        file["/statistics/correlations/acf_x"] = corr.acf_x
        file["/statistics/correlations/acf_y"] = corr.acf_y
        file["/statistics/correlations/cross_xy"] = corr.cross_xy
        file["/statistics/correlations/cross_yx"] = corr.cross_yx
        file["/statistics/correlations/t_decorrelation"] = corr.t_decorrelation

        file["/metadata/model_name"] = "affine_langevin_2d"
        file["/metadata/omega"] = params.omega
        file["/metadata/B0"] = params.B0
        file["/metadata/B1"] = params.B1
        file["/metadata/B2"] = params.B2
        file["/metadata/dt"] = params.dt
        file["/metadata/save_dt"] = params.save_dt
        file["/metadata/ntrajectories"] = float(params.ntrajectories)
    end
    return nothing
end

function render_summary_figure(path::AbstractString, params::SimParams, hist_x::HistogramResult, hist_y::HistogramResult,
                               hist_xy::BivariateHistogramResult, corr::CorrelationResult)
    fig = Figure(; size=(params.figure_width, params.figure_height))
    subtitle = @sprintf("ω=%.2f   dt=%.4f   save_dt=%.4f   t_decorr=%.2f   N_traj=%d",
        params.omega, params.dt, params.save_dt, corr.t_decorrelation, params.ntrajectories)
    figure_title!(fig, "Affine Langevin 2D — observational summary"; subtitle=subtitle)

    ax_px = Axis(fig[1, 1]; title="Marginal PDF: X", xlabel="x", ylabel="density")
    lines!(ax_px, hist_x.centers, hist_x.density; color=STYLE_PRIMARY)

    ax_py = Axis(fig[1, 2]; title="Marginal PDF: Y", xlabel="y", ylabel="density")
    lines!(ax_py, hist_y.centers, hist_y.density; color=STYLE_SECONDARY)

    gl_xy = GridLayout(fig[1, 3])
    ax_xy = Axis(gl_xy[1, 1]; title="Joint PDF (x, y)", xlabel="x", ylabel="y")
    hm = heatmap!(ax_xy, hist_xy.x_grid, hist_xy.y_grid, hist_xy.density;
        colormap=STYLE_SEQUENTIAL_BLUE)
    Colorbar(gl_xy[1, 2], hm; label="density")

    corr_t_max = min(max(10.0 * corr.t_decorrelation, params.save_dt), corr.lags[end])
    lags_acf_x, acf_x = truncate_series(corr.lags, corr.acf_x, corr_t_max)
    lags_acf_y, acf_y = truncate_series(corr.lags, corr.acf_y, corr_t_max)
    lags_xy_t, cross_xy = truncate_series(corr.lags, corr.cross_xy, corr_t_max)
    _, cross_yx = truncate_series(corr.lags, corr.cross_yx, corr_t_max)

    ax_acf_x = Axis(fig[2, 1]; title="Autocorrelation: X", xlabel="lag t", ylabel="C_xx(t)")
    hlines!(ax_acf_x, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_acf_x, lags_acf_x, acf_x; color=STYLE_ACCENT)
    xlims!(ax_acf_x, 0.0, corr_t_max)

    ax_acf_y = Axis(fig[2, 2]; title="Autocorrelation: Y", xlabel="lag t", ylabel="C_yy(t)")
    hlines!(ax_acf_y, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_acf_y, lags_acf_y, acf_y; color=STYLE_HIGHLIGHT)
    xlims!(ax_acf_y, 0.0, corr_t_max)

    ax_cross = Axis(fig[2, 3]; title="Cross-correlation", xlabel="lag t", ylabel="C(t)")
    hlines!(ax_cross, [0.0]; color=STYLE_ZERO, linewidth=1.0, linestyle=:dash)
    lines!(ax_cross, lags_xy_t, cross_xy; color=STYLE_PRIMARY, label="C_xy")
    lines!(ax_cross, lags_xy_t, cross_yx; color=STYLE_VIOLET, label="C_yx")
    xlims!(ax_cross, 0.0, corr_t_max)
    axislegend(ax_cross; position=:rt)

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

    times, trajectories = integrate_affine_langevin_ensemble(params)
    start_idx = burnin_start_index(length(times), params.burnin_fraction)
    hist_x = compute_histogram(trajectories, 1, start_idx, params.histogram_bins)
    hist_y = compute_histogram(trajectories, 2, start_idx, params.histogram_bins)
    hist_xy = compute_bivariate_histogram(trajectories, start_idx, params.histogram_bins)
    _, x_down, y_down = downsample_coordinates(trajectories, start_idx, params.correlation_stride)
    dt_corr = params.save_dt * params.correlation_stride
    corr = compute_correlations(x_down, y_down, dt_corr, params.max_decorrelation_time, params.decorrelation_threshold)

    @printf("Writing HDF5 output to %s\n", output_hdf5)
    save_hdf5(output_hdf5, params, times, trajectories, hist_x, hist_y, hist_xy, corr)

    @printf("Rendering summary figure to %s\n", output_png)
    render_summary_figure(output_png, params, hist_x, hist_y, hist_xy, corr)

    @printf("Completed. Estimated decorrelation time tD = %.6f\n", corr.t_decorrelation)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
