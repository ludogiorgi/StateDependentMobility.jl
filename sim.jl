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

ensure_packages(["KernelDensity", "HDF5"])

using HDF5
using Libdl
using KernelDensity
using Printf
using Random
using Statistics
using TOML

const DEFAULT_PARAM_FILE = joinpath(@__DIR__, "sim.toml")
const LIBZ_CANDIDATES = ["libz.so.1", "libz.so"]

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

mutable struct Canvas
    width::Int
    height::Int
    pixels::Array{UInt8, 3}
end

const COLOR_BLACK = (0x18, 0x18, 0x18)
const COLOR_WHITE = (0xff, 0xff, 0xff)
const COLOR_GRAY = (0x74, 0x74, 0x74)
const COLOR_LIGHT_GRAY = (0xd8, 0xd8, 0xd8)
const COLOR_BLUE = (0x1f, 0x77, 0xb4)
const COLOR_ORANGE = (0xe6, 0x7e, 0x22)
const COLOR_GREEN = (0x2e, 0x86, 0x59)
const COLOR_RED = (0xc0, 0x39, 0x2b)
const COLOR_TEAL = (0x17, 0xa5, 0xa3)
const COLOR_PURPLE = (0x7d, 0x3c, 0x98)
const COLOR_PANEL = (0xf6, 0xf6, 0xf4)

const FONT_5X7 = Dict{Char, NTuple{7, UInt8}}(
    ' ' => (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    '-' => (0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00),
    '.' => (0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c),
    ':' => (0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x00),
    '=' => (0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00, 0x00),
    '?' => (0x1e, 0x01, 0x01, 0x0e, 0x08, 0x00, 0x08),
    '0' => (0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e),
    '1' => (0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e),
    '2' => (0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f),
    '3' => (0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e),
    '4' => (0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02),
    '5' => (0x1f, 0x10, 0x10, 0x1e, 0x01, 0x01, 0x1e),
    '6' => (0x0e, 0x10, 0x10, 0x1e, 0x11, 0x11, 0x0e),
    '7' => (0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    '8' => (0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e),
    '9' => (0x0e, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x0e),
    'A' => (0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11),
    'B' => (0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e),
    'C' => (0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e),
    'D' => (0x1c, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1c),
    'E' => (0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f),
    'F' => (0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10),
    'G' => (0x0e, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0f),
    'H' => (0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11),
    'I' => (0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e),
    'J' => (0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0e),
    'K' => (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    'L' => (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f),
    'M' => (0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11),
    'N' => (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    'O' => (0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e),
    'P' => (0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10),
    'Q' => (0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d),
    'R' => (0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11),
    'S' => (0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e),
    'T' => (0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    'U' => (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e),
    'V' => (0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04),
    'W' => (0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a),
    'X' => (0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11),
    'Y' => (0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04),
    'Z' => (0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f),
)

function resolve_library(candidates::Vector{String}, label::String)
    for name in candidates
        handle = Libdl.dlopen_e(name)
        if handle !== nothing
            Libdl.dlclose(handle)
            return name
        end
    end
    error("Could not find $(label). Checked: $(join(candidates, ", "))")
end

const LIBZ = resolve_library(LIBZ_CANDIDATES, "a zlib shared library")

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

function grid_edges(grid::Vector{Float64})
    if length(grid) <= 1
        return [grid[1] - 0.5, grid[1] + 0.5]
    end

    edges = Vector{Float64}(undef, length(grid) + 1)
    edges[1] = grid[1] - 0.5 * (grid[2] - grid[1])
    for i in 1:(length(grid) - 1)
        edges[i + 1] = 0.5 * (grid[i] + grid[i + 1])
    end
    edges[end] = grid[end] + 0.5 * (grid[end] - grid[end - 1])
    return edges
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

function Canvas(width::Int, height::Int; background=COLOR_WHITE)
    pixels = Array{UInt8}(undef, height, width, 3)
    canvas = Canvas(width, height, pixels)
    fill_canvas!(canvas, background)
    return canvas
end

function fill_canvas!(canvas::Canvas, color)
    @inbounds for y in 1:canvas.height, x in 1:canvas.width
        canvas.pixels[y, x, 1] = UInt8(color[1])
        canvas.pixels[y, x, 2] = UInt8(color[2])
        canvas.pixels[y, x, 3] = UInt8(color[3])
    end
    return nothing
end

function set_pixel!(canvas::Canvas, x::Int, y::Int, color)
    if 1 <= x <= canvas.width && 1 <= y <= canvas.height
        @inbounds begin
            canvas.pixels[y, x, 1] = UInt8(color[1])
            canvas.pixels[y, x, 2] = UInt8(color[2])
            canvas.pixels[y, x, 3] = UInt8(color[3])
        end
    end
    return nothing
end

function fill_rect!(canvas::Canvas, x::Int, y::Int, w::Int, h::Int, color)
    x1 = clamp(x, 1, canvas.width)
    x2 = clamp(x + w - 1, 1, canvas.width)
    y1 = clamp(y, 1, canvas.height)
    y2 = clamp(y + h - 1, 1, canvas.height)
    @inbounds for yy in y1:y2, xx in x1:x2
        canvas.pixels[yy, xx, 1] = UInt8(color[1])
        canvas.pixels[yy, xx, 2] = UInt8(color[2])
        canvas.pixels[yy, xx, 3] = UInt8(color[3])
    end
    return nothing
end

function stroke_rect!(canvas::Canvas, x::Int, y::Int, w::Int, h::Int, color)
    draw_line!(canvas, x, y, x + w - 1, y, color)
    draw_line!(canvas, x, y, x, y + h - 1, color)
    draw_line!(canvas, x + w - 1, y, x + w - 1, y + h - 1, color)
    draw_line!(canvas, x, y + h - 1, x + w - 1, y + h - 1, color)
    return nothing
end

function draw_line!(canvas::Canvas, x0::Int, y0::Int, x1::Int, y1::Int, color)
    dx = abs(x1 - x0)
    sx = x0 < x1 ? 1 : -1
    dy = -abs(y1 - y0)
    sy = y0 < y1 ? 1 : -1
    err = dx + dy
    x = x0
    y = y0

    while true
        set_pixel!(canvas, x, y, color)
        x == x1 && y == y1 && break
        e2 = 2 * err
        if e2 >= dy
            err += dy
            x += sx
        end
        if e2 <= dx
            err += dx
            y += sy
        end
    end
    return nothing
end

function draw_disc!(canvas::Canvas, cx::Int, cy::Int, radius::Int, color)
    r2 = radius * radius
    for dy in -radius:radius, dx in -radius:radius
        if dx * dx + dy * dy <= r2
            set_pixel!(canvas, cx + dx, cy + dy, color)
        end
    end
    return nothing
end

function draw_glyph!(canvas::Canvas, x::Int, y::Int, ch::Char, color; scale::Int=1)
    glyph = get(FONT_5X7, uppercase(ch), FONT_5X7['?'])
    @inbounds for row in 1:7
        bits = glyph[row]
        for col in 1:5
            mask = UInt8(1 << (5 - col))
            if bits & mask != 0
                fill_rect!(canvas, x + (col - 1) * scale, y + (row - 1) * scale, scale, scale, color)
            end
        end
    end
    return nothing
end

function draw_text!(canvas::Canvas, x::Int, y::Int, text::AbstractString, color; scale::Int=1)
    cursor = x
    for ch in text
        draw_glyph!(canvas, cursor, y, ch, color; scale=scale)
        cursor += 6 * scale
    end
    return nothing
end

function nice_ticks(vmin::Float64, vmax::Float64, n::Int)
    if isapprox(vmin, vmax; atol=1e-12)
        delta = abs(vmin) > 1e-12 ? 0.5 * abs(vmin) : 0.5
        vmin -= delta
        vmax += delta
    end
    step = (vmax - vmin) / max(n - 1, 1)
    return [vmin + (i - 1) * step for i in 1:n]
end

function format_tick(value::Float64)
    if abs(value) >= 100.0
        return @sprintf("%.0f", value)
    elseif abs(value) >= 10.0
        return @sprintf("%.1f", value)
    elseif abs(value) >= 1.0
        return @sprintf("%.2f", value)
    else
        return @sprintf("%.3f", value)
    end
end

function plot_axes!(canvas::Canvas, panel_x::Int, panel_y::Int, panel_w::Int, panel_h::Int,
                    x_values::Vector{Float64}, y_values::Vector{Float64};
                    title::String, xlabel::String, ylabel::String,
                    xlim::Tuple{Float64, Float64}, ylim::Tuple{Float64, Float64},
                    background=COLOR_PANEL)
    fill_rect!(canvas, panel_x, panel_y, panel_w, panel_h, background)
    stroke_rect!(canvas, panel_x, panel_y, panel_w, panel_h, COLOR_LIGHT_GRAY)
    draw_text!(canvas, panel_x + 14, panel_y + 12, uppercase(title), COLOR_BLACK; scale=2)

    left = panel_x + 70
    right = panel_x + panel_w - 24
    top = panel_y + 54
    bottom = panel_y + panel_h - 54

    draw_line!(canvas, left, bottom, right, bottom, COLOR_BLACK)
    draw_line!(canvas, left, bottom, left, top, COLOR_BLACK)
    draw_text!(canvas, div(left + right, 2) - 18 * length(xlabel), panel_y + panel_h - 28, uppercase(xlabel), COLOR_BLACK; scale=1)
    draw_text!(canvas, panel_x + 8, div(top + bottom, 2) - 20, uppercase(ylabel), COLOR_BLACK; scale=1)

    x_ticks = nice_ticks(xlim[1], xlim[2], 5)
    for tick in x_ticks
        tx = map_to_pixel(tick, xlim[1], xlim[2], left, right)
        draw_line!(canvas, tx, bottom, tx, bottom + 6, COLOR_BLACK)
        label = format_tick(tick)
        draw_text!(canvas, tx - 3 * length(label), bottom + 10, label, COLOR_GRAY; scale=1)
    end

    y_ticks = nice_ticks(ylim[1], ylim[2], 5)
    for tick in y_ticks
        ty = map_to_pixel_y(tick, ylim[1], ylim[2], top, bottom)
        draw_line!(canvas, left - 6, ty, left, ty, COLOR_BLACK)
        label = format_tick(tick)
        draw_text!(canvas, panel_x + 8, ty - 4, label, COLOR_GRAY; scale=1)
        draw_line!(canvas, left, ty, right, ty, COLOR_LIGHT_GRAY)
    end

    return left, right, top, bottom
end

function map_to_pixel(value::Float64, vmin::Float64, vmax::Float64, pixel_min::Int, pixel_max::Int)
    if isapprox(vmax, vmin; atol=1e-12)
        return div(pixel_min + pixel_max, 2)
    end
    t = (value - vmin) / (vmax - vmin)
    return round(Int, pixel_min + t * (pixel_max - pixel_min))
end

function map_to_pixel_y(value::Float64, vmin::Float64, vmax::Float64, pixel_top::Int, pixel_bottom::Int)
    if isapprox(vmax, vmin; atol=1e-12)
        return div(pixel_top + pixel_bottom, 2)
    end
    t = (value - vmin) / (vmax - vmin)
    return round(Int, pixel_bottom - t * (pixel_bottom - pixel_top))
end

function plot_histogram_panel!(canvas::Canvas, panel_x::Int, panel_y::Int, panel_w::Int, panel_h::Int,
                               hist::HistogramResult, title::String, xlabel::String, color)
    xlim = (hist.centers[1], max(hist.centers[end], hist.centers[1] + 1e-6))
    ylim = (0.0, max(1.1 * maximum(hist.density), 0.05))
    left, right, top, bottom = plot_axes!(canvas, panel_x, panel_y, panel_w, panel_h, hist.centers, hist.density;
        title=title, xlabel=xlabel, ylabel="PDF", xlim=xlim, ylim=ylim)

    @inbounds for i in 1:(length(hist.centers) - 1)
        x0 = map_to_pixel(hist.centers[i], xlim[1], xlim[2], left, right)
        y0 = map_to_pixel_y(hist.density[i], ylim[1], ylim[2], top, bottom)
        x1 = map_to_pixel(hist.centers[i + 1], xlim[1], xlim[2], left, right)
        y1 = map_to_pixel_y(hist.density[i + 1], ylim[1], ylim[2], top, bottom)
        draw_line!(canvas, x0, y0, x1, y1, color)
    end
    return nothing
end

function blend_color(color_a, color_b, t::Float64)
    s = clamp(t, 0.0, 1.0)
    return (
        clamp(round(Int, color_a[1] + s * (color_b[1] - color_a[1])), 0, 255),
        clamp(round(Int, color_a[2] + s * (color_b[2] - color_a[2])), 0, 255),
        clamp(round(Int, color_a[3] + s * (color_b[3] - color_a[3])), 0, 255),
    )
end

function heatmap_color(value::Float64, vmax::Float64)
    if vmax <= 0.0
        return COLOR_WHITE
    end
    t = clamp(value / vmax, 0.0, 1.0)
    if t < 0.5
        return blend_color((0xf4, 0xf7, 0xfb), (0x5d, 0xa5, 0xda), 2.0 * t)
    end
    return blend_color((0x5d, 0xa5, 0xda), (0x0b, 0x39, 0x5b), 2.0 * (t - 0.5))
end

function plot_bivariate_pdf_panel!(canvas::Canvas, panel_x::Int, panel_y::Int, panel_w::Int, panel_h::Int,
                                   hist2d::BivariateHistogramResult, title::String)
    x_edges = grid_edges(hist2d.x_grid)
    y_edges = grid_edges(hist2d.y_grid)
    xlim = (x_edges[1], max(x_edges[end], x_edges[1] + 1e-6))
    ylim = (y_edges[1], max(y_edges[end], y_edges[1] + 1e-6))
    dummy_x = [xlim[1], xlim[2]]
    dummy_y = [ylim[1], ylim[2]]
    left, right, top, bottom = plot_axes!(canvas, panel_x, panel_y, panel_w, panel_h, dummy_x, dummy_y;
        title=title, xlabel="X", ylabel="Y", xlim=xlim, ylim=ylim)

    vmax = maximum(hist2d.density)
    cbar_w = 24
    cbar_gap = 12
    main_right = right - cbar_w - cbar_gap
    nx = length(hist2d.x_grid)
    ny = length(hist2d.y_grid)
    @inbounds for ix in 1:nx
        for iy in 1:ny
            x1 = map_to_pixel(x_edges[ix], xlim[1], xlim[2], left, main_right)
            x2 = map_to_pixel(x_edges[ix + 1], xlim[1], xlim[2], left, main_right)
            y1 = map_to_pixel_y(y_edges[iy], ylim[1], ylim[2], top, bottom)
            y2 = map_to_pixel_y(y_edges[iy + 1], ylim[1], ylim[2], top, bottom)
            color = heatmap_color(hist2d.density[ix, iy], vmax)
            fill_rect!(canvas, min(x1, x2), min(y1, y2), max(abs(x2 - x1), 1), max(abs(y2 - y1), 1), color)
        end
    end

    stroke_rect!(canvas, left, top, main_right - left + 1, bottom - top + 1, COLOR_BLACK)

    cbar_x = main_right + cbar_gap
    cbar_top = top + 18
    cbar_bottom = bottom - 18
    cbar_height = max(cbar_bottom - cbar_top + 1, 1)
    for yy in 0:(cbar_height - 1)
        t = 1.0 - yy / max(cbar_height - 1, 1)
        color = heatmap_color(t * vmax, vmax)
        fill_rect!(canvas, cbar_x, cbar_top + yy, cbar_w, 1, color)
    end
    stroke_rect!(canvas, cbar_x, cbar_top, cbar_w, cbar_height, COLOR_BLACK)
    draw_text!(canvas, cbar_x - 2, cbar_top - 14, "DENSITY", COLOR_BLACK; scale=1)
    draw_text!(canvas, cbar_x + cbar_w + 4, cbar_top - 4, @sprintf("%.2e", vmax), COLOR_GRAY; scale=1)
    draw_text!(canvas, cbar_x + cbar_w + 4, cbar_bottom - 8, "0", COLOR_GRAY; scale=1)
    return nothing
end

function plot_line_panel!(canvas::Canvas, panel_x::Int, panel_y::Int, panel_w::Int, panel_h::Int,
                          xs::Vector{Float64}, ys::Vector{Float64}, title::String;
                          xlabel::String="TIME", ylabel::String="CORR", color=COLOR_BLUE,
                          y_reference::Bool=true, secondary=nothing, secondary_color=COLOR_ORANGE,
                          legend_labels::Tuple{String, String}=("", ""),
                          xlim_override::Union{Nothing, Tuple{Float64, Float64}}=nothing)
    xmin, xmax = xlim_override === nothing ? (minimum(xs), maximum(xs)) : xlim_override
    ymin = minimum(ys)
    ymax = maximum(ys)
    if secondary !== nothing
        ymin = min(ymin, minimum(secondary))
        ymax = max(ymax, maximum(secondary))
    end
    ymin = min(ymin, y_reference ? 0.0 : ymin)
    ymax = max(ymax, 0.0)
    pad = max(0.05 * (ymax - ymin + 1e-12), 0.05)
    ylim = (ymin - pad, ymax + pad)

    left, right, top, bottom = plot_axes!(canvas, panel_x, panel_y, panel_w, panel_h, xs, ys;
        title=title, xlabel=xlabel, ylabel=ylabel, xlim=(xmin, xmax), ylim=ylim)

    if y_reference
        y0 = map_to_pixel_y(0.0, ylim[1], ylim[2], top, bottom)
        draw_line!(canvas, left, y0, right, y0, COLOR_GRAY)
    end

    for i in 1:(length(xs) - 1)
        x0 = map_to_pixel(xs[i], xmin, xmax, left, right)
        y0 = map_to_pixel_y(ys[i], ylim[1], ylim[2], top, bottom)
        x1 = map_to_pixel(xs[i + 1], xmin, xmax, left, right)
        y1 = map_to_pixel_y(ys[i + 1], ylim[1], ylim[2], top, bottom)
        draw_line!(canvas, x0, y0, x1, y1, color)
    end

    if secondary !== nothing
        for i in 1:(length(xs) - 1)
            x0 = map_to_pixel(xs[i], xmin, xmax, left, right)
            y0 = map_to_pixel_y(secondary[i], ylim[1], ylim[2], top, bottom)
            x1 = map_to_pixel(xs[i + 1], xmin, xmax, left, right)
            y1 = map_to_pixel_y(secondary[i + 1], ylim[1], ylim[2], top, bottom)
            draw_line!(canvas, x0, y0, x1, y1, secondary_color)
        end
        fill_rect!(canvas, panel_x + 18, panel_y + 42, 18, 6, color)
        draw_text!(canvas, panel_x + 42, panel_y + 38, uppercase(legend_labels[1]), COLOR_BLACK; scale=1)
        fill_rect!(canvas, panel_x + 18, panel_y + 56, 18, 6, secondary_color)
        draw_text!(canvas, panel_x + 42, panel_y + 52, uppercase(legend_labels[2]), COLOR_BLACK; scale=1)
    end
    return nothing
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

function be32(value::UInt32)
    return UInt8[
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
end

function zlib_compress(data::Vector{UInt8}; level::Int=6)
    compress_bound = ccall((:compressBound, LIBZ), Culong, (Culong,), Culong(length(data)))
    out = Vector{UInt8}(undef, Int(compress_bound))
    out_len = Ref{Culong}(compress_bound)
    status = ccall((:compress2, LIBZ), Cint,
        (Ptr{UInt8}, Ref{Culong}, Ptr{UInt8}, Culong, Cint),
        pointer(out), out_len, pointer(data), Culong(length(data)), Cint(level))
    require(status == 0, "zlib compression failed with status $(status).")
    resize!(out, Int(out_len[]))
    return out
end

function crc32_bytes(chunk_type::NTuple{4, UInt8}, data::Vector{UInt8})
    crc = ccall((:crc32, LIBZ), Culong, (Culong, Ptr{UInt8}, Cuint), 0, C_NULL, 0)
    type_bytes = UInt8[chunk_type[1], chunk_type[2], chunk_type[3], chunk_type[4]]
    crc = ccall((:crc32, LIBZ), Culong, (Culong, Ptr{UInt8}, Cuint), crc, pointer(type_bytes), Cuint(length(type_bytes)))
    crc = ccall((:crc32, LIBZ), Culong, (Culong, Ptr{UInt8}, Cuint), crc, pointer(data), Cuint(length(data)))
    return UInt32(crc)
end

function write_png_chunk(io::IO, chunk_type::NTuple{4, UInt8}, data::Vector{UInt8})
    write(io, be32(UInt32(length(data))))
    type_bytes = UInt8[chunk_type[1], chunk_type[2], chunk_type[3], chunk_type[4]]
    write(io, type_bytes)
    write(io, data)
    write(io, be32(crc32_bytes(chunk_type, data)))
    return nothing
end

function write_png(path::AbstractString, canvas::Canvas)
    raw = Vector{UInt8}(undef, canvas.height * (1 + 3 * canvas.width))
    idx = 1
    @inbounds for y in 1:canvas.height
        raw[idx] = 0x00
        idx += 1
        for x in 1:canvas.width
            raw[idx] = canvas.pixels[y, x, 1]
            raw[idx + 1] = canvas.pixels[y, x, 2]
            raw[idx + 2] = canvas.pixels[y, x, 3]
            idx += 3
        end
    end
    compressed = zlib_compress(raw)
    ihdr = UInt8[]
    append!(ihdr, be32(UInt32(canvas.width)))
    append!(ihdr, be32(UInt32(canvas.height)))
    append!(ihdr, UInt8[0x08, 0x02, 0x00, 0x00, 0x00])

    open(path, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        write_png_chunk(io, (0x49, 0x48, 0x44, 0x52), ihdr)
        write_png_chunk(io, (0x49, 0x44, 0x41, 0x54), compressed)
        write_png_chunk(io, (0x49, 0x45, 0x4e, 0x44), UInt8[])
    end
    return nothing
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
    canvas = Canvas(params.figure_width, params.figure_height; background=COLOR_WHITE)
    fill_rect!(canvas, 0, 0, params.figure_width, 84, (0xf2, 0xf0, 0xe9))
    draw_text!(canvas, 34, 22, "AFFINE LANGEVIN 2D SUMMARY", COLOR_BLACK; scale=3)
    draw_text!(canvas, 36, 56,
        @sprintf("OMEGA=%.2f  DT=%.4f  SAVE_DT=%.4f  TDECORR=%.2f  NTRAJ=%d", params.omega, params.dt, params.save_dt, corr.t_decorrelation, params.ntrajectories),
        COLOR_GRAY; scale=1)

    margin_x = 28
    margin_y = 104
    gap_x = 24
    gap_y = 24
    panel_w = Int(floor((params.figure_width - 2 * margin_x - 2 * gap_x) / 3))
    panel_h = Int(floor((params.figure_height - margin_y - 28 - gap_y) / 2))

    positions = [
        (margin_x, margin_y),
        (margin_x + panel_w + gap_x, margin_y),
        (margin_x + 2 * (panel_w + gap_x), margin_y),
        (margin_x, margin_y + panel_h + gap_y),
        (margin_x + panel_w + gap_x, margin_y + panel_h + gap_y),
        (margin_x + 2 * (panel_w + gap_x), margin_y + panel_h + gap_y),
    ]

    plot_histogram_panel!(canvas, positions[1][1], positions[1][2], panel_w, panel_h, hist_x, "PDF X", "X", COLOR_BLUE)
    plot_histogram_panel!(canvas, positions[2][1], positions[2][2], panel_w, panel_h, hist_y, "PDF Y", "Y", COLOR_ORANGE)
    plot_bivariate_pdf_panel!(canvas, positions[3][1], positions[3][2], panel_w, panel_h, hist_xy, "BIVARIATE PDF")

    lags = corr.lags
    acf_x = corr.acf_x
    acf_y = corr.acf_y
    cross_xy = corr.cross_xy
    cross_yx = corr.cross_yx
    corr_t_max = min(max(10.0 * corr.t_decorrelation, params.save_dt), corr.lags[end])
    lags_x, acf_x = truncate_series(lags, acf_x, corr_t_max)
    lags_y, acf_y = truncate_series(lags, acf_y, corr_t_max)
    lags_xy, cross_xy = truncate_series(lags, cross_xy, corr_t_max)
    _, cross_yx = truncate_series(lags, cross_yx, corr_t_max)

    plot_line_panel!(canvas, positions[4][1], positions[4][2], panel_w, panel_h, lags_x, acf_x, "ACF X";
        xlabel="TIME", ylabel="CORR", color=COLOR_GREEN,
        xlim_override=(0.0, corr_t_max))
    plot_line_panel!(canvas, positions[5][1], positions[5][2], panel_w, panel_h, lags_y, acf_y, "ACF Y";
        xlabel="TIME", ylabel="CORR", color=COLOR_RED,
        xlim_override=(0.0, corr_t_max))
    plot_line_panel!(canvas, positions[6][1], positions[6][2], panel_w, panel_h, lags_xy, cross_xy, "CROSS CORR";
        xlabel="TIME", ylabel="CORR", color=COLOR_TEAL,
        secondary=cross_yx, secondary_color=COLOR_PURPLE,
        legend_labels=("XY", "YX"),
        xlim_override=(0.0, corr_t_max))

    write_png(path, canvas)
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
