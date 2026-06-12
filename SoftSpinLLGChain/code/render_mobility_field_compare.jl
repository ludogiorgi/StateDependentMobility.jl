#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using HDF5
using KernelDensity
using LaTeXStrings
using Printf
using TOML

const DEFAULT_CONFIG = joinpath(@__DIR__, "..", "configs",
    "render_mobility_field_compare.toml")

Base.@kwdef struct MobilityFieldConfig
    observations::String
    score_bson::String = ""
    nn_model::String
    physics_model::String
    phi_target::String = ""
    seed::Int = 20260521
    observation_postburn_fraction::Float64 = 0.10
    frames::Int = 500
    mz_min::Float64 = -1.25
    mz_max::Float64 = 1.25
    mperp_min::Float64 = 0.0
    mperp_max::Float64 = 1.25
    n_mz::Int = 121
    n_mperp::Int = 91
    n_radius::Int = 220
    support_density_fraction::Float64 = 0.025
    width::Int = 5600
    height::Int = 3600
    font_scale::Float64 = 3.35
    legend_font_scale::Float64 = 1.0
    output_png::String
end

function config_path(base::AbstractString, path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(base, path))
end

function load_mobility_field_config(path::AbstractString)
    raw = TOML.parsefile(path)
    base = dirname(abspath(path))
    inputs = raw["inputs"]
    sampling = get(raw, "sampling", Dict{String, Any}())
    grid = get(raw, "grid", Dict{String, Any}())
    figure = get(raw, "figure", Dict{String, Any}())
    output = raw["output"]
    return MobilityFieldConfig(
        observations=config_path(base, String(inputs["observations"])),
        score_bson=haskey(inputs, "score_bson") ?
            config_path(base, String(inputs["score_bson"])) : "",
        nn_model=config_path(base, String(inputs["nn_model"])),
        physics_model=config_path(base, String(inputs["physics_model"])),
        phi_target=haskey(inputs, "phi_target") ?
            config_path(base, String(inputs["phi_target"])) : "",
        seed=Int(get(sampling, "seed", 20260521)),
        observation_postburn_fraction=Float64(get(sampling, "observation_postburn_fraction", 0.10)),
        frames=Int(get(sampling, "frames", 500)),
        mz_min=Float64(get(grid, "mz_min", -1.25)),
        mz_max=Float64(get(grid, "mz_max", 1.25)),
        mperp_min=Float64(get(grid, "mperp_min", 0.0)),
        mperp_max=Float64(get(grid, "mperp_max", 1.25)),
        n_mz=Int(get(grid, "n_mz", 121)),
        n_mperp=Int(get(grid, "n_mperp", 91)),
        n_radius=Int(get(grid, "n_radius", 220)),
        support_density_fraction=Float64(get(grid, "support_density_fraction", 0.025)),
        width=Int(get(figure, "width", 5600)),
        height=Int(get(figure, "height", 3600)),
        font_scale=Float64(get(figure, "font_scale", 3.35)),
        legend_font_scale=Float64(get(figure, "legend_font_scale", 1.0)),
        output_png=config_path(base, String(output["png"])),
    )
end

function choose_observation_frames(path::AbstractString, nframes::Int,
        postburn_fraction::Float64, rng::AbstractRNG)
    h5open(path, "r") do h5
        states = h5["/trajectories/states"]
        nt = size(states, 1)
        first_frame = clamp(floor(Int, postburn_fraction * nt) + 1, 1, nt)
        available = nt - first_frame + 1
        n = min(nframes, available)
        perm = randperm(rng, available)
        return first_frame .+ sort!(perm[1:n] .- 1)
    end
end

function sample_local_spin_support(path::AbstractString, nframes::Int,
        postburn_fraction::Float64, seed::Int)
    rng = MersenneTwister(seed)
    frames = choose_observation_frames(path, nframes, postburn_fraction, rng)
    mperp = Float64[]
    mz = Float64[]
    h5open(path, "r") do h5
        states = h5["/trajectories/states"]
        _, nsites, _, ntraj = size(states)
        sizehint!(mperp, length(frames) * nsites * ntraj)
        sizehint!(mz, length(frames) * nsites * ntraj)
        for frame in frames
            s = Array{Float32, 3}(states[frame, :, :, :])
            @inbounds for tr in 1:ntraj, i in 1:nsites
                x = Float64(s[i, 1, tr])
                y = Float64(s[i, 2, tr])
                z = Float64(s[i, 3, tr])
                push!(mperp, sqrt(x * x + y * y))
                push!(mz, z)
            end
        end
    end
    return mz, mperp, frames
end

function nearest_grid_index(grid::AbstractVector{<:Real}, value::Real)
    hi = searchsortedfirst(grid, value)
    if hi <= firstindex(grid)
        return firstindex(grid)
    elseif hi > lastindex(grid)
        return lastindex(grid)
    end
    lo = hi - 1
    return abs(value - grid[lo]) <= abs(grid[hi] - value) ? lo : hi
end

function observed_support_mask(mz_grid::AbstractVector{<:Real},
        mp_grid::AbstractVector{<:Real}, mz::AbstractVector{<:Real},
        mp::AbstractVector{<:Real}; dilation::Int=1)
    mask = falses(length(mz_grid), length(mp_grid))
    @inbounds for idx in eachindex(mz, mp)
        ix = nearest_grid_index(mz_grid, mz[idx])
        iy = nearest_grid_index(mp_grid, mp[idx])
        for j in max(1, iy - dilation):min(length(mp_grid), iy + dilation),
                i in max(1, ix - dilation):min(length(mz_grid), ix + dilation)
            mask[i, j] = true
        end
    end
    count(mask) > 0 || error("Observed local-spin support mask is empty.")
    return mask
end

function density_support_mask(mz_grid::AbstractVector{<:Real},
        mp_grid::AbstractVector{<:Real}, mz::AbstractVector{<:Real},
        mp::AbstractVector{<:Real}, cfg::MobilityFieldConfig)
    kde_obj = kde((Float64.(mz), Float64.(mp));
        npoints=(length(mz_grid), length(mp_grid)),
        boundary=((cfg.mz_min, cfg.mz_max), (cfg.mperp_min, cfg.mperp_max)))
    max_density = maximum(kde_obj.density)
    max_density > 0 || return observed_support_mask(mz_grid, mp_grid, mz, mp)
    threshold = cfg.support_density_fraction * max_density
    mask = kde_obj.density .>= threshold
    count(mask) > 0 || error("KDE support mask is empty.")
    return BitMatrix(mask)
end

cross_matrix(m::AbstractVector{<:Real}) =
    [0.0 -Float64(m[3]) Float64(m[2]);
     Float64(m[3]) 0.0 -Float64(m[1]);
     -Float64(m[2]) Float64(m[1]) 0.0]

function true_block_from_coeff(m::AbstractVector{<:Real}, coeff::AbstractVector{<:Real})
    x = Float64.(m)
    r2 = dot(x, x)
    I3 = Matrix{Float64}(I, 3, 3)
    return coeff[1] .* I3 .+
        coeff[2] .* (r2 .* I3 .- x * transpose(x)) .+
        coeff[3] .* (x * transpose(x)) .+
        coeff[4] .* cross_matrix(x)
end

function local_scalar_projections(block::AbstractMatrix{<:Real},
        m::AbstractVector{<:Real})
    x = Float64.(m)
    r = norm(x)
    S = 0.5 .* (Matrix{Float64}(block) .+ transpose(Matrix{Float64}(block)))
    K = 0.5 .* (Matrix{Float64}(block) .- transpose(Matrix{Float64}(block)))
    if r < 1.0e-10
        n = [1.0, 0.0, 0.0]
    else
        n = x ./ r
    end
    P = Matrix{Float64}(I, 3, 3) .- n * transpose(n)
    lambda_perp = 0.5 * tr(P * S * P)
    lambda_parallel = dot(n, S * n)
    C = cross_matrix(x)
    denom = sum(abs2, C)
    c_cross = denom > 1.0e-14 ? sum(K .* C) / denom : NaN
    return lambda_perp, lambda_parallel, c_cross
end

function block_from_bp_at(bp, q::Int)
    l11, l21, l22 = Float64(bp.l11[q]), Float64(bp.l21[q]), Float64(bp.l22[q])
    l31, l32, l33 = Float64(bp.l31[q]), Float64(bp.l32[q]), Float64(bp.l33[q])
    k1, k2, k3 = Float64(bp.k1[q]), Float64(bp.k2[q]), Float64(bp.k3[q])
    S = [l11^2 l11*l21 l11*l31;
         l11*l21 l21^2+l22^2 l21*l31+l22*l32;
         l11*l31 l21*l31+l22*l32 l31^2+l32^2+l33^2]
    K = [0.0 -k3 k2; k3 0.0 -k1; -k2 k1 0.0]
    return S .+ K
end

function nn_block_for_points(model::EquivariantMobilityNN,
        points::Vector{NTuple{2, Float64}}, stats::Union{Nothing, DataStats})
    nsites = size(model.mean, 2)
    npoints = length(points)
    raw = Array{Float32, 3}(undef, nsites, 3, npoints)
    @inbounds for (b, (mz, mperp)) in enumerate(points)
        raw[:, 1, b] .= Float32(mperp)
        raw[:, 2, b] .= 0f0
        raw[:, 3, b] .= Float32(mz)
    end
    stats = DataStats(model.mean, model.std)
    xn = apply_stats_tensor(raw, stats)
    bp = block_params(model, xn)
    blocks = Vector{Matrix{Float64}}(undef, npoints)
    @inbounds for b in 1:npoints
        blocks[b] = block_from_bp_at(bp, (b - 1) * nsites + 1)
    end
    return blocks
end

function nn_block_for_points(model::LocalMobilityNN,
        points::Vector{NTuple{2, Float64}}, stats::Union{Nothing, DataStats})
    stats === nothing &&
        error("Local mobility models require score normalization statistics.")
    nsites = size(stats.mean, 2)
    npoints = length(points)
    raw = Array{Float32, 3}(undef, nsites, 3, npoints)
    @inbounds for (b, (mz, mperp)) in enumerate(points)
        raw[:, 1, b] .= Float32(mperp)
        raw[:, 2, b] .= 0f0
        raw[:, 3, b] .= Float32(mz)
    end
    xn = apply_stats_tensor(raw, stats)
    bp = block_params(model, xn)
    blocks = Vector{Matrix{Float64}}(undef, npoints)
    @inbounds for b in 1:npoints
        blocks[b] = block_from_bp_at(bp, (b - 1) * nsites + 1)
    end
    return blocks
end

function scalar_maps(cfg::MobilityFieldConfig, nn_model, nn_stats, true_coeff,
        phys_coeff, phi_block::AbstractMatrix{<:Real})
    mz_grid = collect(range(cfg.mz_min, cfg.mz_max; length=cfg.n_mz))
    mp_grid = collect(range(cfg.mperp_min, cfg.mperp_max; length=cfg.n_mperp))
    points = [(mz, mp) for mp in mp_grid for mz in mz_grid]
    nn_blocks = nn_block_for_points(nn_model, points, nn_stats)

    maps = Dict{Symbol, Vector{Matrix{Float64}}}()
    maps[:truth] = [Matrix{Float64}(undef, cfg.n_mz, cfg.n_mperp) for _ in 1:3]
    maps[:phi] = [Matrix{Float64}(undef, cfg.n_mz, cfg.n_mperp) for _ in 1:3]
    maps[:nn] = [Matrix{Float64}(undef, cfg.n_mz, cfg.n_mperp) for _ in 1:3]
    maps[:phys] = [Matrix{Float64}(undef, cfg.n_mz, cfg.n_mperp) for _ in 1:3]

    @inbounds for (idx, (mz, mp)) in enumerate(points)
        ix = (idx - 1) % cfg.n_mz + 1
        iy = (idx - 1) ÷ cfg.n_mz + 1
        m = [mp, 0.0, mz]
        s_true = local_scalar_projections(true_block_from_coeff(m, true_coeff), m)
        s_phi = local_scalar_projections(phi_block, m)
        s_nn = local_scalar_projections(nn_blocks[idx], m)
        s_phys = local_scalar_projections(true_block_from_coeff(m, phys_coeff), m)
        for r in 1:3
            maps[:truth][r][ix, iy] = s_true[r]
            maps[:phi][r][ix, iy] = s_phi[r]
            maps[:nn][r][ix, iy] = s_nn[r]
            maps[:phys][r][ix, iy] = s_phys[r]
        end
    end
    return mz_grid, mp_grid, maps
end

function finite_extrema(arrays::Vector{Matrix{Float64}})
    vals = reduce(vcat, [vec(A[isfinite.(A)]) for A in arrays])
    isempty(vals) && return (0.0, 1.0)
    lo, hi = extrema(vals)
    if lo == hi
        pad = max(abs(lo), 1.0) * 0.05
        return lo - pad, hi + pad
    end
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad
end

function support_masked_maps(maps::Dict{Symbol, Vector{Matrix{Float64}}},
        support_mask::BitMatrix)
    masked = Dict{Symbol, Vector{Matrix{Float64}}}()
    for (key, row_maps) in maps
        masked[key] = Matrix{Float64}[]
        for A in row_maps
            B = copy(A)
            B[.!support_mask] .= NaN
            push!(masked[key], B)
        end
    end
    return masked
end

function support_error_curves(mz_grid::AbstractVector{<:Real},
        mp_grid::AbstractVector{<:Real}, maps::Dict{Symbol, Vector{Matrix{Float64}}},
        support_mask::BitMatrix, nbins::Int)
    radii = Float64[]
    @inbounds for j in eachindex(mp_grid), i in eachindex(mz_grid)
        support_mask[i, j] || continue
        push!(radii, hypot(Float64(mz_grid[i]), Float64(mp_grid[j])))
    end
    isempty(radii) && error("No support points available for error curves.")
    rlo, rhi = extrema(radii)
    if rlo == rhi
        rlo = max(0.0, rlo - 0.05)
        rhi = rhi + 0.05
    end
    edges = collect(range(rlo, rhi; length=max(nbins, 2) + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    errors = Dict(k => [fill(NaN, length(centers)) for _ in 1:3] for k in (:phi, :nn, :phys))
    for row in 1:3, key in (:phi, :nn, :phys)
        sums = zeros(length(centers))
        denominators = zeros(length(centers))
        @inbounds for j in eachindex(mp_grid), i in eachindex(mz_grid)
            support_mask[i, j] || continue
            truth = maps[:truth][row][i, j]
            pred = maps[key][row][i, j]
            isfinite(truth) && isfinite(pred) || continue
            r = hypot(Float64(mz_grid[i]), Float64(mp_grid[j]))
            bin = searchsortedlast(edges, r)
            bin = clamp(bin, 1, length(centers))
            sums[bin] += abs(pred - truth)
            denominators[bin] += abs(truth)
        end
        @inbounds for kidx in eachindex(centers)
            if denominators[kidx] > eps(Float64)
                errors[key][row][kidx] = sums[kidx] / denominators[kidx]
            end
        end
    end
    return centers, errors
end

function render_mobility_field_compare(cfg::MobilityFieldConfig)
    for path in (cfg.observations, cfg.nn_model, cfg.physics_model)
        isfile(path) || error("Missing input $(path)")
    end
    mkpath(dirname(cfg.output_png))

    p = load_phys(cfg.observations)
    nn_blob = BSON.load(cfg.nn_model)
    nn_model = nn_blob[:host_model]
    Flux.testmode!(nn_model)
    nn_stats = nothing
    if nn_model isa LocalMobilityNN
        isempty(cfg.score_bson) &&
            error("Config input score_bson is required for LocalMobilityNN rendering.")
        isfile(cfg.score_bson) || error("Missing input $(cfg.score_bson)")
        _, nn_stats, _, _ = load_stationary_checkpoint(cfg.score_bson, CPUDevice())
    end
    phys_blob = BSON.load(cfg.physics_model)
    phys_coeff = Vector{Float64}(phys_blob[:coefficients])
    true_coeff = Vector{Float64}(phys_blob[:true_coefficients_expost])
    phi_target_path = cfg.phi_target
    if isempty(phi_target_path)
        haskey(nn_blob, :target_artifact) ||
            error("Config input phi_target is required when the NN checkpoint has no target_artifact.")
        raw_target = String(nn_blob[:target_artifact])
        phi_target_path = isabspath(raw_target) ? raw_target :
            normpath(joinpath(dirname(cfg.nn_model), raw_target))
    end
    isfile(phi_target_path) || error("Missing Phi target artifact $(phi_target_path)")
    phi_blob = BSON.load(phi_target_path)
    phi_block = Matrix{Float64}(phi_blob[:Phi_block])
    true_coeff_from_params = [p.theta * p.eps, p.theta * p.alpha_perp,
        p.theta * p.alpha_parallel, -p.gamma * p.theta]
    if norm(true_coeff .- true_coeff_from_params) > 1.0e-10
        @warn "Saved true coefficients differ from parameters" true_coeff true_coeff_from_params
    end

    support_mz, support_mp, frames = sample_local_spin_support(cfg.observations,
        cfg.frames, cfg.observation_postburn_fraction, cfg.seed)
    @printf("Sampled %d observation frames for support mask (%d local states).\n",
        length(frames), length(support_mz))

    mz_grid, mp_grid, maps = scalar_maps(cfg, nn_model, nn_stats, true_coeff,
        phys_coeff, phi_block)
    support_mask = density_support_mask(mz_grid, mp_grid, support_mz, support_mp, cfg)
    masked_maps = support_masked_maps(maps, support_mask)
    r_grid, support_errors = support_error_curves(mz_grid, mp_grid, maps,
        support_mask, cfg.n_radius)

    row_labels = [latexstring("\\lambda_{\\perp}"),
        latexstring("\\lambda_{\\parallel}"),
        latexstring("c_{\\times}")]
    column_titles = [latexstring("M_{\\mathrm{true}}"),
        latexstring("M=\\Phi"),
        latexstring("M_{\\mathrm{NN}}"),
        latexstring("M_{\\mathrm{phys}}")]
    row_colormaps = [STYLE_SEQUENTIAL_BLUE, STYLE_SEQUENTIAL_BLUE, :balance]
    map_keys = [:truth, :phi, :nn, :phys]

    with_scaled_figure_style(cfg.width, cfg.height; scale_override=cfg.font_scale) do _
        fig = Figure(; size=(cfg.width, cfg.height))
        colorbars = Any[]
        for row in 1:3
            Label(fig[row, 1], row_labels[row]; rotation=pi / 2,
                tellwidth=true, tellheight=false, font=:bold)
            clim = finite_extrema([masked_maps[key][row] for key in map_keys])
            hm = nothing
            for (col, key) in enumerate(map_keys)
                ax = Axis(fig[row, col + 1];
                    title=row == 1 ? column_titles[col] : "",
                    xlabel=row == 3 ? latexstring("m_z") : "",
                    ylabel=col == 1 ? latexstring("m_{\\perp}") : "",
                    xgridvisible=false, ygridvisible=false)
                hm = heatmap!(ax, mz_grid, mp_grid, masked_maps[key][row];
                    colorrange=clim, colormap=row_colormaps[row])
                xlims!(ax, cfg.mz_min, cfg.mz_max)
                ylims!(ax, cfg.mperp_min, cfg.mperp_max)
            end
            push!(colorbars, Colorbar(fig[row, 6], hm; label=row_labels[row],
                ticklabelsize=current_figure_style().colorbar_ticklabelsize * 0.92,
                labelsize=current_figure_style().colorbar_labelsize * 0.92))

            axerr = Axis(fig[row, 7];
                title=row == 1 ? latexstring("\\mathrm{relative\\;error\\;vs.}\\;|m|") : "",
                xlabel=row == 3 ? latexstring("|m|") : "",
                ylabel=latexstring("\\mathrm{relative\\;error}"))
            err_phi = support_errors[:phi][row]
            err_nn = support_errors[:nn][row]
            err_phys = support_errors[:phys][row]
            valid_phi = isfinite.(err_phi)
            valid_nn = isfinite.(err_nn)
            valid_phys = isfinite.(err_phys)
            lines!(axerr, r_grid[valid_phi], err_phi[valid_phi]; color=STYLE_SECONDARY,
                linestyle=:dash, linewidth=curve_linewidth(), label=latexstring("M=\\Phi"))
            lines!(axerr, r_grid[valid_nn], err_nn[valid_nn]; color=STYLE_PRIMARY,
                linewidth=curve_linewidth(), label=latexstring("M_{\\mathrm{NN}}"))
            lines!(axerr, r_grid[valid_phys], err_phys[valid_phys]; color=STYLE_ACCENT,
                linestyle=:dash, linewidth=curve_linewidth(),
                label=latexstring("M_{\\mathrm{phys}}"))
            ylo = 0.0
            finite_errors = vcat(err_phi[valid_phi], err_nn[valid_nn], err_phys[valid_phys])
            yhi = isempty(finite_errors) ? 1.0 : max(maximum(finite_errors), eps(Float64))
            ylims!(axerr, ylo, 1.12 * yhi)
            xlims!(axerr, first(r_grid), last(r_grid))
        end

        legend_elements = [
            LineElement(color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth()),
            LineElement(color=STYLE_PRIMARY, linewidth=curve_linewidth()),
            LineElement(color=STYLE_ACCENT, linestyle=:dash, linewidth=curve_linewidth()),
        ]
        Legend(fig[4, 1:7], legend_elements,
            [latexstring("M=\\Phi"), latexstring("M_{\\mathrm{NN}}"),
                latexstring("M_{\\mathrm{phys}}")];
            orientation=:horizontal, framevisible=false,
            labelsize=current_figure_style().legend_labelsize * cfg.legend_font_scale)

        rowsize!(fig.layout, 1, Relative(1 / 3))
        rowsize!(fig.layout, 2, Relative(1 / 3))
        rowsize!(fig.layout, 3, Relative(1 / 3))
        colsize!(fig.layout, 1, Fixed(120))
        colsize!(fig.layout, 2, Relative(0.18))
        colsize!(fig.layout, 3, Relative(0.18))
        colsize!(fig.layout, 4, Relative(0.18))
        colsize!(fig.layout, 5, Relative(0.18))
        colsize!(fig.layout, 6, Fixed(72))
        colsize!(fig.layout, 7, Relative(0.28))
        rowgap!(fig.layout, 18 * max(current_figure_style().fontsize / 18, 1))
        colgap!(fig.layout, 18 * max(current_figure_style().fontsize / 18, 1))
        save_figure(cfg.output_png, fig)
    end
    @printf("Saved %s\n", cfg.output_png)
    return nothing
end

function main()
    cfg_path = isempty(ARGS) ? DEFAULT_CONFIG : ARGS[1]
    render_mobility_field_compare(load_mobility_field_config(cfg_path))
end

main()
