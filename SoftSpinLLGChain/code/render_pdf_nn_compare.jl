#!/usr/bin/env julia

using HDF5
using KernelDensity
using LaTeXStrings
using Printf
using Random
using Statistics
using TOML

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))

include(joinpath(REPO_ROOT, "2D", "src", "figure_style.jl"))

const DEFAULT_CONFIG = joinpath(SCRIPT_DIR, "..", "configs", "render_pdf_nn_compare.toml")
const COMPONENT_NAMES = ("x", "y", "z")

Base.@kwdef struct PdfCompareConfig
    observations::String
    nn::String
    seed::Int = 20260521
    observation_postburn_fraction::Float64 = 0.10
    nn_postburn_fraction::Float64 = 0.0
    frames::Int = 700
    max_univariate_samples::Int = 600_000
    max_bivariate_samples::Int = 320_000
    kde_points_1d::Int = 320
    kde_points_2d::Int = 112
    quantile_low::Float64 = 0.002
    quantile_high::Float64 = 0.998
    width::Int = 4300
    height::Int = 5200
    font_scale::Float64 = 3.8
    output_png::String
end

function resolve_from_config(config_dir::AbstractString, path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(config_dir, path))
end

function load_config(path::AbstractString)
    parsed = TOML.parsefile(path)
    config_dir = dirname(abspath(path))
    inputs = parsed["inputs"]
    sampling = get(parsed, "sampling", Dict{String, Any}())
    figure = get(parsed, "figure", Dict{String, Any}())
    output = parsed["output"]

    return PdfCompareConfig(
        observations=resolve_from_config(config_dir, String(inputs["observations"])),
        nn=resolve_from_config(config_dir, String(inputs["nn"])),
        seed=Int(get(sampling, "seed", 20260521)),
        observation_postburn_fraction=Float64(get(sampling, "observation_postburn_fraction", 0.10)),
        nn_postburn_fraction=Float64(get(sampling, "nn_postburn_fraction", 0.0)),
        frames=Int(get(sampling, "frames", 700)),
        max_univariate_samples=Int(get(sampling, "max_univariate_samples", 600_000)),
        max_bivariate_samples=Int(get(sampling, "max_bivariate_samples", 320_000)),
        kde_points_1d=Int(get(sampling, "kde_points_1d", 320)),
        kde_points_2d=Int(get(sampling, "kde_points_2d", 112)),
        quantile_low=Float64(get(sampling, "quantile_low", 0.002)),
        quantile_high=Float64(get(sampling, "quantile_high", 0.998)),
        width=Int(get(figure, "width", 4300)),
        height=Int(get(figure, "height", 5200)),
        font_scale=Float64(get(figure, "font_scale", 3.8)),
        output_png=resolve_from_config(config_dir, String(output["png"])),
    )
end

function choose_frames(path::AbstractString, nframes::Int, postburn_fraction::Float64,
        rng::AbstractRNG)
    h5open(path, "r") do h5
        states = h5["/trajectories/states"]
        nt = size(states, 1)
        first_frame = clamp(floor(Int, postburn_fraction * nt) + 1, 1, nt)
        available = nt - first_frame + 1
        n = min(nframes, available)
        perm = randperm(rng, available)
        frames = first_frame .+ sort!(perm[1:n] .- 1)
        return Vector{Int}(frames)
    end
end

function load_state_frames(path::AbstractString, frames::Vector{Int})
    isempty(frames) && error("No frames selected for $(path).")
    h5open(path, "r") do h5
        states = h5["/trajectories/states"]
        nt, nsites, ncomp, ntraj = size(states)
        ncomp == 3 || error("Expected three spin components in $(path), found $(ncomp).")
        all(frame -> 1 <= frame <= nt, frames) ||
            error("Selected frame outside 1:$(nt) for $(path).")
        out = Array{Float32, 4}(undef, length(frames), nsites, ncomp, ntraj)
        @inbounds for (j, frame) in enumerate(frames)
            out[j, :, :, :] .= states[frame, :, :, :]
        end
        times = h5["/trajectories/time"]
        selected_times = [Float64(times[frame]) for frame in frames]
        return out, selected_times
    end
end

function maybe_subsample(values::Vector{Float64}, max_samples::Int, rng::AbstractRNG)
    length(values) <= max_samples && return values
    idx = randperm(rng, length(values))[1:max_samples]
    return values[idx]
end

function maybe_subsample_pair(x::Vector{Float64}, y::Vector{Float64}, max_samples::Int,
        rng::AbstractRNG)
    length(x) == length(y) || error("Pair sample length mismatch.")
    length(x) <= max_samples && return x, y
    idx = randperm(rng, length(x))[1:max_samples]
    return x[idx], y[idx]
end

function component_values(states::AbstractArray{<:Real, 4}, component::Int)
    return Float64.(vec(@view states[:, :, component, :]))
end

function same_site_pair(states::AbstractArray{<:Real, 4}, c1::Int, c2::Int)
    return component_values(states, c1), component_values(states, c2)
end

function neighbor_pair(states::AbstractArray{<:Real, 4}, component::Int, separation::Int)
    _, nsites, _, _ = size(states)
    shifted = [mod1(i + separation, nsites) for i in 1:nsites]
    return Float64.(vec(@view states[:, :, component, :])),
        Float64.(vec(@view states[:, shifted, component, :]))
end

function padded_bounds(values::Vector{Float64}, qlo::Float64, qhi::Float64)
    lo = quantile(values, qlo)
    hi = quantile(values, qhi)
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        center = isfinite(lo) ? lo : 0.0
        return center - 1.0, center + 1.0
    end
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad
end

function common_1d_kdes(obs::Vector{Float64}, model::Vector{Float64},
        cfg::PdfCompareConfig)
    bounds = padded_bounds(vcat(obs, model), cfg.quantile_low, cfg.quantile_high)
    ko = kde(obs; npoints=cfg.kde_points_1d, boundary=bounds)
    km = kde(model; npoints=cfg.kde_points_1d, boundary=bounds)
    return ko, km, bounds
end

function common_2d_kdes(xobs::Vector{Float64}, yobs::Vector{Float64},
        xmodel::Vector{Float64}, ymodel::Vector{Float64}, cfg::PdfCompareConfig)
    xb = padded_bounds(vcat(xobs, xmodel), cfg.quantile_low, cfg.quantile_high)
    yb = padded_bounds(vcat(yobs, ymodel), cfg.quantile_low, cfg.quantile_high)
    ko = kde((xobs, yobs); npoints=(cfg.kde_points_2d, cfg.kde_points_2d),
        boundary=(xb, yb))
    km = kde((xmodel, ymodel); npoints=(cfg.kde_points_2d, cfg.kde_points_2d),
        boundary=(xb, yb))
    return ko, km, xb, yb
end

function density_contour_levels(density_a, density_b)
    max_density = maximum(vcat(vec(density_a.density), vec(density_b.density)))
    max_density > 0.0 || return Float64[]
    return collect(range(0.12 * max_density, 0.90 * max_density; length=6))
end

function render_univariate_panel!(fig, row::Int, col::Int, title, xlabel, obs, model,
        cfg::PdfCompareConfig, rng::AbstractRNG; show_ylabel::Bool=false)
    vo = maybe_subsample(obs, cfg.max_univariate_samples, rng)
    vm = maybe_subsample(model, cfg.max_univariate_samples, rng)
    ko, km, bounds = common_1d_kdes(vo, vm, cfg)
    ax = Axis(fig[row, col]; title=title, xlabel=xlabel,
        ylabel=show_ylabel ? latexstring("p_{\\mathrm{ss}}") : "")
    ref_line = lines!(ax, ko.x, ko.density; color=STYLE_REFERENCE, linewidth=curve_linewidth(),
        label=latexstring("\\mathrm{obs}"))
    model_line = lines!(ax, km.x, km.density; color=STYLE_PRIMARY, linestyle=:dash,
        linewidth=curve_linewidth(), label=latexstring("M_{\\mathrm{NN}}"))
    xlims!(ax, bounds...)
    return ax, ref_line, model_line
end

function render_contour_panel!(fig, row::Int, col::Int, title, xlabel, ylabel,
        obs_pair::Tuple{Vector{Float64}, Vector{Float64}},
        model_pair::Tuple{Vector{Float64}, Vector{Float64}},
        cfg::PdfCompareConfig, rng::AbstractRNG)
    xobs, yobs = maybe_subsample_pair(obs_pair[1], obs_pair[2], cfg.max_bivariate_samples, rng)
    xmodel, ymodel = maybe_subsample_pair(model_pair[1], model_pair[2],
        cfg.max_bivariate_samples, rng)
    ko, km, xb, yb = common_2d_kdes(xobs, yobs, xmodel, ymodel, cfg)
    ax = Axis(fig[row, col]; title=title, xlabel=xlabel, ylabel=ylabel,
        xgridvisible=false, ygridvisible=false)
    levels = density_contour_levels(ko, km)
    if !isempty(levels)
        contour!(ax, ko.x, ko.y, ko.density; levels=levels, color=STYLE_REFERENCE,
            linewidth=curve_linewidth())
        contour!(ax, km.x, km.y, km.density; levels=levels, color=STYLE_PRIMARY,
            linestyle=:dash, linewidth=curve_linewidth())
    end
    xlims!(ax, xb...)
    ylims!(ax, yb...)
    return ax
end

function component_label(component::Int)
    return latexstring("m_{$(COMPONENT_NAMES[component])}")
end

function pair_label(c1::Int, c2::Int)
    return latexstring("m_{$(COMPONENT_NAMES[c1])},m_{$(COMPONENT_NAMES[c2])}")
end

function neighbor_title(component::Int, separation::Int)
    c = COMPONENT_NAMES[component]
    return latexstring("m_{$(c),i},m_{$(c),i+$(separation)}")
end

function render_pdf_compare(cfg::PdfCompareConfig)
    isfile(cfg.observations) || error("Missing observation file $(cfg.observations).")
    isfile(cfg.nn) || error("Missing NN forward file $(cfg.nn).")
    mkpath(dirname(cfg.output_png))

    rng = MersenneTwister(cfg.seed)
    obs_frames = choose_frames(cfg.observations, cfg.frames, cfg.observation_postburn_fraction, rng)
    nn_frames = choose_frames(cfg.nn, cfg.frames, cfg.nn_postburn_fraction, rng)
    obs_states, obs_times = load_state_frames(cfg.observations, obs_frames)
    nn_states, nn_times = load_state_frames(cfg.nn, nn_frames)

    @printf("Observation pool: %d frames, t in [%.3f, %.3f], %d samples/component.\n",
        length(obs_frames), minimum(obs_times), maximum(obs_times),
        length(component_values(obs_states, 1)))
    @printf("M_NN pool: %d frames, t in [%.3f, %.3f], %d samples/component.\n",
        length(nn_frames), minimum(nn_times), maximum(nn_times),
        length(component_values(nn_states, 1)))

    obs_components = [component_values(obs_states, c) for c in 1:3]
    nn_components = [component_values(nn_states, c) for c in 1:3]

    with_scaled_figure_style(cfg.width, cfg.height; scale_override=cfg.font_scale) do _
        fig = Figure(; size=(cfg.width, cfg.height))
        fig_rng = MersenneTwister(cfg.seed + 1000)
        ref_line = nothing
        model_line = nothing

        for c in 1:3
            ax, h1, h2 = render_univariate_panel!(fig, 1, c,
                latexstring("p_{\\mathrm{ss}}($(COMPONENT_NAMES[c]))"),
                component_label(c), obs_components[c], nn_components[c], cfg, fig_rng;
                show_ylabel=(c == 1))
            ref_line === nothing && (ref_line = h1)
            model_line === nothing && (model_line = h2)
        end

        same_site_specs = [(1, 2), (1, 3), (2, 3)]
        for (col, (c1, c2)) in enumerate(same_site_specs)
            render_contour_panel!(fig, 2, col,
                latexstring("p_{\\mathrm{ss}}($(COMPONENT_NAMES[c1]),$(COMPONENT_NAMES[c2]))"),
                component_label(c1), component_label(c2),
                same_site_pair(obs_states, c1, c2),
                same_site_pair(nn_states, c1, c2), cfg, fig_rng)
        end

        for (row_offset, component) in enumerate((1, 2, 3))
            row = row_offset + 2
            for separation in 1:3
                render_contour_panel!(fig, row, separation,
                    neighbor_title(component, separation),
                    latexstring("m_{$(COMPONENT_NAMES[component]),i}"),
                    latexstring("m_{$(COMPONENT_NAMES[component]),i+$(separation)}"),
                    neighbor_pair(obs_states, component, separation),
                    neighbor_pair(nn_states, component, separation), cfg, fig_rng)
            end
        end

        Legend(fig[6, 1:3], Any[ref_line, model_line],
            Any[latexstring("\\mathrm{observations}"),
                latexstring("\\mathrm{Langevin}\\; M_{\\mathrm{NN}}")];
            orientation=:horizontal, tellheight=true, tellwidth=false,
            framevisible=false, nbanks=1)

        apply_publication_grid!(fig.layout, 6, 3;
            row_weights=[0.92, 1.0, 1.0, 1.0, 1.0, 0.16],
            col_weights=[1.0, 1.0, 1.0],
            row_gap=21, col_gap=31)
        save_figure(cfg.output_png, fig)
    end

    @printf("Saved %s\n", cfg.output_png)
    return nothing
end

function main()
    config_path = isempty(ARGS) ? DEFAULT_CONFIG : ARGS[1]
    cfg = load_config(config_path)
    render_pdf_compare(cfg)
end

main()
