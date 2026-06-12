#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))

using HDF5
using LaTeXStrings
using Printf
using Statistics
using TOML

const DEFAULT_HOVMOLLER_CONFIG =
    joinpath(@__DIR__, "..", "configs", "render_hovmoller_forward_compare.toml")

Base.@kwdef struct HovmollerFigureConfig
    observation_h5::String
    phi_h5::String
    nn_h5::String
    physics_h5::String
    duration::Float64
    observation_start_time::Float64
    phi_start_time::Float64
    nn_start_time::Float64
    physics_start_time::Float64
    trajectory_index::Int
    max_frames::Int
    width::Int
    height::Int
    font_scale::Float64
    output_png::String
end

struct HovmollerWindow
    label::LaTeXString
    source_path::String
    start_time::Float64
    stop_time::Float64
    times::Vector{Float64}
    states::Array{Float32, 3}
    stride::Int
end

function string_dict(raw)
    return Dict{String, Any}(String(key) => value for (key, value) in pairs(raw))
end

function load_hovmoller_config(path::AbstractString)
    raw = TOML.parsefile(path)
    base = dirname(abspath(path))
    inputs = string_dict(raw["inputs"])
    window = string_dict(raw["window"])
    figure = string_dict(raw["figure"])
    output = string_dict(raw["output"])
    return HovmollerFigureConfig(
        observation_h5=resolve_path(base, String(inputs["observations"])),
        phi_h5=resolve_path(base, String(inputs["phi"])),
        nn_h5=resolve_path(base, String(inputs["nn"])),
        physics_h5=resolve_path(base, String(inputs["physics"])),
        duration=Float64(window["duration"]),
        observation_start_time=Float64(window["observation_start_time"]),
        phi_start_time=Float64(get(window, "phi_start_time",
            get(window, "forward_start_time", 0.0))),
        nn_start_time=Float64(get(window, "nn_start_time",
            get(window, "forward_start_time", 0.0))),
        physics_start_time=Float64(get(window, "physics_start_time",
            get(window, "forward_start_time", 0.0))),
        trajectory_index=Int(window["trajectory_index"]),
        max_frames=Int(window["max_frames"]),
        width=Int(figure["width"]),
        height=Int(figure["height"]),
        font_scale=Float64(get(figure, "font_scale", 1.35)),
        output_png=resolve_path(base, String(output["png"])),
    )
end

function window_range(times::Vector{Float64}, start_time::Float64,
        duration::Float64, max_frames::Int)
    max_frames >= 4 || error("window.max_frames must be at least 4.")
    stop_time = min(times[end], start_time + duration)
    lo = searchsortedfirst(times, start_time)
    hi = searchsortedlast(times, stop_time)
    if lo > hi
        error(@sprintf("No saved frames in requested window [%.6g, %.6g].",
            start_time, stop_time))
    end
    stride = max(1, cld(hi - lo + 1, max_frames))
    return lo:stride:hi, stride
end

function load_hovmoller_window(path::AbstractString, label::LaTeXString,
        start_time::Float64, duration::Float64, trajectory_index::Int,
        max_frames::Int)
    isfile(path) || error("Missing Hovmoller input HDF5: $(path)")
    h5open(path, "r") do f
        times = Vector{Float64}(read(f["/trajectories/time"]))
        states_dset = f["/trajectories/states"]
        dims = size(states_dset)
        length(dims) == 4 || error("Expected /trajectories/states to have 4 dimensions in $(path).")
        1 <= trajectory_index <= dims[4] ||
            error("trajectory_index=$(trajectory_index) is outside 1:$(dims[4]) for $(path).")
        idx, stride = window_range(times, start_time, duration, max_frames)
        states = Array{Float32, 3}(states_dset[idx, :, :, trajectory_index])
        rel_time = Vector{Float64}(times[idx] .- times[first(idx)])
        return HovmollerWindow(label, path, times[first(idx)], times[last(idx)],
            rel_time, states, stride)
    end
end

function component_limits(windows::Vector{HovmollerWindow}, component::Int)
    values = Float64[]
    for win in windows
        append!(values, abs.(vec(Float64.(@view win.states[:, :, component]))))
    end
    lim = quantile(values, 0.995)
    lim = max(lim, 0.25)
    return (-lim, lim)
end

function render_hovmoller_comparison(cfg::HovmollerFigureConfig)
    windows = [
        load_hovmoller_window(cfg.observation_h5, latexstring("\\mathrm{observations}"),
            cfg.observation_start_time, cfg.duration, cfg.trajectory_index, cfg.max_frames),
        load_hovmoller_window(cfg.phi_h5, latexstring("M=\\Phi"),
            cfg.phi_start_time, cfg.duration, cfg.trajectory_index, cfg.max_frames),
        load_hovmoller_window(cfg.nn_h5, latexstring("M_{\\mathrm{NN}}"),
            cfg.nn_start_time, cfg.duration, cfg.trajectory_index, cfg.max_frames),
        load_hovmoller_window(cfg.physics_h5, latexstring("M_{\\mathrm{phys}}"),
            cfg.physics_start_time, cfg.duration, cfg.trajectory_index, cfg.max_frames),
    ]
    component_names = [latexstring("m_x"), latexstring("m_y"), latexstring("m_z")]
    row_limits = [component_limits(windows, c) for c in 1:3]
    nsites = size(windows[1].states, 2)
    site_ticks = unique([1, cld(nsites, 2), nsites])

    with_scaled_figure_style(cfg.width, cfg.height; scale_override=cfg.font_scale) do _
        fig = Figure(; size=(cfg.width, cfg.height))
        for row in 1:3
            last_hm = nothing
            for (col, win) in enumerate(windows)
                ax = Axis(fig[row, col];
                    title=(row == 1 ? win.label : ""),
                    xlabel=(row == 3 ? latexstring("t") : ""),
                    ylabel=(col == 1 ? "site" : ""),
                    xticklabelsvisible=(row == 3),
                    xticksvisible=(row == 3),
                    yticklabelsvisible=(col == 1),
                    yticksvisible=(col == 1),
                    xgridvisible=false,
                    ygridvisible=false,
                    yticks=site_ticks)
                last_hm = heatmap!(ax, win.times, 1:nsites,
                    Float64.(@view win.states[:, :, row]);
                    colormap=STYLE_DIVERGING,
                    colorrange=row_limits[row])
                xlims!(ax, 0.0, cfg.duration)
                ylims!(ax, 1, nsites)
            end
            Colorbar(fig[row, 5], last_hm;
                label=component_names[row],
                tellheight=false,
                width=18)
        end
        apply_publication_grid!(fig.layout, 3, 5;
            row_weights=[1.0, 1.0, 1.0],
            col_weights=[1.0, 1.0, 1.0, 1.0, 0.075],
            row_gap=24,
            col_gap=24)
        ensure_parent_dir(cfg.output_png)
        save_figure(cfg.output_png, fig)
    end

    @printf("Saved SoftSpin Hovmoller comparison to %s\n", cfg.output_png)
    for win in windows
        @printf("  %-18s start=%.6g stop=%.6g frames=%d stride=%d source=%s\n",
            String(win.label), win.start_time, win.stop_time,
            length(win.times), win.stride, win.source_path)
    end
    return cfg.output_png
end

function main()
    cfg_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_HOVMOLLER_CONFIG
    cfg = load_hovmoller_config(cfg_path)
    render_hovmoller_comparison(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
