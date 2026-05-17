#!/usr/bin/env julia

include(joinpath(@__DIR__, "sim.jl"))

using HDF5
using GLMakie
using Printf
using Statistics

const DEFAULT_H5 = normpath(joinpath(@__DIR__, "..", "data", "fhd_chain_capillary_n32_short.h5"))
const DEFAULT_OUT = normpath(joinpath(@__DIR__, "..", "figures", "rho_m_hovmoller_5tD.png"))

function render_rho_m_hovmoller(h5_path::AbstractString=DEFAULT_H5,
                                out_png::AbstractString=DEFAULT_OUT;
                                traj::Int=1, window_tD::Float64=5.0)
    ensure_parent_dir(out_png)
    h5open(h5_path, "r") do h5
        time = Vector{Float64}(read(h5["/trajectories/time"]))
        save_dt = Float64(read(h5["/metadata/save_dt"]))
        rho0 = Float64(read(h5["/metadata/rho0"]))
        kappa = Float64(read(h5["/metadata/kappa"]))
        xi_over_dx = Float64(read(h5["/metadata/density_correlation_length_dx"]))
        tD = Float64(read(h5["/statistics/correlations/t_decorrelation"]))
        nt = length(time)
        start_idx = floor(Int, 0.1 * nt) + 1
        nframes = round(Int, window_tD * tD / save_dt) + 1
        stop_idx = min(nt, start_idx + nframes - 1)
        idxs = start_idx:stop_idx
        block = Array{Float64}(h5["/trajectories/states"][idxs, :, :, traj])
        tt = (time[idxs] .- first(time[idxs])) ./ tD
        rho_fluc = block[:, :, 1] .- rho0
        mom = block[:, :, 2]
        sites = collect(1:size(rho_fluc, 2))

        fig = Figure(; size=(3000, 1300), fontsize=22)
        Label(fig[1, 1:4], "FHDChainCapillaryN32: physically scaled capillary run";
            fontsize=34, font=:bold, tellwidth=false)
        Label(fig[2, 1:4],
            @sprintf("trajectory %d, %.2f pilot t_D window, %d saved frames x %d sites, xi/dx=%.1f, kappa=%.4f",
                traj, last(tt), length(idxs), length(sites), xi_over_dx, kappa);
            fontsize=23, color=:gray35, tellwidth=false)

        ax1 = Axis(fig[3, 1]; title="density fluctuation rho_i(t)-rho0",
            xlabel="time / pilot t_D", ylabel="site i", titlesize=28,
            xlabelsize=24, ylabelsize=24, yticks=1:4:length(sites))
        ax2 = Axis(fig[3, 3]; title="momentum m_i(t)",
            xlabel="time / pilot t_D", ylabel="site i", titlesize=28,
            xlabelsize=24, ylabelsize=24, yticks=1:4:length(sites))
        rlim = quantile(abs.(vec(rho_fluc)), 0.995)
        mlim = quantile(abs.(vec(mom)), 0.995)
        hm1 = heatmap!(ax1, tt, sites, rho_fluc; colormap=STYLE_DIVERGING,
            colorrange=(-rlim, rlim), interpolate=false)
        hm2 = heatmap!(ax2, tt, sites, mom; colormap=STYLE_DIVERGING,
            colorrange=(-mlim, mlim), interpolate=false)
        Colorbar(fig[3, 2], hm1; label="rho-rho0", labelsize=22, ticklabelsize=20)
        Colorbar(fig[3, 4], hm2; label="m", labelsize=22, ticklabelsize=20)
        xlims!(ax1, first(tt), last(tt))
        xlims!(ax2, first(tt), last(tt))
        ylims!(ax1, 0.5, length(sites) + 0.5)
        ylims!(ax2, 0.5, length(sites) + 0.5)
        colsize!(fig.layout, 1, Relative(0.47))
        colsize!(fig.layout, 3, Relative(0.47))
        colsize!(fig.layout, 2, Fixed(70))
        colsize!(fig.layout, 4, Fixed(70))
        colgap!(fig.layout, 25)
        rowgap!(fig.layout, 10)
        save(out_png, fig; px_per_unit=1.5)
        @printf("saved %s\n", out_png)
        @printf("rho matrix size plotted as time x site = %s, m matrix size = %s\n",
            string(size(rho_fluc)), string(size(mom)))
        @printf("x values=%d, y values=%d, plotted time/tD %.3f\n",
            length(tt), length(sites), last(tt))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    h5_path = length(ARGS) >= 1 ? abspath(ARGS[1]) : DEFAULT_H5
    out_png = length(ARGS) >= 2 ? abspath(ARGS[2]) : DEFAULT_OUT
    render_rho_m_hovmoller(h5_path, out_png)
end
