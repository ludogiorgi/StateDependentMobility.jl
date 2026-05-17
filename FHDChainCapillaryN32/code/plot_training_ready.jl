#!/usr/bin/env julia

include(joinpath(@__DIR__, "sim.jl"))

using HDF5
using KernelDensity
using GLMakie
using LinearAlgebra
using Printf
using Random
using Statistics

const DEFAULT_H5 = normpath(joinpath(@__DIR__, "..", "data", "fhd_chain_capillary_n32_short.h5"))
const DEFAULT_OUT = normpath(joinpath(@__DIR__, "..", "figures", "training_ready_validation.png"))

function normal_pdf(xs::AbstractVector{<:Real}, sigma::Real=1.0)
    c = inv(float(sigma) * sqrt(2pi))
    return [c * exp(-0.5 * (x / sigma)^2) for x in xs]
end

function subset_values(values::AbstractVector{<:Real}, max_count::Int, rng::AbstractRNG)
    n = length(values)
    n <= max_count && return collect(Float64, values)
    idx = rand(rng, 1:n, max_count)
    return Float64[values[i] for i in idx]
end

function channel_scores(states::Array{Float64, 4}, start_idx::Int,
                        dx::Float64, theta::Float64, cs::Float64,
                        rho0::Float64, kappa::Float64)
    nt, nsites, _, ntraj = size(states)
    nvalues = (nt - start_idx + 1) * nsites * ntraj
    srho = Vector{Float64}(undef, nvalues)
    smom = Vector{Float64}(undef, nvalues)
    lap = zeros(Float64, nsites)
    score = zeros(Float64, nsites, 2)
    cursor = 1
    @inbounds for tr in 1:ntraj, t in start_idx:nt
        z = @view states[t, :, :, tr]
        for i in 1:nsites
            ip1 = periodic_index(i + 1, nsites)
            im1 = periodic_index(i - 1, nsites)
            lap[i] = (z[ip1, 1] - 2.0 * z[i, 1] + z[im1, 1]) / dx^2
        end
        mean_rho = 0.0
        mean_mom = 0.0
        for i in 1:nsites
            u = z[i, 2] / z[i, 1]
            score[i, 1] = -dx / theta * (cs^2 * log(z[i, 1] / rho0) - 0.5 * u^2 - kappa * lap[i])
            score[i, 2] = -dx / theta * u
            mean_rho += score[i, 1]
            mean_mom += score[i, 2]
        end
        mean_rho /= nsites
        mean_mom /= nsites
        for i in 1:nsites
            srho[cursor] = score[i, 1] - mean_rho
            smom[cursor] = score[i, 2] - mean_mom
            cursor += 1
        end
    end
    return srho, smom
end

function neighboring_density_pairs(states::Array{Float64, 4}, start_idx::Int,
                                   rho0::Float64, max_count::Int,
                                   rng::AbstractRNG)
    nt, nsites, _, ntraj = size(states)
    total = (nt - start_idx + 1) * nsites * ntraj
    count = min(total, max_count)
    x = Vector{Float64}(undef, count)
    y = Vector{Float64}(undef, count)
    @inbounds for sample in 1:count
        linear = rand(rng, 0:(total - 1))
        time_local = linear % (nt - start_idx + 1)
        tmp = linear ÷ (nt - start_idx + 1)
        site = (tmp % nsites) + 1
        traj = (tmp ÷ nsites) + 1
        t = start_idx + time_local
        jp1 = periodic_index(site + 1, nsites)
        x[sample] = (states[t, site, 1, traj] - rho0) / rho0
        y[sample] = (states[t, jp1, 1, traj] - rho0) / rho0
    end
    return x, y
end

function render_training_ready_validation(h5_path::AbstractString=DEFAULT_H5,
                                          out_png::AbstractString=DEFAULT_OUT)
    ensure_parent_dir(out_png)
    rng = MersenneTwister(90_211)
    h5open(h5_path, "r") do h5
        states = Array{Float64}(read(h5["/trajectories/states"]))
        nsites = Int(read(h5["/metadata/N"]))
        dx_value = Float64(read(h5["/metadata/dx"]))
        theta = Float64(read(h5["/metadata/Theta"]))
        cs = Float64(read(h5["/metadata/sound_speed"]))
        rho0 = Float64(read(h5["/metadata/rho0"]))
        kappa = Float64(read(h5["/metadata/kappa"]))
        xi_over_dx = Float64(read(h5["/metadata/density_correlation_length_dx"]))
        min_rho = Float64(read(h5["/statistics/conservation/min_density"]))
        sf_rmse = Float64(read(h5["/statistics/static_structure_factor/shape_rel_rmse"]))
        sf_corr = Float64(read(h5["/statistics/static_structure_factor/shape_correlation"]))
        tD = Float64(read(h5["/statistics/correlations/t_decorrelation"]))
        start_idx = floor(Int, 0.1 * size(states, 1)) + 1

        rho = vec(states[start_idx:end, :, 1, :])
        mom = vec(states[start_idx:end, :, 2, :])
        vel = mom ./ rho
        rel_rho = (rho .- rho0) ./ rho0
        mach = vel ./ cs

        sigma_rho = std(rho)
        sigma_mom = std(mom)
        sigma_rel_rho = std(rel_rho)
        sigma_mach = std(mach)
        z_rho = (rho .- mean(rho)) ./ sigma_rho
        z_mom = (mom .- mean(mom)) ./ sigma_mom

        srho, smom = channel_scores(states, start_idx, dx_value, theta, cs, rho0, kappa)
        scaled_srho = sigma_rho .* srho
        scaled_smom = sigma_mom .* smom

        theory_mom_std = sqrt((nsites - 1) / nsites * theta * rho0 / dx_value)
        mode_terms = Float64[]
        for n in 1:(nsites - 1)
            lambda_n = 4.0 * sin(pi * n / nsites)^2 / dx_value^2
            push!(mode_terms, inv(dx_value * (cs^2 / rho0 + kappa * lambda_n)))
        end
        theory_rho_std = sqrt(theta / nsites * sum(mode_terms))

        rho_pair_x, rho_pair_y = neighboring_density_pairs(states, start_idx, rho0, 160_000, rng)
        neighbor_corr = cor(rho_pair_x, rho_pair_y)

        rel_kde = kde(subset_values(rel_rho, 220_000, rng); npoints=220)
        mach_kde = kde(subset_values(mach, 220_000, rng); npoints=220)
        zrho_kde = kde(subset_values(z_rho, 220_000, rng); npoints=220)
        zmom_kde = kde(subset_values(z_mom, 220_000, rng); npoints=220)
        ssrho_kde = kde(subset_values(scaled_srho, 220_000, rng); npoints=220)
        ssmom_kde = kde(subset_values(scaled_smom, 220_000, rng); npoints=220)
        pair_kde = kde((rho_pair_x, rho_pair_y); npoints=(130, 130))

        fig = Figure(; size=(2500, 1700), fontsize=22)
        Label(fig[1, 1:3], "FHDChainCapillaryN32 training-ready physical validation";
            fontsize=34, font=:bold, tellwidth=false)
        Label(fig[2, 1:3],
            @sprintf("N=%d, c_s=%.3g, Theta=%.3g, xi/dx=%.1f, kappa=%.6g, t_D=%.3g",
                nsites, cs, theta, xi_over_dx, kappa, tD);
            fontsize=22, color=:gray35, tellwidth=false)

        ax1 = Axis(fig[3, 1]; title="Nondimensional marginals",
            xlabel="value", ylabel="density")
        lines!(ax1, rel_kde.x, rel_kde.density; color=STYLE_PRIMARY, linewidth=3,
            label="(rho-rho0)/rho0")
        lines!(ax1, mach_kde.x, mach_kde.density; color=STYLE_SECONDARY, linewidth=3,
            label="u/c_s")
        axislegend(ax1; position=:rt)

        ax2 = Axis(fig[3, 2]; title="Channel-normalized inputs",
            xlabel="z", ylabel="density")
        lines!(ax2, zrho_kde.x, zrho_kde.density; color=STYLE_PRIMARY, linewidth=3,
            label="rho channel")
        lines!(ax2, zmom_kde.x, zmom_kde.density; color=STYLE_SECONDARY, linewidth=3,
            label="m channel")
        xs = range(-4.5, 4.5; length=240)
        lines!(ax2, xs, normal_pdf(collect(xs)); color=STYLE_ZERO, linewidth=2,
            linestyle=:dash, label="N(0,1)")
        axislegend(ax2; position=:rt)

        ax3 = Axis(fig[3, 3]; title="Score targets in normalized coordinates",
            xlabel="sigma_channel * score_channel", ylabel="density")
        lines!(ax3, ssrho_kde.x, ssrho_kde.density; color=STYLE_PRIMARY, linewidth=3,
            label="density score")
        lines!(ax3, ssmom_kde.x, ssmom_kde.density; color=STYLE_SECONDARY, linewidth=3,
            label="momentum score")
        axislegend(ax3; position=:rt)

        ax4 = Axis(fig[4, 1]; title="Neighbor density coupling",
            xlabel="(rho_i-rho0)/rho0", ylabel="(rho_{i+1}-rho0)/rho0")
        heatmap!(ax4, pair_kde.x, pair_kde.y, pair_kde.density; colormap=STYLE_SEQUENTIAL_BLUE)
        text!(ax4, 0.02, 0.93; text=@sprintf("corr = %.3f", neighbor_corr),
            space=:relative, fontsize=24, color=:black)

        ax5 = Axis(fig[4, 2]; title="Equilibrium scale checks",
            xticks=(1:4, ["rho std", "m std", "Mach rms", "min rho"]),
            ylabel="value")
        values = [sigma_rho, sigma_mom, std(mach), min_rho]
        barplot!(ax5, 1:4, values; color=[STYLE_PRIMARY, STYLE_SECONDARY, STYLE_HIGHLIGHT, STYLE_ACCENT])
        text!(ax5, 0.04, 0.92;
            text=@sprintf("rho std / theory = %.3f\nm std / theory = %.3f",
                sigma_rho / theory_rho_std, sigma_mom / theory_mom_std),
            space=:relative, fontsize=22, color=:black)

        ax6 = Axis(fig[4, 3]; title="Why this is a defensible training set")
        hidedecorations!(ax6)
        hidespines!(ax6)
        lines!(ax6, [0, 1], [0, 0]; color=(:white, 0.0))
        xlims!(ax6, 0, 1)
        ylims!(ax6, 0, 1)
        summary_text = join([
            @sprintf("rho relative std = %.4f", sigma_rel_rho),
            @sprintf("u/c_s rms = %.4f", sigma_mach),
            @sprintf("min rho = %.4f", min_rho),
            @sprintf("S_rho shape corr = %.5f", sf_corr),
            @sprintf("S_rho rel.RMSE = %.3e", sf_rmse),
            @sprintf("neighbor rho corr = %.4f", neighbor_corr),
            @sprintf("std(sigma_rho*s_rho) = %.3f", std(scaled_srho)),
            @sprintf("std(sigma_m*s_m) = %.3f", std(scaled_smom)),
        ], "\n")
        text!(ax6, 0.02, 0.94; text=summary_text, align=(:left, :top),
            fontsize=25, color=:black)

        colgap!(fig.layout, 40)
        rowgap!(fig.layout, 24)
        save(out_png, fig; px_per_unit=1.5)
        @printf("saved %s\n", out_png)
        @printf("rho_rel_std=%.8f mach_rms=%.8f min_rho=%.8f neighbor_corr=%.8f\n",
            sigma_rel_rho, sigma_mach, min_rho, neighbor_corr)
        @printf("scaled_score_std=(%.8f, %.8f), sf_corr=%.8f\n",
            std(scaled_srho), std(scaled_smom), sf_corr)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    h5_path = length(ARGS) >= 1 ? abspath(ARGS[1]) : DEFAULT_H5
    out_png = length(ARGS) >= 2 ? abspath(ARGS[2]) : DEFAULT_OUT
    render_training_ready_validation(h5_path, out_png)
end
