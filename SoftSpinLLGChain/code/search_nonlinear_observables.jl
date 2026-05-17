#!/usr/bin/env julia

if !isdefined(@__MODULE__, :CondScoreConfig)
    include(joinpath(@__DIR__, "cond_score.jl"))
end

using LinearAlgebra
using Printf
using Statistics

const COMP_NAMES = ("mx", "my", "mz")

struct NonlinearLibrary
    names::Vector{String}
end

function nonlinear_candidate_names(family::AbstractString="baseline")
    names = String[]
    for a in COMP_NAMES, b in COMP_NAMES
        push!(names, "$(a)_$(b)2")
    end
    append!(names, ["mx_r2", "my_r2", "mz_r2",
        "mx_mperp2", "my_mperp2", "mz_mperp2",
        "mx_mz2", "my_mz2", "mz_mz2",
        "mx_mz", "my_mz", "mx_my", "r2", "mperp2", "mz2"])
    for a in COMP_NAMES, b in COMP_NAMES
        push!(names, "$(a)_$(b)_p")
    end
    for a in COMP_NAMES, b in COMP_NAMES
        push!(names, "$(a)_$(b)_nnavg")
    end
    append!(names, ["dot_p", "dot_m", "cross_p_x", "cross_p_y", "cross_p_z"])
    for a in COMP_NAMES
        append!(names, ["$(a)_dot_p", "$(a)_dot_m", "$(a)_grad2",
            "$(a)_r2_p", "$(a)_r2_nnavg", "$(a)_Uloc",
            "$(a)_cross_p_x", "$(a)_cross_p_y", "$(a)_cross_p_z"])
    end
    if family in ("poly_high", "all")
        for a in COMP_NAMES
            append!(names, ["$(a)_r4", "$(a)_mperp4", "$(a)_mz4",
                "$(a)_r2_mperp2", "$(a)_r2_mz2", "$(a)_mperp2_mz2",
                "$(a)_amp_dev", "$(a)_amp_dev2", "$(a)_Uloc2"])
            for b in COMP_NAMES, c in COMP_NAMES
                push!(names, "$(a)_$(b)2_$(c)2")
            end
        end
        append!(names, ["r4", "mperp4", "mz4", "r2_mperp2", "r2_mz2",
            "mperp2_mz2", "amp_dev", "amp_dev2", "Uloc2"])
    end
    if family in ("neighbor_high", "all")
        for a in COMP_NAMES
            append!(names, ["$(a)_dot_pm", "$(a)_lap2", "$(a)_neighbor_r2diff",
                "$(a)_neighbor_r2sum", "$(a)_nn_align", "$(a)_twist2"])
            for b in COMP_NAMES
                append!(names, ["$(a)_lap_$(b)", "$(a)_gradp_$(b)",
                    "$(a)_gradm_$(b)", "$(a)_cross_m_$(b)"])
            end
        end
        append!(names, ["dot_pm", "lap2", "neighbor_r2diff", "neighbor_r2sum",
            "nn_align", "twist2", "cross_m_x", "cross_m_y", "cross_m_z"])
    end
    return unique(names)
end

function nonlinear_observables(raw::Array{Float32, 3}, p::SpinParams, lib::NonlinearLibrary)
    N, _, B = size(raw)
    obs = Array{Float32}(undef, N, length(lib.names), B)
    name_to_idx = Dict(name => idx for (idx, name) in enumerate(lib.names))
    @inbounds for b in 1:B
        for i in 1:N
            im = periodic(i - 1, N)
            ip = periodic(i + 1, N)
            x = ntuple(c -> Float64(raw[i, c, b]), 3)
            xm = ntuple(c -> Float64(raw[im, c, b]), 3)
            xp = ntuple(c -> Float64(raw[ip, c, b]), 3)
            r2 = x[1]^2 + x[2]^2 + x[3]^2
            r2p = xp[1]^2 + xp[2]^2 + xp[3]^2
            r2m = xm[1]^2 + xm[2]^2 + xm[3]^2
            mperp2 = x[1]^2 + x[2]^2
            dotp = x[1] * xp[1] + x[2] * xp[2] + x[3] * xp[3]
            dotm = x[1] * xm[1] + x[2] * xm[2] + x[3] * xm[3]
            dotpm = xp[1] * xm[1] + xp[2] * xm[2] + xp[3] * xm[3]
            diffp2 = (xp[1] - x[1])^2 + (xp[2] - x[2])^2 + (xp[3] - x[3])^2
            diffm2 = (x[1] - xm[1])^2 + (x[2] - xm[2])^2 + (x[3] - xm[3])^2
            grad2 = diffp2 + diffm2
            crossp = cross3(x[1], x[2], x[3], xp[1], xp[2], xp[3])
            crossm = cross3(x[1], x[2], x[3], xm[1], xm[2], xm[3])
            lap = ntuple(c -> xp[c] + xm[c] - 2.0 * x[c], 3)
            lap2 = lap[1]^2 + lap[2]^2 + lap[3]^2
            twist2 = crossp[1]^2 + crossp[2]^2 + crossp[3]^2 +
                crossm[1]^2 + crossm[2]^2 + crossm[3]^2
            Uloc = 0.25 * p.lambda * (r2 - p.mstar^2)^2 +
                0.25 * p.J * (diffp2 + diffm2) - 0.5 * p.K * x[3]^2
            amp_dev = r2 - p.mstar^2

            for (ai, a) in enumerate(COMP_NAMES), (bi, cname) in enumerate(COMP_NAMES)
                haskey(name_to_idx, "$(a)_$(cname)2") &&
                    (obs[i, name_to_idx["$(a)_$(cname)2"], b] = Float32(x[ai] * x[bi]^2))
                haskey(name_to_idx, "$(a)_$(cname)_p") &&
                    (obs[i, name_to_idx["$(a)_$(cname)_p"], b] = Float32(x[ai] * xp[bi]))
                haskey(name_to_idx, "$(a)_$(cname)_nnavg") &&
                    (obs[i, name_to_idx["$(a)_$(cname)_nnavg"], b] = Float32(0.5 * x[ai] * (xp[bi] + xm[bi])))
            end

            vals = Dict(
                "mx_r2" => x[1] * r2, "my_r2" => x[2] * r2, "mz_r2" => x[3] * r2,
                "mx_mperp2" => x[1] * mperp2, "my_mperp2" => x[2] * mperp2,
                "mz_mperp2" => x[3] * mperp2,
                "mx_mz2" => x[1] * x[3]^2, "my_mz2" => x[2] * x[3]^2,
                "mz_mz2" => x[3]^3,
                "mx_mz" => x[1] * x[3], "my_mz" => x[2] * x[3],
                "mx_my" => x[1] * x[2],
                "r2" => r2, "mperp2" => mperp2, "mz2" => x[3]^2,
                "dot_p" => dotp, "dot_m" => dotm,
                "cross_p_x" => crossp[1], "cross_p_y" => crossp[2],
                "cross_p_z" => crossp[3],
                "r4" => r2^2, "mperp4" => mperp2^2, "mz4" => x[3]^4,
                "r2_mperp2" => r2 * mperp2, "r2_mz2" => r2 * x[3]^2,
                "mperp2_mz2" => mperp2 * x[3]^2,
                "amp_dev" => amp_dev, "amp_dev2" => amp_dev^2,
                "Uloc2" => Uloc^2,
                "dot_pm" => dotpm, "lap2" => lap2,
                "neighbor_r2diff" => r2p - r2m,
                "neighbor_r2sum" => r2p + r2m,
                "nn_align" => dotp + dotm,
                "twist2" => twist2,
                "cross_m_x" => crossm[1], "cross_m_y" => crossm[2],
                "cross_m_z" => crossm[3],
            )
            for (name, val) in vals
                haskey(name_to_idx, name) && (obs[i, name_to_idx[name], b] = Float32(val))
            end
            for (ai, a) in enumerate(COMP_NAMES)
                vals2 = Dict(
                    "$(a)_dot_p" => x[ai] * dotp,
                    "$(a)_dot_m" => x[ai] * dotm,
                    "$(a)_grad2" => x[ai] * grad2,
                    "$(a)_r2_p" => x[ai] * r2p,
                    "$(a)_r2_nnavg" => 0.5 * x[ai] * (r2p + r2m),
                    "$(a)_Uloc" => x[ai] * Uloc,
                    "$(a)_cross_p_x" => x[ai] * crossp[1],
                    "$(a)_cross_p_y" => x[ai] * crossp[2],
                    "$(a)_cross_p_z" => x[ai] * crossp[3],
                    "$(a)_r4" => x[ai] * r2^2,
                    "$(a)_mperp4" => x[ai] * mperp2^2,
                    "$(a)_mz4" => x[ai] * x[3]^4,
                    "$(a)_r2_mperp2" => x[ai] * r2 * mperp2,
                    "$(a)_r2_mz2" => x[ai] * r2 * x[3]^2,
                    "$(a)_mperp2_mz2" => x[ai] * mperp2 * x[3]^2,
                    "$(a)_amp_dev" => x[ai] * amp_dev,
                    "$(a)_amp_dev2" => x[ai] * amp_dev^2,
                    "$(a)_Uloc2" => x[ai] * Uloc^2,
                    "$(a)_dot_pm" => x[ai] * dotpm,
                    "$(a)_lap2" => x[ai] * lap2,
                    "$(a)_neighbor_r2diff" => x[ai] * (r2p - r2m),
                    "$(a)_neighbor_r2sum" => x[ai] * (r2p + r2m),
                    "$(a)_nn_align" => x[ai] * (dotp + dotm),
                    "$(a)_twist2" => x[ai] * twist2,
                    "$(a)_cross_m_x" => x[ai] * crossm[1],
                    "$(a)_cross_m_y" => x[ai] * crossm[2],
                    "$(a)_cross_m_z" => x[ai] * crossm[3],
                )
                for (name, val) in vals2
                    haskey(name_to_idx, name) && (obs[i, name_to_idx[name], b] = Float32(val))
                end
                for (bi, cname) in enumerate(COMP_NAMES)
                    vals3 = Dict(
                        "$(a)_lap_$(cname)" => x[ai] * lap[bi],
                        "$(a)_gradp_$(cname)" => x[ai] * (xp[bi] - x[bi]),
                        "$(a)_gradm_$(cname)" => x[ai] * (x[bi] - xm[bi]),
                        "$(a)_cross_m_$(cname)" => x[ai] * crossm[bi],
                    )
                    for (name, val) in vals3
                        haskey(name_to_idx, name) && (obs[i, name_to_idx[name], b] = Float32(val))
                    end
                    for (ci, dname) in enumerate(COMP_NAMES)
                        name = "$(a)_$(cname)2_$(dname)2"
                        haskey(name_to_idx, name) &&
                            (obs[i, name_to_idx[name], b] = Float32(x[ai] * x[bi]^2 * x[ci]^2))
                    end
                end
            end
        end
    end
    return obs
end

function sample_fixed_lag_window(sampler::CondPairSampler, lag::Int, npairs::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(sampler.states)
    x0 = Array{Float32}(undef, N, 3, npairs)
    xt = Array{Float32}(undef, N, 3, npairs)
    xp = Array{Float32}(undef, N, 3, npairs)
    xm = Array{Float32}(undef, N, 3, npairs)
    lower = sampler.start_idx + 1
    upper = nt - lag - 1
    require_condition(upper >= lower, "Requested lag/window exceeds available trajectory.")
    @inbounds for b in 1:npairs
        t = rand(rng, lower:upper)
        tr = rand(rng, 1:ntraj)
        x0[:, :, b] .= sampler.states[t, :, :, tr]
        xt[:, :, b] .= sampler.states[t + lag, :, :, tr]
        xp[:, :, b] .= sampler.states[t + lag + 1, :, :, tr]
        xm[:, :, b] .= sampler.states[t + lag - 1, :, :, tr]
    end
    tau_norm = fill(Float32(lag * sampler.save_dt / sampler.tau_max), npairs)
    return x0, xt, xp, xm, tau_norm
end

function estimate_nonlinear_means(sampler::CondPairSampler, p::SpinParams,
        lib::NonlinearLibrary, nsamples::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(sampler.states)
    raw = Array{Float32}(undef, N, 3, nsamples)
    @inbounds for b in 1:nsamples
        t = rand(rng, sampler.start_idx:nt)
        tr = rand(rng, 1:ntraj)
        raw[:, :, b] .= sampler.states[t, :, :, tr]
    end
    obs = nonlinear_observables(raw, p, lib)
    return [mean(Float64, @view obs[:, j, :]) for j in eachindex(lib.names)]
end

function center_observables!(obs::Array{Float32, 3}, means::Vector{Float64})
    @inbounds for j in eachindex(means)
        obs[:, j, :] .-= Float32(means[j])
    end
    return obs
end

function estimate_library_cdot(model, sampler::CondPairSampler, score_model, stats::DataStats,
        score_sigma::Float32, p::SpinParams, params::CondScoreConfig,
        lib::NonlinearLibrary, means::Vector{Float64}, lags::Vector{Int},
        npairs::Int, device::ExecutionDevice)
    rng = MersenneTwister(params.seed + 91_000)
    nobs = length(lib.names)
    Cdata = Array{Float64}(undef, length(lags), sampler.N, nobs, sampler.D)
    Ctrue = similar(Cdata)
    mu = Float64.(mean_flat(stats))
    for (li, lag) in enumerate(lags)
        x0, xt, xp, xm, tau_norm = sample_fixed_lag_window(sampler, lag, npairs, rng)
        x0f = Float64.(flatten_batch(x0))
        x0f .-= mu

        obsp = nonlinear_observables(xp, p, lib)
        obsm = nonlinear_observables(xm, p, lib)
        deriv = (obsp .- obsm) ./ Float32(2.0 * sampler.save_dt)
        deriv_flat = reshape(deriv, sampler.N * nobs, npairs)
        Cdata[li, :, :, :] .= reshape(Matrix{Float64}(deriv_flat) * transpose(x0f) ./ npairs,
            sampler.N, nobs, sampler.D)

        rnorm = evaluate_residual_norm(model, x0, xt, tau_norm, stats, params, device;
            batch_size=params.batch_size, score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        action = true_action_batch(x0, rraw, p)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_flat = reshape(obs, sampler.N * nobs, npairs)
        action_flat = flatten_batch(action)
        Ctrue[li, :, :, :] .= reshape(-Matrix{Float64}(obs_flat) *
            transpose(Matrix{Float64}(action_flat)) ./ npairs, sampler.N, nobs, sampler.D)
        @printf("Nonlinear library lag %.5g (%d/%d), %d observables, %d pairs\n",
            lag * sampler.save_dt, li, length(lags), nobs, npairs)
        GC.gc(false)
    end
    return Cdata, Ctrue
end

function translation_profiles(Craw::Array{Float64, 4})
    nlags, nsites, nobs, dim = size(Craw)
    ncomp = 3
    profiles = zeros(Float64, nlags, nobs, ncomp, nsites)
    counts = zeros(Int, nsites)
    for i in 1:nsites, j in 1:nsites
        offset = mod(j - i, nsites) + 1
        counts[offset] += 1
        for c in 1:ncomp, a in 1:nobs, li in 1:nlags
            col = (j - 1) * ncomp + c
            profiles[li, a, c, offset] += Craw[li, i, a, col]
        end
    end
    for offset in 1:nsites
        profiles[:, :, :, offset] ./= counts[offset]
    end
    return profiles
end

function metric_rows(names, targets, data_prof, true_prof)
    rows = NamedTuple[]
    max_rms = maximum([sqrt(mean((@view data_prof[:, a, c, :]) .^ 2))
        for a in axes(data_prof, 2), c in axes(data_prof, 3)])
    for a in axes(data_prof, 2), c in axes(data_prof, 3)
        D = @view data_prof[:, a, c, :]
        T = @view true_prof[:, a, c, :]
        data_rms = sqrt(mean(D .^ 2))
        true_rms = sqrt(mean(T .^ 2))
        err = sqrt(mean((D .- T) .^ 2))
        corr = dot(vec(D), vec(T)) / max(norm(vec(D)) * norm(vec(T)), eps(Float64))
        push!(rows, (; observable=names[a], observable_index=a,
            target=targets[c], target_index=c, rel_rmse=err / max(data_rms, eps(Float64)),
            corr=corr, data_rms=data_rms, true_rms=true_rms,
            signal_fraction=data_rms / max(max_rms, eps(Float64))))
    end
    return rows
end

function select_rows(rows; corr_min=0.90, rel_max=0.50, signal_fraction_min=0.05)
    return [row.corr >= corr_min && row.rel_rmse <= rel_max &&
        row.signal_fraction >= signal_fraction_min for row in rows]
end

function write_metrics(path, rows, keep; corr_min, rel_max, signal_fraction_min)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain nonlinear observable library search")
        println(io, @sprintf("Selection: corr >= %.3f, rel.RMSE <= %.3f, data_rms >= %.3f of strongest searched nonlinear channel",
            corr_min, rel_max, signal_fraction_min))
        println(io)
        for (row, iskeep) in zip(rows, keep)
            status = iskeep ? "KEEP" : "reject"
            reason = iskeep ? "accepted" :
                row.signal_fraction < signal_fraction_min ? "weak data signal" :
                row.corr < corr_min ? "bad shape correlation" :
                "large relative error"
            println(io, @sprintf("%-6s %-18s -> %-2s rel.RMSE %.8e corr %.8f data_rms %.8e signal_frac %.8e true_rms %.8e  %s",
                status, row.observable, row.target, row.rel_rmse, row.corr,
                row.data_rms, row.signal_fraction, row.true_rms, reason))
        end
    end
end

function write_retained_toml(path, rows, keep; cond_path, corr_min, rel_max, signal_fraction_min)
    open(path, "w") do io
        println(io, "# Generated by code/search_nonlinear_observables.jl.")
        println(io, "# Retained nonlinear observable channels for the conditional-score true-M operator diagnostic.")
        println(io, "# True M is used only for this ex-post diagnostic/filter.")
        println(io, "source_cond_score = \"$(relpath(cond_path, dirname(path)))\"")
        println(io, @sprintf("corr_min = %.8f", corr_min))
        println(io, @sprintf("rel_rmse_max = %.8f", rel_max))
        println(io, @sprintf("signal_fraction_min = %.8f", signal_fraction_min))
        println(io, "translation_offsets = \"all\"")
        println(io)
        for (row, iskeep) in zip(rows, keep)
            iskeep || continue
            println(io, "[[channels]]")
            println(io, "observable = \"$(row.observable)\"")
            println(io, "target_component = \"$(row.target)\"")
            println(io, "observable_index = $(row.observable_index)")
            println(io, "target_component_index = $(row.target_index)")
            println(io, @sprintf("rel_rmse = %.8e", row.rel_rmse))
            println(io, @sprintf("corr = %.8f", row.corr))
            println(io, @sprintf("data_rms = %.8e", row.data_rms))
            println(io, @sprintf("signal_fraction = %.8e", row.signal_fraction))
            println(io)
        end
    end
end

function render_retained(path, rows, keep, data_prof, true_prof, taus)
    kept = findall(keep)
    isempty(kept) && error("No nonlinear observable channels passed selection.")
    sort!(kept; by=i -> (rows[i].rel_rmse, -rows[i].corr))
    shown = kept[1:min(length(kept), 36)]
    palette = Makie.wong_colors()
    ncols = min(3, length(shown))
    nrows = cld(length(shown), ncols)
    fig = Figure(; size=(1200 * ncols, 680 * nrows + 300))
    Label(fig[0, 1:ncols],
        "SoftSpinLLGChain nonlinear observable search: retained Cdot_mn(t)";
        fontsize=30, tellwidth=false)
    for (panel, idx) in enumerate(shown)
        row = rows[idx]
        gr = 1 + (panel - 1) ÷ ncols
        gc = 1 + (panel - 1) % ncols
        ax = Axis(fig[gr, gc];
            title=@sprintf("%s -> %s   rel %.2f corr %.2f",
                row.observable, row.target, row.rel_rmse, row.corr),
            xlabel="tau", ylabel="Cdot")
        for offset in axes(data_prof, 4)
            color = palette[mod1(offset, length(palette))]
            lines!(ax, taus, data_prof[:, row.observable_index, row.target_index, offset];
                color=(color, 0.9), linewidth=1.9)
            lines!(ax, taus, true_prof[:, row.observable_index, row.target_index, offset];
                color=(color, 0.9), linewidth=1.9, linestyle=:dash)
        end
    end
    Legend(fig[nrows + 1, 1:ncols],
        [LineElement(; color=:black, linestyle=:solid, linewidth=3),
            LineElement(; color=:black, linestyle=:dash, linewidth=3)],
        ["data-driven Cdot", "cond score + true M"];
        orientation=:horizontal, tellwidth=false, framevisible=false, labelsize=22)
    Label(fig[nrows + 2, 1:ncols],
        @sprintf("Showing the best %d of %d retained channels by rel.RMSE; the full retained list is in the TOML/metrics files.",
            length(shown), length(kept));
        fontsize=18, tellwidth=false)
    save_figure_checked(path, fig)
end

function render_summary(path, rows, keep)
    sorted = sortperm(collect(eachindex(rows)); by=i -> (keep[i] ? 0 : 1, rows[i].rel_rmse, -rows[i].corr))
    topn = min(length(sorted), 80)
    fig = Figure(; size=(3000, 1800))
    Label(fig[0, 1:2], "Nonlinear observable Cdot search summary";
        fontsize=30, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Agreement scatter", xlabel="rel.RMSE", ylabel="correlation")
    for (i, row) in enumerate(rows)
        color = keep[i] ? STYLE_PRIMARY : (:gray55, 0.45)
        marker = keep[i] ? :circle : :xcross
        scatter!(ax1, [row.rel_rmse], [row.corr]; color=color, marker=marker, markersize=12)
    end
    vlines!(ax1, [0.5]; color=:black, linestyle=:dash)
    hlines!(ax1, [0.9]; color=:black, linestyle=:dash)
    xlims!(ax1, 0, min(8, maximum(row.rel_rmse for row in rows)))
    ax2 = Axis(fig[1, 2]; title="Data signal", xlabel="signal fraction", ylabel="correlation")
    for (i, row) in enumerate(rows)
        scatter!(ax2, [row.signal_fraction], [row.corr];
            color=keep[i] ? STYLE_PRIMARY : (:gray55, 0.45),
            marker=keep[i] ? :circle : :xcross, markersize=12)
    end
    vlines!(ax2, [0.05]; color=:black, linestyle=:dash)
    hlines!(ax2, [0.9]; color=:black, linestyle=:dash)
    ax3 = Axis(fig[2, 1:2]; title="Top searched channels", xlabel="rank", ylabel="rel.RMSE / corr")
    xs = collect(1:topn)
    lines!(ax3, xs, [rows[sorted[i]].rel_rmse for i in 1:topn]; color=STYLE_HIGHLIGHT, linewidth=2, label="rel.RMSE")
    lines!(ax3, xs, [rows[sorted[i]].corr for i in 1:topn]; color=STYLE_PRIMARY, linewidth=2, label="corr")
    axislegend(ax3; position=:rt)
    save_figure_checked(path, fig)
end

function main()
    family = length(ARGS) >= 1 ? String(ARGS[1]) : "baseline"
    suffix = length(ARGS) >= 2 ? String(ARGS[2]) : (family == "baseline" ? "" : "_" * family)
    device_request = length(ARGS) >= 3 ? String(ARGS[3]) : ""
    required_gpu = length(ARGS) >= 4 ? String(ARGS[4]) : ""
    config_path = joinpath(@__DIR__, "..", "configs", "cond_score_gpu0_vA.toml")
    params = load_config(config_path)
    base = dirname(config_path)
    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    cond_path = resolve_path(base, params.output_bson)
    isempty(device_request) && (device_request = params.device)
    isempty(required_gpu) && (required_gpu = params.required_gpu_name)
    device = detect_spin_device(device_request, required_gpu)
    activate_and_describe_device!(device, device_request, required_gpu)

    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    cond = BSON.load(cond_path)
    model = move_model(cond[:host_model], device)
    Flux.testmode!(model)

    lib = NonlinearLibrary(nonlinear_candidate_names(family))
    @printf("Searching nonlinear observable family %s with %d observables.\n",
        family, length(lib.names))
    lags = sampler.lag_steps[1:min(params.operator_lag_count, length(sampler.lag_steps))]
    npairs = params.operator_pairs_per_lag
    means = estimate_nonlinear_means(sampler, p, lib, min(160000, 4npairs),
        MersenneTwister(params.seed + 90_000))
    Cdata, Ctrue = estimate_library_cdot(model, sampler, score_model, stats, score_sigma,
        p, params, lib, means, lags, npairs, device)

    data_prof = translation_profiles(Cdata)
    true_prof = translation_profiles(Ctrue)
    targets = ["mx", "my", "mz"]
    rows = metric_rows(lib.names, targets, data_prof, true_prof)
    corr_min = 0.90
    rel_max = 0.50
    signal_fraction_min = 0.05
    keep = select_rows(rows; corr_min, rel_max, signal_fraction_min)

    metrics_path = joinpath(@__DIR__, "..", "logs", "nonlinear_observable_search$(suffix)_metrics.txt")
    toml_path = joinpath(@__DIR__, "..", "configs", "nonlinear_observable_retained_channels$(suffix).toml")
    retained_png = joinpath(@__DIR__, "..", "figures", "nonlinear_observable_retained_Cdot_profiles$(suffix).png")
    summary_png = joinpath(@__DIR__, "..", "figures", "nonlinear_observable_search_summary$(suffix).png")
    artifact_path = joinpath(@__DIR__, "..", "models", "nonlinear_observable_search$(suffix)_artifacts.bson")

    write_metrics(metrics_path, rows, keep; corr_min, rel_max, signal_fraction_min)
    write_retained_toml(toml_path, rows, keep; cond_path, corr_min, rel_max, signal_fraction_min)
    render_summary(summary_png, rows, keep)
    render_retained(retained_png, rows, keep, data_prof, true_prof, lags .* sampler.save_dt)
    BSON.bson(artifact_path, Dict(:names => lib.names, :means => means, :lags => lags,
        :taus => lags .* sampler.save_dt, :Cdot_data => Cdata, :Cdot_trueM_cond_score => Ctrue,
        :metric_rows => rows, :keep => keep,
        :family => family, :device_request => device_request, :required_gpu => required_gpu,
        :selection => Dict(:corr_min => corr_min, :rel_rmse_max => rel_max,
            :signal_fraction_min => signal_fraction_min),
        :no_cheating_audit => "Observable Cdot_data was estimated from trajectory finite differences. True mobility was used only for ex-post conditional-score operator comparison and channel filtering."))
    @printf("Retained %d/%d nonlinear observable-target channels.\n", count(identity, keep), length(keep))
    @printf("Saved metrics to %s\n", metrics_path)
    @printf("Saved retained channels to %s\n", toml_path)
    @printf("Saved retained figure to %s\n", retained_png)
    @printf("Saved summary figure to %s\n", summary_png)
    @printf("Saved artifacts to %s\n", artifact_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
