#!/usr/bin/env julia

# Fast selected-observable helpers for the 46-channel mobility target branch.
# These routines intentionally evaluate only the observable names requested by
# the retained-channel TOML.  They match nonlinear_observables from
# search_nonlinear_observables.jl for the supported names, but avoid per-site
# Dict allocations in long target-estimation runs.

using Base.Threads

function fill_selected_observables!(obs::Array{Float32, 3},
        raw::Array{Float32, 3}, p::SpinParams, names::Vector{String})
    N, _, B = size(raw)
    size(obs, 1) == N || error("obs site dimension mismatch")
    size(obs, 2) == length(names) || error("obs name dimension mismatch")
    size(obs, 3) == B || error("obs sample dimension mismatch")
    fill!(obs, 0f0)
    name_to_idx = Dict(name => idx for (idx, name) in enumerate(names))
    idx(name) = get(name_to_idx, name, 0)

    i_mz_gradm_my = idx("mz_gradm_my")
    i_mz_lap_my = idx("mz_lap_my")
    i_cross_m_x = idx("cross_m_x")
    i_cross_p_x = idx("cross_p_x")
    i_mx_r4 = idx("mx_r4")
    i_mz_gradp_my = idx("mz_gradp_my")
    i_mx_r2 = idx("mx_r2")
    i_mx_mx2 = idx("mx_mx2")
    i_mx_r2_nnavg = idx("mx_r2_nnavg")
    i_mx_r2_mperp2 = idx("mx_r2_mperp2")
    i_mx_mperp2 = idx("mx_mperp2")
    i_my_r2_mperp2 = idx("my_r2_mperp2")
    i_my_mperp2 = idx("my_mperp2")
    i_my_r4 = idx("my_r4")
    i_my_r2 = idx("my_r2")
    i_my_twist2 = idx("my_twist2")
    i_my_mperp4 = idx("my_mperp4")
    i_my_my2 = idx("my_my2")
    i_my_grad2 = idx("my_grad2")
    i_my_mx2 = idx("my_mx2")
    i_my_lap2 = idx("my_lap2")
    i_my_r2_nnavg = idx("my_r2_nnavg")
    i_my_mx_nnavg = idx("my_mx_nnavg")
    i_cross_p_z = idx("cross_p_z")
    i_cross_m_z = idx("cross_m_z")
    i_mz_lap2 = idx("mz_lap2")
    i_mx_cross_m_my = idx("mx_cross_m_my")
    i_mz_grad2 = idx("mz_grad2")
    i_mx_my_nnavg = idx("mx_my_nnavg")
    i_mx_lap_my = idx("mx_lap_my")
    i_mx_gradm_my = idx("mx_gradm_my")
    i_my_cross_p_x = idx("my_cross_p_x")
    i_mx_cross_p_y = idx("mx_cross_p_y")
    i_mx_my_p = idx("mx_my_p")
    i_mx_mz4 = idx("mx_mz4")
    i_mx_mperp4 = idx("mx_mperp4")
    i_mz_r2_mz2 = idx("mz_r2_mz2")
    i_mz_cross_p_y = idx("mz_cross_p_y")
    i_mx_Uloc = idx("mx_Uloc")
    i_mx_r2_mz2 = idx("mx_r2_mz2")
    i_mx_r2_p = idx("mx_r2_p")
    i_mx_dot_p = idx("mx_dot_p")
    i_mx_grad2 = idx("mx_grad2")
    i_my_amp_dev2 = idx("my_amp_dev2")
    i_mx_amp_dev = idx("mx_amp_dev")
    i_mx_dot_m = idx("mx_dot_m")

    @threads for b in 1:B
        @inbounds for site in 1:N
            im = periodic(site - 1, N)
            ip = periodic(site + 1, N)
            x1 = Float64(raw[site, 1, b])
            x2 = Float64(raw[site, 2, b])
            x3 = Float64(raw[site, 3, b])
            xm1 = Float64(raw[im, 1, b])
            xm2 = Float64(raw[im, 2, b])
            xm3 = Float64(raw[im, 3, b])
            xp1 = Float64(raw[ip, 1, b])
            xp2 = Float64(raw[ip, 2, b])
            xp3 = Float64(raw[ip, 3, b])

            r2 = x1*x1 + x2*x2 + x3*x3
            r2p = xp1*xp1 + xp2*xp2 + xp3*xp3
            r2m = xm1*xm1 + xm2*xm2 + xm3*xm3
            mperp2 = x1*x1 + x2*x2
            dotp = x1*xp1 + x2*xp2 + x3*xp3
            dotm = x1*xm1 + x2*xm2 + x3*xm3
            dxp1 = xp1 - x1
            dxp2 = xp2 - x2
            dxp3 = xp3 - x3
            dxm1 = x1 - xm1
            dxm2 = x2 - xm2
            dxm3 = x3 - xm3
            diffp2 = dxp1*dxp1 + dxp2*dxp2 + dxp3*dxp3
            diffm2 = dxm1*dxm1 + dxm2*dxm2 + dxm3*dxm3
            grad2 = diffp2 + diffm2
            crossp1 = x2*xp3 - x3*xp2
            crossp2 = x3*xp1 - x1*xp3
            crossp3 = x1*xp2 - x2*xp1
            crossm1 = x2*xm3 - x3*xm2
            crossm2 = x3*xm1 - x1*xm3
            crossm3 = x1*xm2 - x2*xm1
            lap1 = xp1 + xm1 - 2.0*x1
            lap2c = xp2 + xm2 - 2.0*x2
            lap3 = xp3 + xm3 - 2.0*x3
            lap2 = lap1*lap1 + lap2c*lap2c + lap3*lap3
            twist2 = crossp1*crossp1 + crossp2*crossp2 + crossp3*crossp3 +
                crossm1*crossm1 + crossm2*crossm2 + crossm3*crossm3
            amp_dev = r2 - p.mstar^2
            Uloc = 0.25*p.lambda*amp_dev*amp_dev +
                0.25*p.J*(diffp2 + diffm2) - 0.5*p.K*x3*x3

            i_mz_gradm_my > 0 && (obs[site, i_mz_gradm_my, b] = Float32(x3 * dxm2))
            i_mz_lap_my > 0 && (obs[site, i_mz_lap_my, b] = Float32(x3 * lap2c))
            i_cross_m_x > 0 && (obs[site, i_cross_m_x, b] = Float32(crossm1))
            i_cross_p_x > 0 && (obs[site, i_cross_p_x, b] = Float32(crossp1))
            i_mx_r4 > 0 && (obs[site, i_mx_r4, b] = Float32(x1 * r2 * r2))
            i_mz_gradp_my > 0 && (obs[site, i_mz_gradp_my, b] = Float32(x3 * dxp2))
            i_mx_r2 > 0 && (obs[site, i_mx_r2, b] = Float32(x1 * r2))
            i_mx_mx2 > 0 && (obs[site, i_mx_mx2, b] = Float32(x1 * x1 * x1))
            i_mx_r2_nnavg > 0 && (obs[site, i_mx_r2_nnavg, b] = Float32(0.5 * x1 * (r2p + r2m)))
            i_mx_r2_mperp2 > 0 && (obs[site, i_mx_r2_mperp2, b] = Float32(x1 * r2 * mperp2))
            i_mx_mperp2 > 0 && (obs[site, i_mx_mperp2, b] = Float32(x1 * mperp2))
            i_my_r2_mperp2 > 0 && (obs[site, i_my_r2_mperp2, b] = Float32(x2 * r2 * mperp2))
            i_my_mperp2 > 0 && (obs[site, i_my_mperp2, b] = Float32(x2 * mperp2))
            i_my_r4 > 0 && (obs[site, i_my_r4, b] = Float32(x2 * r2 * r2))
            i_my_r2 > 0 && (obs[site, i_my_r2, b] = Float32(x2 * r2))
            i_my_twist2 > 0 && (obs[site, i_my_twist2, b] = Float32(x2 * twist2))
            i_my_mperp4 > 0 && (obs[site, i_my_mperp4, b] = Float32(x2 * mperp2 * mperp2))
            i_my_my2 > 0 && (obs[site, i_my_my2, b] = Float32(x2 * x2 * x2))
            i_my_grad2 > 0 && (obs[site, i_my_grad2, b] = Float32(x2 * grad2))
            i_my_mx2 > 0 && (obs[site, i_my_mx2, b] = Float32(x2 * x1 * x1))
            i_my_lap2 > 0 && (obs[site, i_my_lap2, b] = Float32(x2 * lap2))
            i_my_r2_nnavg > 0 && (obs[site, i_my_r2_nnavg, b] = Float32(0.5 * x2 * (r2p + r2m)))
            i_my_mx_nnavg > 0 && (obs[site, i_my_mx_nnavg, b] = Float32(0.5 * x2 * (xp1 + xm1)))
            i_cross_p_z > 0 && (obs[site, i_cross_p_z, b] = Float32(crossp3))
            i_cross_m_z > 0 && (obs[site, i_cross_m_z, b] = Float32(crossm3))
            i_mz_lap2 > 0 && (obs[site, i_mz_lap2, b] = Float32(x3 * lap2))
            i_mx_cross_m_my > 0 && (obs[site, i_mx_cross_m_my, b] = Float32(x1 * crossm2))
            i_mz_grad2 > 0 && (obs[site, i_mz_grad2, b] = Float32(x3 * grad2))
            i_mx_my_nnavg > 0 && (obs[site, i_mx_my_nnavg, b] = Float32(0.5 * x1 * (xp2 + xm2)))
            i_mx_lap_my > 0 && (obs[site, i_mx_lap_my, b] = Float32(x1 * lap2c))
            i_mx_gradm_my > 0 && (obs[site, i_mx_gradm_my, b] = Float32(x1 * dxm2))
            i_my_cross_p_x > 0 && (obs[site, i_my_cross_p_x, b] = Float32(x2 * crossp1))
            i_mx_cross_p_y > 0 && (obs[site, i_mx_cross_p_y, b] = Float32(x1 * crossp2))
            i_mx_my_p > 0 && (obs[site, i_mx_my_p, b] = Float32(x1 * xp2))
            i_mx_mz4 > 0 && (obs[site, i_mx_mz4, b] = Float32(x1 * x3^4))
            i_mx_mperp4 > 0 && (obs[site, i_mx_mperp4, b] = Float32(x1 * mperp2 * mperp2))
            i_mz_r2_mz2 > 0 && (obs[site, i_mz_r2_mz2, b] = Float32(x3 * r2 * x3 * x3))
            i_mz_cross_p_y > 0 && (obs[site, i_mz_cross_p_y, b] = Float32(x3 * crossp2))
            i_mx_Uloc > 0 && (obs[site, i_mx_Uloc, b] = Float32(x1 * Uloc))
            i_mx_r2_mz2 > 0 && (obs[site, i_mx_r2_mz2, b] = Float32(x1 * r2 * x3 * x3))
            i_mx_r2_p > 0 && (obs[site, i_mx_r2_p, b] = Float32(x1 * r2p))
            i_mx_dot_p > 0 && (obs[site, i_mx_dot_p, b] = Float32(x1 * dotp))
            i_mx_grad2 > 0 && (obs[site, i_mx_grad2, b] = Float32(x1 * grad2))
            i_my_amp_dev2 > 0 && (obs[site, i_my_amp_dev2, b] = Float32(x2 * amp_dev * amp_dev))
            i_mx_amp_dev > 0 && (obs[site, i_mx_amp_dev, b] = Float32(x1 * amp_dev))
            i_mx_dot_m > 0 && (obs[site, i_mx_dot_m, b] = Float32(x1 * dotm))
        end
    end
    return obs
end

function selected_observables(raw::Array{Float32, 3}, p::SpinParams,
        names::Vector{String})
    obs = Array{Float32}(undef, size(raw, 1), length(names), size(raw, 3))
    return fill_selected_observables!(obs, raw, p, names)
end

function estimate_selected_observable_means(sampler::CondPairSampler, p::SpinParams,
        names::Vector{String}, nsamples::Int, rng::AbstractRNG; batch_size::Int=50000)
    sums = zeros(Float64, length(names))
    total = 0
    while total < nsamples
        nb = min(batch_size, nsamples - total)
        raw = sample_raw_states_cond(sampler, nb, rng)
        obs = selected_observables(raw, p, names)
        for j in eachindex(names)
            sums[j] += sum(Float64, @view obs[:, j, :])
        end
        total += nb
        GC.gc(false)
    end
    return sums ./ (sampler.N * nsamples)
end

function center_selected_observables!(obs::Array{Float32, 3}, means::Vector{Float64})
    @inbounds for j in eachindex(means)
        obs[:, j, :] .-= Float32(means[j])
    end
    return obs
end

function component_matrix(raw::Array{Float32, 3}, c::Int, mu::AbstractVector{<:Real})
    N, _, B = size(raw)
    out = Matrix{Float64}(undef, N, B)
    @inbounds for b in 1:B, i in 1:N
        out[i, b] = Float64(raw[i, c, b]) - Float64(mu[(i - 1) * 3 + c])
    end
    return out
end

function action_component_matrix(action_flat::AbstractMatrix{<:Real}, N::Int, c::Int)
    B = size(action_flat, 2)
    out = Matrix{Float64}(undef, N, B)
    @inbounds for b in 1:B, i in 1:N
        out[i, b] = Float64(action_flat[(i - 1) * 3 + c, b])
    end
    return out
end

function fill_channel_matrix_vector!(dest::AbstractVector{<:Real},
        start::Int, mat::AbstractMatrix{<:Real})
    N = size(mat, 1)
    k = start
    @inbounds for i in 1:N, j in 1:N
        dest[k] = Float32(mat[i, j])
        k += 1
    end
    return k
end

function onsite_reduce_selected(mat::AbstractMatrix{<:Real},
        selected_channel_ids::AbstractVector{<:Integer}, N::Int, nchannels::Int)
    out = Matrix{Float64}(undef, size(mat, 1), nchannels)
    for cid in 1:nchannels
        inds = findall(==(cid), selected_channel_ids)
        length(inds) == N * N || error("Channel $(cid) has $(length(inds)) entries")
        for li in axes(mat, 1)
            acc = 0.0
            @inbounds for i in 1:N
                acc += Float64(mat[li, inds[(i - 1) * N + i]])
            end
            out[li, cid] = acc / N
        end
    end
    return out
end
