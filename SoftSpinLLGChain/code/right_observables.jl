#!/usr/bin/env julia

using LinearAlgebra

const RIGHT_OBS_DEFAULT = [
    "mx", "my", "mz",
    "r2", "mperp2", "mz2",
    "mx_r2", "my_r2", "mz_r2",
    "mx_mperp2", "my_mperp2", "mz_mperp2",
    "mx_mz", "my_mz", "mx_my",
    "mx_mx2", "my_my2", "mz_mz2",
]

struct RightObservableLibrary
    names::Vector{String}
end

function right_candidate_names(family::AbstractString="core")
    if family == "linear"
        return ["mx", "my", "mz"]
    elseif family == "core"
        return copy(RIGHT_OBS_DEFAULT)
    elseif family == "smooth"
        return [
            "mx", "my", "mz",
            "r2", "mperp2", "mz2",
            "mx_r2", "my_r2", "mz_r2",
            "mx_mz", "my_mz", "mx_my",
        ]
    end
    error("Unknown right-observable family $(family).")
end

function right_observable_value_grad_hess(raw::Array{Float32, 3},
        lib::RightObservableLibrary)
    N, _, B = size(raw)
    R = length(lib.names)
    vals = Array{Float32}(undef, N, R, B)
    grads = zeros(Float32, N, R, 3, B)
    hess = zeros(Float32, N, R, 3, 3, B)
    @inbounds for b in 1:B, i in 1:N
        x = Float64(raw[i, 1, b])
        y = Float64(raw[i, 2, b])
        z = Float64(raw[i, 3, b])
        r2 = x*x + y*y + z*z
        mperp2 = x*x + y*y
        for (a, name) in enumerate(lib.names)
            if name == "mx"
                vals[i, a, b] = Float32(x)
                grads[i, a, 1, b] = 1f0
            elseif name == "my"
                vals[i, a, b] = Float32(y)
                grads[i, a, 2, b] = 1f0
            elseif name == "mz"
                vals[i, a, b] = Float32(z)
                grads[i, a, 3, b] = 1f0
            elseif name == "r2"
                vals[i, a, b] = Float32(r2)
                grads[i, a, 1, b] = Float32(2x)
                grads[i, a, 2, b] = Float32(2y)
                grads[i, a, 3, b] = Float32(2z)
                hess[i, a, 1, 1, b] = 2f0
                hess[i, a, 2, 2, b] = 2f0
                hess[i, a, 3, 3, b] = 2f0
            elseif name == "mperp2"
                vals[i, a, b] = Float32(mperp2)
                grads[i, a, 1, b] = Float32(2x)
                grads[i, a, 2, b] = Float32(2y)
                hess[i, a, 1, 1, b] = 2f0
                hess[i, a, 2, 2, b] = 2f0
            elseif name == "mz2"
                vals[i, a, b] = Float32(z*z)
                grads[i, a, 3, b] = Float32(2z)
                hess[i, a, 3, 3, b] = 2f0
            elseif name in ("mx_r2", "my_r2", "mz_r2")
                k = name == "mx_r2" ? 1 : name == "my_r2" ? 2 : 3
                m = k == 1 ? x : k == 2 ? y : z
                vals[i, a, b] = Float32(m * r2)
                grads[i, a, 1, b] = Float32((k == 1 ? r2 : 0.0) + 2m*x)
                grads[i, a, 2, b] = Float32((k == 2 ? r2 : 0.0) + 2m*y)
                grads[i, a, 3, b] = Float32((k == 3 ? r2 : 0.0) + 2m*z)
                mm = (x, y, z)
                for q in 1:3, r in 1:3
                    hess[i, a, q, r, b] = Float32(
                        2.0 * ((q == k ? mm[r] : 0.0) +
                               (r == k ? mm[q] : 0.0) +
                               (q == r ? m : 0.0)))
                end
            elseif name in ("mx_mperp2", "my_mperp2", "mz_mperp2")
                k = name == "mx_mperp2" ? 1 : name == "my_mperp2" ? 2 : 3
                m = k == 1 ? x : k == 2 ? y : z
                vals[i, a, b] = Float32(m * mperp2)
                grads[i, a, 1, b] = Float32((k == 1 ? mperp2 : 0.0) + 2m*x)
                grads[i, a, 2, b] = Float32((k == 2 ? mperp2 : 0.0) + 2m*y)
                grads[i, a, 3, b] = Float32(k == 3 ? mperp2 : 0.0)
                mm = (x, y, z)
                for q in 1:3, r in 1:3
                    val = 0.0
                    if r <= 2 && q == k
                        val += 2mm[r]
                    end
                    if q <= 2 && r == k
                        val += 2mm[q]
                    end
                    if q == r && q <= 2
                        val += 2m
                    end
                    hess[i, a, q, r, b] = Float32(val)
                end
            elseif name == "mx_mz"
                vals[i, a, b] = Float32(x*z)
                grads[i, a, 1, b] = Float32(z)
                grads[i, a, 3, b] = Float32(x)
                hess[i, a, 1, 3, b] = 1f0
                hess[i, a, 3, 1, b] = 1f0
            elseif name == "my_mz"
                vals[i, a, b] = Float32(y*z)
                grads[i, a, 2, b] = Float32(z)
                grads[i, a, 3, b] = Float32(y)
                hess[i, a, 2, 3, b] = 1f0
                hess[i, a, 3, 2, b] = 1f0
            elseif name == "mx_my"
                vals[i, a, b] = Float32(x*y)
                grads[i, a, 1, b] = Float32(y)
                grads[i, a, 2, b] = Float32(x)
                hess[i, a, 1, 2, b] = 1f0
                hess[i, a, 2, 1, b] = 1f0
            elseif name == "mx_mx2"
                vals[i, a, b] = Float32(x^3)
                grads[i, a, 1, b] = Float32(3x^2)
                hess[i, a, 1, 1, b] = Float32(6x)
            elseif name == "my_my2"
                vals[i, a, b] = Float32(y^3)
                grads[i, a, 2, b] = Float32(3y^2)
                hess[i, a, 2, 2, b] = Float32(6y)
            elseif name == "mz_mz2"
                vals[i, a, b] = Float32(z^3)
                grads[i, a, 3, b] = Float32(3z^2)
                hess[i, a, 3, 3, b] = Float32(6z)
            else
                error("Unsupported right observable $(name).")
            end
        end
    end
    return vals, grads, hess
end

function estimate_right_means(sampler::CondPairSampler, lib::RightObservableLibrary,
        nsamples::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(sampler.states)
    raw = Array{Float32}(undef, N, 3, nsamples)
    @inbounds for b in 1:nsamples
        t = rand(rng, sampler.start_idx:nt)
        tr = rand(rng, 1:ntraj)
        raw[:, :, b] .= sampler.states[t, :, :, tr]
    end
    vals, _, _ = right_observable_value_grad_hess(raw, lib)
    return [mean(Float64, @view vals[:, j, :]) for j in eachindex(lib.names)]
end

function center_right_values!(vals::Array{Float32, 3}, means::Vector{Float64})
    @inbounds for j in eachindex(means)
        vals[:, j, :] .-= Float32(means[j])
    end
    return vals
end

function right_action_from_site_action(action::Array{Float32, 3},
        grads::Array{Float32, 4})
    N, R, _, B = size(grads)
    out = Array{Float32}(undef, N, R, B)
    @inbounds for b in 1:B, a in 1:R, i in 1:N
        out[i, a, b] = action[i, 1, b] * grads[i, a, 1, b] +
            action[i, 2, b] * grads[i, a, 2, b] +
            action[i, 3, b] * grads[i, a, 3, b]
    end
    return out
end

function right_phi_generator_terms(Phi::AbstractMatrix{<:Real},
        sraw::Array{Float32, 3}, grads::Array{Float32, 4},
        hess::Array{Float32, 5})
    N, R, _, B = size(grads)
    action = transpose(Matrix{Float64}(Phi)) * Matrix{Float64}(flatten_batch(sraw))
    out = Array{Float32}(undef, N, R, B)
    @inbounds for b in 1:B, a in 1:R, i in 1:N
        rows = ((i - 1) * 3 + 1):(i * 3)
        val = 0.0
        for c in 1:3
            val += action[rows[c], b] * Float64(grads[i, a, c, b])
        end
        for c in 1:3, d in 1:3
            val += Float64(Phi[rows[c], rows[d]]) * Float64(hess[i, a, d, c, b])
        end
        out[i, a, b] = Float32(val)
    end
    return out
end
