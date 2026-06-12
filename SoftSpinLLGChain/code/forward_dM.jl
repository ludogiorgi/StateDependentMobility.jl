#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using HDF5
using KernelDensity
using Printf
using Statistics

Base.@kwdef struct ForwardConfig
    dt::Float64
    total_time::Float64
    burnin_time::Float64
    save_dt::Float64
    ntraj::Int
    score_clip::Float32
    score_batch_size::Int
    mobility_scale::Float64
    delta_scale::Float64
    skew_scale::Float64
    xy_sym_scale::Float64
    z_sym_scale::Float64
    z_diag_scale::Float64
    xy_skew_scale::Float64
    z_skew_scale::Float64
    lambda_matrix::Union{Nothing, Matrix{Float64}}
end

function parse_lambda_matrix(fwd)
    haskey(fwd, "lambda_matrix") || return nothing
    raw = fwd["lambda_matrix"]
    if raw isa AbstractVector
        if length(raw) == 3 && all(x -> x isa AbstractVector, raw)
            mat = zeros(Float64, 3, 3)
            for i in 1:3
                row = raw[i]
                length(row) == 3 || error("forward.lambda_matrix rows must have length 3.")
                for j in 1:3
                    mat[i, j] = Float64(row[j])
                end
            end
            return mat
        elseif length(raw) == 9
            vals = Float64.(raw)
            return permutedims(reshape(vals, 3, 3))
        else
            error("forward.lambda_matrix must be a flat 9-vector or a 3x3 nested array.")
        end
    else
        error("forward.lambda_matrix must be an array.")
    end
end

function load_forward_config(path::AbstractString)
    raw = TOML.parsefile(path)
    fwd = raw["forward"]
    eval = raw["evaluation"]
    return ForwardConfig(
        dt=Float64(get(fwd, "dt", 0.003)),
        total_time=Float64(get(fwd, "total_time", 180.0)),
        burnin_time=Float64(get(fwd, "burnin_time", 30.0)),
        save_dt=Float64(get(fwd, "save_dt", 0.04)),
        ntraj=Int(get(fwd, "ntrajectories", 72)),
        score_clip=Float32(get(fwd, "score_clip", 80.0)),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
        mobility_scale=Float64(get(fwd, "mobility_scale", 1.0)),
        delta_scale=Float64(get(fwd, "delta_scale", 1.0)),
        skew_scale=Float64(get(fwd, "skew_scale", 1.0)),
        xy_sym_scale=Float64(get(fwd, "xy_sym_scale", 1.0)),
        z_sym_scale=Float64(get(fwd, "z_sym_scale", 1.0)),
        z_diag_scale=Float64(get(fwd, "z_diag_scale", 1.0)),
        xy_skew_scale=Float64(get(fwd, "xy_skew_scale", 1.0)),
        z_skew_scale=Float64(get(fwd, "z_skew_scale", 1.0)),
        lambda_matrix=parse_lambda_matrix(fwd),
    )
end

function feature_matrix_from_xn(xn::Array{Float32, 3}, mode::Symbol)
    feats = feature_tensor(xn, mode)
    F = size(feats, 2)
    return reshape(permutedims(feats, (2, 1, 3)), F, size(xn, 1) * size(xn, 3))
end

function params_to_blocks(y, model::LocalMobilityNN; skew_gain=one(eltype(y)),
        xy_sym_scale=one(eltype(y)), z_sym_scale=one(eltype(y)),
        z_diag_scale=one(eltype(y)), xy_skew_scale=one(eltype(y)),
        z_skew_scale=one(eltype(y)))
    sscale = eltype(y)(model.sym_scale)
    kscale = eltype(y)(model.skew_scale)
    floor = eltype(y)(model.sym_floor)
    xyrow = sqrt(max(eltype(y)(xy_sym_scale), zero(eltype(y))))
    zrow = sqrt(max(eltype(y)(z_sym_scale), zero(eltype(y))))
    zdiag = sqrt(max(eltype(y)(z_diag_scale), zero(eltype(y))))
    l11 = (NNlib.softplus.(y[1, :]) .* sscale .+ floor) .* xyrow
    l21 = y[2, :] .* sscale .* xyrow
    l22 = (NNlib.softplus.(y[3, :]) .* sscale .+ floor) .* xyrow
    l31 = y[4, :] .* sscale .* zrow
    l32 = y[5, :] .* sscale .* zrow
    l33 = (NNlib.softplus.(y[6, :]) .* sscale .+ floor) .* zrow .* zdiag
    k1 = y[7, :] .* kscale .* skew_gain .* z_skew_scale
    k2 = y[8, :] .* kscale .* skew_gain .* z_skew_scale
    k3 = y[9, :] .* kscale .* skew_gain .* xy_skew_scale
    s11 = l11 .* l11
    s12 = l11 .* l21
    s13 = l11 .* l31
    s22 = l21 .* l21 .+ l22 .* l22
    s23 = l21 .* l31 .+ l22 .* l32
    s33 = l31 .* l31 .+ l32 .* l32 .+ l33 .* l33
    return (; l11, l21, l22, l31, l32, l33, k1, k2, k3,
        m11=s11, m12=s12 .- k3, m13=s13 .+ k2,
        m21=s12 .+ k3, m22=s22, m23=s23 .- k1,
        m31=s13 .- k2, m32=s23 .+ k1, m33=s33)
end

function block_params_from_features(model::LocalMobilityNN, features_dev; skew_gain=1.0,
        xy_sym_scale=1.0, z_sym_scale=1.0, z_diag_scale=1.0,
        xy_skew_scale=1.0, z_skew_scale=1.0)
    y = model.mlp(features_dev)
    return params_to_blocks(y, model; skew_gain=eltype(y)(skew_gain),
        xy_sym_scale=eltype(y)(xy_sym_scale), z_sym_scale=eltype(y)(z_sym_scale),
        z_diag_scale=eltype(y)(z_diag_scale), xy_skew_scale=eltype(y)(xy_skew_scale),
        z_skew_scale=eltype(y)(z_skew_scale))
end

function central_feature_row(mode::Symbol, c::Int)
    mode = base_feature_mode(mode)
    mode == :local && return c
    mode == :local_r2 && return c
    mode == :neighbor && return 3 + c
    mode == :neighbor_r2 && return 3 + c
    mode == :neighbor_all_r2 && return 3 + c
    error("Unknown feature mode $(mode)")
end

function host_blocks_from_params(bp; xy_sym_scale::Float64=1.0,
        z_sym_scale::Float64=1.0, z_diag_scale::Float64=1.0,
        xy_skew_scale::Float64=1.0, z_skew_scale::Float64=1.0)
    xyrow = Float32(sqrt(max(xy_sym_scale, 0.0)))
    zrow = Float32(sqrt(max(z_sym_scale, 0.0)))
    zdiag = Float32(sqrt(max(z_diag_scale, 0.0)))
    kzgain = Float32(z_skew_scale)
    l11 = Array(bp.l11) .* xyrow
    l21 = Array(bp.l21) .* xyrow
    l22 = Array(bp.l22) .* xyrow
    l31 = Array(bp.l31) .* zrow
    l32 = Array(bp.l32) .* zrow
    l33 = Array(bp.l33) .* zrow .* zdiag
    k1 = Array(bp.k1) .* kzgain
    k2 = Array(bp.k2) .* kzgain
    k3 = Array(bp.k3) .* Float32(xy_skew_scale)
    return (;
        l11, l21, l22, l31, l32, l33, k1, k2, k3,
        m11=l11 .* l11,
        m12=l11 .* l21 .- k3,
        m13=l11 .* l31 .+ k2,
        m21=l11 .* l21 .+ k3,
        m22=l21 .* l21 .+ l22 .* l22,
        m23=l21 .* l31 .+ l22 .* l32 .- k1,
        m31=l11 .* l31 .- k2,
        m32=l21 .* l31 .+ l22 .* l32 .+ k1,
        m33=l31 .* l31 .+ l32 .* l32 .+ l33 .* l33)
end

function host_entries_from_params(bp)
    return (;
        m11=Array(bp.m11), m12=Array(bp.m12), m13=Array(bp.m13),
        m21=Array(bp.m21), m22=Array(bp.m22), m23=Array(bp.m23),
        m31=Array(bp.m31), m32=Array(bp.m32), m33=Array(bp.m33))
end

function lambda_corrected_entries(host, phi_block::AbstractMatrix,
        lambda_matrix::AbstractMatrix)
    phi = Float32.(phi_block)
    lam = Float32.(lambda_matrix)
    return (;
        m11=phi[1, 1] .+ lam[1, 1] .* (host.m11 .- phi[1, 1]),
        m12=phi[1, 2] .+ lam[1, 2] .* (host.m12 .- phi[1, 2]),
        m13=phi[1, 3] .+ lam[1, 3] .* (host.m13 .- phi[1, 3]),
        m21=phi[2, 1] .+ lam[2, 1] .* (host.m21 .- phi[2, 1]),
        m22=phi[2, 2] .+ lam[2, 2] .* (host.m22 .- phi[2, 2]),
        m23=phi[2, 3] .+ lam[2, 3] .* (host.m23 .- phi[2, 3]),
        m31=phi[3, 1] .+ lam[3, 1] .* (host.m31 .- phi[3, 1]),
        m32=phi[3, 2] .+ lam[3, 2] .* (host.m32 .- phi[3, 2]),
        m33=phi[3, 3] .+ lam[3, 3] .* (host.m33 .- phi[3, 3]))
end

function lambda_corrected_host(host, phi_block::AbstractMatrix,
        lambda_matrix::AbstractMatrix)
    entries = lambda_corrected_entries(host, phi_block, lambda_matrix)
    Q = length(entries.m11)
    l11 = Vector{Float32}(undef, Q)
    l21 = Vector{Float32}(undef, Q)
    l22 = Vector{Float32}(undef, Q)
    l31 = Vector{Float32}(undef, Q)
    l32 = Vector{Float32}(undef, Q)
    l33 = Vector{Float32}(undef, Q)
    tiny = 1f-8
    @inbounds for q in 1:Q
        s11 = entries.m11[q]
        s12 = 0.5f0 * (entries.m12[q] + entries.m21[q])
        s13 = 0.5f0 * (entries.m13[q] + entries.m31[q])
        s22 = entries.m22[q]
        s23 = 0.5f0 * (entries.m23[q] + entries.m32[q])
        s33 = entries.m33[q]
        l11[q] = sqrt(max(s11, tiny))
        l21[q] = s12 / l11[q]
        l31[q] = s13 / l11[q]
        rem22 = s22 - l21[q] * l21[q]
        l22[q] = sqrt(max(rem22, tiny))
        l32[q] = (s23 - l21[q] * l31[q]) / l22[q]
        rem33 = s33 - l31[q] * l31[q] - l32[q] * l32[q]
        require_condition(rem22 > -1f-5 && rem33 > -1f-5,
            "lambda_matrix produced a non-PSD symmetric mobility block.")
        l33[q] = sqrt(max(rem33, tiny))
    end
    return merge(entries, (; l11, l21, l22, l31, l32, l33))
end

function std_for_flat(stats::DataStats, N::Int, B::Int, c::Int)
    out = Vector{Float32}(undef, N * B)
    @inbounds for b in 1:B, i in 1:N
        out[(b - 1) * N + i] = stats.std[c, i]
    end
    return out
end

function normalized_component_flat(xn::Array{Float32, 3}, c::Int)
    N, _, B = size(xn)
    out = Vector{Float32}(undef, N * B)
    @inbounds for b in 1:B, i in 1:N
        out[(b - 1) * N + i] = xn[i, c, b]
    end
    return out
end

function divergence_blocks(model::LocalMobilityNN, features::Matrix{Float32},
        xn::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice;
        eps_raw::Float32=1f-3, skew_gain::Float64=1.0,
        xy_sym_scale::Float64=1.0, z_sym_scale::Float64=1.0,
        z_diag_scale::Float64=1.0, xy_skew_scale::Float64=1.0,
        z_skew_scale::Float64=1.0,
        phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        lambda_matrix::Union{Nothing, Matrix{Float64}}=nothing)
    if parity_symmetrized_mode(model.feature_mode)
        return divergence_blocks_parity(model, xn, stats, device;
            eps_raw=eps_raw, skew_gain=skew_gain,
            xy_sym_scale=xy_sym_scale, z_sym_scale=z_sym_scale,
            z_diag_scale=z_diag_scale, xy_skew_scale=xy_skew_scale,
            z_skew_scale=z_skew_scale, phi_block=phi_block,
            lambda_matrix=lambda_matrix)
    end
    N, _, B = size(xn)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        stdv = std_for_flat(stats, N, B, c)
        epsn = eps_raw ./ stdv
        row = central_feature_row(model.feature_mode, c)
        fplus = copy(features)
        fminus = copy(features)
        xnc = normalized_component_flat(xn, c)
        @inbounds for q in 1:Q
            fplus[row, q] += epsn[q]
            fminus[row, q] -= epsn[q]
            if model.feature_mode == :local_r2
                fplus[4, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[4, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            elseif model.feature_mode == :neighbor_r2
                fplus[10, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[10, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            elseif model.feature_mode == :neighbor_all_r2
                fplus[11, q] += 2f0 * xnc[q] * epsn[q] + epsn[q]^2
                fminus[11, q] += -2f0 * xnc[q] * epsn[q] + epsn[q]^2
            end
        end
        bp = block_params_from_features(model, move_array(fplus, device);
            skew_gain=skew_gain, xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        bm = block_params_from_features(model, move_array(fminus, device);
            skew_gain=skew_gain, xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        hp = host_entries_from_params(bp)
        hm = host_entries_from_params(bm)
        if lambda_matrix !== nothing
            require_condition(phi_block !== nothing,
                "lambda_matrix divergence requires Phi block.")
            hp = lambda_corrected_entries(hp, phi_block, lambda_matrix)
            hm = lambda_corrected_entries(hm, phi_block, lambda_matrix)
        end
        cols_p = (hp.m11, hp.m21, hp.m31,
            hp.m12, hp.m22, hp.m32,
            hp.m13, hp.m23, hp.m33)
        cols_m = (hm.m11, hm.m21, hm.m31,
            hm.m12, hm.m22, hm.m32,
            hm.m13, hm.m23, hm.m33)
        # Add derivative of column c to every output row.
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function divergence_blocks_parity(model::LocalMobilityNN, xn::Array{Float32, 3},
        stats::DataStats, device::ExecutionDevice; eps_raw::Float32=1f-3,
        skew_gain::Float64=1.0, xy_sym_scale::Float64=1.0,
        z_sym_scale::Float64=1.0, z_diag_scale::Float64=1.0,
        xy_skew_scale::Float64=1.0, z_skew_scale::Float64=1.0,
        phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        lambda_matrix::Union{Nothing, Matrix{Float64}}=nothing)
    N, _, B = size(xn)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        epsn = reshape(eps_raw ./ stats.std[c, :], N, 1, 1)
        xplus = copy(xn)
        xminus = copy(xn)
        @views xplus[:, c:c, :] .+= epsn
        @views xminus[:, c:c, :] .-= epsn
        bp = block_params(model, move_array(xplus, device); skew_gain=skew_gain)
        bm = block_params(model, move_array(xminus, device); skew_gain=skew_gain)
        hp = host_blocks_from_params(bp; xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        hm = host_blocks_from_params(bm; xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        if lambda_matrix !== nothing
            require_condition(phi_block !== nothing,
                "lambda_matrix divergence requires Phi block.")
            hp = lambda_corrected_entries(hp, phi_block, lambda_matrix)
            hm = lambda_corrected_entries(hm, phi_block, lambda_matrix)
        end
        cols_p = (hp.m11, hp.m21, hp.m31,
            hp.m12, hp.m22, hp.m32,
            hp.m13, hp.m23, hp.m33)
        cols_m = (hm.m11, hm.m21, hm.m31,
            hm.m12, hm.m22, hm.m32,
            hm.m13, hm.m23, hm.m33)
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function divergence_blocks(model::EquivariantMobilityNN, xn::Array{Float32, 3},
        stats::DataStats, device::ExecutionDevice; eps_raw::Float32=1f-3,
        skew_gain::Float64=1.0, xy_sym_scale::Float64=1.0,
        z_sym_scale::Float64=1.0, z_diag_scale::Float64=1.0,
        xy_skew_scale::Float64=1.0, z_skew_scale::Float64=1.0,
        phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        lambda_matrix::Union{Nothing, Matrix{Float64}}=nothing)
    N, _, B = size(xn)
    Q = N * B
    div = zeros(Float32, 3, Q)
    for c in 1:3
        epsn = reshape(eps_raw ./ stats.std[c, :], N, 1, 1)
        xplus = copy(xn)
        xminus = copy(xn)
        @views xplus[:, c:c, :] .+= epsn
        @views xminus[:, c:c, :] .-= epsn
        bp = block_params(model, move_array(xplus, device); skew_gain=skew_gain)
        bm = block_params(model, move_array(xminus, device); skew_gain=skew_gain)
        hp = host_blocks_from_params(bp; xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        hm = host_blocks_from_params(bm; xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        if lambda_matrix !== nothing
            require_condition(phi_block !== nothing,
                "lambda_matrix divergence requires Phi block.")
            hp = lambda_corrected_entries(hp, phi_block, lambda_matrix)
            hm = lambda_corrected_entries(hm, phi_block, lambda_matrix)
        end
        cols_p = (hp.m11, hp.m21, hp.m31,
            hp.m12, hp.m22, hp.m32,
            hp.m13, hp.m23, hp.m33)
        cols_m = (hm.m11, hm.m21, hm.m31,
            hm.m12, hm.m22, hm.m32,
            hm.m13, hm.m23, hm.m33)
        for a in 1:3
            idx = (c - 1) * 3 + a
            @. div[a, :] += (cols_p[idx] - cols_m[idx]) / (2f0 * eps_raw)
        end
    end
    return div
end

function action_noise_divergence(model::LocalMobilityNN, raw::Array{Float32, 3},
        score_raw::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice,
        rng::AbstractRNG; phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        sqrt_phi_block::Union{Nothing, Matrix{Float32}}=nothing, delta_scale::Float64=1.0,
        skew_gain::Float64=1.0, xy_sym_scale::Float64=1.0,
        z_sym_scale::Float64=1.0, z_diag_scale::Float64=1.0,
        xy_skew_scale::Float64=1.0, z_skew_scale::Float64=1.0,
        lambda_matrix::Union{Nothing, Matrix{Float64}}=nothing)
    N, _, B = size(raw)
    xn = apply_stats_tensor(raw, stats)
    features = feature_matrix_from_xn(xn, model.feature_mode)
    if parity_symmetrized_mode(model.feature_mode)
        bp = block_params(model, move_array(xn, device); skew_gain=skew_gain)
        host = host_blocks_from_params(bp; xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
    else
        bp = block_params_from_features(model, move_array(features, device);
            skew_gain=skew_gain, xy_sym_scale=xy_sym_scale,
            z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
            xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
        host = (;
            l11=Array(bp.l11), l21=Array(bp.l21), l22=Array(bp.l22),
            l31=Array(bp.l31), l32=Array(bp.l32), l33=Array(bp.l33),
            m11=Array(bp.m11), m12=Array(bp.m12), m13=Array(bp.m13),
            m21=Array(bp.m21), m22=Array(bp.m22), m23=Array(bp.m23),
            m31=Array(bp.m31), m32=Array(bp.m32), m33=Array(bp.m33))
    end
    if lambda_matrix !== nothing
        require_condition(phi_block !== nothing,
            "lambda_matrix forward integration requires Phi block.")
        host = lambda_corrected_host(host, phi_block, lambda_matrix)
    end
    Q = N * B
    sflat = reshape(permutedims(score_raw, (2, 1, 3)), 3, Q)
    drift_flat = Array{Float32}(undef, 3, Q)
    noise_flat = Array{Float32}(undef, 3, Q)
    z = randn(rng, Float32, 3, Q)
    beta = Float32(delta_scale)
    @inbounds for q in 1:Q
        s1, s2, s3 = sflat[1, q], sflat[2, q], sflat[3, q]
        drift_flat[1, q] = host.m11[q] * s1 + host.m12[q] * s2 + host.m13[q] * s3
        drift_flat[2, q] = host.m21[q] * s1 + host.m22[q] * s2 + host.m23[q] * s3
        drift_flat[3, q] = host.m31[q] * s1 + host.m32[q] * s2 + host.m33[q] * s3
        # Noise factor is the Cholesky factor of the symmetric PSD part.
        noise_flat[1, q] = host.l11[q] * z[1, q]
        noise_flat[2, q] = host.l21[q] * z[1, q] + host.l22[q] * z[2, q]
        noise_flat[3, q] = host.l31[q] * z[1, q] + host.l32[q] * z[2, q] + host.l33[q] * z[3, q]
    end
    div = divergence_blocks(model, features, xn, stats, device; skew_gain=skew_gain,
        xy_sym_scale=xy_sym_scale, z_sym_scale=z_sym_scale,
        z_diag_scale=z_diag_scale, xy_skew_scale=xy_skew_scale,
        z_skew_scale=z_skew_scale, phi_block=phi_block, lambda_matrix=lambda_matrix)
    drift_flat .+= div
    if delta_scale != 1.0
        require_condition(phi_block !== nothing && sqrt_phi_block !== nothing,
            "delta_scale requires Phi block and square-root Phi block.")
        require_condition(delta_scale >= 0.0,
            "delta_scale must be nonnegative.")
        phi = phi_block::Matrix{Float32}
        phi_drift = phi * sflat
        @. drift_flat = beta * drift_flat + (1f0 - beta) * phi_drift
        if delta_scale <= 1.0
            zphi = randn(rng, Float32, 3, Q)
            phi_noise = (sqrt_phi_block::Matrix{Float32}) * zphi
            @. noise_flat = sqrt(beta) * noise_flat + sqrt(1f0 - beta) * phi_noise
        else
            sym_phi = 0.5f0 .* (phi .+ transpose(phi))
            zextra = randn(rng, Float32, 3, Q)
            tiny = 1f-8
            @inbounds for q in 1:Q
                s11 = host.l11[q] * host.l11[q]
                s12 = host.l11[q] * host.l21[q]
                s13 = host.l11[q] * host.l31[q]
                s22 = host.l21[q] * host.l21[q] + host.l22[q] * host.l22[q]
                s23 = host.l21[q] * host.l31[q] + host.l22[q] * host.l32[q]
                s33 = host.l31[q] * host.l31[q] + host.l32[q] * host.l32[q] + host.l33[q] * host.l33[q]
                e11 = beta * s11 + (1f0 - beta) * sym_phi[1, 1]
                e12 = beta * s12 + (1f0 - beta) * sym_phi[1, 2]
                e13 = beta * s13 + (1f0 - beta) * sym_phi[1, 3]
                e22 = beta * s22 + (1f0 - beta) * sym_phi[2, 2]
                e23 = beta * s23 + (1f0 - beta) * sym_phi[2, 3]
                e33 = beta * s33 + (1f0 - beta) * sym_phi[3, 3]
                l11 = sqrt(max(e11, tiny))
                l21 = e12 / l11
                l31 = e13 / l11
                rem22 = e22 - l21 * l21
                l22 = sqrt(max(rem22, tiny))
                l32 = (e23 - l21 * l31) / l22
                rem33 = e33 - l31 * l31 - l32 * l32
                require_condition(rem22 > -1f-5 && rem33 > -1f-5,
                    "delta_scale=$(delta_scale) produced a non-PSD symmetric mobility block.")
                l33 = sqrt(max(rem33, tiny))
                noise_flat[1, q] = l11 * zextra[1, q]
                noise_flat[2, q] = l21 * zextra[1, q] + l22 * zextra[2, q]
                noise_flat[3, q] = l31 * zextra[1, q] + l32 * zextra[2, q] + l33 * zextra[3, q]
            end
        end
    end
    drift = permutedims(reshape(drift_flat, 3, N, B), (2, 1, 3))
    noise = permutedims(reshape(noise_flat, 3, N, B), (2, 1, 3))
    return drift, noise
end

function action_noise_divergence(model::EquivariantMobilityNN, raw::Array{Float32, 3},
        score_raw::Array{Float32, 3}, stats::DataStats, device::ExecutionDevice,
        rng::AbstractRNG; phi_block::Union{Nothing, Matrix{Float32}}=nothing,
        sqrt_phi_block::Union{Nothing, Matrix{Float32}}=nothing, delta_scale::Float64=1.0,
        skew_gain::Float64=1.0, xy_sym_scale::Float64=1.0,
        z_sym_scale::Float64=1.0, z_diag_scale::Float64=1.0,
        xy_skew_scale::Float64=1.0, z_skew_scale::Float64=1.0,
        lambda_matrix::Union{Nothing, Matrix{Float64}}=nothing)
    N, _, B = size(raw)
    xn = apply_stats_tensor(raw, stats)
    bp = block_params(model, move_array(xn, device); skew_gain=skew_gain)
    host = host_blocks_from_params(bp; xy_sym_scale=xy_sym_scale,
        z_sym_scale=z_sym_scale, z_diag_scale=z_diag_scale,
        xy_skew_scale=xy_skew_scale, z_skew_scale=z_skew_scale)
    if lambda_matrix !== nothing
        require_condition(phi_block !== nothing,
            "lambda_matrix forward integration requires Phi block.")
        host = lambda_corrected_host(host, phi_block, lambda_matrix)
    end
    Q = N * B
    sflat = reshape(permutedims(score_raw, (2, 1, 3)), 3, Q)
    drift_flat = Array{Float32}(undef, 3, Q)
    noise_flat = Array{Float32}(undef, 3, Q)
    z = randn(rng, Float32, 3, Q)
    beta = Float32(delta_scale)
    @inbounds for q in 1:Q
        s1, s2, s3 = sflat[1, q], sflat[2, q], sflat[3, q]
        drift_flat[1, q] = host.m11[q] * s1 + host.m12[q] * s2 + host.m13[q] * s3
        drift_flat[2, q] = host.m21[q] * s1 + host.m22[q] * s2 + host.m23[q] * s3
        drift_flat[3, q] = host.m31[q] * s1 + host.m32[q] * s2 + host.m33[q] * s3
        noise_flat[1, q] = host.l11[q] * z[1, q]
        noise_flat[2, q] = host.l21[q] * z[1, q] + host.l22[q] * z[2, q]
        noise_flat[3, q] = host.l31[q] * z[1, q] + host.l32[q] * z[2, q] + host.l33[q] * z[3, q]
    end
    div = divergence_blocks(model, xn, stats, device; skew_gain=skew_gain,
        xy_sym_scale=xy_sym_scale, z_sym_scale=z_sym_scale,
        z_diag_scale=z_diag_scale, xy_skew_scale=xy_skew_scale,
        z_skew_scale=z_skew_scale, phi_block=phi_block, lambda_matrix=lambda_matrix)
    drift_flat .+= div
    if delta_scale != 1.0
        require_condition(phi_block !== nothing && sqrt_phi_block !== nothing,
            "delta_scale requires Phi block and square-root Phi block.")
        require_condition(delta_scale >= 0.0,
            "delta_scale must be nonnegative.")
        phi = phi_block::Matrix{Float32}
        phi_drift = phi * sflat
        @. drift_flat = beta * drift_flat + (1f0 - beta) * phi_drift
        if delta_scale <= 1.0
            zphi = randn(rng, Float32, 3, Q)
            phi_noise = (sqrt_phi_block::Matrix{Float32}) * zphi
            @. noise_flat = sqrt(beta) * noise_flat + sqrt(1f0 - beta) * phi_noise
        else
            sym_phi = 0.5f0 .* (phi .+ transpose(phi))
            zextra = randn(rng, Float32, 3, Q)
            tiny = 1f-8
            @inbounds for q in 1:Q
                s11 = host.l11[q] * host.l11[q]
                s12 = host.l11[q] * host.l21[q]
                s13 = host.l11[q] * host.l31[q]
                s22 = host.l21[q] * host.l21[q] + host.l22[q] * host.l22[q]
                s23 = host.l21[q] * host.l31[q] + host.l22[q] * host.l32[q]
                s33 = host.l31[q] * host.l31[q] + host.l32[q] * host.l32[q] + host.l33[q] * host.l33[q]
                e11 = beta * s11 + (1f0 - beta) * sym_phi[1, 1]
                e12 = beta * s12 + (1f0 - beta) * sym_phi[1, 2]
                e13 = beta * s13 + (1f0 - beta) * sym_phi[1, 3]
                e22 = beta * s22 + (1f0 - beta) * sym_phi[2, 2]
                e23 = beta * s23 + (1f0 - beta) * sym_phi[2, 3]
                e33 = beta * s33 + (1f0 - beta) * sym_phi[3, 3]
                l11 = sqrt(max(e11, tiny))
                l21 = e12 / l11
                l31 = e13 / l11
                rem22 = e22 - l21 * l21
                l22 = sqrt(max(rem22, tiny))
                l32 = (e23 - l21 * l31) / l22
                rem33 = e33 - l31 * l31 - l32 * l32
                require_condition(rem22 > -1f-5 && rem33 > -1f-5,
                    "delta_scale=$(delta_scale) produced a non-PSD symmetric mobility block.")
                l33 = sqrt(max(rem33, tiny))
                noise_flat[1, q] = l11 * zextra[1, q]
                noise_flat[2, q] = l21 * zextra[1, q] + l22 * zextra[2, q]
                noise_flat[3, q] = l31 * zextra[1, q] + l32 * zextra[2, q] + l33 * zextra[3, q]
            end
        end
    end
    drift = permutedims(reshape(drift_flat, 3, N, B), (2, 1, 3))
    noise = permutedims(reshape(noise_flat, 3, N, B), (2, 1, 3))
    return drift, noise
end

function initial_raw_batch_cond(sampler::CondPairSampler, ntraj::Int, rng::AbstractRNG)
    raw = sample_raw_states_cond(sampler, ntraj, rng)
    return Float64.(raw)
end

function integrate_dm_forward(dm_cfg_path::AbstractString, phi_cfg_path::AbstractString;
        output_h5::AbstractString="", mobility_scale_override::Float64=NaN,
        dt_override::Float64=NaN, delta_scale_override::Float64=NaN,
        skew_scale_override::Float64=NaN)
    base = dirname(dm_cfg_path)
    dm_cfg = load_dm_config(dm_cfg_path)
    fcfg = load_forward_config(phi_cfg_path)
    mobility_scale = isfinite(mobility_scale_override) ? mobility_scale_override : fcfg.mobility_scale
    delta_scale = isfinite(delta_scale_override) ? delta_scale_override : fcfg.delta_scale
    skew_scale = isfinite(skew_scale_override) ? skew_scale_override : fcfg.skew_scale
    require_condition(fcfg.xy_sym_scale >= 0.0, "forward.xy_sym_scale must be nonnegative.")
    require_condition(fcfg.z_sym_scale >= 0.0, "forward.z_sym_scale must be nonnegative.")
    require_condition(fcfg.z_diag_scale >= 0.0, "forward.z_diag_scale must be nonnegative.")
    if fcfg.lambda_matrix !== nothing
        require_condition(fcfg.delta_scale == 1.0,
            "forward.lambda_matrix cannot be combined with delta_scale.")
        require_condition(fcfg.mobility_scale == 1.0,
            "forward.lambda_matrix cannot be combined with mobility_scale.")
    end
    dt = isfinite(dt_override) ? dt_override : fcfg.dt
    device = detect_spin_device(dm_cfg.device, dm_cfg.required_gpu_name)
    activate_and_describe_device!(device, dm_cfg.device, dm_cfg.required_gpu_name)
    data_h5 = resolve_path(base, dm_cfg.input_hdf5)
    score_path = resolve_path(base, dm_cfg.score_bson)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, dm_cfg.burnin_fraction,
        dm_cfg.tau_max_decorrelation_multiples, dm_cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    blob = BSON.load(resolve_path(base, dm_cfg.output_bson))
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    phi_block = nothing
    sqrt_phi_block = nothing
    if delta_scale != 1.0 || fcfg.lambda_matrix !== nothing
        target = BSON.load(resolve_path(base, dm_cfg.target_artifact_bson))
        phi_block = Matrix{Float32}(target[:Phi_block])
        sym_phi = 0.5f0 .* (phi_block .+ transpose(phi_block))
        sqrt_phi_block = Matrix{Float32}(cholesky(Symmetric(Matrix{Float64}(sym_phi))).L)
    end
    rng = MersenneTwister(dm_cfg.seed + 1500)
    z = initial_raw_batch_cond(sampler, fcfg.ntraj, rng)
    nsteps = ceil(Int, fcfg.total_time / dt)
    burn_steps = floor(Int, fcfg.burnin_time / dt)
    save_every = max(1, round(Int, fcfg.save_dt / dt))
    nsaved = max(1, (nsteps - burn_steps) ÷ save_every + 1)
    saved = Array{Float32}(undef, nsaved, sampler.N, 3, fcfg.ntraj)
    times = Vector{Float64}(undef, nsaved)
    save_idx = 0
    progress = ProgressMeter.Progress(nsteps; desc="Forward $(basename(dm_cfg_path))")
    for step in 0:nsteps
        if step >= burn_steps && (step - burn_steps) % save_every == 0
            save_idx += 1
            saved[save_idx, :, :, :] .= Float32.(z)
            times[save_idx] = (step - burn_steps) * dt
        end
        step == nsteps && break
        raw32 = Float32.(z)
        score_raw = evaluate_raw_score_local(score_model, raw32, stats, score_sigma,
            device; batch_size=fcfg.score_batch_size)
        clamp!(score_raw, -fcfg.score_clip, fcfg.score_clip)
        if delta_scale != 1.0
            drift, noise = action_noise_divergence(model, raw32, score_raw, stats, device, rng;
                phi_block=phi_block, sqrt_phi_block=sqrt_phi_block,
                delta_scale=delta_scale, skew_gain=skew_scale,
                xy_sym_scale=fcfg.xy_sym_scale, z_sym_scale=fcfg.z_sym_scale,
                z_diag_scale=fcfg.z_diag_scale, xy_skew_scale=fcfg.xy_skew_scale,
                z_skew_scale=fcfg.z_skew_scale, lambda_matrix=fcfg.lambda_matrix)
        else
            drift, noise = action_noise_divergence(model, raw32, score_raw, stats, device, rng;
                skew_gain=skew_scale,
                xy_sym_scale=fcfg.xy_sym_scale, z_sym_scale=fcfg.z_sym_scale,
                z_diag_scale=fcfg.z_diag_scale, xy_skew_scale=fcfg.xy_skew_scale,
                z_skew_scale=fcfg.z_skew_scale, phi_block=phi_block,
                lambda_matrix=fcfg.lambda_matrix)
        end
        if mobility_scale != 1.0
            drift .*= Float32(mobility_scale)
            noise .*= Float32(sqrt(mobility_scale))
        end
        @. z = z + dt * Float64(drift) + sqrt(2.0 * dt) * Float64(noise)
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    isempty(output_h5) && (output_h5 = joinpath(@__DIR__, "..", "data",
        replace(splitext(basename(dm_cfg_path))[1], "fit_dM_" => "forward_dM_") * ".h5"))
    ensure_parent_dir(output_h5)
    h5open(output_h5, "w") do f
        f["/trajectories/time"] = times[1:save_idx]
        f["/trajectories/states"] = saved[1:save_idx, :, :, :]
        f["/metadata/model"] = "learned local mobility NN score-based Langevin"
        f["/metadata/source_config"] = dm_cfg_path
        f["/metadata/source_model"] = resolve_path(base, dm_cfg.output_bson)
        f["/metadata/divergence"] = "feature-space central finite differences"
        f["/metadata/mobility_scale"] = mobility_scale
        f["/metadata/delta_scale"] = delta_scale
        f["/metadata/skew_scale"] = skew_scale
        f["/metadata/xy_sym_scale"] = fcfg.xy_sym_scale
        f["/metadata/z_sym_scale"] = fcfg.z_sym_scale
        f["/metadata/z_diag_scale"] = fcfg.z_diag_scale
        f["/metadata/xy_skew_scale"] = fcfg.xy_skew_scale
        f["/metadata/z_skew_scale"] = fcfg.z_skew_scale
        if fcfg.lambda_matrix !== nothing
            f["/metadata/lambda_matrix"] = fcfg.lambda_matrix
            f["/metadata/lambda_calibration"] = "M_Lambda = Phi + Lambda .* (M_NN - Phi)"
        end
        f["/metadata/dt"] = dt
    end
    @printf("Saved learned-M forward trajectories to %s\n", output_h5)
    return output_h5
end

function main()
    dm_cfg = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "fit_dM_gpu2_vC.toml")
    phi_cfg = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "configs", "fit_Phi.toml")
    out = length(ARGS) >= 3 ? ARGS[3] : ""
    scale = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : NaN
    dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : NaN
    delta_scale = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : NaN
    skew_scale = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : NaN
    integrate_dm_forward(dm_cfg, phi_cfg; output_h5=out,
        mobility_scale_override=scale, dt_override=dt, delta_scale_override=delta_scale,
        skew_scale_override=skew_scale)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
