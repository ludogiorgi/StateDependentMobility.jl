#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using LinearAlgebra
using Printf
using Statistics

const PHYS_ANSATZ_DEFAULT_CONFIG = normpath(joinpath(@__DIR__, "..", "configs",
    "fit_dM_phys_ansatz_clean37_mean10_gpu0.toml"))

Base.@kwdef struct PhysicalAnsatzConfig
    dm::DMConfig
    active_first::Int
    active_last::Int
    design_pairs_per_lag::Int
    eval_pairs_per_lag::Int
    mean_samples::Int
    ridge::Float64
    mean_penalty_weight::Float64
    fit_intercept::Bool
    force_psd_grid::Bool
    symmetric_coeff_floor::Float64
    output_bson::String
    metrics_txt::String
    figure_png::String
end

function load_physical_ansatz_config(path::AbstractString)
    raw = TOML.parsefile(path)
    ans = raw["physical_ansatz"]
    out = raw["output"]
    return PhysicalAnsatzConfig(
        dm=load_dm_config(path),
        active_first=Int(get(ans, "first_lag_index", get(raw["training"], "first_lag_index", 7))),
        active_last=Int(get(ans, "last_lag_index", get(raw["training"], "last_lag_index", 0))),
        design_pairs_per_lag=Int(get(ans, "design_pairs_per_lag", 80000)),
        eval_pairs_per_lag=Int(get(ans, "eval_pairs_per_lag", 50000)),
        mean_samples=Int(get(ans, "mean_samples", 120000)),
        ridge=Float64(get(ans, "ridge", 1.0e-8)),
        mean_penalty_weight=Float64(get(ans, "mean_penalty_weight", 10.0)),
        fit_intercept=Bool(get(ans, "fit_intercept", false)),
        force_psd_grid=Bool(get(ans, "force_psd_grid", true)),
        symmetric_coeff_floor=Float64(get(ans, "symmetric_coeff_floor", 1.0e-8)),
        output_bson=String(out["model_bson"]),
        metrics_txt=String(out["metrics_txt"]),
        figure_png=String(out["figure_png"]),
    )
end

function physical_active_lag_indices(cfg::PhysicalAnsatzConfig, nlags::Int)
    last = cfg.active_last <= 0 ? nlags : min(cfg.active_last, nlags)
    require_condition(1 <= cfg.active_first <= last <= nlags,
        "Invalid physical ansatz active lag window.")
    return collect(cfg.active_first:last)
end

function physical_basis_actions_transpose(raw::Array{Float32, 3}, rraw::Array{Float32, 3})
    N, _, B = size(raw)
    D = 3N
    out = [Matrix{Float32}(undef, D, B) for _ in 1:4]
    @inbounds for b in 1:B, i in 1:N
        x1, x2, x3 = raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]
        r1, r2v, r3 = rraw[i, 1, b], rraw[i, 2, b], rraw[i, 3, b]
        rsq = x1 * x1 + x2 * x2 + x3 * x3
        dotxr = x1 * r1 + x2 * r2v + x3 * r3
        # C r = x cross r for C=[x]_x; the paper residual uses M^T r.
        c1 = x2 * r3 - x3 * r2v
        c2 = x3 * r1 - x1 * r3
        c3 = x1 * r2v - x2 * r1
        q = (i - 1) * 3
        out[1][q + 1, b] = r1
        out[1][q + 2, b] = r2v
        out[1][q + 3, b] = r3
        out[2][q + 1, b] = rsq * r1 - x1 * dotxr
        out[2][q + 2, b] = rsq * r2v - x2 * dotxr
        out[2][q + 3, b] = rsq * r3 - x3 * dotxr
        out[3][q + 1, b] = x1 * dotxr
        out[3][q + 2, b] = x2 * dotxr
        out[3][q + 3, b] = x3 * dotxr
        out[4][q + 1, b] = -c1
        out[4][q + 2, b] = -c2
        out[4][q + 3, b] = -c3
    end
    return out
end

function selected_from_action(obs_flat::Matrix{Float32}, action_flat::Matrix{Float32},
        selected_indices)
    B = size(obs_flat, 2)
    mat = -(obs_flat * transpose(action_flat)) ./ Float32(B)
    return Float64.(vec(mat)[selected_indices])
end

function zero_model_offset(obs_flat::Matrix{Float32}, rraw::Array{Float32, 3},
        Phi::Matrix{Float32}, selected_indices)
    B = size(obs_flat, 2)
    rflat = reshape(permutedims(rraw, (2, 1, 3)), size(Phi, 1), B)
    phi_action = Phi' * rflat
    mat = (obs_flat * transpose(phi_action)) ./ Float32(B)
    return Float64.(vec(mat)[selected_indices])
end

function accumulate_physical_design(cfg::PhysicalAnsatzConfig, sampler, score_model,
        stats, score_sigma, cond_model, cond_cfg, cond_kind::Symbol, target, p,
        device; npairs::Int, active_lag_indices)
    rng = MersenneTwister(cfg.dm.seed + npairs + 9100)
    lib = NonlinearLibrary(Vector{String}(target[:names]))
    means = Vector{Float64}(target[:observable_means])
    lags = Vector{Int}(target[:lags])
    selected = target[:selected_indices]
    Phi = Matrix{Float32}(target[:Phi])
    nsel = length(selected)
    nrows = nsel * length(active_lag_indices)
    X = Matrix{Float64}(undef, nrows, 4)
    y = Vector{Float64}(undef, nrows)
    offset = Vector{Float64}(undef, nrows)
    scale = Vector{Float64}(undef, nrows)
    row0 = 0
    for li in active_lag_indices
        lag = lags[li]
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, npairs, rng)
        rnorm = evaluate_transition_norm(cond_kind, cond_model, x0, xt, tau_norm, stats,
            cond_cfg, device; batch_size=min(cond_cfg.batch_size, npairs),
            score_model=score_model, score_sigma=score_sigma)
        rraw = normalized_residual_to_raw(rnorm, stats)
        obs = nonlinear_observables(xt, p, lib)
        center_observables!(obs, means)
        obs_flat = reshape(obs, size(obs, 1) * size(obs, 2), npairs)
        basis = physical_basis_actions_transpose(x0, rraw)
        rows = (row0 + 1):(row0 + nsel)
        for k in 1:4
            X[rows, k] .= selected_from_action(obs_flat, basis[k], selected)
        end
        offset[rows] .= zero_model_offset(obs_flat, rraw, Phi, selected)
        y[rows] .= Float64.(vec(Array{Float32}(target[:target_vec][li, :])))
        if haskey(target, :scale_vec)
            scale[rows] .= Float64.(target[:scale_vec])
        else
            scale[rows] .= max.(sqrt.(mean(abs2, Float64.(target[:target_vec]); dims=1))[:], 1e-6)
        end
        row0 += nsel
        @printf("Physical ansatz design lag %.5g (%d/%d), pairs=%d\n",
            lag * sampler.save_dt, findfirst(==(li), active_lag_indices),
            length(active_lag_indices), npairs)
        GC.gc()
    end
    return (; X, y, offset, scale)
end

function physical_mean_basis(raw::Array{Float32, 3})
    N, _, B = size(raw)
    basis = [zeros(Float64, 3, 3) for _ in 1:4]
    @inbounds for b in 1:B, i in 1:N
        x = Float64[raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]]
        rsq = dot(x, x)
        C = [0.0 -x[3] x[2]; x[3] 0.0 -x[1]; -x[2] x[1] 0.0]
        basis[1] .+= I(3)
        basis[2] .+= rsq .* Matrix{Float64}(I, 3, 3) .- x * x'
        basis[3] .+= x * x'
        basis[4] .+= C
    end
    denom = N * B
    return [Bmat ./ denom for Bmat in basis]
end

function fit_coefficients(design, target, mean_basis, phi_block::Matrix{Float32},
        cfg::PhysicalAnsatzConfig)
    X = copy(design.X)
    y = design.y .- design.offset
    scale = max.(design.scale, 1e-7)
    Xw = X ./ scale
    yw = y ./ scale
    ncoef = size(X, 2)
    if cfg.mean_penalty_weight > 0
        phi = vec(Matrix{Float64}(phi_block))
        norm_phi = max(norm(phi), eps(Float64))
        Xm = zeros(Float64, 9, ncoef)
        for k in 1:ncoef
            Xm[:, k] .= vec(mean_basis[k])
        end
        wm = sqrt(cfg.mean_penalty_weight) / norm_phi
        Xw = vcat(Xw, wm .* Xm)
        yw = vcat(yw, wm .* phi)
    end
    if cfg.fit_intercept
        Xw = hcat(Xw, ones(size(Xw, 1)))
    end
    ridge = cfg.ridge .* I(size(Xw, 2))
    coeff_aug = (transpose(Xw) * Xw + ridge) \ (transpose(Xw) * yw)
    coeff = coeff_aug[1:ncoef]
    intercept = cfg.fit_intercept ? coeff_aug[end] : 0.0
    if cfg.force_psd_grid && any(coeff[1:3] .< 0)
        @printf("Warning: unconstrained fit has non-PSD coefficients %s; applying nonnegative symmetric projection.\n",
            repr(coeff[1:3]))
        coeff[1:3] .= max.(coeff[1:3], cfg.symmetric_coeff_floor)
    end
    pred = design.offset .+ design.X * coeff .+ intercept
    rel, corr = agreement(pred, design.y)
    return (; coeff, intercept, relative_rmse=rel, correlation=corr, pred)
end

function physical_coeff_true(p::SpinParams)
    return Float64[p.theta * p.eps, p.theta * p.alpha_perp,
        p.theta * p.alpha_parallel, -p.gamma * p.theta]
end

function physical_blocks_from_coeff(raw::Array{Float32, 3}, coeff::AbstractVector{<:Real})
    N, _, B = size(raw)
    blocks = Array{Float64}(undef, 3, 3, N * B)
    @inbounds for b in 1:B, i in 1:N
        x = Float64[raw[i, 1, b], raw[i, 2, b], raw[i, 3, b]]
        rsq = dot(x, x)
        C = [0.0 -x[3] x[2]; x[3] 0.0 -x[1]; -x[2] x[1] 0.0]
        blocks[:, :, (b - 1) * N + i] .= coeff[1] .* Matrix{Float64}(I, 3, 3) .+
            coeff[2] .* (rsq .* Matrix{Float64}(I, 3, 3) .- x * x') .+
            coeff[3] .* (x * x') .+ coeff[4] .* C
    end
    return blocks
end

function physical_true_diagnostics(coeff, sampler, target, p, nsamples::Int, seed::Int)
    raw = sample_raw_states_cond(sampler, nsamples, MersenneTwister(seed))
    pred = physical_blocks_from_coeff(raw, coeff)
    N, _, B = size(raw)
    truth = similar(pred)
    @inbounds for b in 1:B, i in 1:N
        M = true_mobility_matrix(@view(raw[:, :, b]), p)
        rows = ((i - 1) * 3 + 1):(i * 3)
        truth[:, :, (b - 1) * N + i] .= M[rows, rows]
    end
    rel = norm(vec(pred .- truth)) / max(norm(vec(truth)), eps(Float64))
    corr = dot(vec(pred), vec(truth)) / max(norm(vec(pred)) * norm(vec(truth)), eps(Float64))
    mean_pred = mean(pred; dims=3)[:, :, 1]
    mean_true = mean(truth; dims=3)[:, :, 1]
    mean_phi = Matrix{Float64}(target[:Phi_block])
    return (; relative_rmse=rel, correlation=corr, mean_pred, mean_true, mean_phi,
        mean_phi_rel=norm(mean_pred - mean_phi) / max(norm(mean_phi), eps(Float64)))
end

function physical_psd_diagnostics(coeff, raw::Array{Float32, 3})
    c0, cp, cq = coeff[1], coeff[2], coeff[3]
    vals_perp = Float64[]
    vals_para = Float64[]
    @inbounds for b in axes(raw, 3), i in axes(raw, 1)
        rsq = sum(abs2, Float64.(raw[i, :, b]))
        push!(vals_perp, c0 + cp * rsq)
        push!(vals_para, c0 + cq * rsq)
    end
    return (; min_perp=minimum(vals_perp), min_parallel=minimum(vals_para),
        mean_perp=mean(vals_perp), mean_parallel=mean(vals_para))
end

function render_physical_ansatz_figure(path, fit_train, fit_eval, true_metrics,
        coeff, true_coeff, psd_diag)
    fig = Figure(; size=(2400, 1600))
    Label(fig[0, 1:2], "SoftSpinLLG physics-informed mobility ansatz";
        fontsize=30, tellwidth=false)
    ax1 = Axis(fig[1, 1]; title="Coefficient values", xticks=(1:4, ["I", "perp", "parallel", "skew"]))
    barplot!(ax1, 1:4, coeff; color=STYLE_PRIMARY, label="fit")
    scatter!(ax1, 1:4, true_coeff; color=STYLE_HIGHLIGHT, markersize=18, label="true ex-post")
    axislegend(ax1; position=:lt)
    ax2 = Axis(fig[1, 2]; title="A prediction vs target", xlabel="target", ylabel="prediction")
    idx = round.(Int, range(1, length(fit_eval.target), length=min(25000, length(fit_eval.target))))
    scatter!(ax2, fit_eval.target[idx], fit_eval.pred[idx]; markersize=2, color=(:black, 0.25))
    lo = minimum(fit_eval.target[idx]); hi = maximum(fit_eval.target[idx])
    lines!(ax2, [lo, hi], [lo, hi]; color=STYLE_HIGHLIGHT, linewidth=3)
    mats = [true_metrics.mean_pred, true_metrics.mean_true, true_metrics.mean_phi,
        true_metrics.mean_pred - true_metrics.mean_true]
    titles = ["mean M_phys", "mean M_true", "Phi onsite", "M_phys - M_true"]
    for j in 1:4
        ax = Axis(fig[2 + (j - 1) ÷ 2, 1 + (j - 1) % 2]; title=titles[j])
        heatmap!(ax, mats[j]; colormap=:balance)
    end
    Label(fig[4, 1:2],
        @sprintf("train A rel %.4f corr %.4f; eval A rel %.4f corr %.4f; true-M rel %.4f corr %.4f; min sym eig proxies %.4g/%.4g",
            fit_train.relative_rmse, fit_train.correlation,
            fit_eval.relative_rmse, fit_eval.correlation,
            true_metrics.relative_rmse, true_metrics.correlation,
            psd_diag.min_perp, psd_diag.min_parallel);
        fontsize=21, tellwidth=false)
    save_figure_checked(path, fig)
end

function fit_physical_ansatz(cfg_path::AbstractString)
    base = dirname(cfg_path)
    cfg = load_physical_ansatz_config(cfg_path)
    dm = cfg.dm
    device = detect_spin_device(dm.device, dm.required_gpu_name)
    activate_and_describe_device!(device, dm.device, dm.required_gpu_name)
    data_h5 = resolve_path(base, dm.input_hdf5)
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, dm.burnin_fraction,
        dm.tau_max_decorrelation_multiples, dm.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(resolve_path(base, dm.score_bson), device)
    cond_kind = configured_cond_score_kind(cfg_path)
    cond_model, cond_cfg = load_transition_source(cond_kind, cfg_path,
        resolve_path(base, dm.cond_score_bson), base, device)
    target = BSON.load(resolve_path(base, dm.target_artifact_bson))
    active = physical_active_lag_indices(cfg, length(Vector{Int}(target[:lags])))
    train_design = accumulate_physical_design(cfg, sampler, score_model, stats,
        score_sigma, cond_model, cond_cfg, cond_kind, target, p, device;
        npairs=cfg.design_pairs_per_lag, active_lag_indices=active)
    mean_raw = sample_raw_states_cond(sampler, cfg.mean_samples, MersenneTwister(dm.seed + 9200))
    mean_basis = physical_mean_basis(mean_raw)
    fit_train = fit_coefficients(train_design, target, mean_basis,
        Matrix{Float32}(target[:Phi_block]), cfg)
    eval_design = accumulate_physical_design(cfg, sampler, score_model, stats,
        score_sigma, cond_model, cond_cfg, cond_kind, target, p, device;
        npairs=cfg.eval_pairs_per_lag, active_lag_indices=active)
    pred_eval = eval_design.offset .+ eval_design.X * fit_train.coeff .+ fit_train.intercept
    rel_eval, corr_eval = agreement(pred_eval, eval_design.y)
    fit_eval = (; relative_rmse=rel_eval, correlation=corr_eval,
        pred=pred_eval, target=eval_design.y)
    true_metrics = physical_true_diagnostics(fit_train.coeff, sampler, target, p,
        min(cfg.mean_samples, 120000), dm.seed + 9300)
    psd_diag = physical_psd_diagnostics(fit_train.coeff, mean_raw)
    true_coeff = physical_coeff_true(p)
    out_model = resolve_path(base, cfg.output_bson)
    ensure_parent_dir(out_model)
    BSON.bson(out_model, Dict(:coefficients => fit_train.coeff,
        :coefficient_names => ["theta_eps", "theta_alpha_perp", "theta_alpha_parallel", "minus_gamma_theta"],
        :intercept => fit_train.intercept,
        :cfg_path => cfg_path,
        :dm_config => dm,
        :fit_train => fit_train,
        :fit_eval => fit_eval,
        :true_metrics => true_metrics,
        :true_coefficients_expost => true_coeff,
        :psd_diag => psd_diag,
        :mean_basis => mean_basis,
        :target_artifact => dm.target_artifact_bson,
        :active_lag_indices => active,
        :no_cheating_audit => "Physics-informed ansatz used the true mobility tensor form as a declared structural prior. Coefficients were fitted from clean37 data-only A targets, data-only Phi, learned stationary score, and learned transition score. True coefficient values were saved only as ex-post diagnostics."))
    metrics_path = resolve_path(base, cfg.metrics_txt)
    ensure_parent_dir(metrics_path)
    open(metrics_path, "w") do io
        println(io, "SoftSpinLLGChain physics-informed M ansatz metrics")
        println(io, "config = $(basename(cfg_path))")
        println(io, "active_lag_indices = $(first(active)):$(last(active)) / $(length(Vector{Int}(target[:lags])))")
        println(io, "design_pairs_per_lag = $(cfg.design_pairs_per_lag)")
        println(io, "eval_pairs_per_lag = $(cfg.eval_pairs_per_lag)")
        println(io, "mean_penalty_weight = $(cfg.mean_penalty_weight)")
        println(io, "ridge = $(cfg.ridge)")
        println(io, @sprintf("coeff I = %.12e", fit_train.coeff[1]))
        println(io, @sprintf("coeff perp = %.12e", fit_train.coeff[2]))
        println(io, @sprintf("coeff parallel = %.12e", fit_train.coeff[3]))
        println(io, @sprintf("coeff skew = %.12e", fit_train.coeff[4]))
        println(io, "true coefficients ex-post = $(repr(true_coeff))")
        println(io, @sprintf("train A rel.RMSE = %.8e", fit_train.relative_rmse))
        println(io, @sprintf("train A corr = %.8e", fit_train.correlation))
        println(io, @sprintf("eval A rel.RMSE = %.8e", fit_eval.relative_rmse))
        println(io, @sprintf("eval A corr = %.8e", fit_eval.correlation))
        println(io, @sprintf("true M block rel.RMSE ex-post = %.8e", true_metrics.relative_rmse))
        println(io, @sprintf("true M block corr ex-post = %.8e", true_metrics.correlation))
        println(io, @sprintf("mean M_phys vs Phi onsite rel.RMSE = %.8e", true_metrics.mean_phi_rel))
        println(io, @sprintf("min lambda_perp proxy = %.8e", psd_diag.min_perp))
        println(io, @sprintf("min lambda_parallel proxy = %.8e", psd_diag.min_parallel))
        println(io, "No Langevin equation was run by this script.")
        println(io, "Audit: true functional form was used as a declared physics prior; true coefficients were not used in the fit.")
    end
    render_physical_ansatz_figure(resolve_path(base, cfg.figure_png), fit_train,
        fit_eval, true_metrics, fit_train.coeff, true_coeff, psd_diag)
    @printf("Saved physical ansatz model to %s\n", out_model)
    @printf("Saved metrics to %s\n", metrics_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_physical_ansatz(length(ARGS) >= 1 ? ARGS[1] : PHYS_ANSATZ_DEFAULT_CONFIG)
end
