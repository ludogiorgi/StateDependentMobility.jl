#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using LinearAlgebra
using Printf
using Statistics

function block_arrays(model, raw::Array{Float32, 3}, stats::DataStats,
        p::SpinParams, device::ExecutionDevice)
    xn = move_array(apply_stats_tensor(raw, stats), device)
    bp = block_params(model, xn)
    l11, l21, l22 = Array(bp.l11), Array(bp.l21), Array(bp.l22)
    l31, l32, l33 = Array(bp.l31), Array(bp.l32), Array(bp.l33)
    k1, k2, k3 = Array(bp.k1), Array(bp.k2), Array(bp.k3)
    nb = length(l11)
    pred = Array{Float64}(undef, 3, 3, nb)
    for q in 1:nb
        S = [l11[q]^2 l11[q]*l21[q] l11[q]*l31[q];
             l11[q]*l21[q] l21[q]^2 + l22[q]^2 l21[q]*l31[q] + l22[q]*l32[q];
             l11[q]*l31[q] l21[q]*l31[q] + l22[q]*l32[q] l31[q]^2 + l32[q]^2 + l33[q]^2]
        K = [0.0 -k3[q] k2[q]; k3[q] 0.0 -k1[q]; -k2[q] k1[q] 0.0]
        pred[:, :, q] .= S .+ K
    end
    N, _, B = size(raw)
    truth = Array{Float64}(undef, 3, 3, N * B)
    for b in 1:B, i in 1:N
        M = true_mobility_matrix(@view(raw[:, :, b]), p)
        rows = ((i - 1) * 3 + 1):(i * 3)
        truth[:, :, (b - 1) * N + i] .= M[rows, rows]
    end
    return pred, truth
end

function per_entry_metrics(pred, truth)
    rel = zeros(Float64, 3, 3)
    corr = zeros(Float64, 3, 3)
    for i in 1:3, j in 1:3
        p = vec(pred[i, j, :])
        t = vec(truth[i, j, :])
        rel[i, j] = sqrt(mean((p .- t) .^ 2)) / max(sqrt(mean(t .^ 2)), eps(Float64))
        corr[i, j] = dot(p, t) / max(norm(p) * norm(t), eps(Float64))
    end
    return rel, corr
end

function evaluate_A_per_lag(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_params, target, p, device)
    rng = MersenneTwister(cfg.seed + 920)
    lags = Vector{Int}(target[:lags])
    rel = zeros(Float64, length(lags))
    corr = zeros(Float64, length(lags))
    all_pred = Float32[]
    all_target = Float32[]
    for (li, lag) in enumerate(lags)
        x0, xt, _, _, tau_norm = sample_fixed_lag_window(sampler, lag, cfg.eval_pairs_per_lag, rng)
        pred = selected_prediction(model, x0, xt, tau_norm, score_model, stats, score_sigma,
            cond_model, cond_params, target, p, cfg, device)
        ref = vec(Array{Float32}(target[:target_vec][li, :]))
        rel[li], corr[li] = agreement(pred, ref)
        append!(all_pred, Float32.(Array(pred)))
        append!(all_target, ref)
    end
    total_rel, total_corr = agreement(all_pred, all_target)
    return (; taus=Vector{Float64}(target[:taus]), rel, corr,
        total_rel, total_corr, pred=all_pred, target=all_target)
end

function render_full_diagnostics(path, variant, Aeval, pred_blocks, true_blocks, target)
    entry_rel, entry_corr = per_entry_metrics(pred_blocks, true_blocks)
    mean_pred = mean(pred_blocks; dims=3)[:, :, 1]
    mean_true = mean(true_blocks; dims=3)[:, :, 1]
    mean_phi = Matrix{Float64}(target[:Phi_block])
    sym_eigs = [minimum(eigvals(Symmetric(0.5 .* (pred_blocks[:, :, q] .+ pred_blocks[:, :, q]'))))
        for q in axes(pred_blocks, 3)]
    labels = ["11" "12" "13"; "21" "22" "23"; "31" "32" "33"]
    fig = Figure(; size=(3600, 3000))
    Label(fig[0, 1:3], "Mobility NN $(variant): A targets and true-M ex-post diagnostics";
        fontsize=30, tellwidth=false)

    ax1 = Axis(fig[1, 1]; title="A target per-lag rel.RMSE", xlabel="tau", ylabel="rel.RMSE")
    lines!(ax1, Aeval.taus, Aeval.rel; color=STYLE_HIGHLIGHT, linewidth=3)
    ax2 = Axis(fig[1, 2]; title="A target per-lag correlation", xlabel="tau", ylabel="corr")
    lines!(ax2, Aeval.taus, Aeval.corr; color=STYLE_PRIMARY, linewidth=3)
    ax3 = Axis(fig[1, 3]; title="A selected entries", xlabel="target", ylabel="prediction")
    idx = range(1, length(Aeval.target), length=min(20000, length(Aeval.target)))
    scatter!(ax3, Aeval.target[round.(Int, collect(idx))], Aeval.pred[round.(Int, collect(idx))];
        color=(:black, 0.16), markersize=3)
    lines!(ax3, [-0.25, 0.25], [-0.25, 0.25]; color=STYLE_HIGHLIGHT, linewidth=2)

    mats = [mean_pred, mean_true, mean_phi, mean_pred - mean_true, entry_rel, entry_corr]
    titles = ["mean M_NN", "mean M_true", "Phi onsite", "mean M_NN - mean M_true",
        "entry rel.RMSE", "entry corr"]
    for k in 1:6
        ax = Axis(fig[2 + (k - 1) ÷ 3, 1 + (k - 1) % 3]; title=titles[k])
        heatmap!(ax, mats[k]; colormap=:balance)
        for i in 1:3, j in 1:3
            text!(ax, j, i; text=@sprintf("%.2g", mats[k][i, j]), align=(:center, :center),
                color=:black, fontsize=18)
        end
        ax.xticks = (1:3, ["x", "y", "z"])
        ax.yticks = (1:3, ["x", "y", "z"])
    end

    for k in 1:9
        i = 1 + (k - 1) ÷ 3
        j = 1 + (k - 1) % 3
        ax = Axis(fig[4 + (k - 1) ÷ 3, 1 + (k - 1) % 3];
            title="M$(labels[i,j]) scatter", xlabel="true", ylabel="NN")
        vals_t = vec(true_blocks[i, j, :])
        vals_p = vec(pred_blocks[i, j, :])
        stride = max(1, length(vals_t) ÷ 12000)
        scatter!(ax, vals_t[1:stride:end], vals_p[1:stride:end];
            color=(STYLE_PRIMARY, 0.18), markersize=3)
        lo = min(minimum(vals_t), minimum(vals_p))
        hi = max(maximum(vals_t), maximum(vals_p))
        lines!(ax, [lo, hi], [lo, hi]; color=:black, linestyle=:dash)
    end

    axeig = Axis(fig[7, 1:3]; title="Symmetric-part minimum eigenvalue of M_NN blocks",
        xlabel="sample", ylabel="min eig")
    lines!(axeig, collect(eachindex(sym_eigs)), sort(sym_eigs); color=STYLE_ACCENT, linewidth=2)
    Label(fig[8, 1:3],
        @sprintf("A overall rel.RMSE %.4f corr %.4f; block true-M rel.RMSE %.4f corr %.4f; mean-vs-true rel %.4f; mean-vs-Phi rel %.4f",
            Aeval.total_rel, Aeval.total_corr,
            norm(vec(pred_blocks .- true_blocks)) / max(norm(vec(true_blocks)), eps(Float64)),
            dot(vec(pred_blocks), vec(true_blocks)) / max(norm(vec(pred_blocks)) * norm(vec(true_blocks)), eps(Float64)),
            norm(mean_pred - mean_true) / max(norm(mean_true), eps(Float64)),
            norm(mean_pred - mean_phi) / max(norm(mean_phi), eps(Float64)));
        fontsize=22, tellwidth=false)
    save_figure_checked(path, fig)
end

function main()
    cfg_path = length(ARGS) >= 1 ? ARGS[1] :
        normpath(joinpath(@__DIR__, "..", "configs", "fit_dM_gpu2_vC.toml"))
    model_path_arg = length(ARGS) >= 2 ? ARGS[2] : ""
    variant = splitext(basename(cfg_path))[1]
    base = dirname(cfg_path)
    cfg = load_dm_config(cfg_path)
    device = detect_spin_device(cfg.device, cfg.required_gpu_name)
    activate_and_describe_device!(device, cfg.device, cfg.required_gpu_name)
    data_h5 = resolve_path(base, cfg.input_hdf5)
    score_path = resolve_path(base, cfg.score_bson)
    cond_path = resolve_path(base, cfg.cond_score_bson)
    model_path = isempty(model_path_arg) ? resolve_path(base, cfg.output_bson) : model_path_arg
    p = load_phys(data_h5)
    sampler = build_cond_sampler(data_h5, cfg.burnin_fraction,
        cfg.tau_max_decorrelation_multiples, cfg.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(score_path, device)
    cond_cfg = load_config(resolve_path(base, "cond_score_gpu0_vA.toml"))
    cond_blob = BSON.load(cond_path)
    cond_model = move_model(cond_blob[:host_model], device)
    Flux.testmode!(cond_model)
    target = BSON.load(resolve_path(base, cfg.target_artifact_bson))
    blob = BSON.load(model_path)
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)

    Aeval = evaluate_A_per_lag(model, cfg, sampler, score_model, stats, score_sigma,
        cond_model, cond_cfg, target, p, device)
    raw = sample_raw_states_cond(sampler, 20000, MersenneTwister(cfg.seed + 930))
    pred_blocks, true_blocks = block_arrays(model, raw, stats, p, device)
    entry_rel, entry_corr = per_entry_metrics(pred_blocks, true_blocks)
    mean_pred = mean(pred_blocks; dims=3)[:, :, 1]
    mean_true = mean(true_blocks; dims=3)[:, :, 1]
    mean_phi = Matrix{Float64}(target[:Phi_block])
    rel_blocks = norm(vec(pred_blocks .- true_blocks)) / max(norm(vec(true_blocks)), eps(Float64))
    corr_blocks = dot(vec(pred_blocks), vec(true_blocks)) / max(norm(vec(pred_blocks)) * norm(vec(true_blocks)), eps(Float64))
    out_fig = joinpath(@__DIR__, "..", "figures", "$(variant)_full_trueM_diagnostics.png")
    out_metrics = joinpath(@__DIR__, "..", "logs", "$(variant)_full_trueM_diagnostics.txt")
    render_full_diagnostics(out_fig, variant, Aeval, pred_blocks, true_blocks, target)
    open(out_metrics, "w") do io
        println(io, "Full mobility NN true-M diagnostics")
        println(io, "variant = $(variant)")
        println(io, @sprintf("A overall rel.RMSE = %.8e", Aeval.total_rel))
        println(io, @sprintf("A overall corr = %.8e", Aeval.total_corr))
        println(io, @sprintf("M block rel.RMSE ex-post = %.8e", rel_blocks))
        println(io, @sprintf("M block corr ex-post = %.8e", corr_blocks))
        println(io, @sprintf("mean M_NN vs mean M_true rel.RMSE = %.8e",
            norm(mean_pred - mean_true) / max(norm(mean_true), eps(Float64))))
        println(io, @sprintf("mean M_NN vs Phi onsite rel.RMSE = %.8e",
            norm(mean_pred - mean_phi) / max(norm(mean_phi), eps(Float64))))
        println(io, "Per-entry rel.RMSE:")
        for i in 1:3
            println(io, join([@sprintf("%.8e", entry_rel[i, j]) for j in 1:3], " "))
        end
        println(io, "Per-entry corr:")
        for i in 1:3
            println(io, join([@sprintf("%.8e", entry_corr[i, j]) for j in 1:3], " "))
        end
        println(io, "No Langevin equation was run.")
        println(io, "No-cheating audit: true M was used only for this ex-post diagnostic.")
    end
    @printf("Saved full dM diagnostics to %s\n", out_fig)
    @printf("Saved full dM metrics to %s\n", out_metrics)
end

main()
