#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dm.jl"))

const DEFAULT_REGENERATE_PARAM_FILE = joinpath(@__DIR__, "regenerate_run_figures.toml")

Base.@kwdef struct RegenerateRunFiguresParams
    run_dir::String
    output_dir::String
    overwrite::Bool = true
    fit_width::Union{Nothing, Int} = nothing
    fit_height::Union{Nothing, Int} = nothing
    forward_width::Union{Nothing, Int} = nothing
    forward_height::Union{Nothing, Int} = nothing
    use_latex::Bool = true
    show_global_titles::Bool = false
    show_metric_titles::Bool = false
    show_summary_panels::Bool = false
end

function history_from_saved_dict(raw)
    return MobilityNNHistory(
        Int.(vec(raw[:epochs])),
        Float64.(vec(raw[:train_loss])),
        Float64.(vec(raw[:normalized_rmse])),
        Float64.(vec(raw[:physical_rmse])),
        Float64.(vec(raw[:mean_abs_delta])),
        Float64.(vec(raw[:mean_delta11])),
        Float64.(vec(raw[:mean_delta12])),
        Float64.(vec(raw[:mean_delta21])),
        Float64.(vec(raw[:mean_delta22])),
        Float64.(vec(raw[:weight_l2])),
    )
end

function string_dict(raw)
    return Dict{String, Any}(String(key) => value for (key, value) in pairs(raw))
end

function resolve_config_path(config_path::AbstractString, path::AbstractString)
    base_dir = dirname(abspath(config_path))
    return isabspath(path) ? normpath(String(path)) : normpath(joinpath(base_dir, String(path)))
end

function load_regenerate_params(path::AbstractString)
    raw = TOML.parsefile(path)
    input_cfg = string_dict(raw["input"])
    output_cfg = haskey(raw, "output") ? string_dict(raw["output"]) : Dict{String, Any}()
    fit_cfg = haskey(raw, "fit") ? string_dict(raw["fit"]) : Dict{String, Any}()
    forward_cfg = haskey(raw, "forward") ? string_dict(raw["forward"]) : Dict{String, Any}()
    theme_cfg = haskey(raw, "theme") ? string_dict(raw["theme"]) : Dict{String, Any}()
    render_cfg = haskey(raw, "render") ? string_dict(raw["render"]) : Dict{String, Any}()

    params = RegenerateRunFiguresParams(
        run_dir=resolve_config_path(path, String(input_cfg["run_dir"])),
        output_dir=resolve_config_path(path, String(get(output_cfg, "figure_dir", "figures"))),
        overwrite=Bool(get(output_cfg, "overwrite", true)),
        fit_width=haskey(fit_cfg, "width") ? Int(fit_cfg["width"]) : nothing,
        fit_height=haskey(fit_cfg, "height") ? Int(fit_cfg["height"]) : nothing,
        forward_width=haskey(forward_cfg, "width") ? Int(forward_cfg["width"]) : nothing,
        forward_height=haskey(forward_cfg, "height") ? Int(forward_cfg["height"]) : nothing,
        use_latex=Bool(get(render_cfg, "use_latex", true)),
        show_global_titles=Bool(get(render_cfg, "show_global_titles", false)),
        show_metric_titles=Bool(get(render_cfg, "show_metric_titles", false)),
        show_summary_panels=Bool(get(render_cfg, "show_summary_panels", false)),
    )
    return params, theme_cfg
end

function apply_theme_overrides!(theme_cfg::Dict{String, Any})
    configure_figure_style!(
        px_per_unit=get(theme_cfg, "px_per_unit", nothing),
        fontsize=get(theme_cfg, "fontsize", nothing),
        axis_ticklabelsize=get(theme_cfg, "axis_ticklabelsize", nothing),
        axis_labelsize=get(theme_cfg, "axis_labelsize", nothing),
        axis_titlesize=get(theme_cfg, "axis_titlesize", nothing),
        legend_labelsize=get(theme_cfg, "legend_labelsize", nothing),
        legend_titlesize=get(theme_cfg, "legend_titlesize", nothing),
        colorbar_labelsize=get(theme_cfg, "colorbar_labelsize", nothing),
        colorbar_ticklabelsize=get(theme_cfg, "colorbar_ticklabelsize", nothing),
        label_fontsize=get(theme_cfg, "label_fontsize", nothing),
        text_fontsize=get(theme_cfg, "text_fontsize", nothing),
        summary_fontsize=get(theme_cfg, "summary_fontsize", nothing),
        summary_title_fontsize=get(theme_cfg, "summary_title_fontsize", nothing),
        figure_title_fontsize=get(theme_cfg, "figure_title_fontsize", nothing),
        figure_subtitle_fontsize=get(theme_cfg, "figure_subtitle_fontsize", nothing),
    )
    return nothing
end

function maybe_remove_output(path::AbstractString, overwrite::Bool)
    if isfile(path)
        overwrite || error("Refusing to overwrite existing output $(path). Set output.overwrite = true.")
        rm(path; force=true)
    end
    return nothing
end

function output_path_in_dir(output_dir::AbstractString, original_path::AbstractString, overwrite::Bool)
    mkpath(output_dir)
    path = joinpath(output_dir, basename(original_path))
    maybe_remove_output(path, overwrite)
    return path
end

function compute_rmse_tables(data_arr::Array{Float64, 3}, ref_arr::Array{Float64, 3}, nn_arr::Array{Float64, 3})
    nobs = size(data_arr, 2)
    rmse_data_ref = zeros(Float64, nobs, nobs)
    rmse_data_nn = zeros(Float64, nobs, nobs)
    rmse_ref_nn = zeros(Float64, nobs, nobs)
    for m in 1:nobs, n in 1:nobs
        rmse_data_ref[m, n] = sqrt(mean((data_arr[:, m, n] .- ref_arr[:, m, n]) .^ 2))
        rmse_data_nn[m, n] = sqrt(mean((data_arr[:, m, n] .- nn_arr[:, m, n]) .^ 2))
        rmse_ref_nn[m, n] = sqrt(mean((ref_arr[:, m, n] .- nn_arr[:, m, n]) .^ 2))
    end
    return rmse_data_ref, rmse_data_nn, rmse_ref_nn
end

function parse_training_channel_metrics(metrics_txt::AbstractString, channel_labels::Vector{String})
    pattern = r"^(.+?)\s+\|\s+RMSE data/NN = ([^|]+)\s+\|\s+normalized = ([^|]+)\s+\|\s+RMSE true/NN = ([^|]+)\s+\|\s+scale = (.+)$"
    parsed = Dict{String, NTuple{4, Float64}}()
    for line in eachline(metrics_txt)
        match_obj = match(pattern, line)
        match_obj === nothing && continue
        label = strip(match_obj.captures[1])
        parsed[label] = (
            parse(Float64, strip(match_obj.captures[2])),
            parse(Float64, strip(match_obj.captures[3])),
            parse(Float64, strip(match_obj.captures[4])),
            parse(Float64, strip(match_obj.captures[5])),
        )
    end

    nchannels = length(channel_labels)
    rmse_phys = zeros(Float64, nchannels, 1)
    rmse_norm = zeros(Float64, nchannels, 1)
    target_scale = zeros(Float64, nchannels, 1)
    for (idx, label) in enumerate(channel_labels)
        haskey(parsed, label) || error("Could not find diagnostics line for training channel $(label) in $(metrics_txt).")
        rmse_phys[idx, 1] = parsed[label][1]
        rmse_norm[idx, 1] = parsed[label][2]
        target_scale[idx, 1] = parsed[label][4]
    end
    return rmse_phys, rmse_norm, target_scale
end

function lookup_metric_lines(metrics_txt::AbstractString, prefixes::Vector{String})
    lines = readlines(metrics_txt)
    selected = String[]
    for prefix in prefixes
        idx = findfirst(line -> startswith(line, prefix), lines)
        idx === nothing || push!(selected, strip(lines[idx]))
    end
    return selected
end

function fit_summary_lines(metrics_txt::AbstractString, params::FitDMParams, artifact_data, model_data)
    metric_lines = lookup_metric_lines(metrics_txt, [
        "Mean training-channel RMSE data/NN",
        "Mean training-channel normalized RMSE",
        "Mean training-channel RMSE true/NN",
        "Heatmap-on-support RMSE ΔM total",
        "Heatmap-on-support relative RMSE ΔM",
        "Heatmap-on-support R2 ΔM",
        "Pointwise RMSE ΔM total",
        "Pointwise relative RMSE ΔM",
        "||<ΔM>_NN-<ΔM>_true||_F",
    ])
    train_pairs = Tuple{Int, Int}[(Int(pair[1]), Int(pair[2])) for pair in artifact_data[:training_pairs]]
    train_pair_weights = Float64.(artifact_data[:training_pair_weights])
    lines = String[
        "Training target = " * String(get(artifact_data, :training_target_source, "unknown")),
        metric_lines...,
        "# training channels = " * string(length(model_data[:training_channel_labels])),
        training_pairs_line(train_pairs),
        training_pair_weights_line(train_pair_weights),
        @sprintf("pairs/tau = %d", params.mobility_nn_pairs_per_tau),
        @sprintf("tau batch = %d", params.mobility_nn_tau_batch),
        @sprintf("anchor points = %d", params.mobility_nn_anchor_points),
        @sprintf("epochs = %d", params.mobility_nn_epochs),
        @sprintf("lr = %.2e", params.mobility_nn_learning_rate),
        @sprintf("lag weight = %.2f", params.mobility_nn_lag_weight_power),
        @sprintf("mean penalty = %.2e", params.mobility_nn_mean_penalty_weight),
        @sprintf("anchor rms penalty = %.2e", params.mobility_nn_anchor_rms_penalty_weight),
        @sprintf("weight decay = %.2e", params.mobility_nn_weight_decay),
        "checkpoint metric = " * params.mobility_nn_checkpoint_metric,
        isempty(params.mobility_nn_widths) ? "widths = affine-only" : "widths = " * join(params.mobility_nn_widths, " x "),
    ]
    if params.mobility_nn_reliability_mode != "none"
        push!(lines, "reliability weighting = " * params.mobility_nn_reliability_mode)
        push!(lines, @sprintf("reliability splits = %d", params.mobility_nn_reliability_splits))
        push!(lines, @sprintf("reliability fraction = %.2f", params.mobility_nn_reliability_fraction))
        push!(lines, @sprintf("reliability min weight = %.2f", params.mobility_nn_reliability_min_weight))
        push!(lines, @sprintf("reliability strength = %.2f", params.mobility_nn_reliability_strength))
    end
    if params.mobility_nn_mean_penalty_final_scale != 1.0
        push!(lines, @sprintf("mean penalty final scale = %.2f", params.mobility_nn_mean_penalty_final_scale))
    end
    if params.mobility_nn_anchor_rms_penalty_final_scale != 1.0
        push!(lines, @sprintf("anchor penalty final scale = %.2f", params.mobility_nn_anchor_rms_penalty_final_scale))
    end
    return lines
end

function density1d_from_arrays(centers, density)
    centers64 = Float64.(vec(centers))
    return Density1D(centers64, Float64.(vec(density)), grid_boundary(centers64))
end

function density2d_from_arrays(xgrid, ygrid, density)
    x64 = Float64.(vec(xgrid))
    y64 = Float64.(vec(ygrid))
    return Density2D(x64, y64, Float64.(density), grid_boundary(x64), grid_boundary(y64))
end

function symbol_float_dict(raw)
    return Dict{Symbol, Float64}(Symbol(key) => Float64(value) for (key, value) in pairs(raw))
end

function rebuild_forward_plot_inputs(diag)
    corr_metrics = symbol_float_dict(diag[:corr_metrics])
    corr_metrics_phi = symbol_float_dict(diag[:corr_metrics_phi])
    observed_corr_metrics = symbol_float_dict(diag[:observed_corr_metrics])
    observed_corr_metrics_phi = symbol_float_dict(diag[:observed_corr_metrics_phi])
    pdf_metrics = symbol_float_dict(diag[:pdf_metrics])
    observed_pdf_metrics = symbol_float_dict(diag[:observed_pdf_metrics])

    corr_lags = Float64.(vec(diag[:corr_lags]))
    observed_corr_lags = Float64.(vec(diag[:observed_corr_lags]))

    pdf_data_full = Dict(
        :true_x => density1d_from_arrays(diag[:pdf_true_x_centers], diag[:pdf_true_x_density]),
        :pred_x => density1d_from_arrays(diag[:pdf_true_x_centers], diag[:pdf_pred_x_density]),
        :true_y => density1d_from_arrays(diag[:pdf_true_y_centers], diag[:pdf_true_y_density]),
        :pred_y => density1d_from_arrays(diag[:pdf_true_y_centers], diag[:pdf_pred_y_density]),
        :true_xy => density2d_from_arrays(diag[:pdf_xy_xgrid], diag[:pdf_xy_ygrid], diag[:pdf_true_xy_density]),
        :pred_xy => density2d_from_arrays(diag[:pdf_xy_xgrid], diag[:pdf_xy_ygrid], diag[:pdf_pred_xy_density]),
        :metrics => pdf_metrics,
    )

    observed_pdf_data_full = Dict(
        :true_x => density1d_from_arrays(diag[:observed_pdf_x_centers], diag[:observed_pdf_x_density]),
        :pred_x => density1d_from_arrays(diag[:observed_pdf_x_centers], diag[:observed_pred_x_density]),
        :true_y => density1d_from_arrays(diag[:observed_pdf_y_centers], diag[:observed_pdf_y_density]),
        :pred_y => density1d_from_arrays(diag[:observed_pdf_y_centers], diag[:observed_pred_y_density]),
        :true_xy => density2d_from_arrays(diag[:observed_pdf_xy_xgrid], diag[:observed_pdf_xy_ygrid], diag[:observed_pdf_xy_density]),
        :pred_xy => density2d_from_arrays(diag[:observed_pdf_xy_xgrid], diag[:observed_pdf_xy_ygrid], diag[:observed_pred_xy_density]),
        :metrics => observed_pdf_metrics,
    )

    corr_data_full = Dict(
        :true => CorrelationSummary(
            corr_lags,
            Float64.(vec(diag[:corr_true_acf_x])),
            Float64.(vec(diag[:corr_true_acf_y])),
            Float64.(vec(diag[:corr_true_cross_xy])),
            Float64.(vec(diag[:corr_true_cross_yx])),
            corr_metrics[:tdec_true],
        ),
        :pred => CorrelationSummary(
            corr_lags,
            Float64.(vec(diag[:corr_pred_acf_x])),
            Float64.(vec(diag[:corr_pred_acf_y])),
            Float64.(vec(diag[:corr_pred_cross_xy])),
            Float64.(vec(diag[:corr_pred_cross_yx])),
            corr_metrics[:tdec_pred],
        ),
        :metrics => corr_metrics,
    )

    corr_data_phi = Dict(
        :pred => CorrelationSummary(
            corr_lags,
            Float64.(vec(diag[:corr_pred_phi_acf_x])),
            Float64.(vec(diag[:corr_pred_phi_acf_y])),
            Float64.(vec(diag[:corr_pred_phi_cross_xy])),
            Float64.(vec(diag[:corr_pred_phi_cross_yx])),
            corr_metrics_phi[:tdec_pred],
        ),
        :metrics => corr_metrics_phi,
    )

    observed_corr_data_full = Dict(
        :true => CorrelationSummary(
            observed_corr_lags,
            Float64.(vec(diag[:observed_corr_true_acf_x])),
            Float64.(vec(diag[:observed_corr_true_acf_y])),
            Float64.(vec(diag[:observed_corr_true_cross_xy])),
            Float64.(vec(diag[:observed_corr_true_cross_yx])),
            observed_corr_metrics[:tdec_true],
        ),
        :pred => CorrelationSummary(
            observed_corr_lags,
            Float64.(vec(diag[:observed_corr_pred_acf_x])),
            Float64.(vec(diag[:observed_corr_pred_acf_y])),
            Float64.(vec(diag[:observed_corr_pred_cross_xy])),
            Float64.(vec(diag[:observed_corr_pred_cross_yx])),
            observed_corr_metrics[:tdec_pred],
        ),
        :metrics => observed_corr_metrics,
    )

    observed_corr_data_phi = Dict(
        :pred => CorrelationSummary(
            observed_corr_lags,
            Float64.(vec(diag[:observed_corr_pred_phi_acf_x])),
            Float64.(vec(diag[:observed_corr_pred_phi_acf_y])),
            Float64.(vec(diag[:observed_corr_pred_phi_cross_xy])),
            Float64.(vec(diag[:observed_corr_pred_phi_cross_yx])),
            observed_corr_metrics_phi[:tdec_pred],
        ),
        :metrics => observed_corr_metrics_phi,
    )

    cphi_data_full = Dict(
        :lag_times => Float64.(vec(diag[:cphi_lag_times])),
        :true => Float64.(diag[:cphi_true]),
        :pred => Float64.(diag[:cphi_pred]),
        :mean_rmse => Float64(diag[:cphi_mean_rmse]),
        :channel_rmse => Float64.(vec(diag[:cphi_channel_rmse])),
    )

    cphi_data_phi = Dict(
        :lag_times => Float64.(vec(diag[:cphi_lag_times])),
        :pred => Float64.(diag[:cphi_pred_phi]),
        :mean_rmse => Float64(diag[:cphi_mean_rmse_phi]),
        :channel_rmse => Float64.(vec(diag[:cphi_channel_rmse_phi])),
    )

    observed_cphi_data_full = Dict(
        :lag_times => Float64.(vec(diag[:cphi_lag_times])),
        :true => Float64.(diag[:observed_cphi_true]),
        :pred => Float64.(diag[:observed_cphi_pred]),
        :mean_rmse => Float64(diag[:observed_cphi_mean_rmse]),
        :channel_rmse => Float64.(vec(diag[:observed_cphi_channel_rmse])),
    )

    observed_cphi_data_phi = Dict(
        :lag_times => Float64.(vec(diag[:cphi_lag_times])),
        :pred => Float64.(diag[:observed_cphi_pred_phi]),
        :mean_rmse => Float64(diag[:observed_cphi_mean_rmse_phi]),
        :channel_rmse => Float64.(vec(diag[:observed_cphi_channel_rmse_phi])),
    )

    aux_data_full = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in pairs(diag[:auxiliary]))
    aux_data_phi = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in pairs(diag[:auxiliary_phi]))

    return pdf_data_full, corr_data_full, corr_data_phi,
        observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi,
        cphi_data_full, cphi_data_phi, observed_cphi_data_full, observed_cphi_data_phi,
        aux_data_full, aux_data_phi
end

function regenerate_reference_latex_tag(reference_label::String)
    lowered = lowercase(reference_label)
    if occursin("observed", lowered)
        return "\\mathrm{obs}"
    elseif occursin("true", lowered)
        return "\\mathrm{ref}"
    end
    return "\\mathrm{ref}"
end

function create_regenerated_reference_stats_figure(pdf_data, corr_data_full, corr_data_phi,
        aux_data_full::Dict{Symbol, Any}, aux_data_phi::Dict{Symbol, Any},
        output_path::AbstractString, width::Int, height::Int;
        reference_label::String, reference_title::String, extra_summary_lines::Vector{String}=String[],
        show_global_title::Bool=true,
        show_metrics::Bool=true,
        show_summary::Bool=true,
        use_latex::Bool=false)
    if show_summary
        return create_reference_stats_figure(pdf_data, corr_data_full, corr_data_phi, aux_data_full, aux_data_phi,
            output_path, width, height;
            reference_label=reference_label,
            reference_title=reference_title,
            extra_summary_lines=extra_summary_lines,
            show_global_title=show_global_title,
            show_metrics=show_metrics,
            show_summary=show_summary,
            use_latex=use_latex)
    end

    pdf_true_x = pdf_data[:true_x]
    pdf_pred_x = pdf_data[:pred_x]
    pdf_true_y = pdf_data[:true_y]
    pdf_pred_y = pdf_data[:pred_y]
    pdf_true_xy = pdf_data[:true_xy]
    pdf_pred_xy = pdf_data[:pred_xy]
    pdf_metrics = pdf_data[:metrics]
    corr_true = corr_data_full[:true]
    corr_pred_full = corr_data_full[:pred]
    corr_pred_phi = corr_data_phi[:pred]
    corr_metrics_full = corr_data_full[:metrics]
    corr_metrics_phi = corr_data_phi[:metrics]

    fw, fh = publication_panel_figure_size(4, 2;
        base_w=max(width, 4200),
        base_h=max(height, 4600),
        panel_w=1550,
        panel_h=900,
        min_w=4000,
        min_h=4600,
        max_w=5600,
        max_h=7600)

    with_scaled_figure_style(fw, fh) do _
        fig = Figure(; size=(fw, fh))
        show_global_title && figure_title!(fig, reference_title * "  vs learned dynamics")

        ref_tag = use_latex ? regenerate_reference_latex_tag(reference_label) : nothing
        x_label = use_latex ? latexstring("x") : "x"
        y_label = use_latex ? latexstring("y") : "y"
        lag_label = use_latex ? latexstring("\\tau") : "lag"
        density_label = use_latex ? latexstring("p") : "density"
        acf_label = use_latex ? latexstring("\\mathrm{ACF}") : "ACF"
        corr_label = use_latex ? latexstring("C") : "correlation"

        px_title = show_metrics ? @sprintf("Marginal p(x)    RMSE %.3e  |  KL %.3e",
            pdf_metrics[:rmse_x], pdf_metrics[:kl_x]) : (use_latex ? latexstring("p(x)") : "Marginal p(x)")
        py_title = show_metrics ? @sprintf("Marginal p(y)    RMSE %.3e  |  KL %.3e",
            pdf_metrics[:rmse_y], pdf_metrics[:kl_y]) : (use_latex ? latexstring("p(y)") : "Marginal p(y)")
        ref_pdf_label = use_latex ? latexstring("p_{" * ref_tag * "}") : reference_label
        nn_pdf_label = use_latex ? latexstring("p_{\\mathrm{NN}}") : "learned"
        ref_label = use_latex ? latexstring("\\mathrm{ref}") : reference_label
        full_label = use_latex ? latexstring("\\mathrm{full}") : "learned full"
        phi_label = use_latex ? latexstring("\\Phi") : "Φ-only"

        ax_px = Axis(fig[1, 1]; xlabel=x_label, ylabel=density_label, title=px_title)
        lines!(ax_px, pdf_true_x.centers, pdf_true_x.density; color=STYLE_REFERENCE, label=ref_pdf_label)
        lines!(ax_px, pdf_pred_x.centers, pdf_pred_x.density; color=STYLE_PRIMARY, linestyle=:dash, label=nn_pdf_label)
        axislegend(ax_px; position=:rt)

        ax_py = Axis(fig[1, 2]; xlabel=y_label, ylabel=density_label, title=py_title)
        lines!(ax_py, pdf_true_y.centers, pdf_true_y.density; color=STYLE_REFERENCE, label=ref_pdf_label)
        lines!(ax_py, pdf_pred_y.centers, pdf_pred_y.density; color=STYLE_PRIMARY, linestyle=:dash, label=nn_pdf_label)
        axislegend(ax_py; position=:rt)

        max_xy = maximum(vcat(vec(pdf_true_xy.density), vec(pdf_pred_xy.density)))
        density_heatmap_panel!(fig[2, 1], pdf_true_xy,
            use_latex ? latexstring("p_{" * ref_tag * "}(x,y)") : reference_title * "  p(x,y)";
            clims=(0.0, max_xy), xlabel=x_label, ylabel=y_label)
        density_heatmap_panel!(fig[2, 2], pdf_pred_xy,
            use_latex ? latexstring("p_{\\mathrm{NN}}(x,y)") : "Learned  p(x,y)";
            clims=(0.0, max_xy), xlabel=x_label, ylabel=y_label)

        acf_x_title = show_metrics ? @sprintf("ACF  x     full %.3e  |  Φ %.3e",
            corr_metrics_full[:rmse_acf_x], corr_metrics_phi[:rmse_acf_x]) :
            (use_latex ? latexstring("\\mathrm{ACF}_x(\\tau)") : "ACF x")
        acf_y_title = show_metrics ? @sprintf("ACF  y     full %.3e  |  Φ %.3e",
            corr_metrics_full[:rmse_acf_y], corr_metrics_phi[:rmse_acf_y]) :
            (use_latex ? latexstring("\\mathrm{ACF}_y(\\tau)") : "ACF y")
        cross_xy_title = show_metrics ? @sprintf("Cross  xy     full %.3e  |  Φ %.3e",
            corr_metrics_full[:rmse_cross_xy], corr_metrics_phi[:rmse_cross_xy]) :
            (use_latex ? latexstring("C_{xy}(\\tau)") : "Cross xy")
        cross_yx_title = show_metrics ? @sprintf("Cross  yx     full %.3e  |  Φ %.3e",
            corr_metrics_full[:rmse_cross_yx], corr_metrics_phi[:rmse_cross_yx]) :
            (use_latex ? latexstring("C_{yx}(\\tau)") : "Cross yx")

        ax_acf_x = Axis(fig[3, 1]; xlabel=lag_label, ylabel=acf_label, title=acf_x_title)
        hlines!(ax_acf_x, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
        lines!(ax_acf_x, corr_true.lags, corr_true.acf_x; color=STYLE_REFERENCE, label=ref_label)
        lines!(ax_acf_x, corr_pred_full.lags, corr_pred_full.acf_x; color=STYLE_PRIMARY, linestyle=:dash, label=full_label)
        lines!(ax_acf_x, corr_pred_phi.lags, corr_pred_phi.acf_x; color=STYLE_SECONDARY, linestyle=:dot, label=phi_label)
        axislegend(ax_acf_x; position=:rt)

        ax_acf_y = Axis(fig[3, 2]; xlabel=lag_label, ylabel=acf_label, title=acf_y_title)
        hlines!(ax_acf_y, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
        lines!(ax_acf_y, corr_true.lags, corr_true.acf_y; color=STYLE_REFERENCE, label=ref_label)
        lines!(ax_acf_y, corr_pred_full.lags, corr_pred_full.acf_y; color=STYLE_PRIMARY, linestyle=:dash, label=full_label)
        lines!(ax_acf_y, corr_pred_phi.lags, corr_pred_phi.acf_y; color=STYLE_SECONDARY, linestyle=:dot, label=phi_label)

        ax_cross_xy = Axis(fig[4, 1]; xlabel=lag_label, ylabel=corr_label, title=cross_xy_title)
        hlines!(ax_cross_xy, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
        lines!(ax_cross_xy, corr_true.lags, corr_true.cross_xy; color=STYLE_REFERENCE, label=ref_label)
        lines!(ax_cross_xy, corr_pred_full.lags, corr_pred_full.cross_xy; color=STYLE_PRIMARY, linestyle=:dash, label=full_label)
        lines!(ax_cross_xy, corr_pred_phi.lags, corr_pred_phi.cross_xy; color=STYLE_SECONDARY, linestyle=:dot, label=phi_label)
        axislegend(ax_cross_xy; position=:rt)

        ax_cross_yx = Axis(fig[4, 2]; xlabel=lag_label, ylabel=corr_label, title=cross_yx_title)
        hlines!(ax_cross_yx, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
        lines!(ax_cross_yx, corr_true.lags, corr_true.cross_yx; color=STYLE_REFERENCE, label=ref_label)
        lines!(ax_cross_yx, corr_pred_full.lags, corr_pred_full.cross_yx; color=STYLE_PRIMARY, linestyle=:dash, label=full_label)
        lines!(ax_cross_yx, corr_pred_phi.lags, corr_pred_phi.cross_yx; color=STYLE_SECONDARY, linestyle=:dot, label=phi_label)

        apply_publication_grid!(fig.layout, 4, 2;
            row_weights=[0.9, 1.05, 1.0, 1.0],
            col_weights=[1.0, 1.0],
            row_gap=30, col_gap=28)

        save_figure(output_path, fig)
    end
    return nothing
end

function regenerate_fit_figures(params::RegenerateRunFiguresParams)
    fit_stage_path = joinpath(params.run_dir, "fit_stage.toml")
    isfile(fit_stage_path) || error("Missing fit stage config at $(fit_stage_path).")

    fit_params = load_params(fit_stage_path)
    width = params.fit_width === nothing ? fit_params.figure_width : params.fit_width
    height = params.fit_height === nothing ? fit_params.figure_height : params.fit_height
    artifact_data = BSON.load(fit_params.output_artifact_bson)
    model_data = BSON.load(fit_params.output_mobility_bson)

    labels = String.(artifact_data[:observable_labels])
    a_data = Float64.(artifact_data[:a_data])
    a_true = Float64.(artifact_data[:a_true])
    a_nn = Float64.(artifact_data[:a_nn])
    cdot_data = Float64.(artifact_data[:cdot_data])
    cdot_true = Float64.(artifact_data[:cdot_true])
    cdot_nn = Float64.(artifact_data[:cdot_nn])
    train_pairs = Tuple{Int, Int}[(Int(pair[1]), Int(pair[2])) for pair in artifact_data[:training_pairs]]
    channel_labels = String.(model_data[:training_channel_labels])
    history = history_from_saved_dict(model_data[:history])

    _, a_rmse_data_nn, a_rmse_true_nn = compute_rmse_tables(a_data, a_true, a_nn)
    cdot_rmse_data_true, _, cdot_rmse_true_nn = compute_rmse_tables(cdot_data, cdot_true, cdot_nn)
    rmse_phys, rmse_norm, target_scale = parse_training_channel_metrics(fit_params.output_metrics_txt, channel_labels)
    summary_lines = params.show_summary_panels ? fit_summary_lines(fit_params.output_metrics_txt, fit_params, artifact_data, model_data) : String[]

    out_a = output_path_in_dir(params.output_dir, fit_params.output_a_png, params.overwrite)
    out_cdot = output_path_in_dir(params.output_dir, fit_params.output_cphi_png, params.overwrite)
    out_training = output_path_in_dir(params.output_dir, fit_params.output_training_png, params.overwrite)
    out_mobility = output_path_in_dir(params.output_dir, fit_params.output_mobility_png, params.overwrite)

    create_a_figure(Float64.(artifact_data[:lag_times]), labels, a_data, a_true, a_nn,
        a_rmse_data_nn, a_rmse_true_nn, train_pairs, width, height, out_a;
        show_global_title=params.show_global_titles,
        show_metrics=params.show_metric_titles,
        use_latex=params.use_latex)
    create_cdot_figure(Float64.(artifact_data[:lag_times]), labels, cdot_data, cdot_true, cdot_nn,
        cdot_rmse_data_true, cdot_rmse_true_nn, train_pairs, width, height, out_cdot;
        show_global_title=params.show_global_titles,
        show_metrics=params.show_metric_titles,
        use_latex=params.use_latex)
    create_training_diagnostics_figure(history, channel_labels, rmse_phys, rmse_norm, target_scale,
        summary_lines, out_training, width, height;
        show_global_title=params.show_global_titles,
        show_summary=params.show_summary_panels,
        use_latex=params.use_latex)

    device = detect_device()
    plain_data = BSON.load(fit_params.plain_score_bson)
    plain_model = to_device(plain_data[:host_model], device)
    sampler = build_pair_sampler(fit_params.input_hdf5, fit_params.burnin_fraction, fit_params.tau_min, fit_params.lag_stride)
    meta = load_affine_model_metadata(fit_params.input_hdf5)
    field = estimate_r_field(plain_model, sampler.states, sampler.start_idx, meta,
        fit_params.mobility_grid_nx, fit_params.mobility_grid_ny, fit_params.mobility_ridge, fit_params.grid_pad_fraction,
        fit_params.score_batch_size, device)
    mobility_runtime, _ = load_mobility_runtime(fit_params.output_mobility_bson, fit_params.score_batch_size, device, fit_params.mobility_nn_psd_jitter)
    ref_mats = mobility_reference_matrices(field)
    nn_mats = mobility_predicted_matrices(mobility_runtime.host_model, field, mobility_runtime.μ, mobility_runtime.σ,
        fit_params.mobility_nn_psd_jitter, fit_params.score_batch_size, device)
    support_mask = observed_support_mask(field, sampler.states, sampler.start_idx)
    create_mobility_heatmap_figure(field, ref_mats, nn_mats, support_mask,
        Float64.(artifact_data[:field_component_rmse]), out_mobility;
        show_global_title=params.show_global_titles,
        show_metrics=params.show_metric_titles,
        use_latex=params.use_latex)

    return [out_a, out_cdot, out_training, out_mobility]
end

function regenerate_forward_validation_figures(params::RegenerateRunFiguresParams)
    forward_stage_path = joinpath(params.run_dir, "forward_validation_stage.toml")
    if !isfile(forward_stage_path)
        return String[]
    end

    forward_params = load_forward_validation_params(forward_stage_path)
    width = params.forward_width === nothing ? forward_params.figure_width : params.forward_width
    height = params.forward_height === nothing ? forward_params.figure_height : params.forward_height
    diagnostics = BSON.load(forward_params.diagnostics_bson)
    mobility_artifact = BSON.load(forward_params.mobility_artifact_bson)

    pdf_data_full, corr_data_full, corr_data_phi,
    observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi,
    cphi_data_full, cphi_data_phi, observed_cphi_data_full, observed_cphi_data_phi,
    aux_data_full, aux_data_phi = rebuild_forward_plot_inputs(diagnostics)

    device = detect_device()
    mobility_runtime, _ = load_mobility_runtime(forward_params.mobility_model_bson, forward_params.eval_batch_size, device, forward_params.mobility_psd_jitter)
    cphi_pairs = cphi_training_pairs(mobility_runtime, mobility_artifact)
    cphi_labels = cphi_display_labels(cphi_pairs)

    outputs = String[]
    out_stats = output_path_in_dir(params.output_dir, forward_params.figure_stats_png, params.overwrite)
    create_regenerated_reference_stats_figure(pdf_data_full, corr_data_full, corr_data_phi, aux_data_full, aux_data_phi,
        out_stats, width, height;
        reference_label="true rollout", reference_title="True Rollout",
        show_global_title=params.show_global_titles,
        show_metrics=params.show_metric_titles,
        show_summary=params.show_summary_panels,
        use_latex=params.use_latex)
    push!(outputs, out_stats)

    if forward_params.figure_observed_png !== nothing
        out_observed = output_path_in_dir(params.output_dir, forward_params.figure_observed_png, params.overwrite)
        observed_extra_lines = params.show_summary_panels ? [
            @sprintf("obs Cphi RMSE full/Phi = %.3e / %.3e", observed_cphi_data_full[:mean_rmse], observed_cphi_data_phi[:mean_rmse]),
            @sprintf("obs Cphi improvement over Phi = %.2f%%", rmse_improvement_percent(observed_cphi_data_phi[:mean_rmse], observed_cphi_data_full[:mean_rmse])),
        ] : String[]
        create_regenerated_reference_stats_figure(observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi, aux_data_full, aux_data_phi,
            out_observed, width, height;
            reference_label="observed", reference_title="Observed Data",
            extra_summary_lines=observed_extra_lines,
            show_global_title=params.show_global_titles,
            show_metrics=params.show_metric_titles,
            show_summary=params.show_summary_panels,
            use_latex=params.use_latex)
        push!(outputs, out_observed)
    end

    out_cphi = output_path_in_dir(params.output_dir, forward_params.figure_cphi_png, params.overwrite)
    create_cphi_figure(cphi_data_full, cphi_data_phi, cphi_labels, out_cphi, width, height;
        show_global_title=params.show_global_titles,
        show_metrics=params.show_metric_titles,
        use_latex=params.use_latex)
    push!(outputs, out_cphi)

    return outputs
end

function regenerate_run_figures(param_file::AbstractString)
    params, theme_cfg = load_regenerate_params(param_file)
    isdir(params.run_dir) || error("Run directory $(params.run_dir) does not exist.")
    mkpath(params.output_dir)
    apply_theme_overrides!(theme_cfg)

    @printf("Regenerating figures from %s\n", params.run_dir)
    @printf("Saving regenerated figures into %s\n", params.output_dir)
    generated = String[]
    append!(generated, regenerate_fit_figures(params))
    append!(generated, regenerate_forward_validation_figures(params))
    @printf("Regenerated %d figures.\n", length(generated))
    for path in generated
        @printf("  %s\n", path)
    end
    return generated
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_REGENERATE_PARAM_FILE : abspath(ARGS[1])
    regenerate_run_figures(param_file)
end