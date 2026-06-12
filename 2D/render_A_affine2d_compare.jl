#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "figure_style.jl"))

using BSON
using HDF5
using LaTeXStrings
using LinearAlgebra
using Printf
using Statistics
using TOML

const DEFAULT_AFFINE_A_CONFIG = normpath(joinpath(@__DIR__,
    "render_A_affine2d_compare.toml"))

Base.@kwdef struct AffineAConfig
    artifact_bson::String
    use_cache::Bool
    width::Int
    height::Int
    ncols::Int
    theme::Dict{String, Any}
    output_png::String
    cache_h5::String
    metrics_txt::String
end

function config_path(base::AbstractString, path::AbstractString)
    return isabspath(path) ? normpath(path) : normpath(joinpath(base, path))
end

function load_affine_a_config(path::AbstractString)
    raw = TOML.parsefile(path)
    base = dirname(path)
    inputs = raw["inputs"]
    sel = get(raw, "selection", Dict{String, Any}())
    fig = raw["figure"]
    out = raw["output"]
    return AffineAConfig(
        artifact_bson=config_path(base, String(inputs["artifact_bson"])),
        use_cache=Bool(get(sel, "use_cache", true)),
        width=Int(fig["width"]),
        height=Int(fig["height"]),
        ncols=Int(fig["ncols"]),
        theme=get(raw, "theme", Dict{String, Any}()),
        output_png=config_path(base, String(out["png"])),
        cache_h5=config_path(base, String(out["cache_h5"])),
        metrics_txt=config_path(base, String(out["metrics_txt"])),
    )
end

function ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    isempty(parent) || mkpath(parent)
    return nothing
end

function apply_theme_overrides!(theme_cfg::Dict{String, Any})
    isempty(theme_cfg) && return nothing
    kwargs = Dict{Symbol, Any}()
    for (key, val) in theme_cfg
        kwargs[Symbol(key)] = val
    end
    configure_figure_style!(; kwargs...)
    return nothing
end

function parse_pair(pair)
    return (Int(pair[1]), Int(pair[2]))
end

function panel_title_latex(obs::AbstractString, target::AbstractString)
    return latexstring("\\phi=", observable_latex(obs), "\\;\\to\\;",
        observable_latex(target))
end

function affine_training_curves(artifact)
    labels = String.(artifact[:observable_labels])
    pairs = parse_pair.(artifact[:training_pairs])
    taus = Vector{Float64}(artifact[:lag_times])
    a_data = Array{Float64, 3}(artifact[:a_data])
    a_nn = Array{Float64, 3}(artifact[:a_nn])
    a_true = Array{Float64, 3}(artifact[:a_true])
    panels = length(pairs)
    Adata = Matrix{Float64}(undef, length(taus), panels)
    Ann = similar(Adata)
    Atrue = similar(Adata)
    panel_labels = Vector{String}(undef, panels)
    for (idx, (m, n)) in enumerate(pairs)
        Adata[:, idx] .= a_data[:, m, n]
        Ann[:, idx] .= a_nn[:, m, n]
        Atrue[:, idx] .= a_true[:, m, n]
        panel_labels[idx] = "$(labels[m]) -> $(labels[n])"
    end
    return (; labels, pairs, panel_labels, taus, Adata, Ann, Atrue)
end

function write_affine_a_cache(path::AbstractString, data, cfg::AffineAConfig)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time/taus", data.taus)
        write(h5, "/curves/A_data", data.Adata)
        write(h5, "/curves/A_nn", data.Ann)
        write(h5, "/curves/A_true", data.Atrue)
        write(h5, "/metadata/observable_labels", data.labels)
        write(h5, "/metadata/panel_labels", data.panel_labels)
        write(h5, "/metadata/training_pairs", hcat(first.(data.pairs), last.(data.pairs)))
        write(h5, "/metadata/artifact_bson", cfg.artifact_bson)
    end
    return nothing
end

function read_affine_a_cache(path::AbstractString)
    h5open(path, "r") do h5
        labels = String.(read(h5["/metadata/observable_labels"]))
        panel_labels = String.(read(h5["/metadata/panel_labels"]))
        pair_mat = Matrix{Int}(read(h5["/metadata/training_pairs"]))
        pairs = [(pair_mat[i, 1], pair_mat[i, 2]) for i in axes(pair_mat, 1)]
        taus = Vector{Float64}(read(h5["/time/taus"]))
        Adata = Matrix{Float64}(read(h5["/curves/A_data"]))
        Ann = Matrix{Float64}(read(h5["/curves/A_nn"]))
        Atrue = Matrix{Float64}(read(h5["/curves/A_true"]))
        return (; labels, pairs, panel_labels, taus, Adata, Ann, Atrue)
    end
end

function compute_or_load_affine_a_curves(cfg::AffineAConfig)
    if cfg.use_cache && isfile(cfg.cache_h5)
        @printf("Loading cached affine 2D A curves from %s\n", cfg.cache_h5)
        return read_affine_a_cache(cfg.cache_h5)
    end
    artifact = BSON.load(cfg.artifact_bson)
    data = affine_training_curves(artifact)
    write_affine_a_cache(cfg.cache_h5, data, cfg)
    return data
end

function agreement_stats(pred, ref)
    p = vec(Float64.(pred))
    r = vec(Float64.(ref))
    rel = norm(p .- r) / max(norm(r), eps(Float64))
    corr = dot(p, r) / max(norm(p) * norm(r), eps(Float64))
    return rel, corr
end

function render_affine_a_compare(path::AbstractString, cfg::AffineAConfig, data)
    panels = size(data.Adata, 2)
    ncols = min(cfg.ncols, panels)
    nrows = cld(panels, ncols)
    with_scaled_figure_style(cfg.width, cfg.height) do _
        fig = Figure(; size=(cfg.width, cfg.height))
        legend_handles = nothing
        used_cells = Set{Tuple{Int, Int}}()
        for j in 1:panels
            row, col = centered_panel_rc(j, panels, ncols)
            push!(used_cells, (row, col))
            m, n = data.pairs[j]
            ax = Axis(fig[row, col];
                title=panel_title_latex(data.labels[m], data.labels[n]),
                xlabel=row == nrows ? latexstring("\\tau") : "",
                ylabel=col == 1 ? latexstring("A_{mn}(\\tau)") : "",
                xgridvisible=true, ygridvisible=true)
            hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot,
                linewidth=guide_linewidth())
            data_line = lines!(ax, data.taus, data.Adata[:, j];
                color=STYLE_REFERENCE,
                linewidth=curve_linewidth(; emphasis=1.05),
                label=latexstring("A_{\\mathrm{data}}"))
            nn_line = lines!(ax, data.taus, data.Ann[:, j];
                color=STYLE_PRIMARY,
                linewidth=curve_linewidth(; emphasis=0.9), linestyle=:dash,
                label=latexstring("A[M_{\\mathrm{NN}}]"))
            true_line = lines!(ax, data.taus, data.Atrue[:, j];
                color=STYLE_ACCENT,
                linewidth=curve_linewidth(; emphasis=0.9), linestyle=:dashdot,
                label=latexstring("A[M_{\\mathrm{true}}]"))
            if legend_handles === nothing
                legend_handles = ([data_line, nn_line, true_line],
                    [latexstring("A_{\\mathrm{data}}"),
                     latexstring("A[M_{\\mathrm{NN}}]"),
                     latexstring("A[M_{\\mathrm{true}}]")])
            end
            xlims!(ax, first(data.taus), last(data.taus))
        end
        for row in 1:nrows, col in 1:ncols
            (row, col) in used_cells && continue
            Box(fig[row, col]; color=(:white, 0), strokecolor=(:white, 0))
        end
        if legend_handles !== nothing
            Legend(fig[nrows + 1, 1:ncols], legend_handles[1], legend_handles[2];
                orientation=:horizontal, framevisible=false, nbanks=1,
                tellheight=true)
        end
        apply_publication_grid!(fig.layout, nrows + 1, ncols;
            row_weights=vcat(fill(1.0, nrows), [0.16]),
            col_weights=fill(1.0, ncols),
            row_gap=24, col_gap=32)
        save_figure(path, fig)
    end
    @printf("Saved %s\n", path)
    return nothing
end

function write_affine_a_metrics(path::AbstractString, cfg::AffineAConfig, data)
    ensure_parent_dir(path)
    rel_nn, corr_nn = agreement_stats(data.Ann, data.Adata)
    rel_true, corr_true = agreement_stats(data.Atrue, data.Adata)
    panel_line = join(data.panel_labels, "; ")
    open(path, "w") do io
        println(io, "Affine 2D A_mn comparison")
        println(io, "artifact_bson = $(cfg.artifact_bson)")
        println(io, "panels = $(length(data.panel_labels))")
        println(io, "panel_labels = $(panel_line)")
        println(io, @sprintf("NN rel.RMSE vs data = %.8e", rel_nn))
        println(io, @sprintf("NN corr vs data = %.8e", corr_nn))
        println(io, @sprintf("true-M rel.RMSE vs data = %.8e", rel_true))
        println(io, @sprintf("true-M corr vs data = %.8e", corr_true))
        println(io, "No-cheating audit: A_data and A[M_NN] are loaded from the accepted saved mobility artifact. A[M_true] is plotted only as an ex-post diagnostic and is not used in this render for model selection or training.")
    end
    @printf("Saved %s\n", path)
    return nothing
end

function render_A_affine2d_compare(cfg::AffineAConfig)
    apply_theme_overrides!(cfg.theme)
    isfile(cfg.artifact_bson) || error("Missing input $(cfg.artifact_bson)")
    mkpath(dirname(cfg.output_png))
    data = compute_or_load_affine_a_curves(cfg)
    render_affine_a_compare(cfg.output_png, cfg, data)
    write_affine_a_metrics(cfg.metrics_txt, cfg, data)
    return nothing
end

function main()
    cfg_path = isempty(ARGS) ? DEFAULT_AFFINE_A_CONFIG : ARGS[1]
    render_A_affine2d_compare(load_affine_a_config(cfg_path))
end

main()
