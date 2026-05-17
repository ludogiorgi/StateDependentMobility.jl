#! Shared GLMakie theme + helpers used by sim.jl, score.jl, joint_score.jl,
#! fit_dm.jl, and src/mobility_forward_validation.jl. Provides:
#!   - `ensure_display!()`     : transparently spins up Xvfb on headless boxes
#!                               so GLMakie can create an OpenGL context.
#!   - `STYLE_*` color constants
#!   - `STYLE_DIVERGING` / `STYLE_SEQUENTIAL` colormaps
#!   - `unified_theme()`       : the publication-quality Makie theme used everywhere
#!   - `panel_grid_dims`       : adaptive (rows, cols) for variable-panel layouts
#!   - `text_panel!`           : drops a multi-line annotation block into a Figure cell
#!   - `figure_title!`         : adds a centered, bold figure-level title
#!   - `with_scaled_figure_style`: temporarily rescales typography for large figures
#!   - `observable_latex` / `latex_channel_label`: safe LaTeX label helpers
#!   - `apply_unified_theme!()`: convenience that calls `set_theme!(unified_theme())`

function ensure_display!()
    if haskey(ENV, "DISPLAY") && !isempty(ENV["DISPLAY"])
        return nothing
    end
    xvfb = Sys.which("Xvfb")
    if xvfb === nothing
        @warn "No DISPLAY set and Xvfb is unavailable; GLMakie may fail to load. " *
              "Run via `xvfb-run -a julia ...` to provide a virtual display."
        return nothing
    end
    display_num = get(ENV, "STATEDEP_XVFB_DISPLAY", ":99")
    lock_path = "/tmp/.X" * lstrip(display_num, ':') * "-lock"
    if !isfile(lock_path)
        @info "Starting Xvfb for headless GLMakie rendering" display=display_num
        try
            run(pipeline(`$xvfb $display_num -screen 0 1920x1200x24 -nolisten tcp`,
                stdout=devnull, stderr=devnull); wait=false)
            sleep(1.5)
        catch err
            @warn "Failed to start Xvfb; expect GLMakie load errors." err
        end
    end
    ENV["DISPLAY"] = display_num
    return nothing
end

ensure_display!()

using GLMakie
using LaTeXStrings

# --- Unified publication palette --------------------------------------------
const STYLE_REFERENCE = RGBf(0.102, 0.129, 0.184)   # ink / truth / data
const STYLE_PRIMARY   = RGBf(0.122, 0.306, 0.541)   # learned / NN / model A
const STYLE_SECONDARY = RGBf(0.698, 0.286, 0.227)   # alternative / Φ-only
const STYLE_ACCENT    = RGBf(0.180, 0.545, 0.494)   # tertiary / accent
const STYLE_HIGHLIGHT = RGBf(0.820, 0.557, 0.102)   # warm highlight / reference model
const STYLE_VIOLET    = RGBf(0.416, 0.298, 0.573)
const STYLE_MUTED     = RGBf(0.420, 0.447, 0.502)
const STYLE_SOFT      = RGBf(0.753, 0.784, 0.820)
const STYLE_ZERO      = RGBf(0.55, 0.58, 0.64)
const STYLE_GRID      = RGBf(0.85, 0.86, 0.88)
const STYLE_SPINE     = RGBf(0.30, 0.33, 0.37)

# Palettes
const STYLE_DIVERGING       = Reverse(:RdBu)
const STYLE_SEQUENTIAL      = :viridis
const STYLE_SEQUENTIAL_BLUE = cgrad([RGBf(0.95, 0.96, 0.98),
                                     RGBf(0.122, 0.306, 0.541),
                                     RGBf(0.04, 0.10, 0.22)])
const STYLE_DIVERGING_SOFT  = cgrad([RGBf(0.239, 0.368, 0.576),
                                     RGBf(0.647, 0.757, 0.839),
                                     RGBf(0.965, 0.965, 0.965),
                                     RGBf(0.937, 0.749, 0.663),
                                     RGBf(0.698, 0.286, 0.227)])

# Cycle used implicitly by `lines!` when no explicit color is given.
const STYLE_LINE_CYCLE = [STYLE_PRIMARY, STYLE_SECONDARY, STYLE_ACCENT,
                          STYLE_HIGHLIGHT, STYLE_VIOLET, STYLE_REFERENCE]

Base.@kwdef struct FigureStyleConfig
    px_per_unit::Float64 = 1.5
    fontsize::Float64 = 18.0
    axis_ticklabelsize::Float64 = 16.0
    axis_labelsize::Float64 = 19.0
    axis_titlesize::Float64 = 21.0
    legend_labelsize::Float64 = 16.0
    legend_titlesize::Float64 = 17.0
    colorbar_labelsize::Float64 = 17.0
    colorbar_ticklabelsize::Float64 = 14.0
    label_fontsize::Float64 = 19.0
    text_fontsize::Float64 = 16.0
    summary_fontsize::Float64 = 16.0
    summary_title_fontsize::Float64 = 18.0
    figure_title_fontsize::Float64 = 26.0
    figure_subtitle_fontsize::Float64 = 17.0
end

const CURRENT_FIGURE_STYLE = Ref(FigureStyleConfig())

current_figure_style() = CURRENT_FIGURE_STYLE[]

function curve_linewidth(; emphasis::Real=1.0)
    layout_scale = max(current_figure_style().fontsize / 18.0, 1.0)
    return round(3.8 * sqrt(layout_scale) * Float64(emphasis); digits=2)
end

function guide_linewidth()
    return round(max(1.8, 0.45 * curve_linewidth()); digits=2)
end

function scaled_figure_style(cfg::FigureStyleConfig, scale::Real)
    factor = Float64(scale)
    return FigureStyleConfig(
        px_per_unit = cfg.px_per_unit,
        fontsize = cfg.fontsize * factor,
        axis_ticklabelsize = cfg.axis_ticklabelsize * factor,
        axis_labelsize = cfg.axis_labelsize * factor,
        axis_titlesize = cfg.axis_titlesize * factor,
        legend_labelsize = cfg.legend_labelsize * factor,
        legend_titlesize = cfg.legend_titlesize * factor,
        colorbar_labelsize = cfg.colorbar_labelsize * factor,
        colorbar_ticklabelsize = cfg.colorbar_ticklabelsize * factor,
        label_fontsize = cfg.label_fontsize * factor,
        text_fontsize = cfg.text_fontsize * factor,
        summary_fontsize = cfg.summary_fontsize * factor,
        summary_title_fontsize = cfg.summary_title_fontsize * factor,
        figure_title_fontsize = cfg.figure_title_fontsize * factor,
        figure_subtitle_fontsize = cfg.figure_subtitle_fontsize * factor,
    )
end

function figure_font_scale(width::Real, height::Real;
        reference_width::Real=2200,
        min_scale::Real=1.0,
        max_scale::Real=2.6)
    width_ratio = max(Float64(width) / Float64(reference_width), eps(Float64))
    return clamp(width_ratio, min_scale, max_scale)
end

function with_scaled_figure_style(f::Function, width::Real, height::Real;
        reference_width::Real=2200,
        min_scale::Real=1.0,
        max_scale::Real=2.6,
        scale_override=nothing)
    base = current_figure_style()
    scale = scale_override === nothing ? figure_font_scale(width, height;
        reference_width=reference_width,
        min_scale=min_scale,
        max_scale=max_scale) : Float64(scale_override)
    CURRENT_FIGURE_STYLE[] = scaled_figure_style(base, scale)
    apply_unified_theme!()
    try
        return f(scale)
    finally
        CURRENT_FIGURE_STYLE[] = base
        apply_unified_theme!()
    end
end

function observable_latex(label::AbstractString)
    text = strip(String(label))
    isempty(text) && return text

    pieces = String[]
    idx = firstindex(text)
    while idx <= lastindex(text)
        ch = text[idx]
        if ch == 'x' || ch == 'y'
            symbol = string(ch)
            idx = nextind(text, idx)
            if idx <= lastindex(text) && text[idx] == '^'
                idx = nextind(text, idx)
                exp_start = idx
                while idx <= lastindex(text) && isdigit(text[idx])
                    idx = nextind(text, idx)
                end
                exponent = exp_start < idx ? text[exp_start:prevind(text, idx)] : "1"
                push!(pieces, symbol * "^{" * exponent * "}")
            else
                push!(pieces, symbol)
            end
        else
            idx = nextind(text, idx)
        end
    end

    return isempty(pieces) ? replace(text, "_" => "\\_") : join(pieces, " ")
end

function latex_channel_label(head::AbstractString, first::AbstractString, second::AbstractString;
        argument::Union{Nothing, AbstractString}=nothing)
    expr = head * "_{" * observable_latex(first) * ", " * observable_latex(second) * "}"
    argument === nothing || (expr *= "(" * String(argument) * ")")
    return latexstring(expr)
end

function configure_figure_style!(;
        px_per_unit=nothing,
        fontsize=nothing,
        axis_ticklabelsize=nothing,
        axis_labelsize=nothing,
        axis_titlesize=nothing,
        legend_labelsize=nothing,
        legend_titlesize=nothing,
        colorbar_labelsize=nothing,
        colorbar_ticklabelsize=nothing,
        label_fontsize=nothing,
        text_fontsize=nothing,
        summary_fontsize=nothing,
        summary_title_fontsize=nothing,
        figure_title_fontsize=nothing,
        figure_subtitle_fontsize=nothing)
    current = current_figure_style()
    CURRENT_FIGURE_STYLE[] = FigureStyleConfig(
        px_per_unit = px_per_unit === nothing ? current.px_per_unit : Float64(px_per_unit),
        fontsize = fontsize === nothing ? current.fontsize : Float64(fontsize),
        axis_ticklabelsize = axis_ticklabelsize === nothing ? current.axis_ticklabelsize : Float64(axis_ticklabelsize),
        axis_labelsize = axis_labelsize === nothing ? current.axis_labelsize : Float64(axis_labelsize),
        axis_titlesize = axis_titlesize === nothing ? current.axis_titlesize : Float64(axis_titlesize),
        legend_labelsize = legend_labelsize === nothing ? current.legend_labelsize : Float64(legend_labelsize),
        legend_titlesize = legend_titlesize === nothing ? current.legend_titlesize : Float64(legend_titlesize),
        colorbar_labelsize = colorbar_labelsize === nothing ? current.colorbar_labelsize : Float64(colorbar_labelsize),
        colorbar_ticklabelsize = colorbar_ticklabelsize === nothing ? current.colorbar_ticklabelsize : Float64(colorbar_ticklabelsize),
        label_fontsize = label_fontsize === nothing ? current.label_fontsize : Float64(label_fontsize),
        text_fontsize = text_fontsize === nothing ? current.text_fontsize : Float64(text_fontsize),
        summary_fontsize = summary_fontsize === nothing ? current.summary_fontsize : Float64(summary_fontsize),
        summary_title_fontsize = summary_title_fontsize === nothing ? current.summary_title_fontsize : Float64(summary_title_fontsize),
        figure_title_fontsize = figure_title_fontsize === nothing ? current.figure_title_fontsize : Float64(figure_title_fontsize),
        figure_subtitle_fontsize = figure_subtitle_fontsize === nothing ? current.figure_subtitle_fontsize : Float64(figure_subtitle_fontsize),
    )
    apply_unified_theme!()
    return CURRENT_FIGURE_STYLE[]
end

# --- Theme ------------------------------------------------------------------
function unified_theme()
    cfg = current_figure_style()
    layout_scale = max(cfg.fontsize / 18.0, 1.0)
    line_width = curve_linewidth()
    return Theme(
        fontsize        = cfg.fontsize,
        backgroundcolor = :white,
        figure_padding  = (
            round(Int, 28 * layout_scale),
            round(Int, 28 * layout_scale),
            round(Int, 22 * layout_scale),
            round(Int, 22 * layout_scale),
        ),
        Axis = (
            backgroundcolor   = :white,
            xgridcolor        = (STYLE_GRID, 0.85),
            ygridcolor        = (STYLE_GRID, 0.85),
            xgridwidth        = 0.9,
            ygridwidth        = 0.9,
            xminorgridvisible = false,
            yminorgridvisible = false,
            xticksize         = 6,
            yticksize         = 6,
            xtickwidth        = 1.0,
            ytickwidth        = 1.0,
            xtickalign        = 0.0,
            ytickalign        = 0.0,
            spinewidth        = 1.2,
            xticklabelsize    = cfg.axis_ticklabelsize,
            yticklabelsize    = cfg.axis_ticklabelsize,
            xlabelsize        = cfg.axis_labelsize,
            ylabelsize        = cfg.axis_labelsize,
            titlesize         = cfg.axis_titlesize,
            titlegap          = round(Int, 10 * layout_scale),
            titlefont         = :bold,
            xlabelpadding     = round(Int, 6 * layout_scale),
            ylabelpadding     = round(Int, 6 * layout_scale),
            leftspinecolor    = STYLE_SPINE,
            rightspinecolor   = STYLE_SPINE,
            topspinecolor     = STYLE_SPINE,
            bottomspinecolor  = STYLE_SPINE,
            xtickcolor        = STYLE_SPINE,
            ytickcolor        = STYLE_SPINE,
            xticklabelcolor   = STYLE_REFERENCE,
            yticklabelcolor   = STYLE_REFERENCE,
            xlabelcolor       = STYLE_REFERENCE,
            ylabelcolor       = STYLE_REFERENCE,
            titlecolor        = STYLE_REFERENCE,
            palette           = (color = STYLE_LINE_CYCLE,),
        ),
        Legend = (
            framecolor       = (STYLE_MUTED, 0.45),
            framewidth       = 1.0,
            backgroundcolor  = (:white, 0.88),
            labelsize        = cfg.legend_labelsize,
            titlesize        = cfg.legend_titlesize,
            patchsize        = (round(Int, 30 * layout_scale), round(Int, 16 * layout_scale)),
            rowgap           = round(Int, 4 * layout_scale),
            padding          = (
                round(Int, 12 * layout_scale),
                round(Int, 12 * layout_scale),
                round(Int, 8 * layout_scale),
                round(Int, 8 * layout_scale),
            ),
            labelcolor       = STYLE_REFERENCE,
            titlecolor       = STYLE_REFERENCE,
        ),
        Lines = (
            linewidth = line_width,
        ),
        Scatter = (
            markersize  = 11,
            strokewidth = 0.6,
            strokecolor = (:black, 0.4),
        ),
        Heatmap = (
            colormap = STYLE_SEQUENTIAL,
        ),
        Contour = (
            linewidth = round(0.75 * line_width; digits=2),
        ),
        BarPlot = (
            color       = STYLE_PRIMARY,
            strokecolor = (STYLE_REFERENCE, 0.0),
            strokewidth = 0.0,
            gap         = 0.18,
        ),
        Colorbar = (
            labelsize     = cfg.colorbar_labelsize,
            ticklabelsize = cfg.colorbar_ticklabelsize,
            size          = round(Int, 18 * layout_scale),
            spinewidth    = 1.0,
            ticksize      = round(Int, 5 * layout_scale),
            tickalign     = 0.0,
            ticklabelpad  = round(Int, 4 * layout_scale),
            labelcolor    = STYLE_REFERENCE,
            ticklabelcolor= STYLE_REFERENCE,
        ),
        Label = (
            fontsize  = cfg.label_fontsize,
            color     = STYLE_REFERENCE,
        ),
        Text = (
            color    = STYLE_REFERENCE,
            fontsize = cfg.text_fontsize,
        ),
    )
end

apply_unified_theme!() = set_theme!(unified_theme())

# --- Layout helpers ---------------------------------------------------------
"""
    panel_grid_dims(npanels; max_cols=4) -> (nrows, ncols)

Pick a near-square layout so the figure stays balanced for any panel count.
"""
function panel_grid_dims(npanels::Int; max_cols::Int=4)
    npanels >= 1 || error("panel_grid_dims: need at least one panel.")
    max_cols >= 1 || error("panel_grid_dims: max_cols must be ≥ 1.")
    ncols = min(max_cols, ceil(Int, sqrt(npanels)))
    nrows = cld(npanels, ncols)
    return nrows, ncols
end

panel_rc(idx::Int, ncols::Int) = ((idx - 1) ÷ ncols + 1, (idx - 1) % ncols + 1)

function centered_panel_rc(idx::Int, npanels::Int, ncols::Int)
    nrows = cld(npanels, ncols)
    r, c = panel_rc(idx, ncols)
    if r < nrows
        return r, c
    end

    last_row_count = npanels - ncols * (nrows - 1)
    if last_row_count == ncols
        return r, c
    end

    last_row_idx = idx - ncols * (nrows - 1)
    offset = fld(ncols - last_row_count, 2)
    return r, offset + last_row_idx
end

function normalized_grid_weights(weights::AbstractVector{<:Real}, label::AbstractString)
    values = Float64.(collect(weights))
    isempty(values) && error(label * " must contain at least one entry.")
    all(value -> value > 0.0, values) || error(label * " must be strictly positive.")
    values ./= sum(values)
    return values
end

"""
    apply_publication_grid!(layout, nrows, ncols; row_weights, col_weights, row_gap, col_gap)

Force a Makie grid layout to occupy its parent region with explicit relative row
and column weights. This avoids auto-sized cells collapsing to inconsistent
panel widths or leaving large blank regions inside oversized figures.
"""
function apply_publication_grid!(layout, nrows::Int, ncols::Int;
        row_weights::AbstractVector{<:Real}=fill(1.0, nrows),
        col_weights::AbstractVector{<:Real}=fill(1.0, ncols),
        row_gap::Real=28, col_gap::Real=30)
    length(row_weights) == nrows || error("row_weights length must match nrows.")
    length(col_weights) == ncols || error("col_weights length must match ncols.")

    # Force the grid to allocate the requested extent before we assign explicit
    # row/column sizes. Makie throws if rowsize!/colsize! target dimensions that
    # have not been materialized yet.
    layout[1, 1]
    layout[nrows, ncols]

    row_fracs = normalized_grid_weights(row_weights, "row_weights")
    col_fracs = normalized_grid_weights(col_weights, "col_weights")

    for r in 1:nrows
        rowsize!(layout, r, Relative(row_fracs[r]))
    end
    for c in 1:ncols
        colsize!(layout, c, Relative(col_fracs[c]))
    end
    gap_scale = max(current_figure_style().fontsize / 18.0, 1.0)
    rowgap!(layout, row_gap * gap_scale)
    colgap!(layout, col_gap * gap_scale)
    return nothing
end

"""
    publication_panel_figure_size(nrows, ncols; base_w, base_h, panel_w, panel_h, ...)

Choose a readable overall figure size for panel-heavy figures. `base_w` and
`base_h` are the caller's requested minimum overall size, while `panel_w` and
`panel_h` encode a comfortable target panel size. The result is clamped to keep
very large panel grids from becoming unwieldy on screen.
"""
function publication_panel_figure_size(nrows::Int, ncols::Int;
        base_w::Int=2200, base_h::Int=1800,
        panel_w::Int=980, panel_h::Int=620,
        min_w::Int=1700, min_h::Int=1100,
    max_w::Int=6000, max_h::Int=5200)
    width = clamp(max(base_w, panel_w * ncols), min_w, max(max_w, min_w))
    height = clamp(max(base_h, panel_h * nrows), min_h, max(max_h, min_h))
    return width, height
end

"""
    text_panel!(parent, lines; title="", fontsize=15)

Add a left-aligned multi-line annotation block to `parent` (a Figure cell or
GridLayout slot). Used for compact summary cards.
"""
function text_panel!(parent, lines::AbstractVector{<:AbstractString};
        title::AbstractString="", fontsize::Union{Nothing, Real}=nothing,
        color=STYLE_REFERENCE, titlefontsize::Union{Nothing, Real}=nothing)
    cfg = current_figure_style()
    body_fontsize = fontsize === nothing ? cfg.summary_fontsize : fontsize
    heading_fontsize = titlefontsize === nothing ? cfg.summary_title_fontsize : titlefontsize
    body = join(lines, "\n")
    gl = GridLayout(parent)
    if !isempty(title)
        Label(gl[1, 1], title; halign=:left, valign=:top,
              fontsize=heading_fontsize, color=color, font=:bold,
              padding=(0, 0, 6, 0))
        Label(gl[2, 1], body; halign=:left, valign=:top, justification=:left,
              fontsize=body_fontsize, color=color)
        rowgap!(gl, 4)
    else
        Label(gl[1, 1], body; halign=:left, valign=:top, justification=:left,
              fontsize=body_fontsize, color=color)
    end
    return gl
end

function split_text_lines(lines::AbstractVector{<:AbstractString}, ncols::Int)
    ncols >= 1 || error("ncols must be at least 1.")
    isempty(lines) && return [String[] for _ in 1:ncols]
    chunk_size = cld(length(lines), ncols)
    chunks = Vector{Vector{String}}(undef, ncols)
    for col in 1:ncols
        start_idx = (col - 1) * chunk_size + 1
        end_idx = min(col * chunk_size, length(lines))
        chunks[col] = start_idx <= length(lines) ? String.(lines[start_idx:end_idx]) : String[]
    end
    return chunks
end

"""
    text_columns_panel!(parent, columns; title="", fontsize=15)

Render one summary card containing multiple text columns. Useful when a single
compact annotation block becomes too dense to read comfortably.
"""
function text_columns_panel!(parent, columns::AbstractVector{<:AbstractVector{<:AbstractString}};
        title::AbstractString="", fontsize::Union{Nothing, Real}=nothing, titlefontsize::Union{Nothing, Real}=nothing,
        color=STYLE_REFERENCE, colgap::Real=26)
    cfg = current_figure_style()
    body_fontsize = fontsize === nothing ? cfg.summary_fontsize : fontsize
    heading_fontsize = titlefontsize === nothing ? cfg.summary_title_fontsize : titlefontsize
    gl = GridLayout(parent)
    body_row = 1
    if !isempty(title)
        Label(gl[1, 1:length(columns)], title; halign=:left, valign=:top,
              fontsize=heading_fontsize, color=color, font=:bold,
              padding=(0, 0, 8, 0))
        body_row = 2
    end

    for (col_idx, col_lines) in enumerate(columns)
        body = join(col_lines, "\n")
        Label(gl[body_row, col_idx], body; halign=:left, valign=:top,
              justification=:left, fontsize=body_fontsize, color=color)
    end
    colgap!(gl, colgap)
    rowgap!(gl, 6)
    return gl
end

"""
    figure_title!(fig, text; subtitle="")

Add a centered figure-level title (and optional subtitle) at the top.
"""
function figure_title!(fig::Figure, title::AbstractString;
        subtitle::AbstractString="", fontsize::Union{Nothing, Real}=nothing,
        subtitle_fontsize::Union{Nothing, Real}=nothing)
    cfg = current_figure_style()
    title_size = fontsize === nothing ? cfg.figure_title_fontsize : fontsize
    subtitle_size = subtitle_fontsize === nothing ? cfg.figure_subtitle_fontsize : subtitle_fontsize
    if isempty(subtitle)
        Label(fig[0, :], title; fontsize=title_size, font=:bold,
              color=STYLE_REFERENCE, halign=:center,
              padding=(0, 0, 8, 6))
    else
        gl = GridLayout(fig[0, :]; tellheight=true)
        Label(gl[1, 1], title; fontsize=title_size, font=:bold,
              color=STYLE_REFERENCE, halign=:center,
              padding=(0, 0, 0, 0))
        Label(gl[2, 1], subtitle; fontsize=subtitle_size,
              color=STYLE_MUTED, halign=:center,
              padding=(0, 0, 6, 4))
        rowgap!(gl, 2)
    end
    return nothing
end

# Convenience wrapper around `Makie.save` with the standard pixel scaling we
# want for publication-quality PNGs.
function save_figure(path::AbstractString, fig::Figure; px_per_unit::Union{Nothing, Real}=nothing)
    scale = px_per_unit === nothing ? current_figure_style().px_per_unit : px_per_unit
    save(path, fig; px_per_unit=scale)
    return nothing
end

apply_unified_theme!()
