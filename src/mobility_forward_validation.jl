#! Internal helper included by fit_dm.jl.

# mobility_forward_validation.jl
# - Loads the learned stationary score and mobility artifacts produced by the
#   affine multiplicative-noise pipeline.
# - Builds the learned Ito drift f(x) = M(x) s(x) + div M(x), with the
#   divergence of the full mobility field computed from the learned network.
# - Integrates the learned reduced Langevin model and the true affine-noise
#   benchmark with matched practical settings.
# - Writes comparison figures, diagnostics, and trajectory outputs for forward
#   validation of the learned model.

ensure_packages(["ForwardDiff", "KernelDensity", "ProgressMeter"])

using ForwardDiff
using KernelDensity
using ProgressMeter

const DEFAULT_CPHI_CHANNELS = [(3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)]

struct ForwardValidationParams
    input_hdf5::String
    score_bson::String
    mobility_model_bson::String
    mobility_artifact_bson::String
    phi_bson::Union{Nothing, String}
    data_burnin_fraction::Float64
    dt::Float64
    save_stride::Int
    total_time::Float64
    burnin_time::Float64
    ntrajectories::Int
    seed::Int
    use_common_random_numbers::Bool
    eval_batch_size::Int
    clamp_eval_to_support::Bool
    hard_clamp_state::Bool
    support_pad_fraction::Float64
    mobility_psd_jitter::Float64
    diffusion_floor::Float64
    diffusion_cap_quantile::Float64
    diffusion_cap_multiplier::Float64
    pdf_bins::Int
    pdf_max_samples::Int
    correlation_stride::Int
    correlation_max_time::Float64
    correlation_threshold::Float64
    cphi_use_artifact_lags::Bool
    cphi_max_time::Float64
    cphi_stride::Int
    auxiliary_max_samples::Int
    divergence_method::String
    divergence_grid_nx::Int
    divergence_grid_ny::Int
    divergence_fd_eps::Float64
    figure_width::Int
    figure_height::Int
    figure_stats_png::String
    figure_observed_png::Union{Nothing, String}
    figure_cphi_png::String
    metrics_txt::String
    diagnostics_bson::String
    trajectories_hdf5::String
end

struct ScoreRuntime
    model
    device::DeviceConfig
    batch_size::Int
end

struct MobilityRuntime
    host_model
    device_model
    device::DeviceConfig
    μ::Vector{Float32}
    σ::Vector{Float32}
    μ_dev
    σ_dev
    μ64::Vector{Float64}
    σ64::Vector{Float64}
    psd_jitter::Float64
    batch_size::Int
    phi_est::Matrix{Float64}
    training_target_source::String
    training_pairs::Vector{Tuple{Int, Int}}
    training_channel_labels::Vector{String}
end

struct PhiBaselineRuntime
    phi::Matrix{Float64}
    λ1::Float64
    λ2::Float64
    c::Float64
    s::Float64
    floor_hits_per_state::Int
    cap_hits_per_state::Int
end

struct DivergenceGrid
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    div1::Matrix{Float64}
    div2::Matrix{Float64}
end

struct Density1D
    centers::Vector{Float64}
    density::Vector{Float64}
    boundary::Tuple{Float64, Float64}
end

struct Density2D
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    density::Matrix{Float64}
    xboundary::Tuple{Float64, Float64}
    yboundary::Tuple{Float64, Float64}
end

struct CorrelationSummary
    lags::Vector{Float64}
    acf_x::Vector{Float64}
    acf_y::Vector{Float64}
    cross_xy::Vector{Float64}
    cross_yx::Vector{Float64}
    t_decorrelation::Float64
end

struct ObservedStatisticsReference
    pdf_x::Density1D
    pdf_y::Density1D
    pdf_xy::Density2D
    corr::CorrelationSummary
end

function optional_string(value)
    text = strip(String(value))
    return isempty(text) ? nothing : text
end

function resolve_optional_path(base_dir::AbstractString, path::Union{Nothing, String})
    path === nothing && return nothing
    return resolve_path(base_dir, path)
end

function load_forward_validation_params(path::AbstractString)
    raw = TOML.parsefile(path)
    input_cfg = raw["inputs"]
    sim_cfg = raw["simulation"]
    analysis_cfg = raw["analysis"]
    divergence_cfg = raw["divergence"]
    figure_cfg = raw["figure"]
    output_cfg = raw["output"]

    params = ForwardValidationParams(
        String(input_cfg["input_hdf5"]),
        String(input_cfg["score_bson"]),
        String(input_cfg["mobility_model_bson"]),
        String(input_cfg["mobility_artifact_bson"]),
        optional_string(get(input_cfg, "phi_bson", "")),
        Float64(input_cfg["burnin_fraction"]),
        Float64(sim_cfg["dt"]),
        Int(sim_cfg["save_stride"]),
        Float64(sim_cfg["total_time"]),
        Float64(sim_cfg["burnin_time"]),
        Int(sim_cfg["ntrajectories"]),
        Int(sim_cfg["seed"]),
        Bool(sim_cfg["use_common_random_numbers"]),
        Int(sim_cfg["eval_batch_size"]),
        Bool(sim_cfg["clamp_eval_to_support"]),
        Bool(sim_cfg["hard_clamp_state"]),
        Float64(sim_cfg["support_pad_fraction"]),
        Float64(sim_cfg["mobility_psd_jitter"]),
        Float64(sim_cfg["diffusion_floor"]),
        Float64(sim_cfg["diffusion_cap_quantile"]),
        Float64(sim_cfg["diffusion_cap_multiplier"]),
        Int(analysis_cfg["pdf_bins"]),
        Int(analysis_cfg["pdf_max_samples"]),
        Int(analysis_cfg["correlation_stride"]),
        Float64(analysis_cfg["correlation_max_time"]),
        Float64(analysis_cfg["correlation_threshold"]),
        Bool(analysis_cfg["use_artifact_lags"]),
        Float64(analysis_cfg["cphi_max_time"]),
        Int(analysis_cfg["cphi_stride"]),
        Int(analysis_cfg["auxiliary_max_samples"]),
        String(divergence_cfg["method"]),
        Int(divergence_cfg["grid_nx"]),
        Int(divergence_cfg["grid_ny"]),
        Float64(divergence_cfg["finite_difference_eps"]),
        Int(figure_cfg["width"]),
        Int(figure_cfg["height"]),
        String(output_cfg["figure_stats_png"]),
        optional_string(get(output_cfg, "figure_observed_png", "")),
        String(output_cfg["figure_cphi_png"]),
        String(output_cfg["metrics_txt"]),
        String(output_cfg["diagnostics_bson"]),
        String(output_cfg["trajectories_hdf5"]),
    )

    require_condition(0.0 <= params.data_burnin_fraction < 1.0, "burnin_fraction must lie in [0, 1).")
    require_condition(params.dt > 0.0, "dt must be positive.")
    require_condition(params.save_stride >= 1, "save_stride must be at least 1.")
    require_condition(params.total_time > params.burnin_time >= 0.0, "Need total_time > burnin_time >= 0.")
    require_condition(params.ntrajectories >= 32, "ntrajectories must be at least 32.")
    require_condition(params.eval_batch_size >= 32, "eval_batch_size must be at least 32.")
    require_condition(params.support_pad_fraction >= 0.0, "support_pad_fraction must be nonnegative.")
    require_condition(params.mobility_psd_jitter > 0.0, "mobility_psd_jitter must be positive.")
    require_condition(params.diffusion_floor > 0.0, "diffusion_floor must be positive.")
    require_condition(0.0 < params.diffusion_cap_quantile <= 1.0, "diffusion_cap_quantile must lie in (0, 1].")
    require_condition(params.diffusion_cap_multiplier >= 1.0, "diffusion_cap_multiplier must be at least 1.")
    require_condition(params.pdf_bins >= 32, "pdf_bins must be at least 32.")
    require_condition(params.pdf_max_samples >= 10_000, "pdf_max_samples must be at least 10000.")
    require_condition(params.correlation_stride >= 1, "correlation_stride must be at least 1.")
    require_condition(params.correlation_max_time > 0.0, "correlation_max_time must be positive.")
    require_condition(params.correlation_threshold > 0.0, "correlation_threshold must be positive.")
    require_condition(params.cphi_max_time > 0.0, "cphi_max_time must be positive.")
    require_condition(params.cphi_stride >= 1, "cphi_stride must be at least 1.")
    require_condition(params.auxiliary_max_samples >= 5_000, "auxiliary_max_samples must be at least 5000.")
    require_condition(params.divergence_method in ("forwarddiff_grid", "finite_difference_grid"),
        "divergence.method must be one of: forwarddiff_grid, finite_difference_grid.")
    require_condition(params.divergence_grid_nx >= 32 && params.divergence_grid_ny >= 32,
        "divergence grid dimensions must each be at least 32.")
    require_condition(params.divergence_fd_eps > 0.0, "finite_difference_eps must be positive.")
    require_condition(params.figure_width >= 1600 && params.figure_height >= 1200, "figure dimensions are too small.")
    return params
end

function load_score_runtime(path::AbstractString, batch_size::Int, device::DeviceConfig)
    data = BSON.load(path)
    require_condition(haskey(data, :host_model), "Expected :host_model in $(path).")
    model_dev = to_device(data[:host_model], device)
    return ScoreRuntime(model_dev, device, batch_size), data
end

function load_mobility_runtime(path::AbstractString, batch_size::Int, device::DeviceConfig, psd_jitter::Float64)
    data = BSON.load(path)
    actual_kind = String(get(data, :model_kind, ""))
    require_condition(actual_kind == "full_mobility_affine_benchmark",
        "Expected model_kind=full_mobility_affine_benchmark in $(path), found $(actual_kind).")
    require_condition(haskey(data, :host_model), "Expected :host_model in $(path).")
    require_condition(haskey(data, :input_mean) && haskey(data, :input_scale), "Missing input normalization metadata in $(path).")
    require_condition(haskey(data, :phi_est), "Missing :phi_est in $(path).")

    host_model = cpu(data[:host_model])
    μ = Float32.(vec(data[:input_mean]))
    σ = Float32.(vec(data[:input_scale]))
    training_pairs = haskey(data, :training_pairs) ? Tuple{Int, Int}[(Int(pair[1]), Int(pair[2])) for pair in data[:training_pairs]] :
        Tuple{Int, Int}[DEFAULT_CPHI_CHANNELS...]
    runtime = MobilityRuntime(
        host_model,
        to_device(host_model, device),
        device,
        μ,
        σ,
        to_device(μ, device),
        to_device(σ, device),
        Float64.(μ),
        Float64.(σ),
        psd_jitter,
        batch_size,
        Float64.(Matrix(data[:phi_est])),
        String(get(data, :training_target_source, "unknown")),
        training_pairs,
        haskey(data, :training_channel_labels) ? String.(data[:training_channel_labels]) : String[],
    )
    return runtime, data
end

function load_phi_override(path::Union{Nothing, String})
    path === nothing && return nothing
    data = BSON.load(path)
    require_condition(haskey(data, :phi_est), "Expected :phi_est in $(path).")
    return Float64.(Matrix(data[:phi_est]))
end

function grid_boundary(grid::Vector{Float64})
    halfwidth = length(grid) > 1 ? 0.5 * (grid[2] - grid[1]) : 0.5
    return (grid[1] - halfwidth, grid[end] + halfwidth)
end

function load_observed_statistics_reference(path::AbstractString)
    xcenters = Float64.(h5read(path, "/statistics/pdf/x_centers"))
    xdensity = Float64.(h5read(path, "/statistics/pdf/x_density"))
    ycenters = Float64.(h5read(path, "/statistics/pdf/y_centers"))
    ydensity = Float64.(h5read(path, "/statistics/pdf/y_density"))
    xgrid = Float64.(h5read(path, "/statistics/pdf/xy_x_grid"))
    ygrid = Float64.(h5read(path, "/statistics/pdf/xy_y_grid"))
    density_xy = Float64.(h5read(path, "/statistics/pdf/xy_density"))
    lags = Float64.(h5read(path, "/statistics/correlations/lags"))
    acf_x = Float64.(h5read(path, "/statistics/correlations/acf_x"))
    acf_y = Float64.(h5read(path, "/statistics/correlations/acf_y"))
    cross_xy = Float64.(h5read(path, "/statistics/correlations/cross_xy"))
    cross_yx = Float64.(h5read(path, "/statistics/correlations/cross_yx"))
    t_decorrelation = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))

    return ObservedStatisticsReference(
        Density1D(xcenters, xdensity, grid_boundary(xcenters)),
        Density1D(ycenters, ydensity, grid_boundary(ycenters)),
        Density2D(xgrid, ygrid, density_xy, grid_boundary(xgrid), grid_boundary(ygrid)),
        CorrelationSummary(lags, acf_x, acf_y, cross_xy, cross_yx, t_decorrelation),
    )
end

function mobility_vector_at_point(model, point::AbstractVector, μ::AbstractVector, σ::AbstractVector, psd_jitter::Real)
    normalized = reshape((point .- μ) ./ σ, 2, 1)
    m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(model, normalized, psd_jitter)
    return [m11[1], m12[1], m21[1], m22[1]]
end

function divergence_components_forwarddiff(runtime::MobilityRuntime, x::Float64, y::Float64)
    point = Float64[x, y]
    jac = ForwardDiff.jacobian(z -> mobility_vector_at_point(runtime.host_model, z, runtime.μ64, runtime.σ64, runtime.psd_jitter), point)
    return Float64(jac[1, 1] + jac[2, 2]), Float64(jac[3, 1] + jac[4, 2])
end

function divergence_components_fd(runtime::MobilityRuntime, x::Float64, y::Float64, eps_fd::Float64)
    plus_x = mobility_vector_at_point(runtime.host_model, [x + eps_fd, y], runtime.μ64, runtime.σ64, runtime.psd_jitter)
    minus_x = mobility_vector_at_point(runtime.host_model, [x - eps_fd, y], runtime.μ64, runtime.σ64, runtime.psd_jitter)
    plus_y = mobility_vector_at_point(runtime.host_model, [x, y + eps_fd], runtime.μ64, runtime.σ64, runtime.psd_jitter)
    minus_y = mobility_vector_at_point(runtime.host_model, [x, y - eps_fd], runtime.μ64, runtime.σ64, runtime.psd_jitter)
    inv_2eps = 0.5 / eps_fd
    div1 = (plus_x[1] - minus_x[1]) * inv_2eps + (plus_y[2] - minus_y[2]) * inv_2eps
    div2 = (plus_x[3] - minus_x[3]) * inv_2eps + (plus_y[4] - minus_y[4]) * inv_2eps
    return div1, div2
end

function build_divergence_grid(runtime::MobilityRuntime, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64},
        method::String, nx::Int, ny::Int, eps_fd::Float64)
    xgrid = collect(range(xbounds[1], xbounds[2]; length=nx))
    ygrid = collect(range(ybounds[1], ybounds[2]; length=ny))
    div1 = Matrix{Float64}(undef, nx, ny)
    div2 = Matrix{Float64}(undef, nx, ny)

    @printf("Building divergence grid (%s) on %d x %d support points\n", method, nx, ny)
    progress = Progress(ny; desc="div(M) grid ", dt=0.5)
    for j in 1:ny
        y = ygrid[j]
        @inbounds for i in 1:nx
            x = xgrid[i]
            d1, d2 = if method == "forwarddiff_grid"
                divergence_components_forwarddiff(runtime, x, y)
            else
                divergence_components_fd(runtime, x, y, eps_fd)
            end
            div1[i, j] = d1
            div2[i, j] = d2
        end
        next!(progress)
    end
    finish!(progress)
    return DivergenceGrid(xgrid, ygrid, div1, div2)
end

function clamp_to_bounds!(points::AbstractMatrix{Float32}, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64})
    @views points[1, :] .= clamp.(points[1, :], Float32(xbounds[1]), Float32(xbounds[2]))
    @views points[2, :] .= clamp.(points[2, :], Float32(ybounds[1]), Float32(ybounds[2]))
    return nothing
end

function bilinear_value(grid::Matrix{Float64}, xgrid::Vector{Float64}, ygrid::Vector{Float64}, x::Float64, y::Float64)
    xc = clamp(x, xgrid[1], xgrid[end])
    yc = clamp(y, ygrid[1], ygrid[end])
    ix_hi = searchsortedfirst(xgrid, xc)
    iy_hi = searchsortedfirst(ygrid, yc)
    ix_hi = clamp(ix_hi, 2, length(xgrid))
    iy_hi = clamp(iy_hi, 2, length(ygrid))
    ix_lo = ix_hi - 1
    iy_lo = iy_hi - 1

    x0 = xgrid[ix_lo]
    x1 = xgrid[ix_hi]
    y0 = ygrid[iy_lo]
    y1 = ygrid[iy_hi]
    tx = x1 > x0 ? (xc - x0) / (x1 - x0) : 0.0
    ty = y1 > y0 ? (yc - y0) / (y1 - y0) : 0.0

    v00 = grid[ix_lo, iy_lo]
    v10 = grid[ix_hi, iy_lo]
    v01 = grid[ix_lo, iy_hi]
    v11 = grid[ix_hi, iy_hi]
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11
end

function interpolate_divergence!(out::Matrix{Float64}, grid::DivergenceGrid, points::Matrix{Float64})
    n = size(points, 2)
    @inbounds for idx in 1:n
        x = points[1, idx]
        y = points[2, idx]
        out[1, idx] = bilinear_value(grid.div1, grid.xgrid, grid.ygrid, x, y)
        out[2, idx] = bilinear_value(grid.div2, grid.xgrid, grid.ygrid, x, y)
    end
    return out
end

function sample_initial_states(states::Array{Float64, 3}, start_idx::Int, nsamples::Int, rng::AbstractRNG)
    nt, _, ntraj = size(states)
    init = Matrix{Float64}(undef, 2, nsamples)
    @inbounds for idx in 1:nsamples
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, start_idx:nt)
        init[1, idx] = states[time_idx, 1, traj_idx]
        init[2, idx] = states[time_idx, 2, traj_idx]
    end
    return init
end

function evaluate_score_batch(runtime::ScoreRuntime, points::Matrix{Float32})
    out = Matrix{Float64}(undef, 2, size(points, 2))
    for start in 1:runtime.batch_size:size(points, 2)
        stop = min(start + runtime.batch_size - 1, size(points, 2))
        batch = @view points[:, start:stop]
        pred = runtime.model(to_device(batch, runtime.device))
        out[:, start:stop] .= Float64.(to_host(pred))
    end
    return out
end

function evaluate_mobility_batch(runtime::MobilityRuntime, points::Matrix{Float32})
    out = Matrix{Float64}(undef, 4, size(points, 2))
    for start in 1:runtime.batch_size:size(points, 2)
        stop = min(start + runtime.batch_size - 1, size(points, 2))
        batch = @view points[:, start:stop]
        m11, m12, m21, m22, _, _, _ = evaluate_mobility_outputs(runtime.device_model,
            normalize_points(to_device(batch, runtime.device), runtime.μ_dev, runtime.σ_dev), runtime.psd_jitter)
        out[1, start:stop] .= Float64.(to_host(m11))
        out[2, start:stop] .= Float64.(to_host(m12))
        out[3, start:stop] .= Float64.(to_host(m21))
        out[4, start:stop] .= Float64.(to_host(m22))
    end
    return out
end

function prepare_eval_points(state::Matrix{Float64}, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64}, clamp_eval::Bool)
    points64 = copy(state)
    outside = 0
    if clamp_eval
        @inbounds for idx in 1:size(points64, 2)
            x = points64[1, idx]
            y = points64[2, idx]
            if x < xbounds[1] || x > xbounds[2] || y < ybounds[1] || y > ybounds[2]
                outside += 1
            end
            points64[1, idx] = clamp(x, xbounds[1], xbounds[2])
            points64[2, idx] = clamp(y, ybounds[1], ybounds[2])
        end
    else
        @inbounds for idx in 1:size(points64, 2)
            x = points64[1, idx]
            y = points64[2, idx]
            if x < xbounds[1] || x > xbounds[2] || y < ybounds[1] || y > ybounds[2]
                outside += 1
            end
        end
    end
    return points64, Float32.(points64), outside
end

function diffusion_eigvals(a::Float64, b::Float64, d::Float64)
    disc = sqrt((a - d)^2 + 4.0 * b^2 + 1.0e-15)
    return 0.5 * (a + d + disc), 0.5 * (a + d - disc)
end

function diffusion_cap_from_observed(runtime::MobilityRuntime, states::Array{Float64, 3}, start_idx::Int,
        quantile_level::Float64, multiplier::Float64)
    navailable = (size(states, 1) - start_idx + 1) * size(states, 3)
    nsample = min(navailable, 50_000)
    rng = MersenneTwister(20260413)
    points = Matrix{Float32}(undef, 2, nsample)
    nt, _, ntraj = size(states)
    @inbounds for idx in 1:nsample
        traj_idx = rand(rng, 1:ntraj)
        time_idx = rand(rng, start_idx:nt)
        points[1, idx] = Float32(states[time_idx, 1, traj_idx])
        points[2, idx] = Float32(states[time_idx, 2, traj_idx])
    end
    mats = evaluate_mobility_batch(runtime, points)
    λmax = Vector{Float64}(undef, nsample)
    @inbounds for idx in 1:nsample
        a = mats[1, idx]
        b = 0.5 * (mats[2, idx] + mats[3, idx])
        d = mats[4, idx]
        λ1, λ2 = diffusion_eigvals(a, b, d)
        λmax[idx] = max(λ1, λ2)
    end
    sort!(λmax)
    qidx = clamp(round(Int, quantile_level * length(λmax)), 1, length(λmax))
    return max(λmax[qidx] * multiplier, 5.0 * runtime.psd_jitter)
end

function build_phi_baseline_runtime(phi::Matrix{Float64}, diffusion_floor::Float64, diffusion_cap::Float64)
    sym = 0.5 .* (phi .+ phi')
    a = sym[1, 1]
    b = sym[1, 2]
    d = sym[2, 2]
    λ1, λ2 = diffusion_eigvals(a, b, d)
    λ1_clamped = clamp(λ1, diffusion_floor, diffusion_cap)
    λ2_clamped = clamp(λ2, diffusion_floor, diffusion_cap)
    θ = 0.5 * atan(2.0 * b, a - d)
    return PhiBaselineRuntime(
        copy(phi),
        λ1_clamped,
        λ2_clamped,
        cos(θ),
        sin(θ),
        Int(λ1_clamped == diffusion_floor) + Int(λ2_clamped == diffusion_floor),
        Int(λ1_clamped == diffusion_cap) + Int(λ2_clamped == diffusion_cap),
    )
end

function predicted_full_step!(state::Matrix{Float64}, noise_buf::Matrix{Float64}, score_runtime::ScoreRuntime, mobility_runtime::MobilityRuntime,
        divergence_grid::DivergenceGrid, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64},
        dt::Float64, diffusion_floor::Float64, diffusion_cap::Float64, clamp_eval::Bool, hard_clamp_state::Bool)
    ntraj = size(state, 2)
    eval_points64, eval_points32, outside_eval = prepare_eval_points(state, xbounds, ybounds, clamp_eval)
    score = evaluate_score_batch(score_runtime, eval_points32)
    mats = evaluate_mobility_batch(mobility_runtime, eval_points32)
    divergence = interpolate_divergence!(zeros(Float64, 2, ntraj), divergence_grid, eval_points64)

    floor_hits = 0
    cap_hits = 0
    hard_clamps = 0
    sqrt_2dt = sqrt(2.0 * dt)

    @inbounds for idx in 1:ntraj
        m11 = mats[1, idx]
        m12 = mats[2, idx]
        m21 = mats[3, idx]
        m22 = mats[4, idx]
        s1 = score[1, idx]
        s2 = score[2, idx]
        drift1 = m11 * s1 + m12 * s2 + divergence[1, idx]
        drift2 = m21 * s1 + m22 * s2 + divergence[2, idx]

        a = m11
        b = 0.5 * (m12 + m21)
        d = m22
        λ1, λ2 = diffusion_eigvals(a, b, d)
        λ1_clamped = clamp(λ1, diffusion_floor, diffusion_cap)
        λ2_clamped = clamp(λ2, diffusion_floor, diffusion_cap)
        floor_hits += (λ1_clamped == diffusion_floor) + (λ2_clamped == diffusion_floor)
        cap_hits += (λ1_clamped == diffusion_cap) + (λ2_clamped == diffusion_cap)

        θ = 0.5 * atan(2.0 * b, a - d)
        c = cos(θ)
        s = sin(θ)
        n1 = sqrt(λ1_clamped) * noise_buf[1, idx]
        n2 = sqrt(λ2_clamped) * noise_buf[2, idx]
        noise1 = c * n1 - s * n2
        noise2 = s * n1 + c * n2

        state[1, idx] += dt * drift1 + sqrt_2dt * noise1
        state[2, idx] += dt * drift2 + sqrt_2dt * noise2
        if hard_clamp_state
            clamped_x = clamp(state[1, idx], xbounds[1], xbounds[2])
            clamped_y = clamp(state[2, idx], ybounds[1], ybounds[2])
            hard_clamps += (clamped_x != state[1, idx]) || (clamped_y != state[2, idx])
            state[1, idx] = clamped_x
            state[2, idx] = clamped_y
        end
        require_condition(isfinite(state[1, idx]) && isfinite(state[2, idx]), "Predicted trajectory produced a non-finite state.")
    end
    return outside_eval, floor_hits, cap_hits, hard_clamps
end

function predicted_phi_step!(state::Matrix{Float64}, noise_buf::Matrix{Float64}, score_runtime::ScoreRuntime,
        phi_runtime::PhiBaselineRuntime, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64},
        dt::Float64, clamp_eval::Bool, hard_clamp_state::Bool)
    eval_points64, eval_points32, outside_eval = prepare_eval_points(state, xbounds, ybounds, clamp_eval)
    score = evaluate_score_batch(score_runtime, eval_points32)

    hard_clamps = 0
    sqrt_2dt = sqrt(2.0 * dt)
    phi11 = phi_runtime.phi[1, 1]
    phi12 = phi_runtime.phi[1, 2]
    phi21 = phi_runtime.phi[2, 1]
    phi22 = phi_runtime.phi[2, 2]

    @inbounds for idx in 1:size(state, 2)
        s1 = score[1, idx]
        s2 = score[2, idx]
        drift1 = phi11 * s1 + phi12 * s2
        drift2 = phi21 * s1 + phi22 * s2

        n1 = sqrt(phi_runtime.λ1) * noise_buf[1, idx]
        n2 = sqrt(phi_runtime.λ2) * noise_buf[2, idx]
        noise1 = phi_runtime.c * n1 - phi_runtime.s * n2
        noise2 = phi_runtime.s * n1 + phi_runtime.c * n2

        state[1, idx] += dt * drift1 + sqrt_2dt * noise1
        state[2, idx] += dt * drift2 + sqrt_2dt * noise2
        if hard_clamp_state
            clamped_x = clamp(state[1, idx], xbounds[1], xbounds[2])
            clamped_y = clamp(state[2, idx], ybounds[1], ybounds[2])
            hard_clamps += (clamped_x != state[1, idx]) || (clamped_y != state[2, idx])
            state[1, idx] = clamped_x
            state[2, idx] = clamped_y
        end
        require_condition(isfinite(state[1, idx]) && isfinite(state[2, idx]), "Phi-only trajectory produced a non-finite state.")
    end
    floor_hits = phi_runtime.floor_hits_per_state * size(state, 2)
    cap_hits = phi_runtime.cap_hits_per_state * size(state, 2)
    return outside_eval, floor_hits, cap_hits, hard_clamps
end

function true_step!(state::Matrix{Float64}, noise_buf::Matrix{Float64}, meta::AffineModelMetadata, dt::Float64)
    sqrt_dt = sqrt(dt)
    @inbounds for idx in 1:size(state, 2)
        x = state[1, idx]
        y = state[2, idx]
        drift = affine_model_drift(x, y, meta)
        bmat = affine_noise_matrix(x, y, meta)
        state[1, idx] += dt * drift[1] + sqrt_dt * (bmat[1, 1] * noise_buf[1, idx] + bmat[1, 2] * noise_buf[2, idx])
        state[2, idx] += dt * drift[2] + sqrt_dt * (bmat[2, 1] * noise_buf[1, idx] + bmat[2, 2] * noise_buf[2, idx])
        require_condition(isfinite(state[1, idx]) && isfinite(state[2, idx]), "True trajectory produced a non-finite state.")
    end
    return nothing
end

function integrate_validation_models(params::ForwardValidationParams, init_states::Matrix{Float64}, score_runtime::ScoreRuntime,
    mobility_runtime::MobilityRuntime, phi_runtime::PhiBaselineRuntime, divergence_grid::DivergenceGrid,
    meta::AffineModelMetadata, xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64}, diffusion_cap::Float64)
    total_steps = round(Int, params.total_time / params.dt)
    burnin_steps = round(Int, params.burnin_time / params.dt)
    require_condition(abs(total_steps * params.dt - params.total_time) <= 1.0e-10, "total_time must be an integer multiple of dt.")
    require_condition(abs(burnin_steps * params.dt - params.burnin_time) <= 1.0e-10, "burnin_time must be an integer multiple of dt.")
    require_condition(total_steps > burnin_steps, "Need at least one post-burn-in integration step.")

    nsaved = 1 + fld(total_steps - burnin_steps, params.save_stride)
    times = Vector{Float64}(undef, nsaved)
    true_states = Array{Float64}(undef, nsaved, 2, params.ntrajectories)
    pred_states_full = Array{Float64}(undef, nsaved, 2, params.ntrajectories)
    pred_states_phi = Array{Float64}(undef, nsaved, 2, params.ntrajectories)

    true_state = copy(init_states)
    pred_state_full = copy(init_states)
    pred_state_phi = copy(init_states)
    rng_true = MersenneTwister(params.seed)
    rng_pred_full = MersenneTwister(params.seed + 1)
    rng_pred_phi = MersenneTwister(params.seed + 2)
    common_rng = MersenneTwister(params.seed + 2)
    noise_true = Matrix{Float64}(undef, 2, params.ntrajectories)
    noise_pred_full = Matrix{Float64}(undef, 2, params.ntrajectories)
    noise_pred_phi = Matrix{Float64}(undef, 2, params.ntrajectories)

    eval_clamp_count_full = 0
    hard_clamp_count_full = 0
    floor_hit_count_full = 0
    cap_hit_count_full = 0
    eval_clamp_count_phi = 0
    hard_clamp_count_phi = 0
    floor_hit_count_phi = 0
    cap_hit_count_phi = 0

    save_cursor = 0
    progress = Progress(total_steps; desc="Forward sim ", dt=0.5)
    for step in 1:total_steps
        if params.use_common_random_numbers
            randn!(common_rng, noise_true)
            noise_pred_full .= noise_true
            noise_pred_phi .= noise_true
        else
            randn!(rng_true, noise_true)
            randn!(rng_pred_full, noise_pred_full)
            randn!(rng_pred_phi, noise_pred_phi)
        end

        true_step!(true_state, noise_true, meta, params.dt)
        outside_eval_full, floor_hits_full, cap_hits_full, hard_clamps_full = predicted_full_step!(pred_state_full, noise_pred_full, score_runtime, mobility_runtime,
            divergence_grid, xbounds, ybounds, params.dt, params.diffusion_floor, diffusion_cap,
            params.clamp_eval_to_support, params.hard_clamp_state)
        outside_eval_phi, floor_hits_phi, cap_hits_phi, hard_clamps_phi = predicted_phi_step!(pred_state_phi, noise_pred_phi, score_runtime,
            phi_runtime, xbounds, ybounds, params.dt, params.clamp_eval_to_support, params.hard_clamp_state)

        eval_clamp_count_full += outside_eval_full
        hard_clamp_count_full += hard_clamps_full
        floor_hit_count_full += floor_hits_full
        cap_hit_count_full += cap_hits_full
        eval_clamp_count_phi += outside_eval_phi
        hard_clamp_count_phi += hard_clamps_phi
        floor_hit_count_phi += floor_hits_phi
        cap_hit_count_phi += cap_hits_phi

        if step >= burnin_steps && (step - burnin_steps) % params.save_stride == 0
            save_cursor += 1
            times[save_cursor] = (step - burnin_steps) * params.dt
            true_states[save_cursor, :, :] .= true_state
            pred_states_full[save_cursor, :, :] .= pred_state_full
            pred_states_phi[save_cursor, :, :] .= pred_state_phi
        end
        next!(progress)
    end
    finish!(progress)

    stats_full = Dict{Symbol, Any}(
        :total_steps => total_steps,
        :burnin_steps => burnin_steps,
        :saved_steps => nsaved,
        :saved_dt => params.dt * params.save_stride,
        :eval_clamp_fraction => eval_clamp_count_full / (total_steps * params.ntrajectories),
        :hard_clamp_fraction => hard_clamp_count_full / (total_steps * params.ntrajectories),
        :diffusion_floor_fraction => floor_hit_count_full / (2 * total_steps * params.ntrajectories),
        :diffusion_cap_fraction => cap_hit_count_full / (2 * total_steps * params.ntrajectories),
        :diffusion_cap => diffusion_cap,
    )
    stats_phi = Dict{Symbol, Any}(
        :total_steps => total_steps,
        :burnin_steps => burnin_steps,
        :saved_steps => nsaved,
        :saved_dt => params.dt * params.save_stride,
        :eval_clamp_fraction => eval_clamp_count_phi / (total_steps * params.ntrajectories),
        :hard_clamp_fraction => hard_clamp_count_phi / (total_steps * params.ntrajectories),
        :diffusion_floor_fraction => floor_hit_count_phi / (2 * total_steps * params.ntrajectories),
        :diffusion_cap_fraction => cap_hit_count_phi / (2 * total_steps * params.ntrajectories),
        :diffusion_cap => diffusion_cap,
        :phi_matrix => copy(phi_runtime.phi),
        :lambda_1 => phi_runtime.λ1,
        :lambda_2 => phi_runtime.λ2,
    )
    return times, true_states, pred_states_full, pred_states_phi, stats_full, stats_phi
end

function flatten_coordinates(states::Array{Float64, 3})
    return vec(@view states[:, 1, :]), vec(@view states[:, 2, :])
end

function maybe_subsample_pair(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, max_samples::Int, rng::AbstractRNG)
    n = length(x)
    if n <= max_samples
        return Float64.(x), Float64.(y)
    end
    keep = randperm(rng, n)[1:max_samples]
    return Float64.(x[keep]), Float64.(y[keep])
end

function kde_range(values::AbstractVector{<:Real})
    vmin = minimum(values)
    vmax = maximum(values)
    span = max(vmax - vmin, 1.0e-6)
    pad = max(0.05 * span, 1.0e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function combined_kde_range(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    vmin = min(minimum(a), minimum(b))
    vmax = max(maximum(a), maximum(b))
    span = max(vmax - vmin, 1.0e-6)
    pad = max(0.05 * span, 1.0e-3)
    return (Float64(vmin - pad), Float64(vmax + pad))
end

function compute_density_1d(data::AbstractVector{<:Real}, bins::Int; boundary=nothing)
    actual_boundary = boundary === nothing ? kde_range(data) : boundary
    kde_result = kde(Float64.(data); npoints=bins, boundary=actual_boundary)
    return Density1D(collect(kde_result.x), collect(kde_result.density), actual_boundary)
end

function compute_density_2d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, bins::Int; xboundary=nothing, yboundary=nothing)
    actual_x = xboundary === nothing ? kde_range(x) : xboundary
    actual_y = yboundary === nothing ? kde_range(y) : yboundary
    kde_result = kde((Float64.(x), Float64.(y)); npoints=(bins, bins), boundary=(actual_x, actual_y))
    return Density2D(collect(kde_result.x), collect(kde_result.y), Array(kde_result.density), actual_x, actual_y)
end

function density_width(grid::Vector{Float64})
    return length(grid) > 1 ? (grid[2] - grid[1]) : 1.0
end

function rmse(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    return sqrt(mean((Float64.(a) .- Float64.(b)) .^ 2))
end

function kl_divergence_1d(p_density::Vector{Float64}, q_density::Vector{Float64}, width::Float64)
    eps = 1.0e-12
    p = p_density .* width
    q = q_density .* width
    p .+= eps
    q .+= eps
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function kl_divergence_2d(p_density::Matrix{Float64}, q_density::Matrix{Float64}, xwidth::Float64, ywidth::Float64)
    eps = 1.0e-12
    p = vec(p_density .* (xwidth * ywidth))
    q = vec(q_density .* (xwidth * ywidth))
    p .+= eps
    q .+= eps
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function pdf_comparison(true_states::Array{Float64, 3}, pred_states::Array{Float64, 3}, bins::Int, max_samples::Int, rng::AbstractRNG)
    true_x_all, true_y_all = flatten_coordinates(true_states)
    pred_x_all, pred_y_all = flatten_coordinates(pred_states)
    true_x, true_y = maybe_subsample_pair(true_x_all, true_y_all, max_samples, rng)
    pred_x, pred_y = maybe_subsample_pair(pred_x_all, pred_y_all, max_samples, rng)

    xboundary = combined_kde_range(true_x, pred_x)
    yboundary = combined_kde_range(true_y, pred_y)

    dens_true_x = compute_density_1d(true_x, bins; boundary=xboundary)
    dens_pred_x = compute_density_1d(pred_x, bins; boundary=xboundary)
    dens_true_y = compute_density_1d(true_y, bins; boundary=yboundary)
    dens_pred_y = compute_density_1d(pred_y, bins; boundary=yboundary)
    dens_true_xy = compute_density_2d(true_x, true_y, bins; xboundary=xboundary, yboundary=yboundary)
    dens_pred_xy = compute_density_2d(pred_x, pred_y, bins; xboundary=xboundary, yboundary=yboundary)

    xwidth = density_width(dens_true_x.centers)
    ywidth = density_width(dens_true_y.centers)
    xwidth2 = density_width(dens_true_xy.xgrid)
    ywidth2 = density_width(dens_true_xy.ygrid)

    metrics = Dict{Symbol, Float64}(
        :rmse_x => rmse(dens_true_x.density, dens_pred_x.density),
        :rmse_y => rmse(dens_true_y.density, dens_pred_y.density),
        :rmse_xy => rmse(dens_true_xy.density, dens_pred_xy.density),
        :kl_x => kl_divergence_1d(dens_true_x.density, dens_pred_x.density, xwidth),
        :kl_y => kl_divergence_1d(dens_true_y.density, dens_pred_y.density, ywidth),
        :kl_xy => kl_divergence_2d(dens_true_xy.density, dens_pred_xy.density, xwidth2, ywidth2),
    )
    return Dict(
        :true_x => dens_true_x,
        :pred_x => dens_pred_x,
        :true_y => dens_true_y,
        :pred_y => dens_pred_y,
        :true_xy => dens_true_xy,
        :pred_xy => dens_pred_xy,
        :metrics => metrics,
    )
end

function pdf_comparison_against_reference(reference::ObservedStatisticsReference, pred_states::Array{Float64, 3},
        max_samples::Int, rng::AbstractRNG)
    pred_x_all, pred_y_all = flatten_coordinates(pred_states)
    pred_x, pred_y = maybe_subsample_pair(pred_x_all, pred_y_all, max_samples, rng)
    dens_pred_x = compute_density_1d(pred_x, length(reference.pdf_x.centers); boundary=reference.pdf_x.boundary)
    dens_pred_y = compute_density_1d(pred_y, length(reference.pdf_y.centers); boundary=reference.pdf_y.boundary)
    dens_pred_xy = compute_density_2d(pred_x, pred_y, length(reference.pdf_xy.xgrid);
        xboundary=reference.pdf_xy.xboundary, yboundary=reference.pdf_xy.yboundary)

    xwidth = density_width(reference.pdf_x.centers)
    ywidth = density_width(reference.pdf_y.centers)
    xwidth2 = density_width(reference.pdf_xy.xgrid)
    ywidth2 = density_width(reference.pdf_xy.ygrid)

    metrics = Dict{Symbol, Float64}(
        :rmse_x => rmse(reference.pdf_x.density, dens_pred_x.density),
        :rmse_y => rmse(reference.pdf_y.density, dens_pred_y.density),
        :rmse_xy => rmse(reference.pdf_xy.density, dens_pred_xy.density),
        :kl_x => kl_divergence_1d(reference.pdf_x.density, dens_pred_x.density, xwidth),
        :kl_y => kl_divergence_1d(reference.pdf_y.density, dens_pred_y.density, ywidth),
        :kl_xy => kl_divergence_2d(reference.pdf_xy.density, dens_pred_xy.density, xwidth2, ywidth2),
    )
    return Dict(
        :true_x => reference.pdf_x,
        :pred_x => dens_pred_x,
        :true_y => reference.pdf_y,
        :pred_y => dens_pred_y,
        :true_xy => reference.pdf_xy,
        :pred_xy => dens_pred_xy,
        :metrics => metrics,
    )
end

function matrix_mean_and_variance(data::Matrix{Float64})
    total = 0.0
    count = 0
    @inbounds for j in axes(data, 2), i in axes(data, 1)
        total += data[i, j]
        count += 1
    end
    mean_value = total / count
    sumsq = 0.0
    @inbounds for j in axes(data, 2), i in axes(data, 1)
        delta = data[i, j] - mean_value
        sumsq += delta * delta
    end
    return mean_value, sumsq / count
end

function estimate_decorrelation_time(lags::Vector{Float64}, acf_x::Vector{Float64}, acf_y::Vector{Float64}, threshold::Float64)
    envelope = max.(abs.(acf_x), abs.(acf_y))
    running_max = copy(envelope)
    for idx in (length(envelope) - 1):-1:1
        running_max[idx] = max(running_max[idx], running_max[idx + 1])
    end
    for idx in 2:length(lags)
        if running_max[idx] <= threshold
            return lags[idx]
        end
    end
    return lags[end]
end

function lag_steps_from_times(lag_times::Vector{Float64}, saved_dt::Float64; allow_zero::Bool=false)
    min_step = allow_zero ? 0 : 1
    steps = Int[]
    for tau in lag_times
        step = round(Int, tau / saved_dt)
        require_condition(step >= min_step, "Lag times must map to admissible step counts.")
        require_condition(abs(step * saved_dt - tau) <= max(1.0e-8, 1.0e-4 * saved_dt),
            @sprintf("Lag %.6f is incompatible with saved_dt = %.6f.", tau, saved_dt))
        push!(steps, step)
    end
    return steps
end

function compute_correlations(x::Matrix{Float64}, y::Matrix{Float64}, dt_corr::Float64, max_time::Float64, threshold::Float64)
    ntime, ntraj = size(x)
    mean_x, var_x = matrix_mean_and_variance(x)
    mean_y, var_y = matrix_mean_and_variance(y)
    denom_xy = sqrt(var_x * var_y)
    max_lag = min(ntime - 1, floor(Int, max_time / dt_corr))
    acf_x = zeros(Float64, max_lag + 1)
    acf_y = zeros(Float64, max_lag + 1)
    cross_xy = zeros(Float64, max_lag + 1)
    cross_yx = zeros(Float64, max_lag + 1)

    Threads.@threads for lag in 0:max_lag
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0
        sum_yx = 0.0
        count = 0
        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for i in 1:upper
                x0 = x[i, traj_idx] - mean_x
                y0 = y[i, traj_idx] - mean_y
                x1 = x[i + lag, traj_idx] - mean_x
                y1 = y[i + lag, traj_idx] - mean_y
                sum_xx += x1 * x0
                sum_yy += y1 * y0
                sum_xy += x1 * y0
                sum_yx += y1 * x0
            end
            count += upper
        end
        acf_x[lag + 1] = sum_xx / (count * var_x)
        acf_y[lag + 1] = sum_yy / (count * var_y)
        cross_xy[lag + 1] = sum_xy / (count * denom_xy)
        cross_yx[lag + 1] = sum_yx / (count * denom_xy)
    end

    lags = collect(0:max_lag) .* dt_corr
    return CorrelationSummary(lags, acf_x, acf_y, cross_xy, cross_yx, estimate_decorrelation_time(lags, acf_x, acf_y, threshold))
end

function compute_correlations_at_lag_steps(x::Matrix{Float64}, y::Matrix{Float64}, saved_dt::Float64,
        lag_steps::Vector{Int}, threshold::Float64)
    ntime, ntraj = size(x)
    mean_x, var_x = matrix_mean_and_variance(x)
    mean_y, var_y = matrix_mean_and_variance(y)
    denom_xy = sqrt(var_x * var_y)
    acf_x = zeros(Float64, length(lag_steps))
    acf_y = zeros(Float64, length(lag_steps))
    cross_xy = zeros(Float64, length(lag_steps))
    cross_yx = zeros(Float64, length(lag_steps))

    Threads.@threads for lag_idx in eachindex(lag_steps)
        lag = lag_steps[lag_idx]
        require_condition(lag < ntime, "Requested lag exceeds the available saved model trajectory length.")
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0
        sum_yx = 0.0
        count = 0
        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for i in 1:upper
                x0 = x[i, traj_idx] - mean_x
                y0 = y[i, traj_idx] - mean_y
                x1 = x[i + lag, traj_idx] - mean_x
                y1 = y[i + lag, traj_idx] - mean_y
                sum_xx += x1 * x0
                sum_yy += y1 * y0
                sum_xy += x1 * y0
                sum_yx += y1 * x0
            end
            count += upper
        end
        acf_x[lag_idx] = sum_xx / (count * var_x)
        acf_y[lag_idx] = sum_yy / (count * var_y)
        cross_xy[lag_idx] = sum_xy / (count * denom_xy)
        cross_yx[lag_idx] = sum_yx / (count * denom_xy)
    end

    lags = lag_steps .* saved_dt
    return CorrelationSummary(lags, acf_x, acf_y, cross_xy, cross_yx, estimate_decorrelation_time(lags, acf_x, acf_y, threshold))
end

function saved_coordinate_matrices(states::Array{Float64, 3}, stride::Int)
    indices = collect(1:stride:size(states, 1))
    ntime = length(indices)
    ntraj = size(states, 3)
    x = Matrix{Float64}(undef, ntime, ntraj)
    y = Matrix{Float64}(undef, ntime, ntraj)
    @inbounds for traj_idx in 1:ntraj
        for (local_idx, global_idx) in enumerate(indices)
            x[local_idx, traj_idx] = states[global_idx, 1, traj_idx]
            y[local_idx, traj_idx] = states[global_idx, 2, traj_idx]
        end
    end
    return indices, x, y
end

function correlation_comparison(true_states::Array{Float64, 3}, pred_states::Array{Float64, 3},
        saved_dt::Float64, stride::Int, max_time::Float64, threshold::Float64)
    _, true_x, true_y = saved_coordinate_matrices(true_states, stride)
    _, pred_x, pred_y = saved_coordinate_matrices(pred_states, stride)
    dt_corr = saved_dt * stride
    corr_true = compute_correlations(true_x, true_y, dt_corr, max_time, threshold)
    corr_pred = compute_correlations(pred_x, pred_y, dt_corr, max_time, threshold)
    metrics = Dict{Symbol, Float64}(
        :rmse_acf_x => rmse(corr_true.acf_x, corr_pred.acf_x),
        :rmse_acf_y => rmse(corr_true.acf_y, corr_pred.acf_y),
        :rmse_cross_xy => rmse(corr_true.cross_xy, corr_pred.cross_xy),
        :rmse_cross_yx => rmse(corr_true.cross_yx, corr_pred.cross_yx),
        :tdec_true => corr_true.t_decorrelation,
        :tdec_pred => corr_pred.t_decorrelation,
    )
    return Dict(:true => corr_true, :pred => corr_pred, :metrics => metrics)
end

function correlation_comparison_against_reference(reference::CorrelationSummary, pred_states::Array{Float64, 3},
        pred_saved_dt::Float64, max_time::Float64, threshold::Float64)
    _, pred_x, pred_y = saved_coordinate_matrices(pred_states, 1)
    max_available_time = pred_saved_dt * (size(pred_states, 1) - 1)
    max_compare_time = min(reference.lags[end], max_time, max_available_time)
    keep = findall(t -> t <= max_compare_time + 1.0e-12, reference.lags)
    require_condition(length(keep) >= 2, "No shared lag window remains for observed/model correlation comparison.")
    ref_lags = reference.lags[keep]
    ref_acf_x = reference.acf_x[keep]
    ref_acf_y = reference.acf_y[keep]
    ref_cross_xy = reference.cross_xy[keep]
    ref_cross_yx = reference.cross_yx[keep]
    truncated_reference = CorrelationSummary(
        ref_lags,
        ref_acf_x,
        ref_acf_y,
        ref_cross_xy,
        ref_cross_yx,
        estimate_decorrelation_time(ref_lags, ref_acf_x, ref_acf_y, threshold),
    )
    lag_steps = lag_steps_from_times(truncated_reference.lags, pred_saved_dt; allow_zero=true)
    corr_pred = compute_correlations_at_lag_steps(pred_x, pred_y, pred_saved_dt, lag_steps, threshold)
    metrics = Dict{Symbol, Float64}(
        :rmse_acf_x => rmse(truncated_reference.acf_x, corr_pred.acf_x),
        :rmse_acf_y => rmse(truncated_reference.acf_y, corr_pred.acf_y),
        :rmse_cross_xy => rmse(truncated_reference.cross_xy, corr_pred.cross_xy),
        :rmse_cross_yx => rmse(truncated_reference.cross_yx, corr_pred.cross_yx),
        :tdec_true => truncated_reference.t_decorrelation,
        :tdec_pred => corr_pred.t_decorrelation,
    )
    return Dict(:true => truncated_reference, :pred => corr_pred, :metrics => metrics)
end

function cphi_training_pairs(runtime::MobilityRuntime, artifact_data)
    if !isempty(runtime.training_pairs)
        return runtime.training_pairs
    end
    if haskey(artifact_data, :training_pairs)
        return Tuple{Int, Int}[(Int(pair[1]), Int(pair[2])) for pair in artifact_data[:training_pairs]]
    end
    return Tuple{Int, Int}[DEFAULT_CPHI_CHANNELS...]
end

function cphi_channel_labels(train_pairs::Vector{Tuple{Int, Int}}, artifact_labels::Vector{String})
    if length(artifact_labels) == length(train_pairs)
        return artifact_labels
    end
    labels = observable_labels()
    return [pair_channel_label("A", pair, labels) for pair in train_pairs]
end

function cphi_display_labels(train_pairs::Vector{Tuple{Int, Int}})
    labels = observable_labels()
    return ["C_{$(labels[m]),$(labels[n])}(t)" for (m, n) in train_pairs]
end

function artifact_lag_steps(artifact_data, saved_dt::Float64)
    require_condition(haskey(artifact_data, :lag_times), "Expected :lag_times in mobility artifact diagnostics.")
    lag_times = Float64.(vec(artifact_data[:lag_times]))
    lag_steps = lag_steps_from_times(lag_times, saved_dt; allow_zero=false)
    return lag_times, lag_steps
end

function fallback_lag_steps(max_time::Float64, saved_dt::Float64, stride::Int)
    max_step = floor(Int, max_time / saved_dt)
    lag_steps = collect(1:stride:max_step)
    require_condition(!isempty(lag_steps), "No fallback lag steps remain for Cphi analysis.")
    return lag_steps .* saved_dt, lag_steps
end

function compute_cphi_channels(states::Array{Float64, 3}, lag_steps::Vector{Int}, train_pairs::Vector{Tuple{Int, Int}}; start_idx::Int=1)
    obs_indices, local_pairs, _ = active_observable_subset(train_pairs)
    nchannels = length(train_pairs)
    out = Matrix{Float64}(undef, length(lag_steps), nchannels)
    for (lag_idx, lag) in enumerate(lag_steps)
        upper = size(states, 1) - lag
        require_condition(upper >= start_idx, "Lag exceeds the available post-burn-in trajectory window.")
        x0x = vec(@view states[start_idx:upper, 1, :])
        x0y = vec(@view states[start_idx:upper, 2, :])
        xtx = vec(@view states[(start_idx + lag):end, 1, :])
        xty = vec(@view states[(start_idx + lag):end, 2, :])
        obs0 = observable_basis(x0x, x0y, obs_indices)
        obst = observable_basis(xtx, xty, obs_indices)
        @inbounds for channel_idx in 1:nchannels
            obs_idx, coord_idx = local_pairs[channel_idx]
            source = @view obs0[coord_idx, :]
            channel_values = @view obst[obs_idx, :]
            out[lag_idx, channel_idx] = mean(channel_values .* source)
        end
    end
    return out
end

function cphi_comparison(true_states::Array{Float64, 3}, pred_states::Array{Float64, 3}, lag_times::Vector{Float64}, lag_steps::Vector{Int},
        train_pairs::Vector{Tuple{Int, Int}}, channel_labels::Vector{String})
    cphi_true = compute_cphi_channels(true_states, lag_steps, train_pairs)
    cphi_pred = compute_cphi_channels(pred_states, lag_steps, train_pairs)
    channel_rmse = [rmse(cphi_true[:, idx], cphi_pred[:, idx]) for idx in 1:size(cphi_true, 2)]
    return Dict(
        :lag_times => lag_times,
        :lag_steps => lag_steps,
        :training_pairs => [(pair[1], pair[2]) for pair in train_pairs],
        :channel_labels => copy(channel_labels),
        :true => cphi_true,
        :pred => cphi_pred,
        :channel_rmse => channel_rmse,
        :mean_rmse => mean(channel_rmse),
    )
end

function cphi_comparison_from_saved_dt(reference_states::Array{Float64, 3}, reference_start_idx::Int, reference_saved_dt::Float64,
        pred_states::Array{Float64, 3}, pred_saved_dt::Float64, lag_times::Vector{Float64},
        train_pairs::Vector{Tuple{Int, Int}}, channel_labels::Vector{String})
    reference_lag_steps = lag_steps_from_times(lag_times, reference_saved_dt; allow_zero=false)
    pred_lag_steps = lag_steps_from_times(lag_times, pred_saved_dt; allow_zero=false)
    cphi_ref = compute_cphi_channels(reference_states, reference_lag_steps, train_pairs; start_idx=reference_start_idx)
    cphi_pred = compute_cphi_channels(pred_states, pred_lag_steps, train_pairs)
    channel_rmse = [rmse(cphi_ref[:, idx], cphi_pred[:, idx]) for idx in 1:size(cphi_ref, 2)]
    return Dict(
        :lag_times => lag_times,
        :reference_lag_steps => reference_lag_steps,
        :pred_lag_steps => pred_lag_steps,
        :training_pairs => [(pair[1], pair[2]) for pair in train_pairs],
        :channel_labels => copy(channel_labels),
        :true => cphi_ref,
        :pred => cphi_pred,
        :channel_rmse => channel_rmse,
        :mean_rmse => mean(channel_rmse),
    )
end

function sample_state_columns(states::Array{Float64, 3}, max_samples::Int, rng::AbstractRNG)
    x, y = flatten_coordinates(states)
    total = length(x)
    if total <= max_samples
        return vcat(x', y')
    end
    keep = randperm(rng, total)[1:max_samples]
    return vcat(x[keep]', y[keep]')
end

function auxiliary_diagnostics(pred_states::Array{Float64, 3}, observed_states::Array{Float64, 3}, observed_start_idx::Int,
        score_runtime::ScoreRuntime, mobility_runtime::MobilityRuntime, divergence_grid::DivergenceGrid,
        xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64}, max_samples::Int)
    rng = MersenneTwister(20260421)
    sampled = sample_state_columns(pred_states, max_samples, rng)
    eval_points64, eval_points32, outside_eval = prepare_eval_points(sampled, xbounds, ybounds, true)
    score = evaluate_score_batch(score_runtime, eval_points32)
    mats = evaluate_mobility_batch(mobility_runtime, eval_points32)
    divergence = interpolate_divergence!(zeros(Float64, 2, size(sampled, 2)), divergence_grid, eval_points64)

    drift = Matrix{Float64}(undef, 2, size(sampled, 2))
    λmin = Vector{Float64}(undef, size(sampled, 2))
    @inbounds for idx in 1:size(sampled, 2)
        drift[1, idx] = mats[1, idx] * score[1, idx] + mats[2, idx] * score[2, idx] + divergence[1, idx]
        drift[2, idx] = mats[3, idx] * score[1, idx] + mats[4, idx] * score[2, idx] + divergence[2, idx]
        a = mats[1, idx]
        b = 0.5 * (mats[2, idx] + mats[3, idx])
        d = mats[4, idx]
        λ1, λ2 = diffusion_eigvals(a, b, d)
        λmin[idx] = min(λ1, λ2)
    end

    raw_x, raw_y = flatten_coordinates(pred_states)
    outside_saved = 0
    outside_magnitude = Float64[]
    @inbounds for idx in 1:length(raw_x)
        dx = max(0.0, xbounds[1] - raw_x[idx], raw_x[idx] - xbounds[2])
        dy = max(0.0, ybounds[1] - raw_y[idx], raw_y[idx] - ybounds[2])
        if dx > 0.0 || dy > 0.0
            outside_saved += 1
            push!(outside_magnitude, hypot(dx, dy))
        end
    end

    mobility_mean = vec(mean(mats; dims=2))
    mobility_std = vec(std(mats; dims=2))
    score_norm = sqrt.(vec(sum(abs2, score; dims=1)))
    div_norm = sqrt.(vec(sum(abs2, divergence; dims=1)))
    drift_norm = sqrt.(vec(sum(abs2, drift; dims=1)))
    observed_mean_m = mean_nn_mobility(mobility_runtime.host_model, observed_states, observed_start_idx,
        mobility_runtime.μ, mobility_runtime.σ, mobility_runtime.psd_jitter, mobility_runtime.batch_size, DeviceConfig(false, "CPU"))

    return Dict{Symbol, Any}(
        :sample_count => size(sampled, 2),
        :eval_clamped_fraction => outside_eval / size(sampled, 2),
        :saved_excursion_fraction => outside_saved / length(raw_x),
        :saved_excursion_max => isempty(outside_magnitude) ? 0.0 : maximum(outside_magnitude),
        :saved_excursion_mean => isempty(outside_magnitude) ? 0.0 : mean(outside_magnitude),
        :mobility_mean => mobility_mean,
        :mobility_std => mobility_std,
        :observed_mean_mobility => observed_mean_m,
        :lambda_min_min => minimum(λmin),
        :lambda_min_mean => mean(λmin),
        :lambda_min_q01 => quantile(λmin, 0.01),
        :score_norm_mean => mean(score_norm),
        :score_norm_std => std(score_norm),
        :score_norm_max => maximum(score_norm),
        :div_norm_mean => mean(div_norm),
        :div_norm_std => std(div_norm),
        :div_norm_max => maximum(div_norm),
        :drift_norm_mean => mean(drift_norm),
        :drift_norm_std => std(drift_norm),
        :drift_norm_max => maximum(drift_norm),
    )
end

function auxiliary_diagnostics_phi(pred_states::Array{Float64, 3}, score_runtime::ScoreRuntime, phi_runtime::PhiBaselineRuntime,
        xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64}, max_samples::Int)
    rng = MersenneTwister(20260422)
    sampled = sample_state_columns(pred_states, max_samples, rng)
    eval_points64, eval_points32, outside_eval = prepare_eval_points(sampled, xbounds, ybounds, true)
    score = evaluate_score_batch(score_runtime, eval_points32)
    drift = Matrix{Float64}(undef, 2, size(sampled, 2))
    div_norm = zeros(Float64, size(sampled, 2))
    score_norm = sqrt.(vec(sum(abs2, score; dims=1)))

    phi11 = phi_runtime.phi[1, 1]
    phi12 = phi_runtime.phi[1, 2]
    phi21 = phi_runtime.phi[2, 1]
    phi22 = phi_runtime.phi[2, 2]
    @inbounds for idx in 1:size(sampled, 2)
        s1 = score[1, idx]
        s2 = score[2, idx]
        drift[1, idx] = phi11 * s1 + phi12 * s2
        drift[2, idx] = phi21 * s1 + phi22 * s2
    end

    raw_x, raw_y = flatten_coordinates(pred_states)
    outside_saved = 0
    outside_magnitude = Float64[]
    @inbounds for idx in 1:length(raw_x)
        dx = max(0.0, xbounds[1] - raw_x[idx], raw_x[idx] - xbounds[2])
        dy = max(0.0, ybounds[1] - raw_y[idx], raw_y[idx] - ybounds[2])
        if dx > 0.0 || dy > 0.0
            outside_saved += 1
            push!(outside_magnitude, hypot(dx, dy))
        end
    end

    drift_norm = sqrt.(vec(sum(abs2, drift; dims=1)))
    sym = 0.5 .* (phi_runtime.phi .+ phi_runtime.phi')
    λ1, λ2 = diffusion_eigvals(sym[1, 1], sym[1, 2], sym[2, 2])
    λmin = min(λ1, λ2)
    return Dict{Symbol, Any}(
        :sample_count => size(sampled, 2),
        :eval_clamped_fraction => outside_eval / size(sampled, 2),
        :saved_excursion_fraction => outside_saved / length(raw_x),
        :saved_excursion_max => isempty(outside_magnitude) ? 0.0 : maximum(outside_magnitude),
        :saved_excursion_mean => isempty(outside_magnitude) ? 0.0 : mean(outside_magnitude),
        :mobility_mean => vec(phi_runtime.phi),
        :mobility_std => zeros(Float64, 4),
        :lambda_min_min => λmin,
        :lambda_min_mean => λmin,
        :lambda_min_q01 => λmin,
        :score_norm_mean => mean(score_norm),
        :score_norm_std => std(score_norm),
        :score_norm_max => maximum(score_norm),
        :div_norm_mean => mean(div_norm),
        :div_norm_std => std(div_norm),
        :div_norm_max => maximum(div_norm),
        :drift_norm_mean => mean(drift_norm),
        :drift_norm_std => std(drift_norm),
        :drift_norm_max => maximum(drift_norm),
    )
end

function rmse_improvement_percent(phi_rmse::Float64, full_rmse::Float64)
    return phi_rmse > 0.0 ? 100.0 * (phi_rmse - full_rmse) / phi_rmse : 0.0
end

function summary_text_panel(lines::Vector{String})
    panel = plot(; axis=nothing, framestyle=:none, xlim=(0, 1), ylim=(0, 1))
    y = 0.96
    for line in lines
        annotate!(panel, 0.02, y, text(line, 10, :left, :top, "DejaVu Sans"))
        y -= 0.08
    end
    return panel
end

function density_heatmap_panel(density::Density2D, title::String; clims=nothing, color=:viridis)
    kwargs = Dict{Symbol, Any}(
        :xlabel => "x",
        :ylabel => "y",
        :title => title,
        :aspect_ratio => :equal,
        :colorbar => true,
        :color => color,
    )
    if clims !== nothing
        kwargs[:clims] = clims
    end
    return heatmap(density.xgrid, density.ygrid, density.density'; kwargs...)
end

function create_reference_stats_figure(pdf_data, corr_data_full, corr_data_phi, aux_data_full::Dict{Symbol, Any},
        aux_data_phi::Dict{Symbol, Any}, output_path::AbstractString, width::Int, height::Int;
        reference_label::String, reference_title::String, extra_summary_lines::Vector{String}=String[])
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

    p1 = plot(pdf_true_x.centers, pdf_true_x.density; xlabel="x", ylabel="density", label=reference_label,
        color=:black, title=@sprintf("PDF x | RMSE %.3e | KL %.3e", pdf_metrics[:rmse_x], pdf_metrics[:kl_x]))
    plot!(p1, pdf_pred_x.centers, pdf_pred_x.density; label="learned", color=:royalblue3, linestyle=:dash)

    p2 = plot(pdf_true_y.centers, pdf_true_y.density; xlabel="y", ylabel="density", label=reference_label,
        color=:black, title=@sprintf("PDF y | RMSE %.3e | KL %.3e", pdf_metrics[:rmse_y], pdf_metrics[:kl_y]))
    plot!(p2, pdf_pred_y.centers, pdf_pred_y.density; label="learned", color=:royalblue3, linestyle=:dash)

    max_xy = maximum(vcat(vec(pdf_true_xy.density), vec(pdf_pred_xy.density)))
    p4 = density_heatmap_panel(pdf_true_xy, reference_title * " p(x,y)"; clims=(0.0, max_xy))
    p5 = density_heatmap_panel(pdf_pred_xy, "Learned p(x,y)"; clims=(0.0, max_xy))
    diff_xy = Density2D(pdf_true_xy.xgrid, pdf_true_xy.ygrid, pdf_pred_xy.density .- pdf_true_xy.density,
        pdf_true_xy.xboundary, pdf_true_xy.yboundary)
    diff_clim = maximum(abs.(diff_xy.density))
    p6 = density_heatmap_panel(diff_xy, @sprintf("Density Difference | RMSE %.3e | KL %.3e", pdf_metrics[:rmse_xy], pdf_metrics[:kl_xy]);
        clims=(-diff_clim, diff_clim), color=:balance)

    p7 = plot(corr_true.lags, corr_true.acf_x; xlabel="lag", ylabel="corr", label=reference_label,
        color=:black, title=@sprintf("ACF x | full %.3e | Phi %.3e", corr_metrics_full[:rmse_acf_x], corr_metrics_phi[:rmse_acf_x]))
    plot!(p7, corr_pred_full.lags, corr_pred_full.acf_x; label="learned full", color=:royalblue3, linestyle=:dash)
    plot!(p7, corr_pred_phi.lags, corr_pred_phi.acf_x; label="Phi-only", color=:firebrick3, linestyle=:dot)

    p8 = plot(corr_true.lags, corr_true.acf_y; xlabel="lag", ylabel="corr", label=reference_label,
        color=:black, title=@sprintf("ACF y | full %.3e | Phi %.3e", corr_metrics_full[:rmse_acf_y], corr_metrics_phi[:rmse_acf_y]))
    plot!(p8, corr_pred_full.lags, corr_pred_full.acf_y; label="learned full", color=:royalblue3, linestyle=:dash)
    plot!(p8, corr_pred_phi.lags, corr_pred_phi.acf_y; label="Phi-only", color=:firebrick3, linestyle=:dot)

    p9 = plot(corr_true.lags, corr_true.cross_xy; xlabel="lag", ylabel="corr", label=reference_label * " Cxy",
        color=:black, title=@sprintf("Cross Corr | full xy/yx %.3e / %.3e | Phi %.3e / %.3e",
            corr_metrics_full[:rmse_cross_xy], corr_metrics_full[:rmse_cross_yx],
            corr_metrics_phi[:rmse_cross_xy], corr_metrics_phi[:rmse_cross_yx]))
    plot!(p9, corr_pred_full.lags, corr_pred_full.cross_xy; label="learned full Cxy", color=:royalblue3, linestyle=:dash)
    plot!(p9, corr_pred_phi.lags, corr_pred_phi.cross_xy; label="Phi-only Cxy", color=:firebrick3, linestyle=:dot)
    plot!(p9, corr_true.lags, corr_true.cross_yx; label=reference_label * " Cyx", color=:darkorange)
    plot!(p9, corr_pred_full.lags, corr_pred_full.cross_yx; label="learned full Cyx", color=:seagreen4, linestyle=:dashdot)
    plot!(p9, corr_pred_phi.lags, corr_pred_phi.cross_yx; label="Phi-only Cyx", color=:purple4, linestyle=:dot)

    lines = [
        @sprintf("tD %s/full/Phi = %.3f / %.3f / %.3f", reference_label,
            corr_metrics_full[:tdec_true], corr_metrics_full[:tdec_pred], corr_metrics_phi[:tdec_pred]),
        @sprintf("ACF-x improvement over Phi = %.2f%%", rmse_improvement_percent(corr_metrics_phi[:rmse_acf_x], corr_metrics_full[:rmse_acf_x])),
        @sprintf("ACF-y improvement over Phi = %.2f%%", rmse_improvement_percent(corr_metrics_phi[:rmse_acf_y], corr_metrics_full[:rmse_acf_y])),
        @sprintf("cross-xy improvement over Phi = %.2f%%", rmse_improvement_percent(corr_metrics_phi[:rmse_cross_xy], corr_metrics_full[:rmse_cross_xy])),
        @sprintf("cross-yx improvement over Phi = %.2f%%", rmse_improvement_percent(corr_metrics_phi[:rmse_cross_yx], corr_metrics_full[:rmse_cross_yx])),
        @sprintf("excursion frac full/Phi = %.3e / %.3e", aux_data_full[:saved_excursion_fraction], aux_data_phi[:saved_excursion_fraction]),
        @sprintf("lambda_min(D) full/Phi = %.3e / %.3e", aux_data_full[:lambda_min_min], aux_data_phi[:lambda_min_min]),
        @sprintf("drift norm mean full/Phi = %.3e / %.3e", aux_data_full[:drift_norm_mean], aux_data_phi[:drift_norm_mean]),
    ]
    append!(lines, extra_summary_lines)
    push!(lines, @sprintf("<M> full = [%.3e, %.3e, %.3e, %.3e]", aux_data_full[:mobility_mean]...))
    push!(lines, @sprintf("<M> Phi = [%.3e, %.3e, %.3e, %.3e]", aux_data_phi[:mobility_mean]...))
    p3 = summary_text_panel(lines)

    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9; layout=(3, 3), size=(width, height), margin=6Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function create_cphi_figure(cphi_data_full, cphi_data_phi, labels::Vector{String}, output_path::AbstractString, width::Int, height::Int)
    plots = Any[]
    for idx in 1:length(labels)
        panel = plot(cphi_data_full[:lag_times], cphi_data_full[:true][:, idx]; xlabel="lag", ylabel="Cphi", label="true", color=:black,
            title=@sprintf("%s | full %.3e | Phi %.3e", labels[idx], cphi_data_full[:channel_rmse][idx], cphi_data_phi[:channel_rmse][idx]))
        plot!(panel, cphi_data_full[:lag_times], cphi_data_full[:pred][:, idx]; label="learned full", color=:royalblue3, linestyle=:dash)
        plot!(panel, cphi_data_phi[:lag_times], cphi_data_phi[:pred][:, idx]; label="Phi-only", color=:firebrick3, linestyle=:dot)
        push!(plots, panel)
    end
    nrows, ncols = panel_grid_dims(length(labels))
    fig = plot(plots...; layout=(nrows, ncols), size=(max(width, 1200 * ncols), max(height, 900 * nrows)), margin=6Plots.mm)
    savefig(fig, output_path)
    return nothing
end

function write_metrics_report(path::AbstractString, params::ForwardValidationParams, sim_stats_full::Dict{Symbol, Any},
        sim_stats_phi::Dict{Symbol, Any}, pdf_data_full, corr_data_full, corr_data_phi, cphi_data_full, cphi_data_phi,
        observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi, observed_cphi_data_full, observed_cphi_data_phi,
        aux_data_full::Dict{Symbol, Any}, aux_data_phi::Dict{Symbol, Any}, training_target_source::String,
        phi_matrix::Matrix{Float64}, phi_error::Union{Nothing, Float64})
    pdf_metrics = pdf_data_full[:metrics]
    corr_metrics_full = corr_data_full[:metrics]
    corr_metrics_phi = corr_data_phi[:metrics]
    observed_pdf_metrics = observed_pdf_data_full[:metrics]
    observed_corr_metrics_full = observed_corr_data_full[:metrics]
    observed_corr_metrics_phi = observed_corr_data_phi[:metrics]
    open(path, "w") do io
        println(io, "Forward validation of learned reduced Langevin model")
        println(io, @sprintf("training_target_source = %s", training_target_source))
        println(io, @sprintf("dt = %.6f", params.dt))
        println(io, @sprintf("save_stride = %d", params.save_stride))
        println(io, @sprintf("ntrajectories = %d", params.ntrajectories))
        println(io, @sprintf("total_time = %.6f", params.total_time))
        println(io, @sprintf("burnin_time = %.6f", params.burnin_time))
        println(io, @sprintf("saved_dt = %.6f", sim_stats_full[:saved_dt]))
        println(io, @sprintf("diffusion_cap = %.6e", sim_stats_full[:diffusion_cap]))
        println(io, @sprintf("phi_matrix = [%.6e %.6e; %.6e %.6e]", phi_matrix[1, 1], phi_matrix[1, 2], phi_matrix[2, 1], phi_matrix[2, 2]))
        println(io, @sprintf("eval_clamp_fraction = %.6e", sim_stats_full[:eval_clamp_fraction]))
        println(io, @sprintf("hard_clamp_fraction = %.6e", sim_stats_full[:hard_clamp_fraction]))
        println(io, @sprintf("diffusion_floor_fraction = %.6e", sim_stats_full[:diffusion_floor_fraction]))
        println(io, @sprintf("diffusion_cap_fraction = %.6e", sim_stats_full[:diffusion_cap_fraction]))
        println(io, @sprintf("phi_eval_clamp_fraction = %.6e", sim_stats_phi[:eval_clamp_fraction]))
        println(io, @sprintf("phi_hard_clamp_fraction = %.6e", sim_stats_phi[:hard_clamp_fraction]))
        println(io, @sprintf("phi_diffusion_floor_fraction = %.6e", sim_stats_phi[:diffusion_floor_fraction]))
        println(io, @sprintf("phi_diffusion_cap_fraction = %.6e", sim_stats_phi[:diffusion_cap_fraction]))
        println(io, @sprintf("pdf_rmse_x = %.6e", pdf_metrics[:rmse_x]))
        println(io, @sprintf("pdf_rmse_y = %.6e", pdf_metrics[:rmse_y]))
        println(io, @sprintf("pdf_rmse_xy = %.6e", pdf_metrics[:rmse_xy]))
        println(io, @sprintf("pdf_kl_x = %.6e", pdf_metrics[:kl_x]))
        println(io, @sprintf("pdf_kl_y = %.6e", pdf_metrics[:kl_y]))
        println(io, @sprintf("pdf_kl_xy = %.6e", pdf_metrics[:kl_xy]))
        println(io, @sprintf("acf_rmse_x = %.6e", corr_metrics_full[:rmse_acf_x]))
        println(io, @sprintf("acf_rmse_y = %.6e", corr_metrics_full[:rmse_acf_y]))
        println(io, @sprintf("cross_rmse_xy = %.6e", corr_metrics_full[:rmse_cross_xy]))
        println(io, @sprintf("cross_rmse_yx = %.6e", corr_metrics_full[:rmse_cross_yx]))
        println(io, @sprintf("tdecorrelation_true = %.6e", corr_metrics_full[:tdec_true]))
        println(io, @sprintf("tdecorrelation_pred = %.6e", corr_metrics_full[:tdec_pred]))
        println(io, @sprintf("phi_acf_rmse_x = %.6e", corr_metrics_phi[:rmse_acf_x]))
        println(io, @sprintf("phi_acf_rmse_y = %.6e", corr_metrics_phi[:rmse_acf_y]))
        println(io, @sprintf("phi_cross_rmse_xy = %.6e", corr_metrics_phi[:rmse_cross_xy]))
        println(io, @sprintf("phi_cross_rmse_yx = %.6e", corr_metrics_phi[:rmse_cross_yx]))
        println(io, @sprintf("phi_tdecorrelation_pred = %.6e", corr_metrics_phi[:tdec_pred]))
        println(io, @sprintf("cphi_mean_rmse = %.6e", cphi_data_full[:mean_rmse]))
        for (idx, value) in enumerate(cphi_data_full[:channel_rmse])
            println(io, @sprintf("cphi_channel_%d_rmse = %.6e", idx, value))
        end
        println(io, @sprintf("phi_cphi_mean_rmse = %.6e", cphi_data_phi[:mean_rmse]))
        for (idx, value) in enumerate(cphi_data_phi[:channel_rmse])
            println(io, @sprintf("phi_cphi_channel_%d_rmse = %.6e", idx, value))
        end
        println(io, @sprintf("full_over_phi_acf_x_improvement_pct = %.6f", rmse_improvement_percent(corr_metrics_phi[:rmse_acf_x], corr_metrics_full[:rmse_acf_x])))
        println(io, @sprintf("full_over_phi_acf_y_improvement_pct = %.6f", rmse_improvement_percent(corr_metrics_phi[:rmse_acf_y], corr_metrics_full[:rmse_acf_y])))
        println(io, @sprintf("full_over_phi_cross_xy_improvement_pct = %.6f", rmse_improvement_percent(corr_metrics_phi[:rmse_cross_xy], corr_metrics_full[:rmse_cross_xy])))
        println(io, @sprintf("full_over_phi_cross_yx_improvement_pct = %.6f", rmse_improvement_percent(corr_metrics_phi[:rmse_cross_yx], corr_metrics_full[:rmse_cross_yx])))
        println(io, @sprintf("full_over_phi_cphi_mean_improvement_pct = %.6f", rmse_improvement_percent(cphi_data_phi[:mean_rmse], cphi_data_full[:mean_rmse])))
        println(io, @sprintf("observed_pdf_rmse_x = %.6e", observed_pdf_metrics[:rmse_x]))
        println(io, @sprintf("observed_pdf_rmse_y = %.6e", observed_pdf_metrics[:rmse_y]))
        println(io, @sprintf("observed_pdf_rmse_xy = %.6e", observed_pdf_metrics[:rmse_xy]))
        println(io, @sprintf("observed_pdf_kl_x = %.6e", observed_pdf_metrics[:kl_x]))
        println(io, @sprintf("observed_pdf_kl_y = %.6e", observed_pdf_metrics[:kl_y]))
        println(io, @sprintf("observed_pdf_kl_xy = %.6e", observed_pdf_metrics[:kl_xy]))
        println(io, @sprintf("observed_acf_rmse_x = %.6e", observed_corr_metrics_full[:rmse_acf_x]))
        println(io, @sprintf("observed_acf_rmse_y = %.6e", observed_corr_metrics_full[:rmse_acf_y]))
        println(io, @sprintf("observed_cross_rmse_xy = %.6e", observed_corr_metrics_full[:rmse_cross_xy]))
        println(io, @sprintf("observed_cross_rmse_yx = %.6e", observed_corr_metrics_full[:rmse_cross_yx]))
        println(io, @sprintf("observed_tdecorrelation = %.6e", observed_corr_metrics_full[:tdec_true]))
        println(io, @sprintf("observed_pred_tdecorrelation = %.6e", observed_corr_metrics_full[:tdec_pred]))
        println(io, @sprintf("observed_phi_acf_rmse_x = %.6e", observed_corr_metrics_phi[:rmse_acf_x]))
        println(io, @sprintf("observed_phi_acf_rmse_y = %.6e", observed_corr_metrics_phi[:rmse_acf_y]))
        println(io, @sprintf("observed_phi_cross_rmse_xy = %.6e", observed_corr_metrics_phi[:rmse_cross_xy]))
        println(io, @sprintf("observed_phi_cross_rmse_yx = %.6e", observed_corr_metrics_phi[:rmse_cross_yx]))
        println(io, @sprintf("observed_phi_pred_tdecorrelation = %.6e", observed_corr_metrics_phi[:tdec_pred]))
        println(io, @sprintf("observed_cphi_mean_rmse = %.6e", observed_cphi_data_full[:mean_rmse]))
        for (idx, value) in enumerate(observed_cphi_data_full[:channel_rmse])
            println(io, @sprintf("observed_cphi_channel_%d_rmse = %.6e", idx, value))
        end
        println(io, @sprintf("observed_phi_cphi_mean_rmse = %.6e", observed_cphi_data_phi[:mean_rmse]))
        for (idx, value) in enumerate(observed_cphi_data_phi[:channel_rmse])
            println(io, @sprintf("observed_phi_cphi_channel_%d_rmse = %.6e", idx, value))
        end
        println(io, @sprintf("observed_full_over_phi_acf_x_improvement_pct = %.6f", rmse_improvement_percent(observed_corr_metrics_phi[:rmse_acf_x], observed_corr_metrics_full[:rmse_acf_x])))
        println(io, @sprintf("observed_full_over_phi_acf_y_improvement_pct = %.6f", rmse_improvement_percent(observed_corr_metrics_phi[:rmse_acf_y], observed_corr_metrics_full[:rmse_acf_y])))
        println(io, @sprintf("observed_full_over_phi_cross_xy_improvement_pct = %.6f", rmse_improvement_percent(observed_corr_metrics_phi[:rmse_cross_xy], observed_corr_metrics_full[:rmse_cross_xy])))
        println(io, @sprintf("observed_full_over_phi_cross_yx_improvement_pct = %.6f", rmse_improvement_percent(observed_corr_metrics_phi[:rmse_cross_yx], observed_corr_metrics_full[:rmse_cross_yx])))
        println(io, @sprintf("observed_full_over_phi_cphi_mean_improvement_pct = %.6f", rmse_improvement_percent(observed_cphi_data_phi[:mean_rmse], observed_cphi_data_full[:mean_rmse])))
        println(io, @sprintf("saved_excursion_fraction = %.6e", aux_data_full[:saved_excursion_fraction]))
        println(io, @sprintf("saved_excursion_max = %.6e", aux_data_full[:saved_excursion_max]))
        println(io, @sprintf("lambda_min_min = %.6e", aux_data_full[:lambda_min_min]))
        println(io, @sprintf("lambda_min_mean = %.6e", aux_data_full[:lambda_min_mean]))
        println(io, @sprintf("score_norm_mean = %.6e", aux_data_full[:score_norm_mean]))
        println(io, @sprintf("div_norm_mean = %.6e", aux_data_full[:div_norm_mean]))
        println(io, @sprintf("drift_norm_mean = %.6e", aux_data_full[:drift_norm_mean]))
        println(io, @sprintf("phi_saved_excursion_fraction = %.6e", aux_data_phi[:saved_excursion_fraction]))
        println(io, @sprintf("phi_saved_excursion_max = %.6e", aux_data_phi[:saved_excursion_max]))
        println(io, @sprintf("phi_lambda_min_min = %.6e", aux_data_phi[:lambda_min_min]))
        println(io, @sprintf("phi_lambda_min_mean = %.6e", aux_data_phi[:lambda_min_mean]))
        println(io, @sprintf("phi_score_norm_mean = %.6e", aux_data_phi[:score_norm_mean]))
        println(io, @sprintf("phi_div_norm_mean = %.6e", aux_data_phi[:div_norm_mean]))
        println(io, @sprintf("phi_drift_norm_mean = %.6e", aux_data_phi[:drift_norm_mean]))
        if phi_error !== nothing
            println(io, @sprintf("phi_consistency_error = %.6e", phi_error))
        end
    end
    return nothing
end

function save_trajectory_hdf5(path::AbstractString, times::Vector{Float64}, true_states::Array{Float64, 3},
        pred_states_full::Array{Float64, 3}, pred_states_phi::Array{Float64, 3},
        xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64})
    h5open(path, "w") do file
        write(file, "/time", times)
        write(file, "/true/states", true_states)
        write(file, "/predicted/states", pred_states_full)
        write(file, "/predicted_full/states", pred_states_full)
        write(file, "/predicted_phi/states", pred_states_phi)
        write(file, "/support/x_bounds", collect(xbounds))
        write(file, "/support/y_bounds", collect(ybounds))
    end
    return nothing
end

function diagnostics_dict(params::ForwardValidationParams, sim_stats_full::Dict{Symbol, Any}, sim_stats_phi::Dict{Symbol, Any},
        pdf_data_full, corr_data_full, corr_data_phi, cphi_data_full, cphi_data_phi,
        observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi, observed_cphi_data_full, observed_cphi_data_phi,
        aux_data_full::Dict{Symbol, Any}, aux_data_phi::Dict{Symbol, Any}, divergence_grid::DivergenceGrid,
        xbounds::Tuple{Float64, Float64}, ybounds::Tuple{Float64, Float64}, phi_est::Matrix{Float64},
        training_target_source::String, phi_error::Union{Nothing, Float64})
    return Dict{Symbol, Any}(
        :training_target_source => training_target_source,
        :phi_est => copy(phi_est),
        :phi_consistency_error => phi_error,
        :xbounds => collect(xbounds),
        :ybounds => collect(ybounds),
        :sim_stats => sim_stats_full,
        :sim_stats_phi => sim_stats_phi,
        :pdf_metrics => pdf_data_full[:metrics],
        :corr_metrics => corr_data_full[:metrics],
        :corr_metrics_phi => corr_data_phi[:metrics],
        :cphi_mean_rmse => cphi_data_full[:mean_rmse],
        :cphi_channel_rmse => cphi_data_full[:channel_rmse],
        :cphi_mean_rmse_phi => cphi_data_phi[:mean_rmse],
        :cphi_channel_rmse_phi => cphi_data_phi[:channel_rmse],
        :observed_pdf_metrics => observed_pdf_data_full[:metrics],
        :observed_corr_metrics => observed_corr_data_full[:metrics],
        :observed_corr_metrics_phi => observed_corr_data_phi[:metrics],
        :observed_cphi_mean_rmse => observed_cphi_data_full[:mean_rmse],
        :observed_cphi_channel_rmse => observed_cphi_data_full[:channel_rmse],
        :observed_cphi_mean_rmse_phi => observed_cphi_data_phi[:mean_rmse],
        :observed_cphi_channel_rmse_phi => observed_cphi_data_phi[:channel_rmse],
        :cphi_lag_times => cphi_data_full[:lag_times],
        :cphi_true => cphi_data_full[:true],
        :cphi_pred => cphi_data_full[:pred],
        :cphi_pred_phi => cphi_data_phi[:pred],
        :observed_cphi_true => observed_cphi_data_full[:true],
        :observed_cphi_pred => observed_cphi_data_full[:pred],
        :observed_cphi_pred_phi => observed_cphi_data_phi[:pred],
        :pdf_true_x_centers => pdf_data_full[:true_x].centers,
        :pdf_true_x_density => pdf_data_full[:true_x].density,
        :pdf_pred_x_density => pdf_data_full[:pred_x].density,
        :pdf_true_y_centers => pdf_data_full[:true_y].centers,
        :pdf_true_y_density => pdf_data_full[:true_y].density,
        :pdf_pred_y_density => pdf_data_full[:pred_y].density,
        :pdf_xy_xgrid => pdf_data_full[:true_xy].xgrid,
        :pdf_xy_ygrid => pdf_data_full[:true_xy].ygrid,
        :pdf_true_xy_density => pdf_data_full[:true_xy].density,
        :pdf_pred_xy_density => pdf_data_full[:pred_xy].density,
        :observed_pdf_x_centers => observed_pdf_data_full[:true_x].centers,
        :observed_pdf_x_density => observed_pdf_data_full[:true_x].density,
        :observed_pred_x_density => observed_pdf_data_full[:pred_x].density,
        :observed_pdf_y_centers => observed_pdf_data_full[:true_y].centers,
        :observed_pdf_y_density => observed_pdf_data_full[:true_y].density,
        :observed_pred_y_density => observed_pdf_data_full[:pred_y].density,
        :observed_pdf_xy_xgrid => observed_pdf_data_full[:true_xy].xgrid,
        :observed_pdf_xy_ygrid => observed_pdf_data_full[:true_xy].ygrid,
        :observed_pdf_xy_density => observed_pdf_data_full[:true_xy].density,
        :observed_pred_xy_density => observed_pdf_data_full[:pred_xy].density,
        :corr_lags => corr_data_full[:true].lags,
        :corr_true_acf_x => corr_data_full[:true].acf_x,
        :corr_true_acf_y => corr_data_full[:true].acf_y,
        :corr_true_cross_xy => corr_data_full[:true].cross_xy,
        :corr_true_cross_yx => corr_data_full[:true].cross_yx,
        :corr_pred_acf_x => corr_data_full[:pred].acf_x,
        :corr_pred_acf_y => corr_data_full[:pred].acf_y,
        :corr_pred_cross_xy => corr_data_full[:pred].cross_xy,
        :corr_pred_cross_yx => corr_data_full[:pred].cross_yx,
        :corr_pred_phi_acf_x => corr_data_phi[:pred].acf_x,
        :corr_pred_phi_acf_y => corr_data_phi[:pred].acf_y,
        :corr_pred_phi_cross_xy => corr_data_phi[:pred].cross_xy,
        :corr_pred_phi_cross_yx => corr_data_phi[:pred].cross_yx,
        :observed_corr_lags => observed_corr_data_full[:true].lags,
        :observed_corr_true_acf_x => observed_corr_data_full[:true].acf_x,
        :observed_corr_true_acf_y => observed_corr_data_full[:true].acf_y,
        :observed_corr_true_cross_xy => observed_corr_data_full[:true].cross_xy,
        :observed_corr_true_cross_yx => observed_corr_data_full[:true].cross_yx,
        :observed_corr_pred_acf_x => observed_corr_data_full[:pred].acf_x,
        :observed_corr_pred_acf_y => observed_corr_data_full[:pred].acf_y,
        :observed_corr_pred_cross_xy => observed_corr_data_full[:pred].cross_xy,
        :observed_corr_pred_cross_yx => observed_corr_data_full[:pred].cross_yx,
        :observed_corr_pred_phi_acf_x => observed_corr_data_phi[:pred].acf_x,
        :observed_corr_pred_phi_acf_y => observed_corr_data_phi[:pred].acf_y,
        :observed_corr_pred_phi_cross_xy => observed_corr_data_phi[:pred].cross_xy,
        :observed_corr_pred_phi_cross_yx => observed_corr_data_phi[:pred].cross_yx,
        :auxiliary => aux_data_full,
        :auxiliary_phi => aux_data_phi,
        :divergence_xgrid => divergence_grid.xgrid,
        :divergence_ygrid => divergence_grid.ygrid,
        :divergence_1 => divergence_grid.div1,
        :divergence_2 => divergence_grid.div2,
        :config => Dict(
            :dt => params.dt,
            :save_stride => params.save_stride,
            :total_time => params.total_time,
            :burnin_time => params.burnin_time,
            :ntrajectories => params.ntrajectories,
            :divergence_method => params.divergence_method,
            :pdf_bins => params.pdf_bins,
        ),
    )
end

function run_forward_validation(param_file::AbstractString)
    params = load_forward_validation_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    mobility_model_bson = resolve_path(base_dir, params.mobility_model_bson)
    mobility_artifact_bson = resolve_path(base_dir, params.mobility_artifact_bson)
    phi_bson = resolve_optional_path(base_dir, params.phi_bson)
    figure_stats_png = resolve_path(base_dir, params.figure_stats_png)
    figure_observed_png = resolve_optional_path(base_dir, params.figure_observed_png)
    figure_cphi_png = resolve_path(base_dir, params.figure_cphi_png)
    metrics_txt = resolve_path(base_dir, params.metrics_txt)
    diagnostics_bson = resolve_path(base_dir, params.diagnostics_bson)
    trajectories_hdf5 = resolve_path(base_dir, params.trajectories_hdf5)

    for path in (figure_stats_png, figure_cphi_png, metrics_txt, diagnostics_bson, trajectories_hdf5)
        ensure_parent_dir(path)
    end
    figure_observed_png === nothing || ensure_parent_dir(figure_observed_png)

    device = detect_device()
    @printf("Evaluation device: %s\n", device.name)
    times_obs, observed_states = load_state_tensor(input_hdf5)
    observed_stats = load_observed_statistics_reference(input_hdf5)
    observed_start_idx = burnin_start_index(length(times_obs), params.data_burnin_fraction)
    observed_saved_dt = length(times_obs) > 1 ? (times_obs[2] - times_obs[1]) : 0.0
    require_condition(observed_saved_dt > 0.0, "Observed data must contain at least two saved times.")
    xbounds, ybounds = state_domain(observed_states, observed_start_idx, params.support_pad_fraction)
    meta = load_affine_model_metadata(input_hdf5)

    score_runtime, _ = load_score_runtime(score_bson, params.eval_batch_size, device)
    mobility_runtime, mobility_model_data = load_mobility_runtime(mobility_model_bson, params.eval_batch_size, device, params.mobility_psd_jitter)
    mobility_artifact = BSON.load(mobility_artifact_bson)
    phi_override = load_phi_override(phi_bson)
    phi_error = phi_override === nothing ? nothing : norm(phi_override - mobility_runtime.phi_est)
    if phi_error !== nothing
        @printf("Phi consistency check: ||phi_override - phi_model|| = %.6e\n", phi_error)
    end

    divergence_grid = build_divergence_grid(mobility_runtime, xbounds, ybounds,
        params.divergence_method, params.divergence_grid_nx, params.divergence_grid_ny, params.divergence_fd_eps)

    diffusion_cap = diffusion_cap_from_observed(mobility_runtime, observed_states, observed_start_idx,
        params.diffusion_cap_quantile, params.diffusion_cap_multiplier)
    @printf("Predicted diffusion cap: %.6e\n", diffusion_cap)
    phi_runtime = build_phi_baseline_runtime(mobility_runtime.phi_est, params.diffusion_floor, diffusion_cap)

    rng = MersenneTwister(params.seed + 17)
    init_states = sample_initial_states(observed_states, observed_start_idx, params.ntrajectories, rng)

    times, true_states, pred_states_full, pred_states_phi, sim_stats_full, sim_stats_phi = integrate_validation_models(
        params, init_states, score_runtime, mobility_runtime, phi_runtime, divergence_grid, meta, xbounds, ybounds, diffusion_cap)

    saved_dt = sim_stats_full[:saved_dt]
    lag_times, lag_steps = params.cphi_use_artifact_lags ? artifact_lag_steps(mobility_artifact, saved_dt) :
        fallback_lag_steps(params.cphi_max_time, saved_dt, params.cphi_stride)
    cphi_pairs = cphi_training_pairs(mobility_runtime, mobility_artifact)
    cphi_artifact_labels = cphi_channel_labels(cphi_pairs, mobility_runtime.training_channel_labels)

    pdf_data_full = pdf_comparison(true_states, pred_states_full, params.pdf_bins, params.pdf_max_samples, MersenneTwister(params.seed + 31))
    corr_data_full = correlation_comparison(true_states, pred_states_full, saved_dt, params.correlation_stride,
        params.correlation_max_time, params.correlation_threshold)
    corr_data_phi = correlation_comparison(true_states, pred_states_phi, saved_dt, params.correlation_stride,
        params.correlation_max_time, params.correlation_threshold)
    cphi_data_full = cphi_comparison(true_states, pred_states_full, lag_times, lag_steps, cphi_pairs, cphi_artifact_labels)
    cphi_data_phi = cphi_comparison(true_states, pred_states_phi, lag_times, lag_steps, cphi_pairs, cphi_artifact_labels)
    observed_pdf_data_full = pdf_comparison_against_reference(observed_stats, pred_states_full, params.pdf_max_samples, MersenneTwister(params.seed + 41))
    observed_corr_data_full = correlation_comparison_against_reference(observed_stats.corr, pred_states_full, saved_dt,
        params.correlation_max_time, params.correlation_threshold)
    observed_corr_data_phi = correlation_comparison_against_reference(observed_stats.corr, pred_states_phi, saved_dt,
        params.correlation_max_time, params.correlation_threshold)
    observed_cphi_data_full = cphi_comparison_from_saved_dt(observed_states, observed_start_idx, observed_saved_dt,
        pred_states_full, saved_dt, lag_times, cphi_pairs, cphi_artifact_labels)
    observed_cphi_data_phi = cphi_comparison_from_saved_dt(observed_states, observed_start_idx, observed_saved_dt,
        pred_states_phi, saved_dt, lag_times, cphi_pairs, cphi_artifact_labels)
    aux_data_full = auxiliary_diagnostics(pred_states_full, observed_states, observed_start_idx, score_runtime, mobility_runtime,
        divergence_grid, xbounds, ybounds, params.auxiliary_max_samples)
    aux_data_phi = auxiliary_diagnostics_phi(pred_states_phi, score_runtime, phi_runtime,
        xbounds, ybounds, params.auxiliary_max_samples)

    labels = cphi_display_labels(cphi_pairs)
    create_reference_stats_figure(pdf_data_full, corr_data_full, corr_data_phi, aux_data_full, aux_data_phi,
        figure_stats_png, params.figure_width, params.figure_height;
        reference_label="true rollout", reference_title="True Rollout")
    if figure_observed_png !== nothing
        create_reference_stats_figure(observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi, aux_data_full, aux_data_phi,
            figure_observed_png, params.figure_width, params.figure_height;
            reference_label="observed", reference_title="Observed Data",
            extra_summary_lines=[
                @sprintf("obs Cphi RMSE full/Phi = %.3e / %.3e", observed_cphi_data_full[:mean_rmse], observed_cphi_data_phi[:mean_rmse]),
                @sprintf("obs Cphi improvement over Phi = %.2f%%", rmse_improvement_percent(observed_cphi_data_phi[:mean_rmse], observed_cphi_data_full[:mean_rmse])),
            ])
    end
    create_cphi_figure(cphi_data_full, cphi_data_phi, labels, figure_cphi_png, params.figure_width, max(params.figure_height - 200, 1400))
    write_metrics_report(metrics_txt, params, sim_stats_full, sim_stats_phi, pdf_data_full, corr_data_full, corr_data_phi,
        cphi_data_full, cphi_data_phi, observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi,
        observed_cphi_data_full, observed_cphi_data_phi, aux_data_full, aux_data_phi,
        mobility_runtime.training_target_source, mobility_runtime.phi_est, phi_error)
    save_trajectory_hdf5(trajectories_hdf5, times, true_states, pred_states_full, pred_states_phi, xbounds, ybounds)

    BSON.bson(diagnostics_bson, diagnostics_dict(params, sim_stats_full, sim_stats_phi, pdf_data_full, corr_data_full, corr_data_phi,
        cphi_data_full, cphi_data_phi, observed_pdf_data_full, observed_corr_data_full, observed_corr_data_phi,
        observed_cphi_data_full, observed_cphi_data_phi, aux_data_full, aux_data_phi,
        divergence_grid, xbounds, ybounds, mobility_runtime.phi_est, mobility_runtime.training_target_source, phi_error))

    @printf("Saved basic diagnostics figure to %s\n", figure_stats_png)
    if figure_observed_png !== nothing
        @printf("Saved observed-data comparison figure to %s\n", figure_observed_png)
    end
    @printf("Saved Cphi comparison figure to %s\n", figure_cphi_png)
    @printf("Saved metrics report to %s\n", metrics_txt)
    @printf("Saved diagnostics bundle to %s\n", diagnostics_bson)
    @printf("Saved validation trajectories to %s\n", trajectories_hdf5)
    @printf("Forward validation complete. Mean Cphi RMSE full/Phi = %.6e / %.6e\n", cphi_data_full[:mean_rmse], cphi_data_phi[:mean_rmse])
    return nothing
end
