#!/usr/bin/env julia

include(joinpath(@__DIR__, "render_forward_with_dM.jl"))

using Printf
using LinearAlgebra

const DEFAULT_GRID_TAG = "scale_grid"

function parse_model_specs(base::AbstractString, specs::Vector{String})
    if isempty(specs)
        return [
            ("Phi", resolve_path(base, "../data/phi_forward_langevin.h5"), STYLE_PRIMARY, :dash),
            ("M_NN A x1.0", resolve_path(base, "../data/forward_dM_gpu0_vA.h5"), STYLE_ACCENT, :solid),
            ("M_NN A x1.5", resolve_path(base, "../data/forward_dM_A_scale1p5.h5"), STYLE_HIGHLIGHT, :dashdot),
            ("M_NN A x2.0", resolve_path(base, "../data/forward_dM_A_scale2p0.h5"), STYLE_SECONDARY, :dot),
            ("M_NN A x2.5", resolve_path(base, "../data/forward_dM_A_scale2p5.h5"), :mediumpurple4, :dash),
        ]
    end
    colors = Any[STYLE_PRIMARY, STYLE_ACCENT, STYLE_HIGHLIGHT, STYLE_SECONDARY,
        :mediumpurple4, :brown4, :deepskyblue4, :gray35]
    styles = Any[:dash, :solid, :dashdot, :dot, :dash, :solid, :dashdot, :dot]
    parsed = []
    for (k, spec) in enumerate(specs)
        parts = split(spec, "=", limit=2)
        require_condition(length(parts) == 2,
            "Model specs must have form label=path; got $(spec).")
        label = strip(parts[1])
        path = resolve_path(base, strip(parts[2]))
        push!(parsed, (label, path, colors[1 + mod(k - 1, length(colors))],
            styles[1 + mod(k - 1, length(styles))]))
    end
    return parsed
end

function finite_state_check(label::AbstractString, states::Array{Float32, 4})
    frac_bad = count(!isfinite, states) / length(states)
    max_abs = maximum(abs, states)
    require_condition(frac_bad == 0, "$(label) contains non-finite state values.")
    return max_abs
end

function fast_coordinate_correlations(states::Array{Float32, 4}, save_dt::Float64, max_lags::Int)
    nt, N, _, ntraj = size(states)
    D = 3N
    L = min(max_lags, nt - 1)
    flat = Array{Float64}(undef, D, nt, ntraj)
    @inbounds for tr in 1:ntraj, t in 1:nt
        flat[:, t, tr] .= flatten_state(@view states[t, :, :, tr])
    end
    mu = zeros(Float64, D)
    @inbounds for tr in 1:ntraj, t in 1:nt
        mu .+= @view flat[:, t, tr]
    end
    mu ./= nt * ntraj
    @inbounds for tr in 1:ntraj, t in 1:nt
        @views flat[:, t, tr] .-= mu
    end

    C = Array{Float64}(undef, L + 1, D, D)
    accum = zeros(Float64, D, D)
    for lag in 0:L
        fill!(accum, 0.0)
        @inbounds for tr in 1:ntraj
            X0 = @view flat[:, 1:(nt - lag), tr]
            Xt = @view flat[:, (1 + lag):nt, tr]
            mul!(accum, Xt, transpose(X0), 1.0, 1.0)
        end
        C[lag + 1, :, :] .= accum ./ ((nt - lag) * ntraj)
    end
    return collect(0:L) .* save_dt, C
end

function run_forward_grid(phi_config::AbstractString, tag::AbstractString, specs::Vector{String})
    base = dirname(phi_config)
    params = load_config(phi_config)
    data_h5 = resolve_path(base, params.input_hdf5)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)

    models = ForwardModelStates[]
    max_abs = Dict{String, Float64}()
    for (label, path, color, linestyle) in parse_model_specs(base, specs)
        require_condition(isfile(path), "Missing forward HDF5 for $(label): $(path)")
        states, time = load_forward_h5(path)
        max_abs[label] = finite_state_check(label, states)
        push!(models, ForwardModelStates(label, states, time, color, linestyle))
    end
    require_condition(!isempty(models), "No forward models were provided.")

    stats_path = resolve_path(base, "../figures/forward_stats_$(tag).png")
    obs_stats = @view sampler.states[sampler.start_idx:end, :, :, :]
    cov_obs, cov_models = render_stats_with_dm(stats_path, params, obs_stats, models;
        obs_save_dt=sampler.save_dt)

    obs = sampler.states[sampler.start_idx:end, :, :, :]
    aligned = models
    obs_nt, _, _, obs_ntraj = size(obs)

    cmn_path = resolve_path(base, "../figures/forward_cmn_$(tag).png")
    metrics_path = resolve_path(base, "../logs/forward_$(tag)_metrics.txt")

    corr_lags = min(DEFAULT_FORWARD_CORR_MAX_LAGS, obs_nt - 1,
        minimum(size(model.states, 1) - 1 for model in aligned))
    p = load_phys(data_h5)
    channel_path = resolve_path(base, DEFAULT_FORWARD_CMN_CHANNELS_TOML)
    channels, lib = load_forward_cmn_channels(channel_path)
    channel_labels = channel_label.(channels)
    obs_t, Cobs = retained_channel_correlations(obs, sampler.save_dt, p,
        channels, lib, corr_lags; seed=params.seed + 9000, label="obs")
    model_corrs = []
    for (model_index, model) in enumerate(aligned)
        dt = length(model.time) > 1 ? model.time[2] - model.time[1] : params.forward_save_dt
        t, C = retained_channel_correlations(model.states, dt, p, channels, lib,
            min(DEFAULT_FORWARD_CORR_MAX_LAGS, size(model.states, 1) - 1);
            seed=params.seed + 10_000 + 101 * model_index, label=model.label)
        push!(model_corrs, (model.label, t, C, model.color, model.linestyle))
        GC.gc()
    end
    render_cmn_with_dm(cmn_path, params, obs_t, Cobs, model_corrs;
        channel_labels=channel_labels,
        correlation_kind="retained observable C")
    write_forward_metrics(metrics_path, cov_obs, cov_models, [m.label for m in aligned],
        Cobs, model_corrs; correlation_kind="retained observable C")
    open(metrics_path, "a") do io
        println(io)
        println(io, "Stationary statistics sampling:")
        println(io, @sprintf("PDF sample cap per curve = %d state-site samples", DEFAULT_PDF_SAMPLE_COUNT))
        println(io, @sprintf("Covariance sample cap per curve = %d time-trajectory samples", DEFAULT_COV_SAMPLE_COUNT))
        println(io, "Observation PDFs/covariance use all post-burn observation trajectories.")
        println(io, "Model PDFs/covariance use all saved forward trajectories.")
        println(io)
        println(io, "Finite-state checks:")
        for model in aligned
            println(io, @sprintf("%s max |state| = %.8e", model.label, max_abs[model.label]))
        end
        println(io)
        println(io, "Correlation sampling:")
        println(io, @sprintf("obs_nt = %d, obs_ntraj = %d, save_dt = %.8e, corr_lags = %d",
            obs_nt, obs_ntraj, sampler.save_dt, corr_lags))
        for model in aligned
            println(io, @sprintf("%s nt = %d, ntraj = %d",
                model.label, size(model.states, 1), size(model.states, 4)))
        end
        println(io, @sprintf("corr_tmax = %.8e", corr_lags * sampler.save_dt))
        println(io)
        println(io, "Retained observable correlations:")
        channel_relpath = relpath(channel_path, normpath(joinpath(@__DIR__, "..")))
        println(io, "channels_toml = $(channel_relpath)")
        println(io, @sprintf("channels = %d", length(channels)))
        println(io, @sprintf("pairs_per_lag = %d time-trajectory pairs, all lattice sites", DEFAULT_FORWARD_OBSERVABLE_CORR_PAIRS))
    end
    @printf("Saved forward grid figures:\n  %s\n  %s\n", stats_path, cmn_path)
    @printf("Saved metrics to %s\n", metrics_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    phi_config = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PHI_CONFIG
    tag = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_GRID_TAG
    specs = length(ARGS) >= 3 ? ARGS[3:end] : String[]
    run_forward_grid(phi_config, tag, specs)
end
