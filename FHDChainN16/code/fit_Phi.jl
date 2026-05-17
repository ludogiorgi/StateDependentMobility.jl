#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "fhd_learning_common.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "fit_Phi.toml"))

Base.@kwdef struct FHDFitParams
    input_hdf5::String
    score_bson::String
    cond_score_bson::String
    burnin_fraction::Float64
    tau_min::Float64
    tau_max_decorrelation_multiples::Float64
    lag_stride::Int
    max_fit_lags::Int
    pairs_per_lag_phi::Int
    pairs_per_lag_cdot::Int
    phi_fit_max_lag::Int
    phi_fit_degree::Int
    phi_include_zero_lag::Bool
    phi_zero_lag_samples::Int
    phi_use_stein_correction::Bool
    phi_project_symmetric_psd::Bool
    stein_samples::Int
    true_mobility_samples::Int
    score_batch_size::Int
    train_mobility::Bool
    run_forward_validation::Bool
    mobility_pairs_per_lag::Int
    mobility_epochs::Int
    mobility_learning_rate::Float64
    mobility_mean_penalty::Float64
    mobility_weight_decay::Float64
    mobility_batch_profiles::Int
    mobility_hidden_width::Int
    mobility_cache_bson::String
    forward_dt::Float64
    forward_total_time::Float64
    forward_burnin_time::Float64
    forward_save_dt::Float64
    forward_ntraj::Int
    forward_score_clip::Float32
    n_plot_panels::Int
    figure_width::Int
    figure_height::Int
    artifact_bson::String
    metrics_txt::String
    cdot_figure_png::String
    phi_figure_png::String
    mobility_training_png::String
    mobility_comparison_png::String
    forward_stats_png::String
    forward_cmn_png::String
    forward_hdf5::String
    mobility_bson::String
    device_name::String
    required_gpu_name::String
    seed::Int
    verbose::Bool
end

const PHI_PROJECTION_MODES = IdDict{Any, String}()

function phi_projection_mode(params::FHDFitParams)
    return get(PHI_PROJECTION_MODES, params,
        params.phi_project_symmetric_psd ? "block_circulant_psd" : "block_circulant")
end

Base.@kwdef mutable struct FHDMobilityHistory
    epochs::Vector{Int} = Int[]
    losses::Vector{Float64} = Float64[]
    validation_rmse::Vector{Float64} = Float64[]
    validation_corr::Vector{Float64} = Float64[]
    mean_penalty::Vector{Float64} = Float64[]
    rms_delta::Vector{Float64} = Float64[]
end

struct FHDMobilityTrainingCache
    features::Array{Float32, 4}     # feature, edge/site, pair, lag
    scond::Array{Float32, 4}        # site, component, pair, lag
    observables::Array{Float32, 4}  # site, observable, pair, lag
    mean_features::Matrix{Float32}
    eta_bar::Float32
    skew_scale::Float32
end

struct FHDObservableLibrary
    names::Vector{String}
    means::Vector{Float64}
end

function load_params(path::AbstractString)
    raw = TOML.parsefile(path)
    data = raw["data"]
    eval = raw["evaluation"]
    figure = raw["figure"]
    output = raw["output"]
    run = get(raw, "run", Dict{String, Any}())
    params = FHDFitParams(
        input_hdf5=String(data["input_hdf5"]),
        score_bson=String(data["score_bson"]),
        cond_score_bson=String(get(data, "cond_score_bson", "")),
        burnin_fraction=Float64(get(data, "burnin_fraction", 0.1)),
        tau_min=Float64(get(data, "tau_min", 0.0)),
        tau_max_decorrelation_multiples=Float64(get(data, "tau_max_decorrelation_multiples", 0.67)),
        lag_stride=Int(get(data, "lag_stride", 1)),
        max_fit_lags=Int(get(eval, "max_fit_lags", 60)),
        pairs_per_lag_phi=Int(get(eval, "pairs_per_lag_phi", 250000)),
        pairs_per_lag_cdot=Int(get(eval, "pairs_per_lag_cdot", 120000)),
        phi_fit_max_lag=Int(get(eval, "phi_fit_max_lag", 8)),
        phi_fit_degree=Int(get(eval, "phi_fit_degree", 2)),
        phi_include_zero_lag=Bool(get(eval, "phi_include_zero_lag", false)),
        phi_zero_lag_samples=Int(get(eval, "phi_zero_lag_samples", get(eval, "pairs_per_lag_phi", 250000))),
        phi_use_stein_correction=Bool(get(eval, "phi_use_stein_correction", true)),
        phi_project_symmetric_psd=Bool(get(eval, "phi_project_symmetric_psd", false)),
        stein_samples=Int(get(eval, "stein_samples", 150000)),
        true_mobility_samples=Int(get(eval, "true_mobility_samples", 60000)),
        score_batch_size=Int(get(eval, "score_batch_size", 4096)),
        train_mobility=Bool(get(get(raw, "mobility", Dict{String, Any}()), "train", true)),
        run_forward_validation=Bool(get(get(raw, "forward", Dict{String, Any}()), "run", true)),
        mobility_pairs_per_lag=Int(get(get(raw, "mobility", Dict{String, Any}()), "pairs_per_lag", 2048)),
        mobility_epochs=Int(get(get(raw, "mobility", Dict{String, Any}()), "epochs", 700)),
        mobility_learning_rate=Float64(get(get(raw, "mobility", Dict{String, Any}()), "learning_rate", 5.0e-4)),
        mobility_mean_penalty=Float64(get(get(raw, "mobility", Dict{String, Any}()), "mean_penalty", 1.0e-2)),
        mobility_weight_decay=Float64(get(get(raw, "mobility", Dict{String, Any}()), "weight_decay", 1.0e-7)),
        mobility_batch_profiles=Int(get(get(raw, "mobility", Dict{String, Any}()), "profile_batch_size", 768)),
        mobility_hidden_width=Int(get(get(raw, "mobility", Dict{String, Any}()), "hidden_width", 128)),
        mobility_cache_bson=String(get(get(raw, "mobility", Dict{String, Any}()), "cache_bson", "outputs/fit_dM_mobility_cache.bson")),
        forward_dt=Float64(get(get(raw, "forward", Dict{String, Any}()), "dt", 0.004)),
        forward_total_time=Float64(get(get(raw, "forward", Dict{String, Any}()), "total_time", 36.0)),
        forward_burnin_time=Float64(get(get(raw, "forward", Dict{String, Any}()), "burnin_time", 6.0)),
        forward_save_dt=Float64(get(get(raw, "forward", Dict{String, Any}()), "save_dt", 0.04)),
        forward_ntraj=Int(get(get(raw, "forward", Dict{String, Any}()), "ntrajectories", 96)),
        forward_score_clip=Float32(get(get(raw, "forward", Dict{String, Any}()), "score_clip", 80.0)),
        n_plot_panels=Int(get(figure, "n_plot_panels", 16)),
        figure_width=Int(get(figure, "width", 3600)),
        figure_height=Int(get(figure, "height", 3000)),
        artifact_bson=String(output["artifact_bson"]),
        metrics_txt=String(output["metrics_txt"]),
        cdot_figure_png=String(output["cdot_figure_png"]),
        phi_figure_png=String(output["phi_figure_png"]),
        mobility_training_png=String(get(output, "mobility_training_png", "outputs/fit_dM_mobility_training.png")),
        mobility_comparison_png=String(get(output, "mobility_comparison_png", "outputs/fit_dM_mobility_comparison.png")),
        forward_stats_png=String(get(output, "forward_stats_png", "outputs/fit_dM_forward_validation_stats.png")),
        forward_cmn_png=String(get(output, "forward_cmn_png", "outputs/fit_dM_forward_validation_cmn.png")),
        forward_hdf5=String(get(output, "forward_hdf5", "outputs/fit_dM_forward_trajectories.h5")),
        mobility_bson=String(get(output, "mobility_bson", "outputs/fit_dM_mobility_nn.bson")),
        device_name=String(get(run, "device", "GPU:1")),
        required_gpu_name=String(get(run, "required_gpu_name", "5070")),
        seed=Int(get(run, "seed", 20260622)),
        verbose=Bool(get(run, "verbose", true)),
    )
    PHI_PROJECTION_MODES[params] = String(get(eval, "phi_projection",
        params.phi_project_symmetric_psd ? "block_circulant_psd" : "block_circulant"))
    return params
end

function load_score_checkpoint(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    stats = load_stats_from_bson(blob)
    trainer_cfg = dict_get(blob, :trainer_cfg)
    return model, stats, Float32(trainer_cfg.sigma), blob
end

function load_cond_checkpoint(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(dict_get(blob, :host_model), device)
    Flux.testmode!(model)
    trainer_cfg = dict_get(blob, :trainer_cfg)
    metadata = dict_get(blob, :metadata)
    return model, Float32(trainer_cfg.sigma), metadata, blob
end

function observable_values(z::Array{Float32, 3}, phys::FHDPhysicalParams)
    K, _, B = size(z)
    nobs = 14
    obs = Array{Float32}(undef, K, nobs, B)
    h = Float32(phys.dx)
    @inbounds for b in 1:B
        rho = @view z[:, 1, b]
        m = @view z[:, 2, b]
        eta = edge_viscosity(rho, phys)
        for i in 1:K
            im = periodic(i - 1, K)
            ip = periodic(i + 1, K)
            rho_i = Float64(rho[i])
            m_i = Float64(m[i])
            u_i = m_i / max(rho_i, phys.velocity_density_floor)
            eta_cell = 0.5 * (eta[i] + eta[im])
            obs[i, 1, b] = Float32(rho_i - phys.rho0)
            obs[i, 2, b] = Float32(m_i)
            obs[i, 3, b] = Float32(u_i)
            obs[i, 4, b] = (rho[ip] - rho[im]) / (2h)
            obs[i, 5, b] = (m[ip] - m[im]) / (2h)
            obs[i, 6, b] = (rho[ip] - 2f0 * rho[i] + rho[im]) / (h * h)
            obs[i, 7, b] = (m[ip] - 2f0 * m[i] + m[im]) / (h * h)
            obs[i, 8, b] = Float32(eta_cell)
            obs[i, 9, b] = Float32((rho_i - phys.rho0) * m_i)
            obs[i, 10, b] = Float32((rho_i - phys.rho0)^2)
            obs[i, 11, b] = Float32(m_i^2)
            obs[i, 12, b] = Float32((rho_i - phys.rho0) * u_i)
            obs[i, 13, b] = Float32(m_i * u_i + phys.cs^2 * rho_i)
            obs[i, 14, b] = Float32(eta_cell * (Float64(m[ip]) / max(Float64(rho[ip]), phys.velocity_density_floor) -
                Float64(m[im]) / max(Float64(rho[im]), phys.velocity_density_floor)) / (2.0 * phys.dx))
        end
    end
    return obs
end

function observable_names()
    return ["rho-rho0", "m", "u", "grad rho", "grad m", "lap rho", "lap m",
        "eta cell", "(rho-rho0)m", "(rho-rho0)^2", "m^2", "(rho-rho0)u",
        "momentum flux", "eta grad u"]
end

function estimate_observable_means(sampler::FHDPairSampler, phys::FHDPhysicalParams;
        nsamples::Int, seed::Int)
    rng = MersenneTwister(seed)
    raw = sample_state_tensor(sampler.states, sampler.start_idx, nsamples, rng)
    obs = observable_values(raw, phys)
    nobs = size(obs, 2)
    means = Vector{Float64}(undef, nobs)
    for a in 1:nobs
        means[a] = mean(Float64, @view obs[:, a, :])
    end
    return FHDObservableLibrary(observable_names(), means)
end

function center_observables!(obs::Array{Float32, 3}, lib::FHDObservableLibrary)
    @inbounds for a in 1:length(lib.names)
        obs[:, a, :] .-= Float32(lib.means[a])
    end
    return obs
end

function estimate_data_cdot(sampler::FHDPairSampler, phys::FHDPhysicalParams,
        lib::FHDObservableLibrary, params::FHDFitParams, stats::DataStats)
    lags = sampler.lag_steps[1:min(params.max_fit_lags, length(sampler.lag_steps))]
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cdot = Array{Float64}(undef, length(lags), K, nobs, D)
    rng = MersenneTwister(params.seed + 10)
    mu = Float64.(mean_flat(stats))
    for (li, lag) in enumerate(lags)
        x0, _, xp, xm = random_lag_pairs(sampler, lag, params.pairs_per_lag_cdot, rng;
            centered_window=1)
        x0f = Float64.(raw_flat_batch(x0))
        x0f .-= mu
        obs_p = observable_values(xp, phys)
        obs_m = observable_values(xm, phys)
        deriv = (obs_p .- obs_m) ./ Float32(2.0 * sampler.save_dt)
        deriv_flat = reshape(deriv, K * nobs, params.pairs_per_lag_cdot)
        mat = Matrix{Float64}(deriv_flat) * transpose(x0f) ./ params.pairs_per_lag_cdot
        Cdot[li, :, :, :] .= reshape(mat, K, nobs, D)
        params.verbose && @printf("Data finite-difference Cdot tau %.5g (%d/%d)\n",
            sampler.lag_times[li], li, length(lags))
    end
    return lags, sampler.lag_times[1:length(lags)], Cdot
end

function phi_transpose_action(Phi::AbstractMatrix{<:Real}, score_flat::Matrix{Float64})
    return transpose(Matrix{Float64}(Phi)) * score_flat
end

function estimate_conditional_cdot(sampler::FHDPairSampler, phys::FHDPhysicalParams,
        lib::FHDObservableLibrary, params::FHDFitParams, stats::DataStats,
        cond_model, cond_sigma::Float32, cond_meta, device::ExecutionDevice,
        Phi::Matrix{Float64}, score_model, score_sigma::Float32)
    lags = sampler.lag_steps[1:min(params.max_fit_lags, length(sampler.lag_steps))]
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Ctrue = Array{Float64}(undef, length(lags), K, nobs, D)
    Cphi = similar(Ctrue)
    rng = MersenneTwister(params.seed + 20)
    time_features = String(cond_meta[:time_features])
    nfreq = Int(cond_meta[:time_fourier_frequencies])
    include_delta = Bool(cond_meta[:include_delta_input])
    stdv = Float64.(std_flat(stats))
    for (li, lag) in enumerate(lags)
        x0, xt, _, _ = random_lag_pairs(sampler, lag, params.pairs_per_lag_cdot, rng)
        trans_norm = evaluate_transition_score_norm(cond_model, x0, xt, sampler.lag_tnorm[li],
            stats, cond_sigma, device; batch_size=params.score_batch_size,
            time_features=time_features, time_fourier_frequencies=nfreq,
            include_delta_input=include_delta)
        rflat = Float64.(raw_flat_batch(trans_norm))
        rflat ./= reshape(stdv, :, 1)
        obs = center_observables!(observable_values(xt, phys), lib)
        obs_flat = Matrix{Float64}(reshape(obs, K * nobs, params.pairs_per_lag_cdot))
        action_true = Matrix{Float64}(undef, D, params.pairs_per_lag_cdot)
        @inbounds for b in 1:params.pairs_per_lag_cdot
            action_true[:, b] .= true_mobility_transpose_action_sample(
                @view(x0[:, 1, b]), @view(x0[:, 2, b]), @view(rflat[:, b]), phys)
        end
        stationary_score = raw_stationary_score_flat(score_model, Float64.(x0), stats,
            score_sigma, params, device)
        action_phi_stationary = phi_transpose_action(Phi, stationary_score)
        Ctrue[li, :, :, :] .= reshape(-(obs_flat * transpose(action_true)) ./ params.pairs_per_lag_cdot,
            K, nobs, D)
        Cphi[li, :, :, :] .= reshape((obs_flat * transpose(action_phi_stationary)) ./ params.pairs_per_lag_cdot,
            K, nobs, D)
        params.verbose && @printf("Cdot trueM(cond) and Phi(stationary score) tau %.5g (%d/%d)\n",
            sampler.lag_times[li], li, length(lags))
    end
    return Ctrue, Cphi
end

function estimate_phi_gfdt_cdot(sampler::FHDPairSampler, phys::FHDPhysicalParams,
        lib::FHDObservableLibrary, params::FHDFitParams, stats::DataStats,
        Phi::Matrix{Float64}, score_model, score_sigma::Float32,
        device::ExecutionDevice)
    lags = sampler.lag_steps[1:min(params.max_fit_lags, length(sampler.lag_steps))]
    K = sampler.K
    D = sampler.D
    nobs = length(lib.names)
    Cphi = Array{Float64}(undef, length(lags), K, nobs, D)
    rng = MersenneTwister(params.seed + 20)
    for (li, lag) in enumerate(lags)
        x0, xt, _, _ = random_lag_pairs(sampler, lag, params.pairs_per_lag_cdot, rng)
        obs = center_observables!(observable_values(xt, phys), lib)
        obs_flat = Matrix{Float64}(reshape(obs, K * nobs, params.pairs_per_lag_cdot))
        stationary_score = raw_stationary_score_flat(score_model, Float64.(x0), stats,
            score_sigma, params, device)
        action_phi_stationary = phi_transpose_action(Phi, stationary_score)
        Cphi[li, :, :, :] .= reshape((obs_flat * transpose(action_phi_stationary)) ./ params.pairs_per_lag_cdot,
            K, nobs, D)
        params.verbose && @printf("Cdot Phi(stationary score GFDT) tau %.5g (%d/%d)\n",
            sampler.lag_times[li], li, length(lags))
    end
    return Cphi
end

function translation_profiles(Cdot::Array{Float64, 4}, K::Int)
    nt, _, nobs, D = size(Cdot)
    profiles = zeros(Float64, nt, nobs, 2, K)
    counts = zeros(Int, nobs, 2, K)
    @inbounds for t in 1:nt, i in 1:K, a in 1:nobs, comp in 1:2, r in 0:(K - 1)
        j = periodic(i + r, K)
        col = comp == 1 ? j : K + j
        profiles[t, a, comp, r + 1] += Cdot[t, i, a, col]
        counts[a, comp, r + 1] += 1
    end
    @inbounds for a in 1:nobs, comp in 1:2, r in 1:K
        profiles[:, a, comp, r] ./= max(counts[a, comp, r], 1) ÷ nt
    end
    # counts above includes all time indices; divide once more by nt-safe value.
    profiles ./= nt
    return profiles
end

function translation_profiles_fixed(Cdot::Array{Float64, 4}, K::Int)
    nt, _, nobs, _ = size(Cdot)
    profiles = zeros(Float64, nt, nobs, 2, K)
    @inbounds for t in 1:nt, a in 1:nobs, comp in 1:2, r in 0:(K - 1)
        s = 0.0
        for i in 1:K
            j = periodic(i + r, K)
            col = comp == 1 ? j : K + j
            s += Cdot[t, i, a, col]
        end
        profiles[t, a, comp, r + 1] = s / K
    end
    return profiles
end

function select_profile_channels(data_profiles::Array{Float64, 4}, nplot::Int)
    _, nobs, _, K = size(data_profiles)
    tuples = Tuple{Float64, Int, Int, Int}[]
    for a in 1:nobs, comp in 1:2, r in 1:K
        sig = sqrt(mean(abs2, @view data_profiles[:, a, comp, r]))
        push!(tuples, (sig, a, comp, r))
    end
    sort!(tuples; by=x -> x[1], rev=true)
    return tuples[1:min(nplot, length(tuples))]
end

function project_tangent_symmetric_psd(A::AbstractMatrix{<:Real}, K::Int)
    Q = nonzero_mode_basis(K)
    anti = 0.5 .* (Matrix(A) .- transpose(Matrix(A)))
    E = eigen(Symmetric(Q' * sympart(A) * Q))
    vals = max.(E.values, 0.0)
    Sproj = Q * (E.vectors * Diagonal(vals) * transpose(E.vectors)) * transpose(Q)
    return project_block_circulant(Sproj + anti, K)
end

function project_fhd_mean_stencil(A::AbstractMatrix{<:Real}, K::Int)
    prof = block_profile(A, K)
    out = zeros(Float64, K, 2, 2)
    rp = 2
    rm = K
    adv = 0.25 * (prof[rp, 1, 2] + prof[rp, 2, 1] -
                  prof[rm, 1, 2] - prof[rm, 2, 1])
    visc = mean([prof[1, 2, 2], -2.0 * prof[rp, 2, 2], -2.0 * prof[rm, 2, 2]])
    visc = max(visc, 0.0)
    out[rp, 1, 2] = adv
    out[rm, 1, 2] = -adv
    out[rp, 2, 1] = adv
    out[rm, 2, 1] = -adv
    out[1, 2, 2] = visc
    out[rp, 2, 2] = -0.5 * visc
    out[rm, 2, 2] = -0.5 * visc
    return matrix_from_block_profile(out), (; advective_coupling=adv, viscous_center=visc)
end

function project_phi_matrix(Phi_full::Matrix{Float64}, K::Int, params::FHDFitParams)
    Phi_bc = project_block_circulant(Phi_full, K)
    mode = phi_projection_mode(params)
    if mode == "block_circulant"
        return Phi_bc, Dict{Symbol, Any}(:projection => mode)
    elseif mode == "block_circulant_psd"
        Phi_psd = project_tangent_symmetric_psd(Phi_bc, K)
        return Phi_psd, Dict{Symbol, Any}(:projection => mode)
    elseif mode == "fhd_mean_stencil"
        Phi_stencil, coeffs = project_fhd_mean_stencil(Phi_bc, K)
        if params.phi_project_symmetric_psd
            Phi_stencil = project_tangent_symmetric_psd(Phi_stencil, K)
        end
        return Phi_stencil, Dict{Symbol, Any}(
            :projection => mode,
            :advective_coupling => coeffs.advective_coupling,
            :viscous_center => coeffs.viscous_center,
            :psd_projected => params.phi_project_symmetric_psd,
        )
    else
        error("Unsupported evaluation.phi_projection=$(mode); allowed: block_circulant, block_circulant_psd, fhd_mean_stencil")
    end
end

function render_phi_figure(path::AbstractString, Phi_raw::Matrix{Float64}, V::Matrix{Float64},
        Phi::Matrix{Float64}, Mtrue::Matrix{Float64}, K::Int, metrics::Dict,
        params::FHDFitParams, phi_projection_info=Dict{Symbol, Any}())
    width, height = 3200, 2300
    with_scaled_figure_style(width, height) do _
        fig = Figure(; size=(width, height))
        correction = params.phi_use_stein_correction ? "with learned-score Stein correction" : "raw -Cdot(0)"
        projection = ", projection=$(phi_projection_mode(params))"
        figure_title!(fig, "FHDChain data-only Phi recovery";
            subtitle="zero-lag anchored short-lag coordinate derivative ($(correction)$(projection)); true M is ex-post only")
        mats = [Mtrue, Phi_raw, V, Phi, Phi .- Mtrue]
        titles = ["<M true>", "raw -Cdot(0)", "learned Stein V", "data Phi", "Phi - <M true>"]
        for idx in 1:5
            ax = Axis(fig[1 + (idx > 3), mod1(idx, 3)]; title=titles[idx], xlabel="column", ylabel="row")
            clim = maximum(abs, mats[idx])
            hm = heatmap!(ax, mats[idx]; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
            Colorbar(fig[1 + (idx > 3), mod1(idx, 3), Right()], hm)
        end
        eigs = tangent_eigs(Phi, K)
        ax = Axis(fig[2, 3]; title="sym(Phi) tangent eigenvalues", xlabel="index", ylabel="eigenvalue")
        scatter!(ax, 1:length(eigs), eigs; color=STYLE_PRIMARY)
        hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dash)
        text_panel!(fig[3, 1:3], String[
            @sprintf("Phi vs <M_true> rel.RMSE = %.6e", metrics[:relative_rmse]),
            @sprintf("Phi vs <M_true> correlation = %.6f", metrics[:correlation]),
            @sprintf("min/max tangent eig sym(Phi) = %.6e / %.6e", minimum(eigs), maximum(eigs)),
            @sprintf("projection = %s", phi_projection_mode(params)),
            haskey(phi_projection_info, :advective_coupling) ?
                @sprintf("data-estimated FHD stencil: advective=%.6e, viscous_center=%.6e",
                    phi_projection_info[:advective_coupling], phi_projection_info[:viscous_center]) :
                "data-estimated FHD stencil: not used",
            "The true mobility comparison is diagnostic-only and is not used to construct Phi.",
        ]; title="Summary")
        save_figure_checked(path, fig)
    end
    return nothing
end

function render_cdot_figure(path::AbstractString, taus::Vector{Float64}, tD::Float64,
        lib::FHDObservableLibrary, data_prof, phi_prof, selected, metrics_phi,
        params::FHDFitParams)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        subtitle = @sprintf("Phi rel.RMSE=%.3e corr=%.4f",
            metrics_phi[:relative_rmse], metrics_phi[:correlation])
        figure_title!(fig, "FHDChainN16 Phi Cdot diagnostics";
            subtitle="constant Phi uses the stationary-score GFDT formula, not a conditional score. $(subtitle)")
        rows = ceil(Int, length(selected) / 4)
        for (idx, (_, a, comp, r)) in enumerate(selected)
            row = 1 + (idx - 1) ÷ 4
            col = 1 + (idx - 1) % 4
            cname = comp == 1 ? "rho0" : "m0"
            ax = Axis(fig[row, col];
                title=@sprintf("%s vs %s, offset %d", lib.names[a], cname, r - 1),
                xlabel="tau / t_D", ylabel="Cdot profile")
            x = taus ./ tD
            lines!(ax, x, data_prof[:, a, comp, r]; color=STYLE_REFERENCE,
                linewidth=curve_linewidth(), label="data")
            lines!(ax, x, phi_prof[:, a, comp, r]; color=STYLE_PRIMARY,
                linewidth=curve_linewidth(), linestyle=:dash, label="Phi + stat score")
            if idx == 1
                axislegend(ax; position=:rt)
            end
        end
        save_figure_checked(path, fig)
    end
    return nothing
end

softplus_one_bias() = Float32(log(exp(1.0) - 1.0))

function build_structured_fhd_mobility_nn(width::Int)
    return Chain(
        Dense(4 => width, Flux.swish),
        Dense(width => width, Flux.swish),
        Dense(width => 2),
    )
end

function model_weight_decay(model)
    total = 0.0f0
    count = 0
    for param in Flux.trainables(model)
        total += sum(abs2, param)
        count += length(param)
    end
    return count == 0 ? 0.0f0 : total / count
end

function eta_bar_from_phi(Phi::Matrix{Float64}, phys::FHDPhysicalParams, K::Int)
    prof = block_profile(sympart(Phi), K)
    off = 0.5 * (prof[2, 2, 2] + prof[K, 2, 2])
    eta = -off * phys.dx^2 / phys.theta
    return Float32(max(eta, 1.0e-6))
end

function skew_scale_from_phi(Phi::Matrix{Float64})
    anti = 0.5 .* (Phi .- transpose(Phi))
    return Float32(max(sqrt(mean(abs2, anti)), 1.0e-4))
end

function edge_feature_tensor(x0::Array{Float32, 3}, phys::FHDPhysicalParams)
    K, _, B = size(x0)
    features = Array{Float32}(undef, 4, K, B)
    @inbounds for b in 1:B, i in 1:K
        ip = periodic(i + 1, K)
        rho_i = Float64(x0[i, 1, b])
        rho_ip = Float64(x0[ip, 1, b])
        m_i = Float64(x0[i, 2, b])
        m_ip = Float64(x0[ip, 2, b])
        features[1, i, b] = Float32(0.5 * (rho_i + rho_ip) - phys.rho0)
        features[2, i, b] = Float32(0.5 * (m_i + m_ip))
        features[3, i, b] = Float32(rho_ip - rho_i)
        features[4, i, b] = Float32(m_ip - m_i)
    end
    return features
end

function mean_edge_features(states::Array{Float32, 4}, start_idx::Int, phys::FHDPhysicalParams,
        nsamples::Int, rng::AbstractRNG)
    nt, K, _, ntraj = size(states)
    total = (nt - start_idx + 1) * ntraj
    n = min(nsamples, total)
    x = Array{Float32}(undef, K, 2, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start_idx + (linear % (nt - start_idx + 1))
        tr = (linear ÷ (nt - start_idx + 1)) + 1
        x[:, :, s] .= states[t, :, :, tr]
    end
    return reshape(edge_feature_tensor(x, phys), 4, K * n)
end

function structured_outputs(model, features::Array{Float32, 4},
        eta_bar::Float32, skew_scale::Float32)
    raw = reshape(model(reshape(features, 4, :)), 2, size(features, 2), size(features, 3), size(features, 4))
    eta_delta = eta_bar .* (NNlib.softplus.(raw[1, :, :, :] .+ softplus_one_bias()) .- 1.0f0)
    skew = skew_scale .* raw[2, :, :, :]
    return eta_delta, skew
end

function structured_delta_action_arrays(model, cache::FHDMobilityTrainingCache,
        phys::FHDPhysicalParams)
    K = size(cache.features, 2)
    jm = [periodic(i - 1, K) for i in 1:K]
    jp = [periodic(i + 1, K) for i in 1:K]
    eta_delta, skew = structured_outputs(model, cache.features, cache.eta_bar, cache.skew_scale)
    sr = @view cache.scond[:, 1, :, :]
    sm = @view cache.scond[:, 2, :, :]
    eta_jm = eta_delta[jm, :, :]
    sm_jm = sm[jm, :, :]
    sm_jp = sm[jp, :, :]
    sr_jp = sr[jp, :, :]
    pref = Float32(phys.theta / phys.dx^2)
    d_m = pref .* ((eta_delta .+ eta_jm) .* sm .- eta_delta .* sm_jp .- eta_jm .* sm_jm)
    edge_flux = skew .* sm
    skew_q = edge_flux .- edge_flux[jm, :, :]
    skew_p = .-skew .* (sr .- sr_jp)
    aq = skew_q
    ap = d_m .+ skew_p
    return aq, ap
end

function predict_A_profiles_structured(model, cache::FHDMobilityTrainingCache,
        phys::FHDPhysicalParams)
    K = size(cache.features, 2)
    T = size(cache.features, 4)
    nobs = size(cache.observables, 2)
    aq, ap = structured_delta_action_arrays(model, cache, phys)
    vals = [
        begin
            action = comp == 1 ? aq : ap
            shifted = action[[periodic(i + r0, K) for i in 1:K], :, t]
            -mean(cache.observables[:, a, :, t] .* shifted)
        end
        for t in 1:T, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)
    ]
    return reshape(vals, T, nobs, 2, K)
end

function profile_index_list(T::Int, nobs::Int, K::Int)
    return vec([(t, a, comp, r0) for t in 1:T, a in 1:nobs, comp in 1:2, r0 in 0:(K - 1)])
end

function predict_A_values_structured(model, cache::FHDMobilityTrainingCache,
        phys::FHDPhysicalParams, indices::Vector{NTuple{4, Int}})
    K = size(cache.features, 2)
    aq, ap = structured_delta_action_arrays(model, cache, phys)
    return [
        begin
            t, a, comp, r0 = idx
            action = comp == 1 ? aq : ap
            shifted = action[[periodic(i + r0, K) for i in 1:K], :, t]
            -mean(cache.observables[:, a, :, t] .* shifted)
        end
        for idx in indices
    ]
end

function structured_mean_penalty(model, cache::FHDMobilityTrainingCache)
    raw = model(cache.mean_features)
    eta_delta = cache.eta_bar .* (NNlib.softplus.(raw[1, :] .+ softplus_one_bias()) .- 1.0f0)
    skew = cache.skew_scale .* raw[2, :]
    eta_scale = max(abs(Float64(cache.eta_bar)), 1.0e-8)
    skew_scale = max(abs(Float64(cache.skew_scale)), 1.0e-8)
    return mean((eta_delta ./ Float32(eta_scale)) .^ 2) +
        mean((skew ./ Float32(skew_scale)) .^ 2)
end

function mobility_rms_delta(model, cache::FHDMobilityTrainingCache)
    raw = model(cache.mean_features)
    eta_delta = Float64.(cache.eta_bar .* (NNlib.softplus.(raw[1, :] .+ softplus_one_bias()) .- 1.0f0))
    skew = Float64.(cache.skew_scale .* raw[2, :])
    return sqrt(mean(eta_delta .^ 2) + mean(skew .^ 2))
end

function build_mobility_training_cache(sampler::FHDPairSampler, phys::FHDPhysicalParams,
        lib::FHDObservableLibrary, params::FHDFitParams, stats::DataStats,
        cond_model, cond_sigma::Float32, cond_meta, device::ExecutionDevice,
        Phi::Matrix{Float64})
    cache_path = resolve_path(@__DIR__, params.mobility_cache_bson)
    if isfile(cache_path)
        blob = BSON.load(cache_path)
        if haskey(blob, :cache_version) && blob[:cache_version] == 1 &&
                haskey(blob, :features) && size(blob[:features], 4) == min(params.max_fit_lags, length(sampler.lag_steps)) &&
                size(blob[:features], 3) == params.mobility_pairs_per_lag
            @printf("Loading mobility training cache from %s\n", cache_path)
            return FHDMobilityTrainingCache(blob[:features], blob[:scond], blob[:observables],
                blob[:mean_features], blob[:eta_bar], blob[:skew_scale])
        end
    end
    lags = sampler.lag_steps[1:min(params.max_fit_lags, length(sampler.lag_steps))]
    K = sampler.K
    B = params.mobility_pairs_per_lag
    T = length(lags)
    nobs = length(lib.names)
    features = Array{Float32}(undef, 4, K, B, T)
    scond = Array{Float32}(undef, K, 2, B, T)
    observables = Array{Float32}(undef, K, nobs, B, T)
    rng = MersenneTwister(params.seed + 600)
    time_features = String(cond_meta[:time_features])
    nfreq = Int(cond_meta[:time_fourier_frequencies])
    include_delta = Bool(cond_meta[:include_delta_input])
    stdv = Float64.(std_flat(stats))
    for (li, lag) in enumerate(lags)
        x0, xt, _, _ = random_lag_pairs(sampler, lag, B, rng)
        features[:, :, :, li] .= edge_feature_tensor(x0, phys)
        trans_norm = evaluate_transition_score_norm(cond_model, x0, xt, sampler.lag_tnorm[li],
            stats, cond_sigma, device; batch_size=params.score_batch_size,
            time_features=time_features, time_fourier_frequencies=nfreq,
            include_delta_input=include_delta)
        rflat = Float64.(raw_flat_batch(trans_norm))
        rflat ./= reshape(stdv, :, 1)
        @inbounds for b in 1:B, i in 1:K
            scond[i, 1, b, li] = Float32(rflat[i, b])
            scond[i, 2, b, li] = Float32(rflat[K + i, b])
        end
        obs = center_observables!(observable_values(xt, phys), lib)
        observables[:, :, :, li] .= obs
        params.verbose && @printf("Built mobility cache tau %.5g (%d/%d)\n",
            sampler.lag_times[li], li, T)
    end
    mean_features = mean_edge_features(sampler.states, sampler.start_idx, phys,
        120_000, MersenneTwister(params.seed + 601))
    eta_bar = eta_bar_from_phi(Phi, phys, K)
    skew_scale = skew_scale_from_phi(Phi)
    ensure_parent_dir(cache_path)
    cache_version = 1
    BSON.@save cache_path cache_version features scond observables mean_features eta_bar skew_scale
    @printf("Saved mobility training cache to %s\n", cache_path)
    return FHDMobilityTrainingCache(features, scond, observables, mean_features, eta_bar, skew_scale)
end

function train_structured_mobility_nn(A_target::Array{Float64, 4},
        cache::FHDMobilityTrainingCache, phys::FHDPhysicalParams, params::FHDFitParams)
    Random.seed!(params.seed + 700)
    model = build_structured_fhd_mobility_nn(params.mobility_hidden_width)
    Random.seed!()
    target = Float32.(A_target)
    scale = ones(Float32, size(target))
    for a in 1:size(target, 2)
        vals = vec(@view A_target[:, a, :, :])
        finite_vals = vals[isfinite.(vals)]
        rmsv = isempty(finite_vals) ? 1.0 : sqrt(mean(finite_vals .^ 2))
        scale[:, a, :, :] .= Float32(max(rmsv, 1.0e-7))
    end
    opt_state = Flux.setup(Flux.Adam(params.mobility_learning_rate), model)
    history = FHDMobilityHistory()
    best_model = deepcopy(model)
    best_rmse = Inf
    rng = MersenneTwister(params.seed + 701)
    all_indices_full = profile_index_list(size(target, 1), size(target, 2), size(target, 4))
    scored = Tuple{Float64, NTuple{4, Int}}[]
    for idx in all_indices_full
        t, a, comp, r0 = idx
        push!(scored, (abs(Float64(target[t, a, comp, r0 + 1])) /
            max(Float64(scale[t, a, comp, r0 + 1]), 1.0e-8), idx))
    end
    sort!(scored; by=x -> x[1], rev=true)
    pool_size = min(length(scored), max(4 * params.mobility_batch_profiles,
        ceil(Int, 0.30 * length(scored))))
    all_indices = [scored[i][2] for i in 1:pool_size]
    @printf("Structured mobility training uses %d / %d high-signal A profile entries.\n",
        length(all_indices), length(all_indices_full))
    model, ridge_pred, ridge_rel, ridge_corr = initialize_mobility_last_layer_ridge!(
        model, A_target, cache, phys, all_indices_full; ridge=1.0e-1)
    push!(history.epochs, 0)
    push!(history.losses, ridge_rel^2)
    push!(history.validation_rmse, ridge_rel)
    push!(history.validation_corr, ridge_corr)
    push!(history.mean_penalty, Float64(structured_mean_penalty(model, cache)))
    push!(history.rms_delta, mobility_rms_delta(model, cache))
    best_model = deepcopy(model)
    best_rmse = ridge_rel
    @printf("Structured mobility random-feature ridge init: A_rel %.6e corr %.5f\n",
        ridge_rel, ridge_corr)
    for epoch in 1:params.mobility_epochs
        batch_indices = all_indices[rand(rng, 1:length(all_indices),
            min(params.mobility_batch_profiles, length(all_indices)))]
        target_batch = Float32[target[t, a, comp, r0 + 1] for (t, a, comp, r0) in batch_indices]
        scale_batch = Float32[scale[t, a, comp, r0 + 1] for (t, a, comp, r0) in batch_indices]
        loss, grads = Flux.withgradient(model) do current_model
            pred = predict_A_values_structured(current_model, cache, phys, batch_indices)
            data_loss = mean(((pred .- target_batch) ./ scale_batch) .^ 2)
            mean_pen = structured_mean_penalty(current_model, cache)
            wd = Float32(params.mobility_weight_decay) * model_weight_decay(current_model)
            data_loss + Float32(params.mobility_mean_penalty) * mean_pen + wd
        end
        opt_state, model = Flux.update!(opt_state, model, grads[1])
        if epoch == 1 || epoch % 25 == 0 || epoch == params.mobility_epochs
            pred = Float64.(predict_A_profiles_structured(model, cache, phys))
            metrics = agreement_metrics(A_target, pred)
            mean_pen = Float64(structured_mean_penalty(model, cache))
            rms_delta = mobility_rms_delta(model, cache)
            push!(history.epochs, epoch)
            push!(history.losses, Float64(loss))
            push!(history.validation_rmse, metrics[:relative_rmse])
            push!(history.validation_corr, metrics[:correlation])
            push!(history.mean_penalty, mean_pen)
            push!(history.rms_delta, rms_delta)
            if metrics[:relative_rmse] < best_rmse
                best_rmse = metrics[:relative_rmse]
                best_model = deepcopy(model)
            end
            @printf("Structured mobility epoch %d: loss %.6e A_rel %.6e corr %.5f mean_pen %.3e\n",
                epoch, Float64(loss), metrics[:relative_rmse], metrics[:correlation], mean_pen)
        end
    end
    return best_model, history
end

function initialize_mobility_last_layer_ridge!(model, A_target::Array{Float64, 4},
        cache::FHDMobilityTrainingCache, phys::FHDPhysicalParams,
        fit_indices::Vector{NTuple{4, Int}}; ridge::Float64=1.0e-6)
    # Freeze the first two dense layers as random nonlinear features and solve
    # the final layer for edge viscosity and skew residuals. This is a
    # linearized least-squares minimization of the same paper A-residual loss,
    # constrained to the conservative FHD mobility structure.
    features2 = reshape(cache.features, 4, :)
    hidden = model[2](model[1](features2))
    H0 = size(hidden, 1)
    H = H0 + 1
    hidden_bias = vcat(hidden, ones(Float32, 1, size(hidden, 2)))
    K = size(cache.features, 2)
    B = size(cache.features, 3)
    T = size(cache.features, 4)
    hidden4 = reshape(hidden_bias, H, K, B, T)
    nrows = length(fit_indices)
    ncols = 2H
    X = zeros(Float64, nrows, ncols)
    y = zeros(Float64, nrows)
    jm = [periodic(i - 1, K) for i in 1:K]
    jp = [periodic(i + 1, K) for i in 1:K]
    eta_linear = Float64(cache.eta_bar) * (1.0 - exp(-1.0))
    skew_linear = Float64(cache.skew_scale)
    pref = phys.theta / phys.dx^2
    scale_arr = ones(Float64, size(A_target))
    for a in 1:size(A_target, 2)
        vals = vec(@view A_target[:, a, :, :])
        finite_vals = vals[isfinite.(vals)]
        rmsv = isempty(finite_vals) ? 1.0 : sqrt(mean(finite_vals .^ 2))
        scale_arr[:, a, :, :] .= 1.0 / max(rmsv, 1.0e-7)
    end
    @inbounds for (row, (t, a, comp, r0)) in enumerate(fit_indices)
        row_scale = scale_arr[t, a, comp, r0 + 1]
        y[row] = A_target[t, a, comp, r0 + 1] * row_scale
        for i in 1:K, b in 1:B
            j = periodic(i + r0, K)
            coeff = -row_scale * Float64(cache.observables[i, a, b, t]) / (K * B)
            sr = Float64(cache.scond[j, 1, b, t])
            sm = Float64(cache.scond[j, 2, b, t])
            if comp == 1
                h_j = @view hidden4[:, j, b, t]
                h_jm = @view hidden4[:, jm[j], b, t]
                sm_jm = Float64(cache.scond[jm[j], 2, b, t])
                for h in 1:H
                    X[row, H + h] += coeff * skew_linear *
                        (sm * Float64(h_j[h]) - sm_jm * Float64(h_jm[h]))
                end
            else
                h_j = @view hidden4[:, j, b, t]
                h_jm = @view hidden4[:, jm[j], b, t]
                sm_jm = Float64(cache.scond[jm[j], 2, b, t])
                sm_jp = Float64(cache.scond[jp[j], 2, b, t])
                sr_jp = Float64(cache.scond[jp[j], 1, b, t])
                for h in 1:H
                    X[row, h] += coeff * pref * eta_linear *
                        ((sm - sm_jp) * Float64(h_j[h]) +
                         (sm - sm_jm) * Float64(h_jm[h]))
                    X[row, H + h] += coeff * skew_linear *
                        (-(sr - sr_jp) * Float64(h_j[h]))
                end
            end
        end
    end
    theta = (transpose(X) * X + ridge * I) \ (transpose(X) * y)
    W = zeros(Float32, 2, H0)
    bias = zeros(Float32, 2)
    coeff_mat = reshape(theta, H, 2)
    for out in 1:2
        W[out, :] .= Float32.(coeff_mat[1:H0, out])
        bias[out] = Float32(coeff_mat[H, out])
    end
    model[3].weight .= W
    model[3].bias .= bias
    pred = Float64.(predict_A_profiles_structured(model, cache, phys))
    metrics = agreement_metrics(A_target, pred)
    return model, pred, metrics[:relative_rmse], metrics[:correlation]
end

function structured_delta_action_sample(model, z::AbstractMatrix{<:Real},
        score::AbstractVector{<:Real}, phys::FHDPhysicalParams,
        eta_bar::Float32, skew_scale::Float32; transpose_action::Bool=false)
    K = size(z, 1)
    x = Array{Float32}(undef, K, 2, 1)
    @inbounds for i in 1:K
        x[i, 1, 1] = Float32(z[i, 1])
        x[i, 2, 1] = Float32(z[i, 2])
    end
    features = edge_feature_tensor(x, phys)
    eta_delta, skew = structured_outputs(model, reshape(features, 4, K, 1, 1), eta_bar, skew_scale)
    sr = @view score[1:K]
    sm = @view score[K+1:2K]
    out = zeros(Float64, 2K)
    pref = phys.theta / phys.dx^2
    @inbounds for i in 1:K
        im = periodic(i - 1, K)
        ip = periodic(i + 1, K)
        ei = Float64(eta_delta[i, 1, 1])
        eim = Float64(eta_delta[im, 1, 1])
        out[K + i] += pref * ((ei + eim) * Float64(sm[i]) -
            ei * Float64(sm[ip]) - eim * Float64(sm[im]))
        a = Float64(skew[i, 1, 1])
        if transpose_action
            out[i] += a * Float64(sm[i])
            out[ip] -= a * Float64(sm[i])
            out[K + i] += -a * (Float64(sr[i]) - Float64(sr[ip]))
        else
            out[i] += -a * Float64(sm[i])
            out[ip] += a * Float64(sm[i])
            out[K + i] += a * (Float64(sr[i]) - Float64(sr[ip]))
        end
    end
    return out
end

function structured_delta_matrix_sample(model, z::AbstractMatrix{<:Real},
        phys::FHDPhysicalParams, eta_bar::Float32, skew_scale::Float32)
    K = size(z, 1)
    D = 2K
    M = Matrix{Float64}(undef, D, D)
    for j in 1:D
        e = zeros(Float64, D)
        e[j] = 1.0
        M[:, j] .= structured_delta_action_sample(model, z, e, phys, eta_bar, skew_scale;
            transpose_action=false)
    end
    return M
end

function structured_divergence_sample(model, z::AbstractMatrix{<:Real},
        phys::FHDPhysicalParams, eta_bar::Float32, skew_scale::Float32; eps_fd::Float64=1.0e-4)
    K = size(z, 1)
    D = 2K
    div = zeros(Float64, D)
    zp = Matrix{Float64}(z)
    zm = Matrix{Float64}(z)
    for j in 1:D
        site = j <= K ? j : j - K
        chan = j <= K ? 1 : 2
        zp[site, chan] += eps_fd
        zm[site, chan] -= eps_fd
        Mp = structured_delta_matrix_sample(model, zp, phys, eta_bar, skew_scale)
        Mm = structured_delta_matrix_sample(model, zm, phys, eta_bar, skew_scale)
        div .+= (Mp[:, j] .- Mm[:, j]) ./ (2.0 * eps_fd)
        zp[site, chan] = z[site, chan]
        zm[site, chan] = z[site, chan]
    end
    return div
end

function psd_sqrt(A::AbstractMatrix{<:Real}; floor::Float64=1.0e-9)
    eig = eigen(Symmetric(sympart(A)))
    vals = max.(eig.values, floor)
    return eig.vectors * Diagonal(sqrt.(vals)) * transpose(eig.vectors), minimum(eig.values)
end

function tangent_cholesky_factor(A::AbstractMatrix{<:Real}, K::Int; jitter::Float64=1.0e-10)
    Q = nonzero_mode_basis(K)
    S = Symmetric(Q' * sympart(A) * Q)
    eigvals_s = eigvals(S)
    min_eig = minimum(eigvals_s)
    shift = max(jitter, -min_eig + jitter)
    chol = cholesky(Symmetric(Matrix(S) + shift * I); check=false)
    return Q * Matrix(chol.L), min_eig
end

function raw_stationary_score_flat(score_model, z::Array{Float64, 3}, stats::DataStats,
        sigma::Float32, params::FHDFitParams, device::ExecutionDevice)
    z32 = Float32.(z)
    zn = apply_fhd_stats(z32, stats)
    sn = evaluate_stationary_score_norm(score_model, zn, sigma, device;
        batch_size=params.score_batch_size)
    sr = normalized_score_to_raw(sn, stats)
    return Float64.(raw_flat_batch(sr))
end

function raw_analytic_score_flat(z::Array{Float64, 3}, phys::FHDPhysicalParams)
    score = physical_score_raw(Float32.(z), phys; velocity_floor=phys.velocity_density_floor)
    return Float64.(raw_flat_batch(score))
end

struct EmpiricalLocalScore
    rho_grid::Vector{Float64}
    m_grid::Vector{Float64}
    grad_rho::Matrix{Float64}
    grad_m::Matrix{Float64}
end

function build_empirical_local_score(sampler::FHDPairSampler; nsamples::Int=1_000_000,
        ngrid::Int=320, seed::Int=0)
    rng = MersenneTwister(seed)
    post = @view sampler.states[sampler.start_idx:end, :, :, :]
    nt, K, _, ntraj = size(post)
    n = min(nsamples, nt * K * ntraj)
    rho = Vector{Float64}(undef, n)
    mom = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(nt * K * ntraj - 1))
        t = (linear % nt) + 1
        tmp = linear ÷ nt
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        rho[s] = Float64(post[t, i, 1, tr])
        mom[s] = Float64(post[t, i, 2, tr])
    end
    rlo, rhi = quantile(rho, 0.0002), quantile(rho, 0.9998)
    mlo, mhi = quantile(mom, 0.0002), quantile(mom, 0.9998)
    rpad = 0.25 * (rhi - rlo)
    mpad = 0.25 * (mhi - mlo)
    rr = range(max(1.0e-8, rlo - rpad), rhi + rpad; length=ngrid)
    mr = range(mlo - mpad, mhi + mpad; length=ngrid)
    kd = kde((rho, mom), (rr, mr))
    rg = collect(kd.x)
    mg = collect(kd.y)
    logp = log.(max.(kd.density, maximum(kd.density) * 1.0e-10))
    gr = similar(logp)
    gm = similar(logp)
    dr = rg[2] - rg[1]
    dm = mg[2] - mg[1]
    @inbounds for i in 1:ngrid, j in 1:ngrid
        im = max(i - 1, 1); ip = min(i + 1, ngrid)
        jm = max(j - 1, 1); jp = min(j + 1, ngrid)
        gr[i, j] = (logp[ip, j] - logp[im, j]) / ((ip - im) * dr)
        gm[i, j] = (logp[i, jp] - logp[i, jm]) / ((jp - jm) * dm)
    end
    return EmpiricalLocalScore(rg, mg, gr, gm)
end

function interp_grid_score(grid_x::Vector{Float64}, grid_y::Vector{Float64},
        values::Matrix{Float64}, x::Float64, y::Float64)
    nx = length(grid_x)
    ny = length(grid_y)
    tx = clamp((x - grid_x[1]) / (grid_x[end] - grid_x[1]) * (nx - 1), 0.0, nx - 1 - eps())
    ty = clamp((y - grid_y[1]) / (grid_y[end] - grid_y[1]) * (ny - 1), 0.0, ny - 1 - eps())
    i = floor(Int, tx) + 1
    j = floor(Int, ty) + 1
    wx = tx - floor(tx)
    wy = ty - floor(ty)
    i2 = min(i + 1, nx)
    j2 = min(j + 1, ny)
    return (1 - wx) * (1 - wy) * values[i, j] +
           wx * (1 - wy) * values[i2, j] +
           (1 - wx) * wy * values[i, j2] +
           wx * wy * values[i2, j2]
end

function empirical_local_score_flat(model::EmpiricalLocalScore, z::Array{Float64, 3})
    K, _, B = size(z)
    out = zeros(Float64, 2K, B)
    @inbounds for b in 1:B
        mean_r = 0.0
        mean_m = 0.0
        for i in 1:K
            rho = z[i, 1, b]
            mom = z[i, 2, b]
            sr = interp_grid_score(model.rho_grid, model.m_grid, model.grad_rho, rho, mom)
            sm = interp_grid_score(model.rho_grid, model.m_grid, model.grad_m, rho, mom)
            out[i, b] = sr
            out[K + i, b] = sm
            mean_r += sr
            mean_m += sm
        end
        mean_r /= K
        mean_m /= K
        for i in 1:K
            out[i, b] -= mean_r
            out[K + i, b] -= mean_m
        end
    end
    return out
end

function sample_forward_initial(sampler::FHDPairSampler, ntraj::Int, rng::AbstractRNG)
    nt, K, _, ndata = size(sampler.states)
    z = Array{Float64}(undef, K, 2, ntraj)
    @inbounds for b in 1:ntraj
        tr = rand(rng, 1:ndata)
        t = rand(rng, sampler.start_idx:nt)
        z[:, :, b] .= sampler.states[t, :, :, tr]
    end
    return z
end

function project_raw_zero_modes!(z::Array{Float64, 3}, mass_mean::Float64, momentum_mean::Float64)
    K, _, B = size(z)
    @inbounds for b in 1:B
        rho_shift = mean(@view z[:, 1, b]) - mass_mean
        m_shift = mean(@view z[:, 2, b]) - momentum_mean
        for i in 1:K
            z[i, 1, b] -= rho_shift
            z[i, 2, b] -= m_shift
        end
    end
    return z
end

function forward_support_bounds(sampler::FHDPairSampler)
    post = @view sampler.states[sampler.start_idx:end, :, :, :]
    rho_vals = vec(Float64.(@view post[:, :, 1, :]))
    m_vals = vec(Float64.(@view post[:, :, 2, :]))
    rho_lo = max(quantile(rho_vals, 0.0005), 1.0e-4)
    rho_hi = quantile(rho_vals, 0.9995)
    m_lo = quantile(m_vals, 0.0005)
    m_hi = quantile(m_vals, 0.9995)
    rho_pad = 0.25 * max(rho_hi - rho_lo, 1.0e-6)
    m_pad = 0.25 * max(m_hi - m_lo, 1.0e-6)
    return (rho_lo - 0.1 * rho_pad, rho_hi + rho_pad, m_lo - m_pad, m_hi + m_pad)
end

function clamp_forward_state!(z::Array{Float64, 3}, bounds)
    rho_lo, rho_hi, m_lo, m_hi = bounds
    @inbounds for b in axes(z, 3), i in axes(z, 1)
        z[i, 1, b] = isfinite(z[i, 1, b]) ? clamp(z[i, 1, b], rho_lo, rho_hi) : 0.5 * (rho_lo + rho_hi)
        z[i, 2, b] = isfinite(z[i, 2, b]) ? clamp(z[i, 2, b], m_lo, m_hi) : 0.0
    end
    return z
end

function integrate_forward_langevin(score_model, score_sigma::Float32, stats::DataStats,
        Phi::Matrix{Float64}, mobility_model, cache,
        sampler::FHDPairSampler, phys::FHDPhysicalParams, params::FHDFitParams,
        device::ExecutionDevice; mode::Symbol, score_left_correction=nothing,
        score_clip::Union{Nothing, Real}=nothing)
    nsteps = ceil(Int, params.forward_total_time / params.forward_dt)
    burnin_steps = floor(Int, params.forward_burnin_time / params.forward_dt)
    requested_save_dt = params.forward_save_dt > 0.0 ? params.forward_save_dt : sampler.save_dt
    save_every = max(1, round(Int, requested_save_dt / params.forward_dt))
    nsaved = max(1, fld(max(nsteps - burnin_steps, 0), save_every) + 1)
    K = sampler.K
    D = sampler.D
    rng_offset = mode == :phi ? 800 : mode == :mean_true_analytic ? 850 : 900
    rng = MersenneTwister(params.seed + rng_offset)
    z = sample_forward_initial(sampler, params.forward_ntraj, rng)
    mass_mean = mean(Float64, sampler.states[sampler.start_idx:end, :, 1, :])
    momentum_mean = mean(Float64, sampler.states[sampler.start_idx:end, :, 2, :])
    bounds = forward_support_bounds(sampler)
    project_raw_zero_modes!(z, mass_mean, momentum_mean)
    saved = Array{Float32}(undef, nsaved, K, 2, params.forward_ntraj)
    times = Vector{Float64}(undef, nsaved)
    sqrt_phi, min_phi = tangent_cholesky_factor(Phi, K)
    min_eig = min_phi
    flat = Matrix{Float64}(undef, D, params.forward_ntraj)
    drift = similar(flat)
    noise = similar(flat)
    tangent_noise = Matrix{Float64}(undef, size(sqrt_phi, 2), params.forward_ntraj)
    save_idx = 0
    progress = Progress(100; desc="Forward Langevin $(String(mode))")
    stride = max(1, nsteps ÷ 100)
    for step in 0:nsteps
        if step >= burnin_steps && (step - burnin_steps) % save_every == 0
            save_idx += 1
            times[save_idx] = (step - burnin_steps) * params.forward_dt
            saved[save_idx, :, :, :] .= Float32.(z)
        end
        step == nsteps && break
        score = mode == :mean_true_analytic ?
            raw_analytic_score_flat(z, phys) :
            mode == :empirical_local ?
                empirical_local_score_flat(score_model, z) :
                raw_stationary_score_flat(score_model, z, stats, score_sigma, params, device)
        if mode != :mean_true_analytic && score_left_correction !== nothing
            score .= Matrix{Float64}(score_left_correction) * score
        end
        clip_value = score_clip === nothing ? params.forward_score_clip : Float32(score_clip)
        (mode == :mean_true_analytic || mode == :empirical_local || clip_value <= 0) ||
            clamp!(score, -Float64(clip_value), Float64(clip_value))
        mul!(drift, Phi, score)
        if mode == :phi || mode == :mean_true_analytic || mode == :empirical_local
            randn!(rng, tangent_noise)
            mul!(noise, sqrt_phi, tangent_noise)
        elseif mode == :nn
            randn!(rng, noise)
            for b in 1:params.forward_ntraj
                zb = @view z[:, :, b]
                sb = @view score[:, b]
                drift[:, b] .+= structured_delta_action_sample(mobility_model, zb, sb, phys,
                    cache.eta_bar, cache.skew_scale; transpose_action=false)
                drift[:, b] .+= structured_divergence_sample(mobility_model, zb, phys,
                    cache.eta_bar, cache.skew_scale)
                Mdelta = structured_delta_matrix_sample(mobility_model, zb, phys,
                    cache.eta_bar, cache.skew_scale)
                if any(!isfinite, Mdelta)
                    fill!(Mdelta, 0.0)
                end
                sqrt_nn, mine = psd_sqrt(sympart(Phi + Mdelta); floor=1.0e-9)
                min_eig = min(min_eig, mine)
                noise[:, b] .= sqrt_nn * noise[:, b]
            end
        else
            error("Unknown forward mode $(mode).")
        end
        @inbounds for b in 1:params.forward_ntraj, i in 1:K
            flat[i, b] = z[i, 1, b]
            flat[K + i, b] = z[i, 2, b]
        end
        @. flat = flat + params.forward_dt * drift + sqrt(2.0 * params.forward_dt) * noise
        @inbounds for b in 1:params.forward_ntraj, i in 1:K
            z[i, 1, b] = flat[i, b]
            z[i, 2, b] = flat[K + i, b]
        end
        clamp_forward_state!(z, bounds)
        project_raw_zero_modes!(z, mass_mean, momentum_mean)
        clamp_forward_state!(z, bounds)
        step % stride == 0 && ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)
    return times, saved, min_eig
end

function render_mobility_training_figure(path::AbstractString, history::FHDMobilityHistory,
        taus::Vector{Float64}, lib::FHDObservableLibrary, A_data, A_pred, selected,
        params::FHDFitParams)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        final_rel = isempty(history.validation_rmse) ? NaN : history.validation_rmse[end]
        final_corr = isempty(history.validation_corr) ? NaN : history.validation_corr[end]
        figure_title!(fig, "FHDChain mobility NN training diagnostics";
            subtitle=@sprintf("direct paper residual loss, final A rel.RMSE=%.3e corr=%.4f", final_rel, final_corr))
        ax1 = Axis(fig[1, 1]; title="training loss", xlabel="epoch", ylabel="loss", yscale=log10)
        lines!(ax1, history.epochs, history.losses; color=STYLE_PRIMARY, linewidth=curve_linewidth())
        if length(history.epochs) == 1
            scatter!(ax1, history.epochs, history.losses; color=STYLE_PRIMARY, markersize=18)
            xlims!(ax1, -0.2, 1.0)
        end
        ax2 = Axis(fig[1, 2]; title="A target agreement", xlabel="epoch", ylabel="metric")
        lines!(ax2, history.epochs, history.validation_rmse; color=STYLE_HIGHLIGHT,
            linewidth=curve_linewidth(), label="rel.RMSE")
        lines!(ax2, history.epochs, 1 .- history.validation_corr; color=STYLE_SECONDARY,
            linewidth=curve_linewidth(), linestyle=:dash, label="1-corr")
        if length(history.epochs) == 1
            scatter!(ax2, history.epochs, history.validation_rmse; color=STYLE_HIGHLIGHT, markersize=18)
            scatter!(ax2, history.epochs, 1 .- history.validation_corr; color=STYLE_SECONDARY, markersize=18)
            xlims!(ax2, -0.2, 1.0)
        end
        axislegend(ax2; position=:rt)
        ax3 = Axis(fig[1, 3]; title="regularization diagnostics", xlabel="epoch", ylabel="value", yscale=log10)
        lines!(ax3, history.epochs, history.mean_penalty; color=STYLE_ACCENT,
            linewidth=curve_linewidth(), label="mean penalty")
        lines!(ax3, history.epochs, history.rms_delta; color=STYLE_PRIMARY,
            linewidth=curve_linewidth(), linestyle=:dash, label="rms delta")
        if length(history.epochs) == 1
            scatter!(ax3, history.epochs, history.mean_penalty; color=STYLE_ACCENT, markersize=18)
            scatter!(ax3, history.epochs, history.rms_delta; color=STYLE_PRIMARY, markersize=18)
            xlims!(ax3, -0.2, 1.0)
        end
        axislegend(ax3; position=:rt)
        nshow = min(length(selected), 12)
        rows = 3
        for idx in 1:nshow
            _, a, comp, r = selected[idx]
            row = 2 + (idx - 1) ÷ 4
            col = 1 + (idx - 1) % 4
            cname = comp == 1 ? "rho0" : "m0"
            ax = Axis(fig[row, col];
                title=@sprintf("A: %s vs %s, offset %d", lib.names[a], cname, r - 1),
                xlabel="tau", ylabel="A")
            lines!(ax, taus, A_data[:, a, comp, r]; color=STYLE_REFERENCE,
                linewidth=curve_linewidth(), label="data target")
            lines!(ax, taus, A_pred[:, a, comp, r]; color=STYLE_PRIMARY,
                linewidth=curve_linewidth(), linestyle=:dash, label="NN")
            idx == 1 && axislegend(ax; position=:rt)
        end
        save_figure_checked(path, fig)
    end
    return nothing
end

function mean_structured_mobility(model, Phi::Matrix{Float64}, cache::FHDMobilityTrainingCache,
        sampler::FHDPairSampler, phys::FHDPhysicalParams; nsamples::Int, seed::Int)
    rng = MersenneTwister(seed)
    raw = sample_state_tensor(sampler.states, sampler.start_idx, nsamples, rng)
    M = zeros(Float64, sampler.D, sampler.D)
    for b in 1:nsamples
        M .+= Phi .+ structured_delta_matrix_sample(model, @view(raw[:, :, b]), phys,
            cache.eta_bar, cache.skew_scale)
    end
    return M ./ nsamples
end

function render_mobility_comparison_figure(path::AbstractString, Phi::Matrix{Float64},
        Mnn::Matrix{Float64}, Mtrue::Matrix{Float64}, sampler::FHDPairSampler,
        params::FHDFitParams)
    K = sampler.K
    metrics_nn = agreement_metrics(Mtrue, Mnn)
    metrics_phi = agreement_metrics(Mtrue, Phi)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        figure_title!(fig, "FHDChain learned mobility comparison";
            subtitle=@sprintf("ex-post only: Phi rel=%.3e, NN rel=%.3e, NN corr=%.4f",
                metrics_phi[:relative_rmse], metrics_nn[:relative_rmse], metrics_nn[:correlation]))
        mats = [Mtrue, Phi, Mnn, Mnn .- Mtrue, sympart(Mnn) .- sympart(Mtrue), 0.5 .* (Mnn .- transpose(Mnn)) .- 0.5 .* (Mtrue .- transpose(Mtrue))]
        titles = ["<M true>", "Phi", "<M NN>", "NN - true", "sym error", "skew error"]
        for idx in 1:6
            ax = Axis(fig[1 + (idx - 1) ÷ 3, 1 + (idx - 1) % 3];
                title=titles[idx], xlabel="column", ylabel="row")
            clim = max(maximum(abs, mats[idx]), 1.0e-8)
            hm = heatmap!(ax, mats[idx]; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
            Colorbar(fig[1 + (idx - 1) ÷ 3, 1 + (idx - 1) % 3, Right()], hm)
        end
        prof_true = block_profile(Mtrue, K)
        prof_nn = block_profile(Mnn, K)
        axp = Axis(fig[3, 1:2]; title="block-profile entries", xlabel="offset", ylabel="value")
        offsets = 0:(K - 1)
        for a in 1:2, b in 1:2
            lines!(axp, offsets, prof_true[:, a, b]; color=STYLE_REFERENCE, linewidth=2)
            lines!(axp, offsets, prof_nn[:, a, b]; color=STYLE_PRIMARY, linestyle=:dash, linewidth=2)
        end
        eig_true = tangent_eigs(Mtrue, K)
        eig_nn = tangent_eigs(Mnn, K)
        ax = Axis(fig[3, 3]; title="sym(M) tangent eigenvalues", xlabel="index", ylabel="eig")
        scatter!(ax, 1:length(eig_true), eig_true; color=STYLE_REFERENCE, label="true")
        scatter!(ax, 1:length(eig_nn), eig_nn; color=STYLE_PRIMARY, label="NN")
        axislegend(ax; position=:rt)
        save_figure_checked(path, fig)
    end
    return metrics_nn, metrics_phi
end

function draw_values_fit(states::Array{Float32, 4}, channel::Int, max_samples::Int,
        rng::AbstractRNG; phys::Union{Nothing, FHDPhysicalParams}=nothing, quantity::Symbol=:channel)
    nt, K, _, ntraj = size(states)
    total = nt * K * ntraj
    n = min(max_samples, total)
    vals = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = (linear % nt) + 1
        tmp = linear ÷ nt
        i = (tmp % K) + 1
        tr = (tmp ÷ K) + 1
        if quantity == :velocity
            vals[s] = Float64(states[t, i, 2, tr]) /
                max(Float64(states[t, i, 1, tr]), phys.velocity_density_floor)
        elseif quantity == :eta
            ip = periodic(i + 1, K)
            rho_edge = max(0.5 * (Float64(states[t, i, 1, tr]) + Float64(states[t, ip, 1, tr])), 1.0e-14)
            vals[s] = phys.eta0 * (rho_edge / phys.rho0)^phys.zeta
        else
            vals[s] = Float64(states[t, i, channel, tr])
        end
    end
    return vals
end

function histogram_density_fit(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Float64, length(edges) - 1)
    lo = first(edges)
    hi = last(edges)
    width = (hi - lo) / length(counts)
    @inbounds for v in values
        if lo <= v <= hi
            idx = clamp(floor(Int, (v - lo) / width) + 1, 1, length(counts))
            counts[idx] += 1.0
        end
    end
    return 0.5 .* (edges[1:end-1] .+ edges[2:end]), counts ./ max(sum(counts) * width, eps(Float64))
end

function channel_acf_fit(states::Array{Float32, 4}, channel::Int, max_lag::Int, stride::Int)
    nt, K, _, ntraj = size(states)
    lags = collect(0:stride:max_lag)
    vals = Float64[]
    mu = mean(Float64, states[:, :, channel, :])
    var = mean(x -> (Float64(x) - mu)^2, states[:, :, channel, :])
    for lag in lags
        s = 0.0
        count = 0
        @inbounds for tr in 1:ntraj, t in 1:(nt - lag), i in 1:K
            s += (Float64(states[t + lag, i, channel, tr]) - mu) *
                (Float64(states[t, i, channel, tr]) - mu)
            count += 1
        end
        push!(vals, s / max(count * var, eps(Float64)))
    end
    return lags, vals
end

function render_forward_stats_figure(path::AbstractString, obs_states::Array{Float32, 4},
        phi_states::Array{Float32, 4}, mean_true_states::Union{Nothing, Array{Float32, 4}},
        phys::FHDPhysicalParams, save_dt::Float64, params::FHDFitParams)
    rng = MersenneTwister(params.seed + 1001)
    specs = [
        (:rho, 1, "rho PDF", "rho"),
        (:m, 2, "m PDF", "m"),
        (:velocity, 2, "velocity PDF", "u"),
        (:eta, 1, "edge viscosity PDF", "eta"),
    ]
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        figure_title!(fig, "FHDChainN16 Phi forward Langevin statistics";
            subtitle="observations vs data-estimated Phi and analytic score with <M_true>")
        for (idx, (quantity, channel, title, xlabel)) in enumerate(specs)
            obs = draw_values_fit(obs_states, channel, 220_000, rng; phys=phys, quantity=quantity)
            phi = draw_values_fit(phi_states, channel, 220_000, rng; phys=phys, quantity=quantity)
            mtrue = mean_true_states === nothing ? Float64[] :
                draw_values_fit(mean_true_states, channel, 220_000, rng; phys=phys, quantity=quantity)
            lo = quantile(obs, 0.001); hi = quantile(obs, 0.999)
            pad = 0.08 * max(hi - lo, 1.0e-8)
            edges = collect(range(lo - pad, hi + pad; length=141))
            centers, po = histogram_density_fit(obs, edges)
            _, pp = histogram_density_fit(phi, edges)
            pm = mean_true_states === nothing ? nothing : histogram_density_fit(mtrue, edges)[2]
            ax = Axis(fig[1, idx]; title=title, xlabel=xlabel, ylabel="density")
            lines!(ax, centers, po; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
            lines!(ax, centers, pp; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(), label="M=Phi")
            if pm !== nothing
                lines!(ax, centers, pm; color=STYLE_PRIMARY, linestyle=:dot,
                    linewidth=curve_linewidth(), label="<Mtrue> + analytic score")
            end
            idx == 1 && axislegend(ax; position=:rt)
        end
        max_lag = min(size(obs_states, 1), size(phi_states, 1),
            mean_true_states === nothing ? typemax(Int) : size(mean_true_states, 1)) - 1
        max_lag = min(max_lag, round(Int, 8.0 / save_dt))
        stride = max(1, round(Int, 0.08 / save_dt))
        for (idx, (channel, title)) in enumerate([(1, "rho autocorrelation"), (2, "m autocorrelation")])
            lags, co = channel_acf_fit(obs_states, channel, max_lag, stride)
            _, cp = channel_acf_fit(phi_states, channel, max_lag, stride)
            cm = mean_true_states === nothing ? nothing :
                channel_acf_fit(mean_true_states, channel, max_lag, stride)[2]
            ax = Axis(fig[2, idx]; title=title, xlabel="tau", ylabel="C/C(0)")
            t = lags .* save_dt
            lines!(ax, t, co; color=STYLE_REFERENCE, linewidth=curve_linewidth())
            lines!(ax, t, cp; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth())
            if cm !== nothing
                lines!(ax, t, cm; color=STYLE_PRIMARY, linestyle=:dot, linewidth=curve_linewidth())
            end
        end
        cov_obs, _ = sampled_forward_covariance(obs_states)
        cov_phi, _ = sampled_forward_covariance(phi_states)
        cov_mtrue = mean_true_states === nothing ? nothing : sampled_forward_covariance(mean_true_states)[1]
        for (idx, mat, title) in [(3, cov_obs, "observed covariance"),
                (4, cov_phi .- cov_obs, "Phi covariance error")]
            ax = Axis(fig[2, idx]; title=title, xlabel="column", ylabel="row")
            clim = max(maximum(abs, mat), 1.0e-8)
            cmap = idx == 3 ? :viridis : STYLE_DIVERGING_SOFT
            crange = idx == 3 ? nothing : (-clim, clim)
            hm = crange === nothing ? heatmap!(ax, mat; colormap=cmap) :
                heatmap!(ax, mat; colormap=cmap, colorrange=crange)
            Colorbar(fig[2, idx, Right()], hm)
        end
        if cov_mtrue !== nothing
            ax = Axis(fig[3, 1:2]; title="<Mtrue> analytic-score covariance error",
                xlabel="column", ylabel="row")
            err = cov_mtrue .- cov_obs
            clim = max(maximum(abs, err), 1.0e-8)
            hm = heatmap!(ax, err; colormap=STYLE_DIVERGING_SOFT, colorrange=(-clim, clim))
            Colorbar(fig[3, 1:2, Right()], hm)
            metrics_phi_cov = agreement_metrics(cov_obs, cov_phi)
            metrics_mtrue_cov = agreement_metrics(cov_obs, cov_mtrue)
            text_panel!(fig[3, 3:4], String[
                @sprintf("Covariance rel.RMSE, M=Phi = %.6e", metrics_phi_cov[:relative_rmse]),
                @sprintf("Covariance corr, M=Phi = %.6f", metrics_phi_cov[:correlation]),
                @sprintf("Covariance rel.RMSE, <Mtrue> + analytic score = %.6e", metrics_mtrue_cov[:relative_rmse]),
                @sprintf("Covariance corr, <Mtrue> + analytic score = %.6f", metrics_mtrue_cov[:correlation]),
            ]; title="Forward covariance metrics")
        end
        save_figure_checked(path, fig)
    end
    return nothing
end

function sampled_forward_covariance(states::Array{Float32, 4})
    nt, K, _, ntraj = size(states)
    D = 2K
    ns = nt * ntraj
    X = Matrix{Float64}(undef, D, ns)
    col = 0
    @inbounds for tr in 1:ntraj, t in 1:nt
        col += 1
        for i in 1:K
            X[i, col] = Float64(states[t, i, 1, tr])
            X[K + i, col] = Float64(states[t, i, 2, tr])
        end
    end
    mu = mean(X; dims=2)
    X .-= mu
    return (X * transpose(X)) ./ max(ns - 1, 1), vec(mu)
end

function estimate_c_profiles_from_states_fhd(states::Array{Float32, 4}, save_dt::Float64,
        taus::Vector{Float64}, lib::FHDObservableLibrary, phys::FHDPhysicalParams,
        coord_mean::Vector{Float64}, pairs_per_lag::Int, seed::Int)
    nt, K, _, ntraj = size(states)
    D = 2K
    nobs = length(lib.names)
    C = Array{Float64}(undef, length(taus), K, nobs, D)
    rng = MersenneTwister(seed)
    for (li, tau) in enumerate(taus)
        lag = clamp(round(Int, tau / save_dt), 0, nt - 1)
        total_possible = max(1, (nt - lag) * ntraj)
        n = min(pairs_per_lag, total_possible)
        sums = zeros(Float64, K * nobs, D)
        for _ in 1:n
            tr = rand(rng, 1:ntraj)
            t = rand(rng, 1:(nt - lag))
            x0 = Array{Float32}(undef, K, 2, 1)
            xt = similar(x0)
            x0[:, :, 1] .= states[t, :, :, tr]
            xt[:, :, 1] .= states[t + lag, :, :, tr]
            xflat = Float64.(raw_flat_batch(x0))[:, 1]
            xflat .-= coord_mean
            obs = center_observables!(observable_values(xt, phys), lib)
            obsflat = Float64.(reshape(obs, K * nobs, 1))[:, 1]
            sums .+= obsflat * transpose(xflat)
        end
        C[li, :, :, :] .= reshape(sums ./ n, K, nobs, D)
    end
    return translation_profiles_fixed(C, K)
end

function render_forward_cmn_figure(path::AbstractString, taus::Vector{Float64}, tD::Float64,
        lib::FHDObservableLibrary, C_obs, C_phi, C_mean_true, selected, params::FHDFitParams)
    with_scaled_figure_style(params.figure_width, params.figure_height) do _
        fig = Figure(; size=(params.figure_width, params.figure_height))
        mphi = agreement_metrics(C_obs, C_phi)
        mmtrue = C_mean_true === nothing ? nothing : agreement_metrics(C_obs, C_mean_true)
        figure_title!(fig, "FHDChainN16 Phi forward Langevin Cmn(t)";
            subtitle=mmtrue === nothing ?
                @sprintf("Phi rel=%.3e corr=%.4f", mphi[:relative_rmse], mphi[:correlation]) :
                @sprintf("Phi rel=%.3e corr=%.4f; <Mtrue> analytic rel=%.3e corr=%.4f",
                    mphi[:relative_rmse], mphi[:correlation],
                    mmtrue[:relative_rmse], mmtrue[:correlation]))
        nshow = min(length(selected), 16)
        for idx in 1:nshow
            _, a, comp, r = selected[idx]
            row = 1 + (idx - 1) ÷ 4
            col = 1 + (idx - 1) % 4
            cname = comp == 1 ? "rho0" : "m0"
            ax = Axis(fig[row, col];
                title=@sprintf("%s vs %s, offset %d", lib.names[a], cname, r - 1),
                xlabel="tau / t_D", ylabel="C")
            x = taus ./ tD
            lines!(ax, x, C_obs[:, a, comp, r]; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="data")
            lines!(ax, x, C_phi[:, a, comp, r]; color=STYLE_SECONDARY, linestyle=:dash, linewidth=curve_linewidth(), label="M=Phi")
            if C_mean_true !== nothing
                lines!(ax, x, C_mean_true[:, a, comp, r]; color=STYLE_PRIMARY,
                    linestyle=:dot, linewidth=curve_linewidth(), label="<Mtrue> + analytic score")
            end
            idx == 1 && axislegend(ax; position=:rt)
        end
        save_figure_checked(path, fig)
    end
    return nothing
end

function save_forward_hdf5(path::AbstractString, times::Vector{Float64},
        phi_states::Array{Float32, 4}, min_phi_eig::Float64;
        mean_true_times::Union{Nothing, Vector{Float64}}=nothing,
        mean_true_states::Union{Nothing, Array{Float32, 4}}=nothing,
        min_mean_true_eig::Float64=NaN)
    ensure_parent_dir(path)
    h5open(path, "w") do h5
        write(h5, "/time", times)
        write(h5, "/phi_states", phi_states)
        write(h5, "/min_phi_diffusion_eig", min_phi_eig)
        if mean_true_times !== nothing && mean_true_states !== nothing
            write(h5, "/mean_true_analytic_time", mean_true_times)
            write(h5, "/mean_true_analytic_states", mean_true_states)
            write(h5, "/min_mean_true_analytic_diffusion_eig", min_mean_true_eig)
        end
    end
    @printf("Saved forward trajectories to %s\n", path)
    return nothing
end

function write_metrics(path::AbstractString, Phi::Matrix{Float64}, Mtrue::Matrix{Float64},
        metrics_phi, metrics_phi_cdot, K::Int, params::FHDFitParams)
    ensure_parent_dir(path)
    eigs = tangent_eigs(Phi, K)
    open(path, "w") do io
        println(io, "FHDChainN16 Phi-only diagnostics")
        println(io, "No analytic score, true mobility, or generator formula was used in DSM losses, Cdot_data, or data Phi construction.")
        println(io, "Analytic mobility is used only below for labeled ex-post validation of Phi.")
        println(io, @sprintf("Phi fit: include_zero_lag=%s, positive_lags=%d, degree=%d, Stein correction=%s",
            string(params.phi_include_zero_lag), params.phi_fit_max_lag, params.phi_fit_degree,
            string(params.phi_use_stein_correction)))
        println(io, @sprintf("Phi symmetric PSD projection = %s", string(params.phi_project_symmetric_psd)))
        println(io, @sprintf("Phi projection = %s", phi_projection_mode(params)))
        println(io, @sprintf("Phi vs <M_true> rel.RMSE = %.8e", metrics_phi[:relative_rmse]))
        println(io, @sprintf("Phi vs <M_true> corr = %.8e", metrics_phi[:correlation]))
        println(io, @sprintf("min/max eig sym(Phi) tangent = %.8e / %.8e", minimum(eigs), maximum(eigs)))
        println(io, @sprintf("Cdot Phi+stationary-score GFDT vs data rel.RMSE = %.8e", metrics_phi_cdot[:relative_rmse]))
        println(io, @sprintf("Cdot Phi+stationary-score GFDT vs data corr = %.8e", metrics_phi_cdot[:correlation]))
        println(io, "Phi block profile:")
        show(io, "text/plain", block_profile(Phi, K))
        println(io)
        println(io, "<M_true> block profile:")
        show(io, "text/plain", block_profile(Mtrue, K))
        println(io)
    end
    @printf("Saved metrics to %s\n", path)
    return nothing
end

function run_pipeline(param_file::AbstractString)
    params = load_params(param_file)
    base_dir = dirname(abspath(param_file))
    input_hdf5 = resolve_path(base_dir, params.input_hdf5)
    score_bson = resolve_path(base_dir, params.score_bson)
    artifact_bson = resolve_path(base_dir, params.artifact_bson)
    metrics_txt = resolve_path(base_dir, params.metrics_txt)
    cdot_png = resolve_path(base_dir, params.cdot_figure_png)
    phi_png = resolve_path(base_dir, params.phi_figure_png)
    forward_stats_png = resolve_path(base_dir, params.forward_stats_png)
    forward_cmn_png = resolve_path(base_dir, params.forward_cmn_png)
    forward_h5 = resolve_path(base_dir, params.forward_hdf5)
    for path in (input_hdf5, score_bson)
        require_condition(isfile(path), "Required input missing: $(path)")
    end
    ensure_parent_dir(artifact_bson)
    device = detect_fhd_device(params.device_name, params.required_gpu_name)
    activate_and_describe_device!(device, params.device_name, params.required_gpu_name)

    score_model, stats, score_sigma, _ = load_score_checkpoint(score_bson, device)
    sampler = build_fhd_pair_sampler(input_hdf5, params.burnin_fraction, params.tau_min,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    phys = load_fhd_physical_params(input_hdf5)
    @printf("FHDChainN16 Phi sampler: K=%d, D=%d, lags=%d, tD=%.5g\n",
        sampler.K, sampler.D, length(sampler.lag_steps), sampler.decorrelation_time)

    phi_taus, phi_covariances, Cdot0_raw, Phi_raw = covariance_derivative_phi(sampler, stats;
        pairs_per_lag=params.pairs_per_lag_phi, phi_fit_max_lag=params.phi_fit_max_lag,
        phi_fit_degree=params.phi_fit_degree, seed=params.seed + 1,
        include_zero_lag=params.phi_include_zero_lag,
        zero_lag_samples=params.phi_zero_lag_samples)
    V = estimate_raw_stein_matrix(score_model, sampler, stats, score_sigma, device;
        nsamples=params.stein_samples, batch_size=params.score_batch_size, seed=params.seed + 2)
    Vproj = project_block_circulant(V, sampler.K)
    Phi_full = params.phi_use_stein_correction ? Phi_raw * pinv(Vproj; rtol=1e-5) : Phi_raw
    Phi_pre_psd = project_block_circulant(Phi_full, sampler.K)
    Phi, phi_projection_info = project_phi_matrix(Phi_full, sampler.K, params)
    Mtrue = true_mean_mobility(sampler, phys; nsamples=params.true_mobility_samples, seed=params.seed + 3)
    metrics_phi = agreement_metrics(Mtrue, Phi)
    eigs = tangent_eigs(Phi, sampler.K)
    @printf("Data Phi vs <M_true>: rel.RMSE=%.6e, corr=%.6f\n",
        metrics_phi[:relative_rmse], metrics_phi[:correlation])
    @printf("sym(Phi) tangent eig min/max = %.6e / %.6e\n", minimum(eigs), maximum(eigs))
    render_phi_figure(phi_png, Phi_raw, Vproj, Phi, Mtrue, sampler.K, metrics_phi, params,
        phi_projection_info)

    lib = estimate_observable_means(sampler, phys; nsamples=100_000, seed=params.seed + 4)
    lags, taus, Cdot_data = estimate_data_cdot(sampler, phys, lib, params, stats)
    Cdot_phi = estimate_phi_gfdt_cdot(sampler, phys, lib, params, stats, Phi,
        score_model, score_sigma, device)
    data_prof = translation_profiles_fixed(Cdot_data, sampler.K)
    phi_prof = translation_profiles_fixed(Cdot_phi, sampler.K)
    selected = select_profile_channels(data_prof, params.n_plot_panels)
    metrics_phi_cdot = agreement_metrics(data_prof, phi_prof)
    @printf("Cdot Phi+stationary-score GFDT vs data: rel.RMSE=%.6e, corr=%.6f\n",
        metrics_phi_cdot[:relative_rmse], metrics_phi_cdot[:correlation])
    render_cdot_figure(cdot_png, taus, sampler.decorrelation_time, lib, data_prof,
        phi_prof, selected, metrics_phi_cdot, params)
    write_metrics(metrics_txt, Phi, Mtrue, metrics_phi, metrics_phi_cdot, sampler.K, params)

    phi_forward_states = nothing
    mean_true_forward_states = nothing
    C_obs_forward = nothing
    C_phi_forward = nothing
    C_mean_true_forward = nothing
    min_phi_eig = NaN
    min_mean_true_eig = NaN
    metrics_forward_phi = nothing
    metrics_forward_mean_true = nothing
    if params.run_forward_validation
        @printf("Starting forward Langevin validation for M=Phi and <M_true> analytic-score reference.\n")
        phi_times, phi_forward_states, min_phi_eig = integrate_forward_langevin(score_model,
            score_sigma, stats, Phi, nothing, nothing, sampler, phys, params, device; mode=:phi)
        mean_true_times, mean_true_forward_states, min_mean_true_eig = integrate_forward_langevin(score_model,
            score_sigma, stats, Mtrue, nothing, nothing, sampler, phys, params, device; mode=:mean_true_analytic)
        save_forward_hdf5(forward_h5, phi_times, phi_forward_states, min_phi_eig;
            mean_true_times=mean_true_times, mean_true_states=mean_true_forward_states,
            min_mean_true_eig=min_mean_true_eig)
        obs_stop = min(size(sampler.states, 1), sampler.start_idx + size(phi_forward_states, 1) - 1)
        obs_validation_states = sampler.states[sampler.start_idx:obs_stop, :, :, :]
        forward_actual_save_dt = length(phi_times) > 1 ? phi_times[2] - phi_times[1] :
            (params.forward_save_dt > 0.0 ? params.forward_save_dt : sampler.save_dt)
        render_forward_stats_figure(forward_stats_png, obs_validation_states, phi_forward_states,
            mean_true_forward_states, phys, forward_actual_save_dt, params)
        coord_mean = Float64.(mean_flat(stats))
        C_obs_forward = estimate_c_profiles_from_states_fhd(obs_validation_states, sampler.save_dt,
            taus, lib, phys, coord_mean, 30000, params.seed + 901)
        C_phi_forward = estimate_c_profiles_from_states_fhd(phi_forward_states, forward_actual_save_dt,
            taus, lib, phys, coord_mean, 30000, params.seed + 902)
        mean_true_actual_save_dt = length(mean_true_times) > 1 ? mean_true_times[2] - mean_true_times[1] :
            forward_actual_save_dt
        C_mean_true_forward = estimate_c_profiles_from_states_fhd(mean_true_forward_states,
            mean_true_actual_save_dt, taus, lib, phys, coord_mean, 30000, params.seed + 903)
        metrics_forward_phi = agreement_metrics(C_obs_forward, C_phi_forward)
        metrics_forward_mean_true = agreement_metrics(C_obs_forward, C_mean_true_forward)
        render_forward_cmn_figure(forward_cmn_png, taus, sampler.decorrelation_time, lib,
            C_obs_forward, C_phi_forward, C_mean_true_forward, selected, params)
        open(metrics_txt, "a") do io
            println(io)
            println(io, "Phi and <M_true> analytic-score forward validation")
            println(io, @sprintf("Forward min diffusion eig Phi = %.8e", min_phi_eig))
            println(io, @sprintf("Forward Cmn Phi vs observation rel.RMSE = %.8e", metrics_forward_phi[:relative_rmse]))
            println(io, @sprintf("Forward Cmn Phi vs observation corr = %.8e", metrics_forward_phi[:correlation]))
            println(io, @sprintf("Forward min diffusion eig <M_true> analytic = %.8e", min_mean_true_eig))
            println(io, @sprintf("Forward Cmn <M_true> analytic vs observation rel.RMSE = %.8e", metrics_forward_mean_true[:relative_rmse]))
            println(io, @sprintf("Forward Cmn <M_true> analytic vs observation corr = %.8e", metrics_forward_mean_true[:correlation]))
        end
    end

    BSON.@save artifact_bson params phi_taus phi_covariances Cdot0_raw Phi_raw V Vproj Phi_full Phi_pre_psd Phi phi_projection_info Mtrue metrics_phi lib lags taus Cdot_data Cdot_phi data_prof phi_prof selected metrics_phi_cdot phi_forward_states mean_true_forward_states C_obs_forward C_phi_forward C_mean_true_forward min_phi_eig min_mean_true_eig metrics_forward_phi metrics_forward_mean_true
    @printf("Saved Phi-only artifacts to %s\n", artifact_bson)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    param_file = isempty(ARGS) ? DEFAULT_PARAM_FILE : abspath(ARGS[1])
    run_pipeline(param_file)
end
