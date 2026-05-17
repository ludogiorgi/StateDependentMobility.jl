#!/usr/bin/env julia

include(joinpath(@__DIR__, "src", "spin_common.jl"))

const DEFAULT_PARAM_FILE = normpath(joinpath(@__DIR__, "..", "configs", "sim.toml"))

Base.@kwdef struct SimConfig
    phys::SpinParams
    dt::Float64
    ntrajectories::Int
    requested_threads::Int
    seed::Int
    initial_noise::Float64
    branch_balance::Bool
    pilot_t1::Float64
    pilot_save_dt::Float64
    pilot_burnin_fraction::Float64
    pilot_max_lag_time::Float64
    decorrelation_threshold::Float64
    snapshots_per_decorrelation::Int
    reference_hdf5::String
    fallback_uncorrelated_count::Float64
    burnin_fraction::Float64
    histogram_bins::Int
    max_correlation_lags::Int
    max_pdf_samples::Int
    figure_width::Int
    figure_height::Int
    dynamics_width::Int
    dynamics_height::Int
    trajectory_width::Int
    trajectory_height::Int
    dynamics_window_decorrelation_times::Float64
    dynamics_max_frames::Int
    output_hdf5::String
    output_summary_png::String
    output_dynamics_png::String
    output_trajectories_png::String
end

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    sim = raw["simulation"]
    ic = raw["initial_condition"]
    cal = raw["calibration"]
    ref = raw["reference_previous"]
    stats = raw["statistics"]
    fig = raw["figure"]
    out = raw["output"]
    return SimConfig(
        phys=spin_params_from_table(sim),
        dt=Float64(sim["dt"]),
        ntrajectories=Int(sim["ntrajectories"]),
        requested_threads=Int(get(sim, "requested_threads", Threads.nthreads())),
        seed=Int(sim["seed"]),
        initial_noise=Float64(get(ic, "initial_noise", 0.03)),
        branch_balance=Bool(get(ic, "branch_balance", true)),
        pilot_t1=Float64(get(cal, "pilot_t1", 250.0)),
        pilot_save_dt=Float64(get(cal, "pilot_save_dt", 0.02)),
        pilot_burnin_fraction=Float64(get(cal, "pilot_burnin_fraction", 0.25)),
        pilot_max_lag_time=Float64(get(cal, "pilot_max_lag_time", 150.0)),
        decorrelation_threshold=Float64(get(cal, "decorrelation_threshold", 0.05)),
        snapshots_per_decorrelation=Int(get(sim, "production_snapshots_per_decorrelation", 100)),
        reference_hdf5=String(get(ref, "hdf5_file", "")),
        fallback_uncorrelated_count=Float64(get(ref, "fallback_uncorrelated_count", 2777.7814226519336)),
        burnin_fraction=Float64(get(stats, "burnin_fraction", 0.1)),
        histogram_bins=Int(get(stats, "histogram_bins", 180)),
        max_correlation_lags=Int(get(stats, "max_correlation_lags", 500)),
        max_pdf_samples=Int(get(stats, "max_pdf_samples", 500000)),
        figure_width=Int(get(fig, "width", 2800)),
        figure_height=Int(get(fig, "height", 2200)),
        dynamics_width=Int(get(fig, "dynamics_width", 2600)),
        dynamics_height=Int(get(fig, "dynamics_height", 1800)),
        trajectory_width=Int(get(fig, "trajectory_width", 2600)),
        trajectory_height=Int(get(fig, "trajectory_height", 1800)),
        dynamics_window_decorrelation_times=Float64(get(fig, "dynamics_window_decorrelation_times", 10.0)),
        dynamics_max_frames=Int(get(fig, "dynamics_max_frames", 5000)),
        output_hdf5=String(out["hdf5_file"]),
        output_summary_png=String(out["summary_png"]),
        output_dynamics_png=String(out["dynamics_png"]),
        output_trajectories_png=String(out["trajectories_png"]),
    )
end

function initial_state(p::SpinParams, rng::AbstractRNG, noise::Float64, branch::Int)
    x = zeros(Float64, p.N, SPIN_CHANNELS)
    mz = branch * meq(p)
    @inbounds for i in 1:p.N
        x[i, 1] = noise * randn(rng)
        x[i, 2] = noise * randn(rng)
        x[i, 3] = mz + noise * randn(rng)
    end
    return x
end

function simulate_trajectory(p::SpinParams; dt::Float64, t1::Float64, save_dt::Float64,
        seed::Int, initial_noise::Float64, branch::Int)
    save_every = max(1, round(Int, save_dt / dt))
    actual_save_dt = save_every * dt
    nsteps = ceil(Int, t1 / dt)
    nsaved = 1 + nsteps ÷ save_every
    states = Array{Float32}(undef, nsaved, p.N, SPIN_CHANNELS)
    times = Vector{Float64}(undef, nsaved)
    rng = MersenneTwister(seed)
    x = initial_state(p, rng, initial_noise, branch)
    work = make_work(p)
    saved = 1
    states[saved, :, :] .= Float32.(x)
    times[saved] = 0.0
    for step in 1:nsteps
        em_step!(x, p, dt, rng, work)
        if step % save_every == 0
            saved += 1
            states[saved, :, :] .= Float32.(x)
            times[saved] = step * dt
        end
    end
    return times[1:saved], states[1:saved, :, :]
end

function simulate_ensemble(p::SpinParams, cfg::SimConfig, t1::Float64, save_dt::Float64)
    save_every = max(1, round(Int, save_dt / cfg.dt))
    actual_save_dt = save_every * cfg.dt
    nsteps = ceil(Int, t1 / cfg.dt)
    nsaved = 1 + nsteps ÷ save_every
    states = Array{Float32}(undef, nsaved, p.N, SPIN_CHANNELS, cfg.ntrajectories)
    times = collect(0:actual_save_dt:actual_save_dt * (nsaved - 1))
    Threads.@threads for tr in 1:cfg.ntrajectories
        branch = cfg.branch_balance ? (isodd(tr) ? 1 : -1) : 1
        rng = MersenneTwister(cfg.seed + 1000 * tr)
        x = initial_state(p, rng, cfg.initial_noise, branch)
        work = make_work(p)
        saved = 1
        states[saved, :, :, tr] .= Float32.(x)
        for step in 1:nsteps
            em_step!(x, p, cfg.dt, rng, work)
            if step % save_every == 0
                saved += 1
                states[saved, :, :, tr] .= Float32.(x)
            end
        end
    end
    return times, states, save_every
end

function observable_matrix(states::Array{Float32, 3}, p::SpinParams)
    nt = size(states, 1)
    obs = Matrix{Float64}(undef, nt, 7)
    @inbounds for t in 1:nt
        x = @view states[t, :, :]
        mx = mean(Float64, @view x[:, 1])
        my = mean(Float64, @view x[:, 2])
        mz = mean(Float64, @view x[:, 3])
        r = mean(i -> sqrt(sum(abs2, @view x[i, :])), 1:p.N)
        trans = mean(i -> x[i, 1]^2 + x[i, 2]^2, 1:p.N)
        z2 = mean(i -> x[i, 3]^2, 1:p.N)
        energy = potential_energy(Float64.(x), p) / p.N
        obs[t, :] .= (mx, my, mz, r, trans, z2, energy)
    end
    return obs
end

function acf_1d(v::AbstractVector{<:Real}, maxlag::Int)
    x = Float64.(v)
    x .-= mean(x)
    denom = sum(abs2, x)
    out = Vector{Float64}(undef, maxlag + 1)
    if denom <= eps(Float64)
        fill!(out, 0.0)
        out[1] = 1.0
        return out
    end
    @inbounds for lag in 0:maxlag
        s = 0.0
        n = length(x) - lag
        for i in 1:n
            s += x[i + lag] * x[i]
        end
        out[lag + 1] = s / denom
    end
    return out
end

function estimate_decorrelation_time(times::Vector{Float64}, states::Array{Float32, 3},
        cfg::SimConfig)
    p = cfg.phys
    start = burnin_start_index(length(times), cfg.pilot_burnin_fraction)
    obs = observable_matrix(states[start:end, :, :], p)
    save_dt = times[2] - times[1]
    maxlag = min(size(obs, 1) - 2, floor(Int, cfg.pilot_max_lag_time / save_dt))
    acfs = [acf_1d(obs[:, j], maxlag) for j in 1:size(obs, 2)]
    # The benchmark targets the finite-lag transverse spin/precessional dynamics
    # described in system.txt. Longitudinal branch-mixing observables are plotted
    # as diagnostics, but they are not used to set the saved resolution.
    decorrelation_columns = (1, 2) # global Mx, My
    env = [maximum(abs(acfs[j][k]) for j in decorrelation_columns) for k in 1:(maxlag + 1)]
    idx = findfirst(<(cfg.decorrelation_threshold), env)
    capped = idx === nothing
    lag_idx = capped ? maxlag : idx - 1
    tD = max(lag_idx * save_dt, 10 * save_dt)
    return tD, collect(0:maxlag) .* save_dt, env, acfs, capped
end

function previous_uncorrelated_count(path::AbstractString, fallback::Float64)
    isempty(path) && return fallback, "fallback"
    if isfile(path)
        try
            val = h5read(path, "/metadata/production_uncorrelated_count")
            return Float64(val), path
        catch
            try
                t1 = Float64(h5read(path, "/metadata/t1"))
                tD = Float64(h5read(path, "/statistics/correlations/t_decorrelation"))
                return t1 / tD, path
            catch
            end
        end
    end
    return fallback, "fallback"
end

function marginal_pdf(vals::AbstractVector{<:Real}, nbins::Int)
    kd = kde(Float64.(vals))
    return kd.x, kd.density
end

function collect_flat_samples(states::Array{Float32, 4}, start::Int, max_samples::Int, rng::AbstractRNG)
    nt, N, _, ntraj = size(states)
    total = (nt - start + 1) * ntraj
    n = min(max_samples, total)
    flat = Matrix{Float64}(undef, 3N, n)
    @inbounds for s in 1:n
        linear = rand(rng, 0:(total - 1))
        t = start + (linear % (nt - start + 1))
        tr = (linear ÷ (nt - start + 1)) + 1
        flat[:, s] .= flatten_state(@view states[t, :, :, tr])
    end
    return flat
end

function ensemble_correlations(times::Vector{Float64}, states::Array{Float32, 4}, start::Int,
        max_lags::Int, p::SpinParams)
    nt = size(states, 1)
    ntraj = size(states, 4)
    maxlag = min(max_lags, nt - start - 1)
    labels = ["Mx", "My", "Mz", "|m|", "mperp2", "mz2", "U/N"]
    all_obs = [observable_matrix(states[start:end, :, :, tr], p) for tr in 1:ntraj]
    acfs = Matrix{Float64}(undef, maxlag + 1, length(labels))
    for j in 1:length(labels)
        num = zeros(Float64, maxlag + 1)
        den = 0.0
        for tr in 1:ntraj
            v = all_obs[tr][:, j]
            v = v .- mean(v)
            den += sum(abs2, v)
            for lag in 0:maxlag
                num[lag + 1] += dot(@view(v[1:end-lag]), @view(v[1+lag:end]))
            end
        end
        acfs[:, j] .= num ./ max(den, eps(Float64))
    end
    return collect(0:maxlag) .* (times[2] - times[1]), acfs, labels
end

function save_dataset(path::AbstractString, cfg::SimConfig, times::Vector{Float64},
        states::Array{Float32, 4}, save_every::Int, tD::Float64, ref_count::Float64,
        ref_source::String, corr_lags, acfs, acf_labels, flat_samples)
    p = cfg.phys
    ensure_parent_dir(path)
    states_flat = Array{Float32}(undef, size(states, 1), state_dim(p), size(states, 4))
    @inbounds for tr in 1:size(states, 4), t in 1:size(states, 1)
        states_flat[t, :, tr] .= Float32.(flatten_state(@view states[t, :, :, tr]))
    end
    h5open(path, "w") do f
        f["/trajectories/time"] = times
        f["/trajectories/states"] = states
        f["/trajectories/states_flat"] = states_flat
        f["/trajectories/channel_names"] = channel_names()
        f["/trajectories/flat_order"] = flat_order(p)
        f["/metadata/model_name"] = "SoftSpinLLGChain"
        f["/metadata/state_ordering"] = "site-major: site1_mx, site1_my, site1_mz, site2_mx, ..."
        f["/metadata/N"] = p.N
        f["/metadata/state_dimension"] = state_dim(p)
        f["/metadata/dt"] = cfg.dt
        f["/metadata/save_dt"] = times[2] - times[1]
        f["/metadata/save_every"] = save_every
        f["/metadata/t_D"] = tD
        f["/metadata/t1"] = times[end]
        f["/metadata/T"] = times[end]
        f["/metadata/production_uncorrelated_count"] = times[end] / tD
        f["/metadata/snapshots_per_decorrelation"] = tD / (times[2] - times[1])
        f["/metadata/burnin_fraction"] = cfg.burnin_fraction
        f["/metadata/seed"] = cfg.seed
        f["/metadata/ntrajectories"] = cfg.ntrajectories
        f["/metadata/lambda"] = p.lambda
        f["/metadata/mstar"] = p.mstar
        f["/metadata/J"] = p.J
        f["/metadata/K"] = p.K
        f["/metadata/Theta"] = p.theta
        f["/metadata/gamma"] = p.gamma
        f["/metadata/alpha_perp"] = p.alpha_perp
        f["/metadata/alpha_parallel"] = p.alpha_parallel
        f["/metadata/eps"] = p.eps
        f["/metadata/constraints"] = "none; unconstrained soft spins, periodic lattice"
        f["/metadata/decorrelation_observables"] = "global transverse magnetization Mx, My from pilot trajectory"
        f["/reference_previous/source"] = ref_source
        f["/reference_previous/uncorrelated_count"] = ref_count
        f["/statistics/correlations/lags"] = corr_lags
        f["/statistics/correlations/acfs"] = acfs
        f["/statistics/correlations/labels"] = acf_labels
        f["/statistics/correlations/t_decorrelation"] = tD
        f["/statistics/covariance_flat"] = cov(permutedims(flat_samples))
        norms = vec([sqrt(sum(abs2, states[t, i, :, tr])) for t in axes(states, 1), i in 1:p.N, tr in axes(states, 4)])
        f["/statistics/constraints/min_spin_norm"] = minimum(norms)
        f["/statistics/constraints/max_spin_norm"] = maximum(norms)
        f["/statistics/constraints/mean_spin_norm"] = mean(norms)
    end
    @printf("Saved dataset to %s\n", path)
    return nothing
end

function render_summary(path, cfg, times, states, tD, ref_count, corr_lags, acfs, labels, flat)
    p = cfg.phys
    with_theme() do
        fig = Figure(; size=(cfg.figure_width, cfg.figure_height))
        subtitle = @sprintf("N=%d, D=%d, save_dt=%.5g, t_D=%.5g, T/t_D=%.2f, t_D/save_dt=%.2f",
            p.N, state_dim(p), times[2] - times[1], tD, times[end] / tD, tD / (times[2] - times[1]))
        figure_title!(fig, "Soft-spin LLG simulation summary"; subtitle)
        comp_labels = ["mx", "my", "mz", "|m|"]
        vals = Dict(
            "mx" => vec(Float64.(states[:, :, 1, :])),
            "my" => vec(Float64.(states[:, :, 2, :])),
            "mz" => vec(Float64.(states[:, :, 3, :])),
            "|m|" => vec([sqrt(sum(abs2, states[t, i, :, tr])) for t in axes(states, 1), i in 1:p.N, tr in axes(states, 4)]),
        )
        for (j, lab) in enumerate(comp_labels)
            ax = Axis(fig[1, j]; title="PDF $(lab)", xlabel=lab, ylabel="density")
            xs, ys = marginal_pdf(vals[lab][1:max(1, length(vals[lab]) ÷ min(length(vals[lab]), cfg.max_pdf_samples)):end], cfg.histogram_bins)
            lines!(ax, xs, ys; color=STYLE_PRIMARY, linewidth=2)
        end
        axc = Axis(fig[2, 1:2]; title="Flat covariance", xlabel="flat index", ylabel="flat index")
        heatmap!(axc, cov(permutedims(flat)); colormap=:balance)
        axa = Axis(fig[2, 3:4]; title="ACF envelope and channels", xlabel="lag", ylabel="correlation")
        for j in 1:size(acfs, 2)
            lines!(axa, corr_lags, acfs[:, j]; linewidth=1.5, label=labels[j])
        end
        lines!(axa, corr_lags, vec(maximum(abs.(acfs); dims=2)); color=:black, linewidth=3, label="envelope")
        axislegend(axa; position=:rt)
        axtext = Axis(fig[3, 1:4]; title="Sampling and audit", xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false)
        text!(axtext, 0.02, 0.85; space=:relative, align=(:left, :top), fontsize=22,
            text=@sprintf("Reference uncorrelated count: %.6f\nProduction uncorrelated count: %.6f\nSaved snapshots per decorrelation: %.6f\nPilot t_D observables: global transverse Mx, My\nFlat ordering: site-major [site_i mx,my,mz]\nNo analytic score/mobility was used for data-driven training targets; this script only simulates the reference process.",
                ref_count, times[end] / tD, tD / (times[2] - times[1])))
        hidedecorations!(axtext)
        rowsize!(fig.layout, 3, Fixed(300))
        save_figure_checked(path, fig)
    end
end

function render_dynamics(path, cfg, times, states, tD)
    tr = 1
    window_stop = min(times[end], cfg.dynamics_window_decorrelation_times * tD)
    window_idx = findall(<=(window_stop), times)
    isempty(window_idx) && (window_idx = collect(eachindex(times)))
    stride = max(1, length(window_idx) ÷ cfg.dynamics_max_frames)
    idx = window_idx[1:stride:end]
    fig = Figure(; size=(cfg.dynamics_width, cfg.dynamics_height))
    figure_title!(fig, "Soft-spin LLG full-resolution dynamics window";
        subtitle=@sprintf("Showing every %d saved frame in plotted window; saved resolution dt=%.5g", stride, times[2] - times[1]))
    for c in 1:3
        ax = Axis(fig[c, 1]; title="$(channel_names()[c]) Hovmoller", xlabel="time", ylabel="site")
        heatmap!(ax, times[idx], 1:size(states, 2), Float64.(states[idx, :, c, tr]); colormap=:balance)
    end
    save_figure_checked(path, fig)
end

function render_trajectories(path, cfg, times, states)
    p = cfg.phys
    tr = 1
    sites = unique(clamp.(round.(Int, [1, p.N ÷ 3, 2p.N ÷ 3, p.N]), 1, p.N))
    fig = Figure(; size=(cfg.trajectory_width, cfg.trajectory_height))
    figure_title!(fig, "Soft-spin LLG representative trajectories";
        subtitle=@sprintf("Saved resolution dt=%.5g, trajectory=%d", times[2] - times[1], tr))
    ax1 = Axis(fig[1, 1:2]; title="site traces", xlabel="time", ylabel="component")
    for site in sites, c in 1:3
        lines!(ax1, times, Float64.(states[:, site, c, tr]); linewidth=1.2,
            label="s$(site) $(channel_names()[c])")
    end
    axislegend(ax1; position=:rt, nbanks=2)
    ax2 = Axis(fig[2, 1]; title="site 1 phase portrait", xlabel="mx", ylabel="my")
    lines!(ax2, Float64.(states[:, 1, 1, tr]), Float64.(states[:, 1, 2, tr]); color=STYLE_ACCENT)
    ax3 = Axis(fig[2, 2]; title="spin norm traces", xlabel="time", ylabel="|m_i|")
    for site in sites
        vals = [sqrt(sum(abs2, states[t, site, :, tr])) for t in axes(states, 1)]
        lines!(ax3, times, vals; linewidth=1.4, label="site $(site)")
    end
    axislegend(ax3; position=:rb)
    save_figure_checked(path, fig)
end

function run_pipeline(param_file::AbstractString)
    base = dirname(param_file)
    cfg = load_config(param_file)
    p = cfg.phys
    @printf("SoftSpinLLGChain Step 1 simulation: N=%d, D=%d, threads=%d\n",
        p.N, state_dim(p), Threads.nthreads())
    require_condition(Threads.nthreads() >= min(cfg.requested_threads, cfg.ntrajectories),
        "Julia was started with fewer threads than requested; rerun with --threads $(cfg.requested_threads).")
    pilot_times, pilot_states = simulate_trajectory(p; dt=cfg.dt, t1=cfg.pilot_t1,
        save_dt=cfg.pilot_save_dt, seed=cfg.seed + 11, initial_noise=cfg.initial_noise, branch=1)
    tD, pilot_lags, pilot_env, _, capped = estimate_decorrelation_time(pilot_times, pilot_states, cfg)
    require_condition(!capped, "Pilot ACF window capped the decorrelation estimate; increase pilot_max_lag_time.")
    ref_path = resolve_path(base, cfg.reference_hdf5)
    ref_count, ref_source = previous_uncorrelated_count(ref_path, cfg.fallback_uncorrelated_count)
    save_dt = tD / cfg.snapshots_per_decorrelation
    save_every = max(1, round(Int, save_dt / cfg.dt))
    actual_save_dt = save_every * cfg.dt
    T = ref_count * tD
    nsteps = round(Int, T / cfg.dt)
    T_actual = nsteps * cfg.dt
    @printf("Pilot t_D %.8g; production save_dt %.8g; T %.8g; T/t_D %.8g\n",
        tD, actual_save_dt, T_actual, T_actual / tD)
    times, states, production_save_every = simulate_ensemble(p, cfg, T_actual, actual_save_dt)
    start = burnin_start_index(length(times), cfg.burnin_fraction)
    corr_lags, acfs, labels = ensemble_correlations(times, states, start, cfg.max_correlation_lags, p)
    flat = collect_flat_samples(states, start, cfg.max_pdf_samples, MersenneTwister(cfg.seed + 77))
    h5_path = resolve_path(base, cfg.output_hdf5)
    save_dataset(h5_path, cfg, times, states, production_save_every, tD, ref_count, ref_source,
        corr_lags, acfs, labels, flat)
    render_summary(resolve_path(base, cfg.output_summary_png), cfg, times, states, tD, ref_count,
        corr_lags, acfs, labels, flat)
    render_dynamics(resolve_path(base, cfg.output_dynamics_png), cfg, times, states, tD)
    render_trajectories(resolve_path(base, cfg.output_trajectories_png), cfg, times, states)
    ratio = tD / (times[2] - times[1])
    require_condition(abs(ratio - cfg.snapshots_per_decorrelation) <= 1.0,
        "t_D/save_dt=$(ratio) is not approximately $(cfg.snapshots_per_decorrelation).")
    require_condition(abs(times[end] / tD - ref_count) / ref_count < 5.0e-4,
        "Production T/t_D does not match reference count.")
    @printf("Step 1 complete. Dataset resolution: states has size %s, save_dt %.8g, t_D/save_dt %.4f.\n",
        string(size(states)), times[2] - times[1], ratio)
    @printf("No-cheating audit: simulation generated trajectories only; no analytic quantities were saved as training labels or data-driven estimators.\n")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_PARAM_FILE)
end
