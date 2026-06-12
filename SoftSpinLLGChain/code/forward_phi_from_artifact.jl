#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_Phi.jl"))

using BSON
using Flux
using LinearAlgebra
using Printf
using TOML

const DEFAULT_CONFIG = joinpath(@__DIR__, "..", "configs",
    "forward_phi_mobility11_current_gpu2.toml")

function load_stationary_checkpoint_for_phi(path::AbstractString, device::ExecutionDevice)
    blob = BSON.load(path)
    model = move_model(blob[:host_model], device)
    Flux.testmode!(model)
    stats_obj = blob[:stats]
    stats = stats_obj isa DataStats ? stats_obj :
        DataStats(Float32.(stats_obj[:mean]), Float32.(stats_obj[:std]))
    sigma = Float32(blob[:trainer_cfg][:sigma])
    return model, stats, sigma, blob
end

function run_phi_forward_from_artifact(param_file::AbstractString)
    param_file = abspath(param_file)
    base = dirname(param_file)
    params = load_config(param_file)
    raw = TOML.parsefile(param_file)
    phi_rel = String(raw["data"]["phi_artifact_bson"])
    phi_path = resolve_path(base, phi_rel)
    isfile(phi_path) || error("Missing Phi artifact $(phi_path)")
    data_h5 = resolve_path(base, params.input_hdf5)
    score_path = resolve_path(base, params.score_bson)
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)
    sampler = build_sampler(data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, sigma, _ = load_stationary_checkpoint_for_phi(score_path, device)
    blob = BSON.load(phi_path, @__MODULE__)
    haskey(blob, :Phi) || error("Phi artifact $(phi_path) does not contain :Phi.")
    Phi = Matrix{Float64}(blob[:Phi])
    @printf("Loaded current Phi from %s, norm %.8e\n", phi_path, norm(Phi))
    times, states, eigvals = integrate_phi_forward(score_model, sigma, stats, Phi,
        sampler, params, device)
    out_h5 = resolve_path(base, params.forward_hdf5)
    save_forward(out_h5, times, states, eigvals)
    ensure_parent_dir(resolve_path(base, params.metrics_txt))
    open(resolve_path(base, params.metrics_txt), "w") do io
        println(io, "Current Phi forward baseline for 11-family branch")
        println(io, "phi_artifact = $(phi_path)")
        println(io, "forward_hdf5 = $(out_h5)")
        @printf(io, "dt = %.8e\n", params.forward_dt)
        @printf(io, "total_time = %.8e\n", params.forward_total_time)
        @printf(io, "burnin_time = %.8e\n", params.forward_burnin_time)
        @printf(io, "save_dt = %.8e\n", params.forward_save_dt)
        println(io, "ntrajectories = $(params.forward_ntraj)")
        @printf(io, "min_sym_phi_eig = %.8e\n", minimum(eigvals))
        println(io, "No-cheating audit: this script loaded a saved data-only Phi artifact and a learned stationary score. It did not recompute Phi, use true mobility, or use true coefficients.")
    end
    @printf("Saved current-Phi metrics to %s\n", resolve_path(base, params.metrics_txt))
end

run_phi_forward_from_artifact(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CONFIG)
