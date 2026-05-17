#!/usr/bin/env julia

include(joinpath(@__DIR__, "score.jl"))

using HDF5

function load_score_langevin_states(path::AbstractString)
    h5open(path, "r") do f
        return Float32.(read(f["/trajectories/states"]))
    end
end

function posthoc_score_metrics(cfg_path::AbstractString, score_langevin_h5::AbstractString,
        metrics_txt::AbstractString)
    base = dirname(cfg_path)
    params = load_config(cfg_path)
    data_h5 = resolve_path(base, params.input_hdf5)
    model_path = resolve_path(base, params.output_bson)
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)
    p = load_phys(data_h5)
    dataset, _, states, start = load_score_dataset(data_h5, params, MersenneTwister(params.seed))
    model, _, _ = load_or_train(params, dataset, p, model_path, device)
    diag, _ = diagnostics(model, dataset, p, params, device)
    gen = load_score_langevin_states(score_langevin_h5)
    obs = states[start:end, :, :, :]
    langevin_diag = score_langevin_metrics(obs, gen)
    save_score_metrics(metrics_txt, params, diag, langevin_diag)
    return nothing
end

function main()
    length(ARGS) >= 3 || error("Usage: score_posthoc_metrics.jl CONFIG SCORE_LANGEVIN_H5 OUT_METRICS_TXT")
    posthoc_score_metrics(ARGS[1], ARGS[2], ARGS[3])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
