#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "..", "code", "cond_score.jl"))

using BSON
using Printf

function parse_args(args)
    opts = Dict{String, String}()
    positionals = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if key in ("audit-only", "resume-existing")
                opts[key] = "true"
                i += 1
            else
                i == length(args) && error("Missing value for --$(key)")
                opts[key] = args[i + 1]
                i += 2
            end
        else
            push!(positionals, a)
            i += 1
        end
    end
    length(positionals) >= 1 || error("Usage: finetune_cond_from_checkpoint.jl CONFIG --warm-start CHECKPOINT [--audit-only] [--resume-existing]")
    opts["config"] = positionals[1]
    return opts
end

resolve_cli_path(path::AbstractString) = isabspath(path) ? path : normpath(joinpath(pwd(), path))

function assert_file(path::AbstractString, label::AbstractString)
    isfile(path) || error("Missing $(label): $(path)")
    return abspath(path)
end

function branch_training_state(path::AbstractString)
    isfile(path) || return nothing
    blob = BSON.load(path, @__MODULE__)
    return get(blob, :training_state, nothing)
end

function completed_epoch_from_blob(blob)
    state = get(blob, :training_state, nothing)
    hist = get(blob, :history, Dict{Symbol, Any}(:train_loss => Float64[]))
    return state === nothing ?
        Int(get(get(blob, :metadata, Dict{Symbol, Any}()), :completed_epoch,
            length(get(hist, :train_loss, Float64[])))) :
        Int(get(state, :completed_epoch,
            get(get(blob, :metadata, Dict{Symbol, Any}()), :completed_epoch,
                length(get(hist, :train_loss, Float64[])))))
end

function audit_inputs(config_path::String, warm_start_path::String)
    params = load_config(config_path)
    base = dirname(config_path)
    data_h5 = assert_file(resolve_path(base, params.input_hdf5), "input HDF5")
    score_path = assert_file(resolve_path(base, params.score_bson), "stationary score")
    phi_path = assert_file(resolve_path(base, params.phi_artifact_bson), "Phi artifact")
    model_path = resolve_path(base, params.output_bson)
    warm_start_path = assert_file(warm_start_path, "warm-start conditional checkpoint")

    warm = BSON.load(warm_start_path, @__MODULE__)
    haskey(warm, :host_model) || error("Warm-start checkpoint lacks :host_model")
    haskey(warm, :model_cfg) || error("Warm-start checkpoint lacks :model_cfg")
    warm_meta = get(warm, :metadata, Dict{Symbol, Any}())
    warm_trainer = get(warm, :trainer_cfg, Dict{Symbol, Any}())
    warm_scale = get(warm_meta, :residual_output_scale,
        get(warm_trainer, :residual_output_scale, "unknown"))
    String(warm_scale) == "sigma" ||
        error("Warm-start residual_output_scale must be sigma; got $(warm_scale)")
    get(warm_meta, :conditioning_smoothed, false) == false ||
        error("Warm-start checkpoint says conditioning was smoothed.")

    phi = BSON.load(phi_path, @__MODULE__)
    for key in (:Cdot_data, :Cdot_phi, :Phi, :observable_names, :observable_means, :lags)
        haskey(phi, key) || error("Phi artifact $(phi_path) lacks $(key)")
    end
    length(Vector{Int}(phi[:lags])) >= 24 ||
        error("Phi artifact has fewer than 24 lags: $(length(phi[:lags]))")

    score_blob = BSON.load(score_path, @__MODULE__)
    haskey(score_blob, :host_model) || error("Stationary score lacks :host_model")
    haskey(score_blob, :stats) || error("Stationary score lacks :stats")
    haskey(score_blob, :trainer_cfg) || error("Stationary score lacks :trainer_cfg")

    @printf("Audit OK\n")
    @printf("  config: %s\n", config_path)
    @printf("  data: %s\n", data_h5)
    @printf("  stationary score: %s\n", score_path)
    @printf("  Phi artifact: %s\n", phi_path)
    @printf("  warm start: %s\n", warm_start_path)
    @printf("  output model: %s\n", model_path)
    @printf("  requested device: %s, required GPU: %s\n", params.device, params.required_gpu_name)
    @printf("  epochs: %d, batches/epoch: %d, lr: %.6g\n",
        params.epochs, params.batches_per_epoch, params.learning_rate)
    return (; params, base, data_h5, score_path, phi_path, model_path, warm_start_path)
end

function write_metrics(path::AbstractString, history, diag)
    ensure_parent_dir(path)
    open(path, "w") do io
        println(io, "SoftSpinLLGChain conditional fine-tune metrics")
        if !isempty(get(history, :train_loss, Float64[]))
            println(io, @sprintf("last DSM loss = %.8e", history[:train_loss][end]))
        end
        for key in (:target_rms, :prediction_rms, :null_mse,
                :fractional_improvement, :prediction_target_cosine)
            if haskey(history, key) && !isempty(history[key])
                println(io, @sprintf("last DSM %s = %.8e", String(key), history[key][end]))
            end
        end
        if haskey(diag, :operator_phi_metrics)
            println(io, @sprintf("Phi conditional-vs-GFDT rel.RMSE = %.8e",
                diag[:operator_phi_metrics][:relative_rmse]))
            println(io, @sprintf("Phi conditional-vs-GFDT corr = %.8e",
                diag[:operator_phi_metrics][:correlation]))
        end
        if haskey(diag, :operator_metrics)
            println(io, @sprintf("true-M operator rel.RMSE = %.8e",
                diag[:operator_metrics][:relative_rmse]))
            println(io, @sprintf("true-M operator corr = %.8e",
                diag[:operator_metrics][:correlation]))
        end
        if haskey(diag, :mean_norm)
            println(io, @sprintf("mean lagwise ||E[r]||/sqrt(D) = %.8e",
                mean(diag[:mean_norm])))
        end
        if haskey(diag, :stein_norm)
            println(io, @sprintf("mean lagwise ||E[r x0']||/sqrt(D) = %.8e",
                mean(diag[:stein_norm])))
        end
        println(io, "No-cheating audit: conditional training used only clean trajectory pairs, x0 Gaussian noise, and the learned branch stationary score to form residual DSM targets. True mobility appears only in labeled ex-post diagnostics when diagnostics are enabled.")
    end
    return nothing
end

function run_finetune(config_path::String, warm_start_path::String;
        audit_only::Bool=false, resume_existing::Bool=false)
    paths = audit_inputs(config_path, warm_start_path)
    audit_only && return nothing

    params = paths.params
    device = detect_spin_device(params.device, params.required_gpu_name)
    activate_and_describe_device!(device, params.device, params.required_gpu_name)

    p = load_phys(paths.data_h5)
    sampler = build_cond_sampler(paths.data_h5, params.burnin_fraction,
        params.tau_max_decorrelation_multiples, params.lag_stride)
    score_model, stats, score_sigma, _ = load_stationary_checkpoint(paths.score_path, device)

    model_path = paths.model_path
    diag = Dict{Symbol, Any}()
    if isfile(model_path)
        resume_existing || error("Output model already exists: $(model_path). Pass --resume-existing to resume this branch.")
        blob = BSON.load(model_path, @__MODULE__)
        model = move_model(blob[:host_model], device)
        Flux.testmode!(model)
        model_cfg = blob[:model_cfg]
        history = blob[:history]
        state = get(blob, :training_state, nothing)
        completed = completed_epoch_from_blob(blob)
        if completed >= params.epochs
            @printf("Branch checkpoint already has %d/%d epochs; skipping training.\n",
                completed, params.epochs)
        else
            @printf("Resuming branch %s from epoch %d/%d with branch optimizer state.\n",
                model_path, completed + 1, params.epochs)
            model, model_cfg, history = train_cond_score(sampler, score_model, stats,
                score_sigma, p, params, device; model_path=model_path,
                operator_aux_target_path=resolve_path(paths.base, params.operator_aux_target_bson),
                initial_model=model, initial_model_cfg=model_cfg,
                initial_history=history, initial_state=state,
                start_epoch=completed + 1)
        end
    else
        warm = BSON.load(paths.warm_start_path, @__MODULE__)
        model = move_model(warm[:host_model], device)
        model_cfg = warm[:model_cfg]
        @printf("Warm-starting from %s with reset optimizer/history for new stationary score target.\n",
            paths.warm_start_path)
        model, model_cfg, history = train_cond_score(sampler, score_model, stats,
            score_sigma, p, params, device; model_path=model_path,
            operator_aux_target_path=resolve_path(paths.base, params.operator_aux_target_bson),
            initial_model=model, initial_model_cfg=model_cfg,
            initial_history=nothing, initial_state=nothing, start_epoch=1)
    end

    if params.evaluate
        phi_blob = BSON.load(paths.phi_path, @__MODULE__)
        diag = conditional_diagnostics(model, sampler, score_model, stats,
            score_sigma, p, params, phi_blob, device)
        render_cond_figure(resolve_path(paths.base, params.output_png), params,
            history, diag, phi_blob)
        write_metrics(resolve_path(paths.base, params.metrics_txt), history, diag)
        state = branch_training_state(model_path)
        save_cond_model(model_path, model, model_cfg, stats, params, sampler,
            history, diag; training_state=state)
    end
    return nothing
end

function main(args=ARGS)
    opts = parse_args(args)
    config_path = resolve_cli_path(opts["config"])
    warm_start_path = resolve_cli_path(get(opts, "warm-start",
        "SoftSpinLLGChain/stationary_score_repair/models/cond_repaired_score_gpu2_main_epoch0720.bson"))
    run_finetune(config_path, warm_start_path;
        audit_only=get(opts, "audit-only", "false") == "true",
        resume_existing=get(opts, "resume-existing", "false") == "true")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
