const L96_TRAINING_TARGET_SOURCE = "A_data = Gamma_Phi - Cdot_data"

struct MobilityTrainingChannelSpec
    key::Symbol
    separation::Int
    weight::Float64
end

struct L96MobilityNNCache
    windows::Array{Float32, 4}
    scond::Array{Float32, 3}
    observables::Dict{Symbol, Array{Float32, 3}}
    anchor_windows::Matrix{Float32}
    taus::Vector{Float64}
end

struct L96MobilityNNHistory
    epochs::Vector{Int}
    train_loss::Vector{Float64}
    normalized_rmse::Vector{Float64}
    physical_rmse::Vector{Float64}
    mean_abs_profile::Vector{Float64}
    zero_mean_penalty::Vector{Float64}
    anchor_rms_penalty::Vector{Float64}
    weight_l2::Vector{Float64}
    mean_profiles::Vector{Vector{Float64}}
end

struct L96CurrentActionCache
    windows::Array{Float32, 4}
    score::Array{Float32, 3}
    target::Array{Float32, 3}
end

struct LearnedPipelineSummary
    fit_mean_normalized_rmse::Float64
    fit_mean_cdot_rmse::Float64
    forward_cphi_mean_rmse::Union{Nothing, Float64}
    forward_phi_cphi_mean_rmse::Union{Nothing, Float64}
    acceptable::Bool
end

struct L96PdfReference
    centers::Vector{Float64}
    density::Vector{Float64}
    boundary::Tuple{Float64, Float64}
end

struct L96PairPdfReference
    offset::Int
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    density::Matrix{Float64}
    xboundary::Tuple{Float64, Float64}
    yboundary::Tuple{Float64, Float64}
end

struct L96CorrelationSummaryFV
    lags::Vector{Float64}
    acf_mean::Vector{Float64}
    cross_offsets::Vector{Int}
    cross_mean::Matrix{Float64}
    t_decorrelation::Float64
    mean_value::Float64
    variance_value::Float64
end

struct L96ObservedStatisticsReference
    univariate::L96PdfReference
    pair_pdfs::Dict{Int, L96PairPdfReference}
    corr::L96CorrelationSummaryFV
    Qsqrt::Matrix{Float64}
end

struct L96CurrentDiagnostics
    rmse_phi::Float64
    rmse_ref::Float64
    rmse_nn::Float64
    per_coord_phi::Vector{Float64}
    per_coord_ref::Vector{Float64}
    per_coord_nn::Vector{Float64}
    mean_delta_profile::Vector{Float64}
    mean_full_profile::Vector{Float64}
    phi_consistency_error::Float64
end

trainable_observable_keys() = (:coord, :var, :nn1, :nn2, :adv, :flux)

function training_channel_specs(params::FitDML96Params)
    specs = MobilityTrainingChannelSpec[]
    for sep in params.coordinate_separations
        push!(specs, MobilityTrainingChannelSpec(:coord, sep, 1.0))
    end
    for key in (:var, :nn1, :nn2, :adv, :flux)
        for sep in params.nonlinear_separations
            push!(specs, MobilityTrainingChannelSpec(key, sep, 1.0))
        end
    end
    return specs
end

function channel_spec_label(spec::MobilityTrainingChannelSpec)
    return String(spec.key) * "[r=" * string(spec.separation) * "]"
end

function channel_index_lookup(params::FitDML96Params)
    return Dict(
        :coord => Dict(sep => idx for (idx, sep) in enumerate(params.coordinate_separations)),
        :var => Dict(sep => idx for (idx, sep) in enumerate(params.nonlinear_separations)),
        :nn1 => Dict(sep => idx for (idx, sep) in enumerate(params.nonlinear_separations)),
        :nn2 => Dict(sep => idx for (idx, sep) in enumerate(params.nonlinear_separations)),
        :adv => Dict(sep => idx for (idx, sep) in enumerate(params.nonlinear_separations)),
        :flux => Dict(sep => idx for (idx, sep) in enumerate(params.nonlinear_separations)),
    )
end

function channel_tensor_from_dict(values::Dict{Symbol, Matrix{Float64}},
        channel_specs::Vector{MobilityTrainingChannelSpec}, params::FitDML96Params)
    lookups = channel_index_lookup(params)
    nlags = size(values[:coord], 1)
    out = zeros(Float64, length(channel_specs), 1, nlags)
    for (channel_idx, spec) in enumerate(channel_specs)
        idx = lookups[spec.key][spec.separation]
        out[channel_idx, 1, :] .= values[spec.key][:, idx]
    end
    return out
end

function channel_family_rmse(target::Dict{Symbol, Matrix{Float64}}, pred::Dict{Symbol, Matrix{Float64}})
    out = Dict{Symbol, Float64}()
    for key in trainable_observable_keys()
        haskey(target, key) || continue
        haskey(pred, key) || continue
        out[key] = rmse(target[key], pred[key])
    end
    return out
end

function channel_family_rmse_from_specs(channel_specs::Vector{MobilityTrainingChannelSpec},
        target::Array{Float64, 3}, pred::Array{Float64, 3})
    buckets = Dict{Symbol, Vector{Float64}}()
    for key in trainable_observable_keys()
        buckets[key] = Float64[]
    end
    for (channel_idx, spec) in enumerate(channel_specs)
        push!(buckets[spec.key], rmse(vec(@view(target[channel_idx, 1, :])), vec(@view(pred[channel_idx, 1, :]))))
    end
    return Dict(key => mean(values) for (key, values) in buckets)
end

function mobility_lag_weights(taus::Vector{Float64}, power::Float64)
    if power == 0.0
        return ones(Float32, length(taus))
    end
    tau_ref = minimum(taus)
    weights = Float32.((tau_ref ./ taus) .^ power)
    weights ./= Float32(mean(weights))
    return weights
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

function build_local_offset_model(window_size::Int, widths::Vector{Int}, noffsets::Int)
    layers = Any[]
    in_dim = window_size
    for width in widths
        push!(layers, Dense(in_dim, width, tanh))
        in_dim = width
    end
    push!(layers, Dense(in_dim, noffsets))
    return Chain(layers...)
end

function initialize_local_offset_model!(model::Chain)
    final_layer = model.layers[end]
    final_layer.weight .= zero(eltype(final_layer.weight))
    final_layer.bias .= zero(eltype(final_layer.bias))
    return model
end

function build_window_tensor(u::AbstractMatrix{Float32}, window_offsets::Vector{Int})
    K, B = size(u)
    windows = Array{Float32}(undef, length(window_offsets), K, B)
    @inbounds for b in 1:B, i in 1:K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, i, b] = u[periodic(i + shift, K), b]
        end
    end
    return windows
end

function build_window_tensor(u::AbstractArray{Float32, 3}, window_offsets::Vector{Int})
    K, B, T = size(u)
    windows = Array{Float32}(undef, length(window_offsets), K, B, T)
    @inbounds for t in 1:T, b in 1:B, i in 1:K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, i, b, t] = u[periodic(i + shift, K), b, t]
        end
    end
    return windows
end

function build_window_matrix64(u::AbstractMatrix{Float64}, window_offsets::Vector{Int})
    K, B = size(u)
    windows = Matrix{Float64}(undef, length(window_offsets), K * B)
    @inbounds for b in 1:B, i in 1:K
        col = i + (b - 1) * K
        for (window_idx, shift) in enumerate(window_offsets)
            windows[window_idx, col] = u[periodic(i + shift, K), b]
        end
    end
    return windows
end

function sample_anchor_window_matrix(sampler::PairSampler, mu::Float64, sigma_x::Float64,
        window_offsets::Vector{Int}, nstates::Int, seed::Int)
    K = size(sampler.states, 2)
    x_batch = Matrix{Float32}(undef, K, nstates)
    u_batch = Matrix{Float32}(undef, K, nstates)
    rng = MersenneTwister(seed)
    sample_state_batch!(x_batch, sampler, rng)
    standardize_batch!(u_batch, x_batch, mu, sigma_x)
    return reshape(build_window_tensor(u_batch, window_offsets), length(window_offsets), :)
end

function build_mobility_training_cache(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mu::Float64, sigma_x::Float64, means::ObservableMeans, device::ExecutionDevice;
        pair_seed::Int, anchor_seed::Int)
    K = size(sampler.states, 2)
    npairs = params.mobility_nn_pairs_per_tau
    nlags = length(sampler.lag_steps)
    x0_batch = Matrix{Float32}(undef, K, npairs)
    xt_batch = Matrix{Float32}(undef, K, npairs)
    u0_batch = Matrix{Float32}(undef, K, npairs)
    ut_batch = Matrix{Float32}(undef, K, npairs)
    windows = Array{Float32}(undef, length(params.mobility_nn_window_offsets), K, npairs, nlags)
    scond = Array{Float32}(undef, K, npairs, nlags)
    observables = Dict(
        :coord => Array{Float32}(undef, K, npairs, nlags),
        :var => Array{Float32}(undef, K, npairs, nlags),
        :nn1 => Array{Float32}(undef, K, npairs, nlags),
        :nn2 => Array{Float32}(undef, K, npairs, nlags),
        :adv => Array{Float32}(undef, K, npairs, nlags),
        :flux => Array{Float32}(undef, K, npairs, nlags),
    )
    rng = MersenneTwister(pair_seed)
    for (lag_idx, (lag, tau)) in enumerate(zip(sampler.lag_steps, sampler.lag_times))
        if lag_idx == 1 || lag_idx == nlags || lag_idx % 20 == 0
            @printf("Building mobility training cache %d / %d : tau = %.3f\n", lag_idx, nlags, tau)
        end
        sample_pair_batch!(x0_batch, xt_batch, sampler, lag, rng)
        standardize_batch!(u0_batch, x0_batch, mu, sigma_x)
        standardize_batch!(ut_batch, xt_batch, mu, sigma_x)
        windows[:, :, :, lag_idx] .= build_window_tensor(u0_batch, params.mobility_nn_window_offsets)
        scond[:, :, lag_idx] .= evaluate_conditional_score_x0(models, x0_batch, xt_batch, tau,
            params.score_batch_size, params.joint_batch_size, device)
        obs = compute_observable_batches(ut_batch, means)
        for key in trainable_observable_keys()
            observables[key][:, :, lag_idx] .= obs[key]
        end
    end
    return L96MobilityNNCache(
        windows,
        scond,
        observables,
        sample_anchor_window_matrix(sampler, mu, sigma_x, params.mobility_nn_window_offsets,
            params.mobility_nn_anchor_states, anchor_seed),
        sampler.lag_times,
    )
end

function build_current_action_cache(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mu::Float64, sigma_x::Float64, reference::ReferenceMobilityResult, device::ExecutionDevice;
        seed::Int)
    K = size(sampler.states, 2)
    nstates = params.mobility_nn_anchor_states
    x_batch = Matrix{Float32}(undef, K, nstates)
    u_batch = Matrix{Float32}(undef, K, nstates)
    target = Matrix{Float64}(undef, K, nstates)
    scratch = Matrix{Float64}(undef, K, nstates)
    rng = MersenneTwister(seed)
    sample_state_batch!(x_batch, sampler, rng)
    standardize_batch!(u_batch, x_batch, mu, sigma_x)
    stat_score = evaluate_stationary_score_x(models, x_batch, params.score_batch_size, device)
    apply_local_antisymmetric_action!(target, scratch, u_batch, stat_score,
        reference.offsets, reference.feature_offsets, reference.quadratic_pairs,
        reference.coefficients, sigma_x; include_divergence=false)
    return L96CurrentActionCache(
        reshape(build_window_tensor(u_batch, params.mobility_nn_window_offsets),
            length(params.mobility_nn_window_offsets), K, nstates, 1),
        reshape(Float32.(stat_score), K, nstates, 1),
        reshape(Float32.(target), K, nstates, 1),
    )
end

function move_mobility_training_cache(cache::L96MobilityNNCache, device::ExecutionDevice)
    return L96MobilityNNCache(
        move_array(cache.windows, device),
        move_array(cache.scond, device),
        Dict(key => move_array(value, device) for (key, value) in pairs(cache.observables)),
        move_array(cache.anchor_windows, device),
        cache.taus,
    )
end

function move_mobility_training_chunk(cache::L96MobilityNNCache, tau_chunk, device::ExecutionDevice)
    return (
        windows = move_array(cache.windows[:, :, :, tau_chunk], device),
        scond = move_array(cache.scond[:, :, tau_chunk], device),
        observables = Dict(key => move_array(value[:, :, tau_chunk], device) for (key, value) in pairs(cache.observables)),
    )
end

function move_current_action_chunk(cache::L96CurrentActionCache, state_chunk, device::ExecutionDevice)
    return (
        windows = move_array(cache.windows[:, :, state_chunk, :], device),
        score = move_array(cache.score[:, state_chunk, :], device),
        target = move_array(cache.target[:, state_chunk, :], device),
    )
end

function evaluate_local_offset_coeffs(model, windows::AbstractArray{Float32, 4}, noffsets::Int)
    coeff_flat = model(reshape(windows, size(windows, 1), :))
    return reshape(coeff_flat, noffsets, size(windows, 2), size(windows, 3), size(windows, 4))
end

function local_offset_action(coeffs::AbstractArray, signal::AbstractArray, offsets::Vector{Int})
    K, B, T = size(signal)
    offset_terms = map(enumerate(offsets)) do (offset_idx, offset)
        forward_idx = [periodic(i + offset, K) for i in 1:K]
        backward_idx = [periodic(i - offset, K) for i in 1:K]
        coeff_slice = @view coeffs[offset_idx, :, :, :]
        coeff_slice .* signal[forward_idx, :, :] .- coeff_slice[backward_idx, :, :] .* signal[backward_idx, :, :]
    end
    return reduce(+, offset_terms)
end

function predict_training_chunk(model, cache::L96MobilityNNCache,
        channel_specs::Vector{MobilityTrainingChannelSpec}, tau_chunk,
        offsets::Vector{Int}, sigma_x::Float64)
    windows = cache.windows[:, :, :, tau_chunk]
    scond = cache.scond[:, :, tau_chunk]
    observables = Dict(key => value[:, :, tau_chunk] for (key, value) in pairs(cache.observables))
    return predict_training_chunk(model, windows, scond, observables, channel_specs, offsets, sigma_x)
end

function predict_training_chunk(model, windows::AbstractArray{Float32, 4},
        scond::AbstractArray{Float32, 3}, observables::Dict,
        channel_specs::Vector{MobilityTrainingChannelSpec},
        offsets::Vector{Int}, sigma_x::Float64)
    coeffs = evaluate_local_offset_coeffs(model, windows, length(offsets))
    delta_signal = -local_offset_action(coeffs, scond, offsets) ./ Float32(sigma_x)
    K = size(delta_signal, 1)
    T = size(delta_signal, 3)
    nchannels = length(channel_specs)
    channel_rows = map(channel_specs) do spec
        obs = observables[spec.key]
        shift_idx = [periodic(i + spec.separation, K) for i in 1:K]
        reshape(vec(mean(obs .* delta_signal[shift_idx, :, :]; dims=(1, 2))), 1, :)
    end
    channel_matrix = reduce(vcat, channel_rows)
    return reshape(channel_matrix, nchannels, 1, T)
end

function predict_current_action_chunk(model, windows::AbstractArray{Float32, 4},
        score::AbstractArray{Float32, 3}, offsets::Vector{Int})
    coeffs = evaluate_local_offset_coeffs(model, windows, length(offsets))
    return local_offset_action(coeffs, score, offsets)
end

function current_action_normalized_mse(model, cache::L96CurrentActionCache,
        offsets::Vector{Int}, scale::Vector{Float64})
    pred = Float64.(predict_current_action_chunk(model, cache.windows, cache.score, offsets))
    return mean(((pred .- Float64.(cache.target)) ./ reshape(scale, :, 1, 1)) .^ 2)
end

function evaluate_training_model_on_cache(model, cache::L96MobilityNNCache,
        channel_specs::Vector{MobilityTrainingChannelSpec}, params::FitDML96Params, sigma_x::Float64)
    nlags = size(cache.windows, 4)
    pred = zeros(Float64, length(channel_specs), 1, nlags)
    for start in 1:params.mobility_nn_tau_batch:nlags
        stop = min(start + params.mobility_nn_tau_batch - 1, nlags)
        chunk = start:stop
        pred[:, :, chunk] .= Float64.(predict_training_chunk(model, cache, channel_specs, chunk,
            params.mobility_nn_offsets, sigma_x))
    end
    return pred
end

function mean_offset_profile(model, anchor_windows::AbstractMatrix{Float32}, offsets::Vector{Int}, K::Int)
    coeffs = Float64.(model(anchor_windows))
    mean_coeff = vec(mean(coeffs; dims=2))
    full_profile = zeros(Float64, K)
    @inbounds for (offset_idx, offset) in enumerate(offsets)
        full_profile[offset + 1] += mean_coeff[offset_idx]
        full_profile[K - offset + 1] -= mean_coeff[offset_idx]
    end
    return mean_coeff, full_profile
end

function mean_offset_coefficients(model, anchor_windows::AbstractMatrix{Float32})
    return vec(mean(model(anchor_windows); dims=2))
end

function anchor_rms_penalty_value(model, anchor_windows::AbstractMatrix{Float32})
    coeffs = model(anchor_windows)
    return sum(abs2, coeffs) / length(coeffs)
end

function annealed_penalty_weight(base_weight::Float64, final_scale::Float64, epoch::Int, total_epochs::Int)
    total_epochs <= 1 && return base_weight * final_scale
    frac = (epoch - 1) / (total_epochs - 1)
    scale = 1.0 + (final_scale - 1.0) * frac
    return base_weight * scale
end

function train_mobility_model(cache::L96MobilityNNCache, validation_caches::Vector{L96MobilityNNCache},
    target::Array{Float64, 3}, channel_specs::Vector{MobilityTrainingChannelSpec},
    params::FitDML96Params, sigma_x::Float64, K::Int, device::ExecutionDevice,
    current_action_cache::Union{Nothing, L96CurrentActionCache}=nothing)
    model = build_local_offset_model(size(cache.windows, 1), params.mobility_nn_widths, length(params.mobility_nn_offsets))
    initialize_local_offset_model!(model)
    model = move_model(model, device)
    opt_state = Flux.setup(Flux.Adam(params.mobility_nn_learning_rate), model)
    training_anchor_windows = move_array(cache.anchor_windows, device)
    target_scale = Float64.(sqrt.(mean(target .^ 2; dims=3))[:, 1, 1])
    target_scale .= max.(target_scale, params.mobility_nn_scale_floor)
    target_scale_reshaped = move_array(reshape(Float32.(target_scale), :, 1, 1), device)
    channel_weights = move_array(reshape(Float32.([spec.weight for spec in channel_specs]), :, 1, 1), device)
    lag_weights = reshape(mobility_lag_weights(cache.taus, params.mobility_nn_lag_weight_power), 1, 1, :)
    current_action_scale = current_action_cache === nothing ? Float64[] :
        vec(max.(sqrt.(mean(Float64.(current_action_cache.target) .^ 2; dims=(2, 3))), params.mobility_nn_scale_floor))
    current_action_scale_reshaped = current_action_cache === nothing ? nothing :
        move_array(reshape(Float32.(current_action_scale), :, 1, 1), device)
    current_action_batch_states = current_action_cache === nothing ? 0 : min(256, size(current_action_cache.score, 2))
    history = L96MobilityNNHistory(Int[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Vector{Vector{Float64}}())
    rng = MersenneTwister(params.seed + 551)
    best_metric = Inf
    best_model = move_model(model, CPUDevice())

    for epoch in 1:params.mobility_nn_epochs
        current_zero_mean_penalty = annealed_penalty_weight(
            params.mobility_nn_zero_mean_penalty,
            params.mobility_nn_zero_mean_penalty_final_scale,
            epoch,
            params.mobility_nn_epochs,
        )
        current_anchor_rms_penalty = annealed_penalty_weight(
            params.mobility_nn_anchor_rms_penalty,
            params.mobility_nn_anchor_rms_penalty_final_scale,
            epoch,
            params.mobility_nn_epochs,
        )
        perm = randperm(rng, size(cache.windows, 4))
        current_perm = current_action_cache === nothing ? Int[] : randperm(rng, size(current_action_cache.score, 2))
        current_cursor = 1
        epoch_loss = 0.0
        nbatches = 0
        for start in 1:params.mobility_nn_tau_batch:length(perm)
            stop = min(start + params.mobility_nn_tau_batch - 1, length(perm))
            chunk = perm[start:stop]
            chunk_cache = move_mobility_training_chunk(cache, chunk, device)
            chunk_weights = move_array(lag_weights[:, :, chunk], device)
            target_chunk = move_array(Float32.(target[:, :, chunk]), device)
            current_chunk = nothing
            if current_action_cache !== nothing && params.mobility_nn_current_action_penalty > 0.0
                if current_cursor + current_action_batch_states - 1 > length(current_perm)
                    current_perm = randperm(rng, size(current_action_cache.score, 2))
                    current_cursor = 1
                end
                state_chunk = current_perm[current_cursor:(current_cursor + current_action_batch_states - 1)]
                current_cursor += current_action_batch_states
                current_chunk = move_current_action_chunk(current_action_cache, state_chunk, device)
            end
            loss, grads = Flux.withgradient(model) do current_model
                pred = predict_training_chunk(current_model, chunk_cache.windows, chunk_cache.scond,
                    chunk_cache.observables, channel_specs, params.mobility_nn_offsets, sigma_x)
                data_loss = mean((((pred .- target_chunk) ./ target_scale_reshaped) .^ 2) .* channel_weights .* chunk_weights)
                zero_mean_penalty = zero(data_loss)
                if current_zero_mean_penalty > 0.0
                    mean_coeff = mean_offset_coefficients(current_model, training_anchor_windows)
                    zero_mean_penalty = sum(abs2, mean_coeff) / length(mean_coeff)
                end
                anchor_rms_penalty = zero(data_loss)
                if current_anchor_rms_penalty > 0.0
                    anchor_rms_penalty = anchor_rms_penalty_value(current_model, training_anchor_windows)
                end
                current_action_penalty = zero(data_loss)
                if current_chunk !== nothing
                    pred_current = predict_current_action_chunk(current_model, current_chunk.windows,
                        current_chunk.score, params.mobility_nn_offsets)
                    current_action_penalty = mean(((pred_current .- current_chunk.target) ./ current_action_scale_reshaped) .^ 2)
                end
                reg_penalty = params.mobility_nn_weight_decay == 0.0 ? zero(data_loss) :
                    Float32(params.mobility_nn_weight_decay) * model_weight_decay(current_model)
                data_loss +
                    Float32(current_zero_mean_penalty) * zero_mean_penalty +
                    Float32(current_anchor_rms_penalty) * anchor_rms_penalty +
                    Float32(params.mobility_nn_current_action_penalty) * current_action_penalty +
                    reg_penalty
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float64(loss)
            nbatches += 1
        end

        if epoch == 1 || epoch % params.mobility_nn_eval_every == 0 || epoch == params.mobility_nn_epochs
            host_model = move_model(model, CPUDevice())
            physical_rmse_sum = 0.0
            normalized_rmse_sum = 0.0
            normalized_mse_sum = 0.0
            anchor_rms_penalty_sum = 0.0
            for validation_cache in validation_caches
                pred = evaluate_training_model_on_cache(host_model, validation_cache, channel_specs, params, sigma_x)
                physical_rmse_sum += sqrt(mean((target .- pred) .^ 2))
                normalized_rmse_sum += sqrt(mean((((target .- pred) ./ reshape(target_scale, :, 1, 1)) .^ 2)))
                normalized_mse_sum += mean((((target .- pred) ./ reshape(target_scale, :, 1, 1)) .^ 2))
                anchor_rms_penalty_sum += Float64(anchor_rms_penalty_value(host_model, validation_cache.anchor_windows))
            end
            inv_nvalidation = 1.0 / max(length(validation_caches), 1)
            physical_rmse = physical_rmse_sum * inv_nvalidation
            normalized_rmse = normalized_rmse_sum * inv_nvalidation
            normalized_mse = normalized_mse_sum * inv_nvalidation
            anchor_rms_penalty = anchor_rms_penalty_sum * inv_nvalidation
            mean_profile, _ = mean_offset_profile(host_model, cache.anchor_windows, params.mobility_nn_offsets, K)
            zero_mean_penalty = mean(mean_profile .^ 2)
            current_action_nrmse = current_action_cache === nothing ? NaN :
                sqrt(current_action_normalized_mse(host_model, current_action_cache,
                    params.mobility_nn_offsets, current_action_scale))
            push!(history.epochs, epoch)
            push!(history.train_loss, epoch_loss / max(nbatches, 1))
            push!(history.normalized_rmse, normalized_rmse)
            push!(history.physical_rmse, physical_rmse)
            push!(history.mean_abs_profile, mean(abs.(mean_profile)))
            push!(history.zero_mean_penalty, zero_mean_penalty)
            push!(history.anchor_rms_penalty, anchor_rms_penalty)
            push!(history.weight_l2, Float64(model_weight_decay(host_model)))
            push!(history.mean_profiles, collect(mean_profile))
            @printf("Mobility NN epoch %4d | loss = %.6e | nRMSE = %.6e | RMSE = %.6e | |<delta>| = %.6e | current nRMSE = %.6e\n",
                epoch, history.train_loss[end], normalized_rmse, physical_rmse,
                history.mean_abs_profile[end], current_action_nrmse)
            selection_metric = if params.mobility_nn_checkpoint_metric == "regularized_objective"
                normalized_mse +
                    params.mobility_nn_zero_mean_penalty * zero_mean_penalty +
                    params.mobility_nn_anchor_rms_penalty * anchor_rms_penalty +
                    params.mobility_nn_current_action_penalty * (isfinite(current_action_nrmse) ? current_action_nrmse^2 : 0.0) +
                    params.mobility_nn_weight_decay * history.weight_l2[end]
            else
                normalized_rmse
            end
            if selection_metric < best_metric
                best_metric = selection_metric
                best_model = deepcopy(host_model)
            end
        end
    end

    return best_model, history, target_scale
end

function evaluate_phi_gamma_channels(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mu::Float64, sigma_x::Float64, means::ObservableMeans, phi_raw::Matrix{Float64},
        device::ExecutionDevice)
    K = size(sampler.states, 2)
    nonlinear_index_sets = shift_indices(params.nonlinear_separations, K)
    coord_index_sets = shift_indices(params.coordinate_separations, K)
    gamma = Dict(
        :coord => zeros(Float64, length(sampler.lag_steps), length(params.coordinate_separations)),
        :var => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn1 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn2 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :adv => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :flux => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
    )
    x0_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    ut_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    rng = MersenneTwister(params.seed + 3107)
    for (lag_idx, (lag, tau)) in enumerate(zip(sampler.lag_steps, sampler.lag_times))
        sums_coord = zeros(Float64, length(params.coordinate_separations))
        sums_nonlinear = Dict(key => zeros(Float64, length(params.nonlinear_separations)) for key in (:var, :nn1, :nn2, :adv, :flux))
        remaining = params.pairs_per_tau
        total_pairs = 0
        while remaining > 0
            batch_n = min(remaining, params.eval_batch_size)
            sample_pair_batch!(@view(x0_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), sampler, lag, rng)
            standardize_batch!(@view(ut_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), mu, sigma_x)
            obs_t = compute_observable_batches(@view(ut_batch[:, 1:batch_n]), means)
            stat_score = evaluate_stationary_score_x(models, @view(x0_batch[:, 1:batch_n]), params.score_batch_size, device)
            phi_signal = transpose(phi_raw) * Float64.(stat_score)
            projected_phi = Float32.(phi_signal ./ sigma_x)
            accumulate_translation_channels!(sums_coord, obs_t[:coord], projected_phi, coord_index_sets)
            for key in (:var, :nn1, :nn2, :adv, :flux)
                accumulate_translation_channels!(sums_nonlinear[key], obs_t[key], projected_phi, nonlinear_index_sets)
            end
            total_pairs += batch_n
            remaining -= batch_n
        end
        gamma[:coord][lag_idx, :] .= sums_coord ./ total_pairs
        for key in (:var, :nn1, :nn2, :adv, :flux)
            gamma[key][lag_idx, :] .= sums_nonlinear[key] ./ total_pairs
        end
        if lag_idx == 1 || lag_idx == length(sampler.lag_steps) || lag_idx % 20 == 0
            @printf("Estimated Gamma_Phi channels for tau index %d / %d (tau = %.3f)\n",
                lag_idx, length(sampler.lag_steps), tau)
        end
    end
    return gamma
end

function a_channels_from_gamma_and_cdot(gamma_phi::Dict{Symbol, Matrix{Float64}},
        cdot_positive::Dict{Symbol, Matrix{Float64}})
    out = Dict{Symbol, Matrix{Float64}}()
    for key in keys(gamma_phi)
        out[key] = gamma_phi[key] .- cdot_positive[key]
    end
    return out
end

function positive_lag_channels(cdot_data::Dict{Symbol, Matrix{Float64}})
    out = Dict{Symbol, Matrix{Float64}}()
    for key in trainable_observable_keys()
        out[key] = cdot_data[key][2:end, :]
    end
    return out
end

function evaluate_nn_a_and_cdot_channels(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mu::Float64, sigma_x::Float64, means::ObservableMeans, phi_raw::Matrix{Float64}, model,
        device::ExecutionDevice)
    K = size(sampler.states, 2)
    nonlinear_index_sets = shift_indices(params.nonlinear_separations, K)
    coord_index_sets = shift_indices(params.coordinate_separations, K)
    a_nn = Dict(
        :coord => zeros(Float64, length(sampler.lag_steps), length(params.coordinate_separations)),
        :var => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn1 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn2 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :adv => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :flux => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
    )
    cdot_nn = Dict(
        :coord => zeros(Float64, length(sampler.lag_steps), length(params.coordinate_separations)),
        :var => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn1 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :nn2 => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :adv => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
        :flux => zeros(Float64, length(sampler.lag_steps), length(params.nonlinear_separations)),
    )
    x0_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    xt_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    u0_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    ut_batch = Matrix{Float32}(undef, K, params.eval_batch_size)
    rng = MersenneTwister(params.seed + 3209)
    noffsets = length(params.mobility_nn_offsets)
    for (lag_idx, (lag, tau)) in enumerate(zip(sampler.lag_steps, sampler.lag_times))
        sums_a_coord = zeros(Float64, length(params.coordinate_separations))
        sums_c_coord = zeros(Float64, length(params.coordinate_separations))
        sums_a_nonlinear = Dict(key => zeros(Float64, length(params.nonlinear_separations)) for key in (:var, :nn1, :nn2, :adv, :flux))
        sums_c_nonlinear = Dict(key => zeros(Float64, length(params.nonlinear_separations)) for key in (:var, :nn1, :nn2, :adv, :flux))
        remaining = params.pairs_per_tau
        total_pairs = 0
        while remaining > 0
            batch_n = min(remaining, params.eval_batch_size)
            sample_pair_batch!(@view(x0_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), sampler, lag, rng)
            standardize_batch!(@view(u0_batch[:, 1:batch_n]), @view(x0_batch[:, 1:batch_n]), mu, sigma_x)
            standardize_batch!(@view(ut_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), mu, sigma_x)
            obs_t = compute_observable_batches(@view(ut_batch[:, 1:batch_n]), means)
            scond = evaluate_conditional_score_x0(models,
                @view(x0_batch[:, 1:batch_n]), @view(xt_batch[:, 1:batch_n]), tau,
                params.score_batch_size, params.joint_batch_size, device)
            windows = build_window_tensor(@view(u0_batch[:, 1:batch_n]), params.mobility_nn_window_offsets)
            coeffs = evaluate_local_offset_coeffs(model, reshape(windows, size(windows, 1), size(windows, 2), size(windows, 3), 1), noffsets)
            coeffs = reshape(coeffs, noffsets, K, batch_n, 1)
            delta_signal = Float32.(-local_offset_action(coeffs, reshape(scond, K, batch_n, 1), params.mobility_nn_offsets)[:, :, 1] ./ sigma_x)
            full_signal = (transpose(phi_raw) * Float64.(scond) .+ (-Float64.(local_offset_action(coeffs, reshape(scond, K, batch_n, 1), params.mobility_nn_offsets)[:, :, 1]))) ./ sigma_x
            accumulate_translation_channels!(sums_a_coord, obs_t[:coord], delta_signal, coord_index_sets)
            accumulate_translation_channels!(sums_c_coord, obs_t[:coord], Float32.(full_signal), coord_index_sets)
            for key in (:var, :nn1, :nn2, :adv, :flux)
                accumulate_translation_channels!(sums_a_nonlinear[key], obs_t[key], delta_signal, nonlinear_index_sets)
                accumulate_translation_channels!(sums_c_nonlinear[key], obs_t[key], Float32.(full_signal), nonlinear_index_sets)
            end
            total_pairs += batch_n
            remaining -= batch_n
        end
        a_nn[:coord][lag_idx, :] .= sums_a_coord ./ total_pairs
        cdot_nn[:coord][lag_idx, :] .= .-(sums_c_coord ./ total_pairs)
        for key in (:var, :nn1, :nn2, :adv, :flux)
            a_nn[key][lag_idx, :] .= sums_a_nonlinear[key] ./ total_pairs
            cdot_nn[key][lag_idx, :] .= .-(sums_c_nonlinear[key] ./ total_pairs)
        end
        if lag_idx == 1 || lag_idx == length(sampler.lag_steps) || lag_idx % 20 == 0
            @printf("Estimated learned-M channels for tau index %d / %d (tau = %.3f)\n",
                lag_idx, length(sampler.lag_steps), tau)
        end
    end
    return a_nn, cdot_nn
end

function family_comparison_figure(output_path::AbstractString, lag_times::Vector{Float64},
        data_dict::Dict{Symbol, Matrix{Float64}}, ref_dict::Dict{Symbol, Matrix{Float64}}, nn_dict::Dict{Symbol, Matrix{Float64}},
        params::FitDML96Params, title_text::String, ylabel_text::String,
        family_rmse_data_nn::Dict{Symbol, Float64}, family_rmse_ref_nn::Dict{Symbol, Float64};
        data_label::String, ref_label::String, nn_label::String)
    ensure_figure_support_loaded!()
    return Base.invokelatest(() -> begin
        with_scaled_figure_style(params.figure_width, params.figure_height) do _
            fig = Figure(size=(params.figure_width, params.figure_height))
            figure_title!(fig, title_text)
            panel_specs = [
                (:coord, params.plot_coordinate_separations, fig[1, 1]),
                (:var, params.plot_standard_nonlinear_separations, fig[1, 2]),
                (:nn1, params.plot_standard_nonlinear_separations, fig[2, 1]),
                (:nn2, params.plot_standard_nonlinear_separations, fig[2, 2]),
                (:adv, params.plot_adv_flux_separations, fig[3, 1]),
                (:flux, params.plot_adv_flux_separations, fig[3, 2]),
            ]
            for (key, plotted_seps, parent) in panel_specs
                sep_list = key == :coord ? params.coordinate_separations : params.nonlinear_separations
                sep_to_idx = Dict(sep => idx for (idx, sep) in enumerate(sep_list))
                ax = Axis(parent;
                    xlabel="tau",
                    ylabel=ylabel_text,
                    title=@sprintf("%s | data/NN %.3e | ref/NN %.3e",
                        observable_title(key), family_rmse_data_nn[key], family_rmse_ref_nn[key]))
                hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
                colors = Makie.wong_colors()
                legend_entries = Any[]
                legend_labels = String[]
                for (local_idx, sep) in enumerate(plotted_seps)
                    haskey(sep_to_idx, sep) || continue
                    idx = sep_to_idx[sep]
                    color = colors[mod1(local_idx, length(colors))]
                    data_line = lines!(ax, lag_times, data_dict[key][:, idx]; color=color, linewidth=curve_linewidth(), label=data_label)
                    lines!(ax, lag_times, ref_dict[key][:, idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9), label=ref_label)
                    lines!(ax, lag_times, nn_dict[key][:, idx]; color=color, linestyle=:dashdot, linewidth=curve_linewidth(emphasis=0.9), label=nn_label)
                    push!(legend_entries, data_line)
                    push!(legend_labels, @sprintf("r = %d", sep))
                end
                axislegend(ax, legend_entries, legend_labels; position=:rb, framevisible=false, nbanks=2)
            end
            apply_publication_grid!(fig.layout, 3, 2; row_gap=26, col_gap=26)
            save_figure(output_path, fig)
        end
        nothing
    end)
end

function training_diagnostics_figure(output_path::AbstractString, history::L96MobilityNNHistory,
        channel_specs::Vector{MobilityTrainingChannelSpec}, final_channel_nrmse::Vector{Float64},
        params::FitDML96Params, summary_lines::Vector{String})
    ensure_figure_support_loaded!()
    return Base.invokelatest(() -> begin
        with_scaled_figure_style(params.figure_width, max(params.figure_height - 400, 2200)) do _
            fig = Figure(size=(params.figure_width, max(params.figure_height - 400, 2200)))
            figure_title!(fig, "Learned mobility training diagnostics")

            ax_loss = Axis(fig[1, 1]; xlabel="epoch", ylabel="loss", title="Training objective")
            lines!(ax_loss, history.epochs, history.train_loss; color=STYLE_REFERENCE, linewidth=curve_linewidth())

            ax_rmse = Axis(fig[1, 2]; xlabel="epoch", ylabel="RMSE", title="Validation channel RMSE")
            lines!(ax_rmse, history.epochs, history.physical_rmse; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="physical")
            lines!(ax_rmse, history.epochs, history.normalized_rmse; color=STYLE_HIGHLIGHT, linewidth=curve_linewidth(), label="normalized")
            axislegend(ax_rmse; position=:rt)

            ax_mean = Axis(fig[2, 1]; xlabel="epoch", ylabel="mean offset", title="Mean learned offset profile")
            hlines!(ax_mean, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            if !isempty(history.mean_profiles)
                profile_matrix = reduce(vcat, [reshape(profile, 1, :) for profile in history.mean_profiles])
                colors = Makie.wong_colors()
                for offset_idx in 1:size(profile_matrix, 2)
                    lines!(ax_mean, history.epochs, profile_matrix[:, offset_idx];
                        color=colors[mod1(offset_idx, length(colors))], linewidth=curve_linewidth(),
                        label=@sprintf("offset %d", params.mobility_nn_offsets[offset_idx]))
                end
                axislegend(ax_mean; position=:rt, nbanks=2)
            end

            channel_labels = [channel_spec_label(spec) for spec in channel_specs]
            ax_bar = Axis(fig[2, 2];
                xlabel="channel", ylabel="nRMSE", title="Final normalized channel RMSE",
                xticks=(1:length(channel_labels), channel_labels), xticklabelrotation=pi / 4)
            barplot!(ax_bar, 1:length(channel_labels), final_channel_nrmse; color=STYLE_PRIMARY, gap=0.16)

            summary_columns = split_text_lines(summary_lines, 2)
            text_panel!(fig[3, 1], summary_columns[1]; title="Fit summary")
            text_panel!(fig[3, 2], summary_columns[2]; title="Configuration")

            apply_publication_grid!(fig.layout, 3, 2; row_gap=28, col_gap=24)
            save_figure(output_path, fig)
        end
        nothing
    end)
end

function mobility_summary_figure(output_path::AbstractString, q_profile::Vector{Float64}, phi_profile::Vector{Float64},
        ref_profile::Vector{Float64}, nn_profile::Vector{Float64}, current_diag::L96CurrentDiagnostics,
        params::FitDML96Params, reference::ReferenceMobilityResult)
    ensure_figure_support_loaded!()
    return Base.invokelatest(() -> begin
        with_scaled_figure_style(params.figure_width, max(params.figure_height - 600, 2000)) do _
            fig = Figure(size=(params.figure_width, max(params.figure_height - 600, 2000)))
            figure_title!(fig, "Recovered mobility summary")

            separations = collect(0:(length(q_profile) - 1))
            ax_profile = Axis(fig[1, 1]; xlabel="separation r", ylabel="profile", title="Mean full-mobility profile")
            lines!(ax_profile, separations, q_profile; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label="Q")
            lines!(ax_profile, separations, phi_profile; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="Phi data")
            lines!(ax_profile, separations, ref_profile; color=STYLE_HIGHLIGHT, linestyle=:dash, linewidth=curve_linewidth(), label="reference")
            lines!(ax_profile, separations, nn_profile; color=STYLE_PRIMARY, linestyle=:dashdot, linewidth=curve_linewidth(), label="learned M")
            axislegend(ax_profile; position=:rb, nbanks=2)

            ax_delta = Axis(fig[1, 2]; xlabel="separation r", ylabel="delta profile", title="Mean learned delta profile")
            hlines!(ax_delta, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            lines!(ax_delta, separations, current_diag.mean_delta_profile; color=STYLE_PRIMARY, linewidth=curve_linewidth(), label="learned delta")
            lines!(ax_delta, separations, ref_profile .- phi_profile; color=STYLE_HIGHLIGHT, linestyle=:dash, linewidth=curve_linewidth(), label="reference - Phi")
            axislegend(ax_delta; position=:rb)

            ax_current = Axis(fig[2, 1]; xlabel="coordinate i", ylabel="current RMSE", title="Held-out stationary-current RMSE")
            coords = collect(1:length(current_diag.per_coord_phi))
            lines!(ax_current, coords, current_diag.per_coord_phi; color=STYLE_SECONDARY, linewidth=curve_linewidth(), label="Phi")
            lines!(ax_current, coords, current_diag.per_coord_ref; color=STYLE_HIGHLIGHT, linestyle=:dash, linewidth=curve_linewidth(), label="reference")
            lines!(ax_current, coords, current_diag.per_coord_nn; color=STYLE_PRIMARY, linestyle=:dashdot, linewidth=curve_linewidth(), label="learned")
            axislegend(ax_current; position=:rt)

            summary_lines = [
                @sprintf("Current RMSE Phi/reference/NN = %.3e / %.3e / %.3e", current_diag.rmse_phi, current_diag.rmse_ref, current_diag.rmse_nn),
                @sprintf("Phi symmetry consistency ||sym(Phi)-Q||_F = %.3e", current_diag.phi_consistency_error),
                "Reference mode = " * reference.active_mode,
                "NN offsets = [" * join(string.(params.mobility_nn_offsets), ", ") * "]",
                "NN window offsets = [" * join(string.(params.mobility_nn_window_offsets), ", ") * "]",
            ]
            text_panel!(fig[2, 2], summary_lines; title="Mobility diagnostics")

            apply_publication_grid!(fig.layout, 2, 2; row_gap=24, col_gap=24)
            save_figure(output_path, fig)
        end
        nothing
    end)
end

function save_mobility_model(path::AbstractString, model, history::L96MobilityNNHistory,
        phi_raw::Matrix{Float64}, mu::Float64, sigma_x::Float64, params::FitDML96Params,
        channel_specs::Vector{MobilityTrainingChannelSpec}, target_scale::Vector{Float64})
    BSON.bson(path, Dict{Symbol, Any}(
        :model_kind => "local_window_antisymmetric_delta_mobility",
        :training_target_source => L96_TRAINING_TARGET_SOURCE,
        :host_model => deepcopy(model),
        :phi_raw => copy(phi_raw),
        :standardization_mu => mu,
        :standardization_sigma => sigma_x,
        :offsets => copy(params.mobility_nn_offsets),
        :window_offsets => copy(params.mobility_nn_window_offsets),
        :widths => copy(params.mobility_nn_widths),
        :history => Dict(
            :epochs => copy(history.epochs),
            :train_loss => copy(history.train_loss),
            :normalized_rmse => copy(history.normalized_rmse),
            :physical_rmse => copy(history.physical_rmse),
            :mean_abs_profile => copy(history.mean_abs_profile),
            :zero_mean_penalty => copy(history.zero_mean_penalty),
            :anchor_rms_penalty => copy(history.anchor_rms_penalty),
            :weight_l2 => copy(history.weight_l2),
            :mean_profiles => copy(history.mean_profiles),
        ),
        :channel_labels => [channel_spec_label(spec) for spec in channel_specs],
        :channel_specs => [(String(spec.key), spec.separation, spec.weight) for spec in channel_specs],
        :target_scale => copy(target_scale),
    ))
    return nothing
end

function append_learned_artifacts!(path::AbstractString, learned_dict::Dict{Symbol, Any})
    data = isfile(path) ? BSON.load(path) : Dict{Symbol, Any}()
    for (key, value) in learned_dict
        data[key] = value
    end
    BSON.bson(path, data)
    return nothing
end

function append_learned_metrics_report!(path::AbstractString, history::L96MobilityNNHistory,
        channel_specs::Vector{MobilityTrainingChannelSpec}, target_scale::Vector{Float64},
        train_target::Array{Float64, 3}, train_pred::Array{Float64, 3}, a_family_rmse::Dict{Symbol, Float64},
        cdot_family_rmse::Dict{Symbol, Float64}, current_diag::L96CurrentDiagnostics,
        params::FitDML96Params, learned_summary::LearnedPipelineSummary)
    open(path, "a") do io
        println(io)
        println(io, "Learned mobility fit")
        println(io, "Training target source = " * L96_TRAINING_TARGET_SOURCE)
        println(io, @sprintf("Best validation normalized RMSE = %.8e", minimum(history.normalized_rmse)))
        println(io, @sprintf("Final validation normalized RMSE = %.8e", history.normalized_rmse[end]))
        println(io, @sprintf("Final validation physical RMSE = %.8e", history.physical_rmse[end]))
        println(io, @sprintf("Mean learned channel normalized RMSE = %.8e", learned_summary.fit_mean_normalized_rmse))
        println(io, @sprintf("Mean learned Cdot RMSE = %.8e", learned_summary.fit_mean_cdot_rmse))
        println(io, @sprintf("Current RMSE Phi/reference/NN = %.8e / %.8e / %.8e",
            current_diag.rmse_phi, current_diag.rmse_ref, current_diag.rmse_nn))
        println(io, @sprintf("Phi consistency error ||sym(Phi)-Q||_F = %.8e", current_diag.phi_consistency_error))
        println(io, "NN offsets = [" * join(string.(params.mobility_nn_offsets), ", ") * "]")
        println(io, "NN window offsets = [" * join(string.(params.mobility_nn_window_offsets), ", ") * "]")
        println(io, "NN widths = [" * join(string.(params.mobility_nn_widths), ", ") * "]")
        println(io)
        println(io, "Training channels")
        for (channel_idx, spec) in enumerate(channel_specs)
            chan_rmse = rmse(vec(@view(train_target[channel_idx, 1, :])), vec(@view(train_pred[channel_idx, 1, :])))
            chan_nrmse = chan_rmse / max(target_scale[channel_idx], params.mobility_nn_scale_floor)
            println(io, @sprintf("%-18s | RMSE = %.8e | nRMSE = %.8e | scale = %.8e",
                channel_spec_label(spec), chan_rmse, chan_nrmse, target_scale[channel_idx]))
        end
        println(io)
        println(io, "A-channel family RMSE")
        for key in trainable_observable_keys()
            println(io, @sprintf("%-6s | RMSE = %.8e", String(key), a_family_rmse[key]))
        end
        println(io)
        println(io, "Cdot family RMSE")
        for key in trainable_observable_keys()
            println(io, @sprintf("%-6s | RMSE = %.8e", String(key), cdot_family_rmse[key]))
        end
    end
    return nothing
end

function activation_is_tanh(layer)
    return layer.σ === tanh || string(layer.σ) == "tanh"
end

function activation_is_identity(layer)
    return layer.σ === identity || string(layer.σ) == "identity"
end

function manual_chain_output_and_jacobian(model::Chain, x::AbstractMatrix{Float64})
    nsamples = size(x, 2)
    ninputs = size(x, 1)
    activ = x
    jac = zeros(Float64, size(activ, 1), ninputs, nsamples)
    @inbounds for sample_idx in 1:nsamples, input_idx in 1:ninputs
        jac[input_idx, input_idx, sample_idx] = 1.0
    end
    for layer in model.layers
        weights = Float64.(layer.weight)
        bias = Float64.(layer.bias)
        z = weights * activ .+ bias
        next_jac = Array{Float64}(undef, size(weights, 1), ninputs, nsamples)
        @inbounds for sample_idx in 1:nsamples
            next_jac[:, :, sample_idx] .= weights * @view jac[:, :, sample_idx]
        end
        if activation_is_tanh(layer)
            activ = tanh.(z)
            deriv = 1.0 .- activ .^ 2
            jac = next_jac .* reshape(deriv, size(deriv, 1), 1, nsamples)
        elseif activation_is_identity(layer)
            activ = z
            jac = next_jac
        else
            error("Unsupported activation in local mobility model; expected tanh hidden layers and identity output.")
        end
    end
    return activ, jac
end

function local_coefficients_and_divergence(model, u::AbstractMatrix{Float64},
        offsets::Vector{Int}, window_offsets::Vector{Int}, sigma_x::Float64)
    K, B = size(u)
    windows64 = build_window_matrix64(u, window_offsets)
    coeff_flat = Float64.(model(Float32.(windows64)))
    coeffs = reshape(coeff_flat, length(offsets), K, B)
    _, jac = manual_chain_output_and_jacobian(model, windows64)
    shift_lookup = Dict(shift => idx for (idx, shift) in enumerate(window_offsets))
    require_condition(haskey(shift_lookup, 0), "mobility_nn.window_offsets must contain 0.")
    center_idx = shift_lookup[0]
    divergence = zeros(Float64, K, B)
    inv_sigma = 1.0 / sigma_x
    @inbounds for b in 1:B, i in 1:K
        col_forward = i + (b - 1) * K
        for (offset_idx, offset) in enumerate(offsets)
            if haskey(shift_lookup, offset)
                divergence[i, b] += jac[offset_idx, shift_lookup[offset], col_forward] * inv_sigma
            end
            back_site = periodic(i - offset, K)
            col_back = back_site + (b - 1) * K
            divergence[i, b] -= jac[offset_idx, center_idx, col_back] * inv_sigma
        end
    end
    return coeffs, divergence
end

function local_offset_action(coeffs::AbstractArray{Float64, 3}, signal::AbstractMatrix{Float64}, offsets::Vector{Int})
    K, B = size(signal)
    out = zeros(Float64, K, B)
    @inbounds for b in 1:B, i in 1:K
        accum = 0.0
        for (offset_idx, offset) in enumerate(offsets)
            forward_idx = periodic(i + offset, K)
            backward_idx = periodic(i - offset, K)
            accum += coeffs[offset_idx, i, b] * signal[forward_idx, b]
            accum -= coeffs[offset_idx, backward_idx, b] * signal[backward_idx, b]
        end
        out[i, b] = accum
    end
    return out
end

function evaluate_current_diagnostics(model, sampler::PairSampler, models::LoadedModels,
        params::FitDML96Params, mu::Float64, sigma_x::Float64, reference::ReferenceMobilityResult,
        phi_raw::Matrix{Float64}, device::ExecutionDevice)
    K = size(sampler.states, 2)
    nsamples = min(12_288, max(params.mobility_nn_anchor_states, 4096))
    x_batch = Matrix{Float32}(undef, K, nsamples)
    u_batch = Matrix{Float32}(undef, K, nsamples)
    drift_batch = Matrix{Float64}(undef, K, nsamples)
    scratch = Matrix{Float64}(undef, K, nsamples)
    ref_current = zeros(Float64, K, nsamples)
    rng = MersenneTwister(params.seed + 3713)
    sample_state_batch!(x_batch, sampler, rng)
    standardize_batch!(u_batch, x_batch, mu, sigma_x)
    stat_score = evaluate_stationary_score_x(models, x_batch, params.score_batch_size, device)
    l96_drift!(drift_batch, x_batch, sampler.F)
    g_true = drift_batch .- sampler.Q * Float64.(stat_score)
    phi_current = (phi_raw .- sampler.Q) * Float64.(stat_score)
    if reference.active_mode == "local_antisymmetric_fit" && !isempty(reference.coefficients)
        apply_local_antisymmetric_action!(ref_current, scratch, u_batch, stat_score, reference.offsets,
            reference.feature_offsets, reference.quadratic_pairs, reference.coefficients, sigma_x;
            include_divergence=true)
    else
        fill!(ref_current, 0.0)
    end
    coeffs, divergence = local_coefficients_and_divergence(model, Float64.(u_batch),
        params.mobility_nn_offsets, params.mobility_nn_window_offsets, sigma_x)
    nn_current = phi_current .+ local_offset_action(coeffs, Float64.(stat_score), params.mobility_nn_offsets) .+ divergence
    phi_only_current = phi_current

    per_coord_phi = [rmse(@view(g_true[i, :]), @view(phi_only_current[i, :])) for i in 1:K]
    per_coord_ref = [rmse(@view(g_true[i, :]), @view(ref_current[i, :])) for i in 1:K]
    per_coord_nn = [rmse(@view(g_true[i, :]), @view(nn_current[i, :])) for i in 1:K]
    _, delta_profile = mean_offset_profile(model,
        sample_anchor_window_matrix(sampler, mu, sigma_x, params.mobility_nn_window_offsets,
            params.mobility_nn_anchor_states, params.seed + 3791),
        params.mobility_nn_offsets, K)
    full_profile = circulant_profile(phi_raw) .+ delta_profile
    return L96CurrentDiagnostics(
        rmse(g_true, phi_only_current),
        rmse(g_true, ref_current),
        rmse(g_true, nn_current),
        per_coord_phi,
        per_coord_ref,
        per_coord_nn,
        delta_profile,
        full_profile,
        norm(0.5 .* (phi_raw .+ phi_raw') .- sampler.Q),
    )
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

function grid_boundary(grid::Vector{Float64})
    halfwidth = length(grid) > 1 ? 0.5 * (grid[2] - grid[1]) : 0.5
    return (grid[1] - halfwidth, grid[end] + halfwidth)
end

function load_observed_statistics_reference(path::AbstractString, requested_offsets::Vector{Int})
    centers = Float64.(h5read(path, "/statistics/pdf/univariate_centers"))
    density = Float64.(h5read(path, "/statistics/pdf/univariate_density"))
    offsets = Set(Int.(h5read(path, "/statistics/pdf/bivariate_offsets")))
    pair_refs = Dict{Int, L96PairPdfReference}()
    for offset in requested_offsets
        if offset in offsets
            base = @sprintf("/statistics/pdf/bivariate/offset_%d", offset)
            xgrid = Float64.(h5read(path, string(base, "/x_grid")))
            ygrid = Float64.(h5read(path, string(base, "/y_grid")))
            pair_refs[offset] = L96PairPdfReference(offset, xgrid, ygrid,
                Float64.(h5read(path, string(base, "/density"))), grid_boundary(xgrid), grid_boundary(ygrid))
        end
    end
    corr = L96CorrelationSummaryFV(
        Float64.(h5read(path, "/statistics/correlations/lags")),
        Float64.(h5read(path, "/statistics/correlations/acf_mean")),
        Int.(h5read(path, "/statistics/correlations/cross_offsets")),
        Float64.(h5read(path, "/statistics/correlations/cross_mean")),
        Float64(h5read(path, "/statistics/correlations/t_decorrelation")),
        Float64(h5read(path, "/statistics/correlations/global_mean")),
        Float64(h5read(path, "/statistics/correlations/global_variance")),
    )
    return L96ObservedStatisticsReference(
        L96PdfReference(centers, density, grid_boundary(centers)),
        pair_refs,
        corr,
        Matrix{Float64}(h5read(path, "/diffusion/Qsqrt")),
    )
end

function draw_univariate_samples(states::Array{Float64, 3}, start_idx::Int, max_samples::Int, rng::AbstractRNG)
    nt, K, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        mode_idx = (tmp % K) + 1
        traj_idx = (tmp ÷ K) + 1
        values[sample_idx] = states[start_idx + time_local, mode_idx, traj_idx]
    end
    return values
end

function draw_pair_samples(states::Array{Float64, 3}, start_idx::Int, offset::Int, max_samples::Int, rng::AbstractRNG)
    nt, K, ntraj = size(states)
    npost = nt - start_idx + 1
    total = npost * K * ntraj
    nsamples = min(max_samples, total)
    x_values = Vector{Float64}(undef, nsamples)
    y_values = Vector{Float64}(undef, nsamples)
    @inbounds for sample_idx in 1:nsamples
        linear = rand(rng, 0:(total - 1))
        time_local = linear % npost
        tmp = linear ÷ npost
        mode_idx = (tmp % K) + 1
        traj_idx = (tmp ÷ K) + 1
        paired_idx = periodic(mode_idx + offset, K)
        time_idx = start_idx + time_local
        x_values[sample_idx] = states[time_idx, mode_idx, traj_idx]
        y_values[sample_idx] = states[time_idx, paired_idx, traj_idx]
    end
    return x_values, y_values
end

function compute_univariate_pdf_on_reference(states::Array{Float64, 3}, start_idx::Int,
        reference::L96PdfReference, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed)
    samples = draw_univariate_samples(states, start_idx, max_samples, rng)
    kde_result = kde(samples; npoints=length(reference.centers), boundary=reference.boundary)
    return L96PdfReference(collect(kde_result.x), collect(kde_result.density), reference.boundary)
end

function compute_pair_pdf_on_reference(states::Array{Float64, 3}, start_idx::Int,
        reference::L96PairPdfReference, max_samples::Int, seed::Int)
    rng = MersenneTwister(seed + reference.offset)
    x_values, y_values = draw_pair_samples(states, start_idx, reference.offset, max_samples, rng)
    kde_result = kde((x_values, y_values);
        npoints=(length(reference.xgrid), length(reference.ygrid)),
        boundary=(reference.xboundary, reference.yboundary))
    return L96PairPdfReference(reference.offset, collect(kde_result.x), collect(kde_result.y),
        Array(kde_result.density), reference.xboundary, reference.yboundary)
end

function tensor_mean_and_variance(data::Array{Float64, 3})
    total = 0.0
    count = 0
    @inbounds for traj_idx in axes(data, 3), mode_idx in axes(data, 2), time_idx in axes(data, 1)
        total += data[time_idx, mode_idx, traj_idx]
        count += 1
    end
    mean_value = total / count
    sumsq = 0.0
    @inbounds for traj_idx in axes(data, 3), mode_idx in axes(data, 2), time_idx in axes(data, 1)
        delta = data[time_idx, mode_idx, traj_idx] - mean_value
        sumsq += delta * delta
    end
    return mean_value, sumsq / count
end

function estimate_decorrelation_time(acf::Vector{Float64}, lags::Vector{Float64}, threshold::Float64)
    envelope = abs.(acf)
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

function downsample_states(states::Array{Float64, 3}, stride::Int)
    time_indices = collect(1:stride:size(states, 1))
    return Array(@view states[time_indices, :, :])
end

function compute_lattice_correlations(data::Array{Float64, 3}, dt_corr::Float64,
        max_time::Float64, threshold::Float64, cross_offsets::Vector{Int})
    ntime, K, ntraj = size(data)
    mean_value, variance_value = tensor_mean_and_variance(data)
    require_condition(variance_value > 0.0, "Cannot normalize correlations because the empirical variance is zero.")
    max_lag = min(ntime - 1, floor(Int, max_time / dt_corr))
    require_condition(max_lag >= 1, "The correlation window is empty.")
    lags = collect(0:max_lag) .* dt_corr
    acf_mean = zeros(Float64, max_lag + 1)
    cross_mean = zeros(Float64, max_lag + 1, length(cross_offsets))
    Threads.@threads for lag in 0:max_lag
        sum_acf = 0.0
        sum_cross = zeros(Float64, length(cross_offsets))
        count = 0
        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for time_idx in 1:upper, mode_idx in 1:K
                x0 = data[time_idx, mode_idx, traj_idx] - mean_value
                x_same = data[time_idx + lag, mode_idx, traj_idx] - mean_value
                sum_acf += x_same * x0
                for (offset_idx, offset) in enumerate(cross_offsets)
                    paired_idx = periodic(mode_idx + offset, K)
                    x_pair = data[time_idx + lag, paired_idx, traj_idx] - mean_value
                    sum_cross[offset_idx] += x_pair * x0
                end
                count += 1
            end
        end
        acf_mean[lag + 1] = sum_acf / (count * variance_value)
        for offset_idx in eachindex(cross_offsets)
            cross_mean[lag + 1, offset_idx] = sum_cross[offset_idx] / (count * variance_value)
        end
    end
    return L96CorrelationSummaryFV(lags, acf_mean, copy(cross_offsets), cross_mean,
        estimate_decorrelation_time(acf_mean, lags, threshold), mean_value, variance_value)
end

function compute_lattice_correlations_at_steps(data::Array{Float64, 3}, saved_dt::Float64,
        lag_steps::Vector{Int}, threshold::Float64, cross_offsets::Vector{Int})
    ntime, K, ntraj = size(data)
    mean_value, variance_value = tensor_mean_and_variance(data)
    acf_mean = zeros(Float64, length(lag_steps))
    cross_mean = zeros(Float64, length(lag_steps), length(cross_offsets))
    Threads.@threads for lag_idx in eachindex(lag_steps)
        lag = lag_steps[lag_idx]
        require_condition(lag < ntime, "Requested lag exceeds the saved forward trajectory length.")
        sum_acf = 0.0
        sum_cross = zeros(Float64, length(cross_offsets))
        count = 0
        @inbounds for traj_idx in 1:ntraj
            upper = ntime - lag
            for time_idx in 1:upper, mode_idx in 1:K
                x0 = data[time_idx, mode_idx, traj_idx] - mean_value
                x_same = data[time_idx + lag, mode_idx, traj_idx] - mean_value
                sum_acf += x_same * x0
                for (offset_idx, offset) in enumerate(cross_offsets)
                    paired_idx = periodic(mode_idx + offset, K)
                    x_pair = data[time_idx + lag, paired_idx, traj_idx] - mean_value
                    sum_cross[offset_idx] += x_pair * x0
                end
                count += 1
            end
        end
        acf_mean[lag_idx] = sum_acf / (count * variance_value)
        for offset_idx in eachindex(cross_offsets)
            cross_mean[lag_idx, offset_idx] = sum_cross[offset_idx] / (count * variance_value)
        end
    end
    lags = lag_steps .* saved_dt
    return L96CorrelationSummaryFV(lags, acf_mean, copy(cross_offsets), cross_mean,
        estimate_decorrelation_time(acf_mean, lags, threshold), mean_value, variance_value)
end

function compute_rollout_channel_correlations(states::Array{Float64, 3}, lag_steps::Vector{Int},
        mu::Float64, sigma_x::Float64, means::ObservableMeans,
        coordinate_separations::Vector{Int}, nonlinear_separations::Vector{Int};
        start_idx::Int=1, max_pairs::Int=120_000, seed::Int=20260601)
    K = size(states, 2)
    coord_index_sets = shift_indices(coordinate_separations, K)
    nonlinear_index_sets = shift_indices(nonlinear_separations, K)
    out = Dict(
        :coord => zeros(Float64, length(lag_steps), length(coordinate_separations)),
        :var => zeros(Float64, length(lag_steps), length(nonlinear_separations)),
        :nn1 => zeros(Float64, length(lag_steps), length(nonlinear_separations)),
        :nn2 => zeros(Float64, length(lag_steps), length(nonlinear_separations)),
        :adv => zeros(Float64, length(lag_steps), length(nonlinear_separations)),
        :flux => zeros(Float64, length(lag_steps), length(nonlinear_separations)),
    )
    rng = MersenneTwister(seed)
    for (lag_idx, lag) in enumerate(lag_steps)
        upper = size(states, 1) - lag
        total_pairs = min(max_pairs, (upper - start_idx + 1) * size(states, 3))
        x0 = Matrix{Float32}(undef, K, total_pairs)
        xt = Matrix{Float32}(undef, K, total_pairs)
        @inbounds for pair_idx in 1:total_pairs
            traj_idx = rand(rng, 1:size(states, 3))
            time_idx = rand(rng, start_idx:upper)
            for mode_idx in 1:K
                x0[mode_idx, pair_idx] = Float32(states[time_idx, mode_idx, traj_idx])
                xt[mode_idx, pair_idx] = Float32(states[time_idx + lag, mode_idx, traj_idx])
            end
        end
        u0 = Matrix{Float32}(undef, K, total_pairs)
        ut = Matrix{Float32}(undef, K, total_pairs)
        standardize_batch!(u0, x0, mu, sigma_x)
        standardize_batch!(ut, xt, mu, sigma_x)
        obs_t = compute_observable_batches(ut, means)
        coord_sums = zeros(Float64, length(coordinate_separations))
        nonlinear_sums = Dict(key => zeros(Float64, length(nonlinear_separations)) for key in (:var, :nn1, :nn2, :adv, :flux))
        accumulate_translation_channels!(coord_sums, obs_t[:coord], u0, coord_index_sets)
        for key in (:var, :nn1, :nn2, :adv, :flux)
            accumulate_translation_channels!(nonlinear_sums[key], obs_t[key], u0, nonlinear_index_sets)
        end
        out[:coord][lag_idx, :] .= coord_sums ./ total_pairs
        for key in (:var, :nn1, :nn2, :adv, :flux)
            out[key][lag_idx, :] .= nonlinear_sums[key] ./ total_pairs
        end
    end
    return out
end

function support_bounds(states::Array{Float32, 3}, start_idx::Int, pad_fraction::Float64)
    post = @view states[start_idx:end, :, :]
    xmin = Float64(minimum(post))
    xmax = Float64(maximum(post))
    span = max(xmax - xmin, 1.0e-6)
    pad = max(span * pad_fraction, 1.0e-3)
    return xmin - pad, xmax + pad
end

function clamp_states_for_eval!(points::AbstractMatrix{Float64}, xmin::Float64, xmax::Float64)
    outside = 0
    @inbounds for idx in eachindex(points)
        if points[idx] < xmin || points[idx] > xmax
            outside += 1
            points[idx] = clamp(points[idx], xmin, xmax)
        end
    end
    return outside
end

function symmetric_factor(mat::Matrix{Float64})
    sym = 0.5 .* (mat .+ mat')
    eig = eigen(Symmetric(sym))
    values = clamp.(eig.values, 1.0e-8, Inf)
    return eig.vectors * Diagonal(sqrt.(values)) * eig.vectors'
end

function predicted_full_step!(state::Matrix{Float64}, noise_buf::Matrix{Float64},
        models::LoadedModels, mobility_model, params::FitDML96Params,
        phi_raw::Matrix{Float64}, mu::Float64, sigma_x::Float64,
        eval_min::Float64, eval_max::Float64, device::ExecutionDevice)
    eval_points = copy(state)
    outside_eval = params.forward_validation_clamp_eval_to_support ?
        clamp_states_for_eval!(eval_points, eval_min, eval_max) : 0
    score = Float64.(evaluate_stationary_score_x(models, Float32.(eval_points), params.score_batch_size, device))
    u_eval = Matrix{Float32}(undef, size(eval_points))
    standardize_batch!(u_eval, Float32.(eval_points), mu, sigma_x)
    coeffs, divergence = local_coefficients_and_divergence(mobility_model, Float64.(u_eval),
        params.mobility_nn_offsets, params.mobility_nn_window_offsets, sigma_x)
    r_action = local_offset_action(coeffs, score, params.mobility_nn_offsets)
    drift = phi_raw * score .+ r_action .+ divergence
    sqrt_2dt = sqrt(2.0 * params.forward_validation_dt)
    state .+= params.forward_validation_dt .* drift .+ sqrt_2dt .* noise_buf
    hard_clamps = 0
    if params.forward_validation_hard_clamp_state
        @inbounds for idx in eachindex(state)
            clamped = clamp(state[idx], eval_min, eval_max)
            hard_clamps += clamped != state[idx]
            state[idx] = clamped
        end
    end
    require_condition(all(isfinite, state), "Learned-M forward validation produced a non-finite state.")
    return outside_eval, hard_clamps
end

function predicted_phi_step!(state::Matrix{Float64}, noise_buf::Matrix{Float64},
        models::LoadedModels, params::FitDML96Params, phi_raw::Matrix{Float64},
        eval_min::Float64, eval_max::Float64, device::ExecutionDevice)
    eval_points = copy(state)
    outside_eval = params.forward_validation_clamp_eval_to_support ?
        clamp_states_for_eval!(eval_points, eval_min, eval_max) : 0
    score = Float64.(evaluate_stationary_score_x(models, Float32.(eval_points), params.score_batch_size, device))
    drift = phi_raw * score
    sqrt_2dt = sqrt(2.0 * params.forward_validation_dt)
    state .+= params.forward_validation_dt .* drift .+ sqrt_2dt .* noise_buf
    hard_clamps = 0
    if params.forward_validation_hard_clamp_state
        @inbounds for idx in eachindex(state)
            clamped = clamp(state[idx], eval_min, eval_max)
            hard_clamps += clamped != state[idx]
            state[idx] = clamped
        end
    end
    require_condition(all(isfinite, state), "Phi-only forward validation produced a non-finite state.")
    return outside_eval, hard_clamps
end

function integrate_forward_validation(sampler::PairSampler, models::LoadedModels, params::FitDML96Params,
        mobility_model, phi_raw::Matrix{Float64}, observed_ref::L96ObservedStatisticsReference,
        mu::Float64, sigma_x::Float64, device::ExecutionDevice)
    total_steps = round(Int, params.forward_validation_total_time / params.forward_validation_dt)
    burnin_steps = round(Int, params.forward_validation_burnin_time / params.forward_validation_dt)
    require_condition(abs(total_steps * params.forward_validation_dt - params.forward_validation_total_time) <= 1.0e-10,
        "forward_validation.total_time must be an integer multiple of forward_validation.dt.")
    require_condition(abs(burnin_steps * params.forward_validation_dt - params.forward_validation_burnin_time) <= 1.0e-10,
        "forward_validation.burnin_time must be an integer multiple of forward_validation.dt.")
    nsaved = 1 + fld(total_steps - burnin_steps, params.forward_validation_save_stride)
    times = Vector{Float64}(undef, nsaved)
    pred_full = Array{Float64}(undef, nsaved, size(sampler.states, 2), params.forward_validation_ntrajectories)
    pred_phi = Array{Float64}(undef, nsaved, size(sampler.states, 2), params.forward_validation_ntrajectories)
    rng_init = MersenneTwister(params.forward_validation_seed + 17)
    init_states = Matrix{Float64}(undef, size(sampler.states, 2), params.forward_validation_ntrajectories)
    init_states32 = Matrix{Float32}(undef, size(sampler.states, 2), params.forward_validation_ntrajectories)
    sample_state_batch!(init_states32, sampler, rng_init)
    init_states .= Float64.(init_states32)
    state_full = copy(init_states)
    state_phi = copy(init_states)
    sym_phi = 0.5 .* (phi_raw .+ phi_raw')
    sym_phi_eig = eigvals(Symmetric(sym_phi))
    phi_sqrt = symmetric_factor(phi_raw)
    sym_phi_q_fro = norm(sym_phi .- sampler.Q)
    @printf("Forward validation diffusion: lambda_min(sym(Phi)) = %.8e, lambda_max(sym(Phi)) = %.8e, ||sym(Phi)-Q||_F = %.8e\n",
        minimum(sym_phi_eig), maximum(sym_phi_eig), sym_phi_q_fro)
    noise_full = Matrix{Float64}(undef, size(sampler.states, 2), params.forward_validation_ntrajectories)
    noise_phi = Matrix{Float64}(undef, size(sampler.states, 2), params.forward_validation_ntrajectories)
    correlated_full = similar(noise_full)
    correlated_phi = similar(noise_phi)
    rng_full = MersenneTwister(params.forward_validation_seed + 101)
    rng_phi = MersenneTwister(params.forward_validation_seed + 203)
    rng_common = MersenneTwister(params.forward_validation_seed + 307)
    eval_min, eval_max = support_bounds(sampler.states, sampler.start_idx, params.forward_validation_support_pad_fraction)
    save_cursor = 0
    eval_clamp_full = 0
    eval_clamp_phi = 0
    hard_clamp_full = 0
    hard_clamp_phi = 0
    for step in 1:total_steps
        if params.forward_validation_use_common_random_numbers
            randn!(rng_common, noise_full)
            noise_phi .= noise_full
        else
            randn!(rng_full, noise_full)
            randn!(rng_phi, noise_phi)
        end
        mul!(correlated_full, phi_sqrt, noise_full)
        mul!(correlated_phi, phi_sqrt, noise_phi)
        clamp_full, hard_full = predicted_full_step!(state_full, correlated_full, models, mobility_model,
            params, phi_raw, mu, sigma_x, eval_min, eval_max, device)
        clamp_phi, hard_phi = predicted_phi_step!(state_phi, correlated_phi, models, params, phi_raw,
            eval_min, eval_max, device)
        eval_clamp_full += clamp_full
        eval_clamp_phi += clamp_phi
        hard_clamp_full += hard_full
        hard_clamp_phi += hard_phi
        if step >= burnin_steps && (step - burnin_steps) % params.forward_validation_save_stride == 0
            save_cursor += 1
            times[save_cursor] = (step - burnin_steps) * params.forward_validation_dt
            pred_full[save_cursor, :, :] .= state_full
            pred_phi[save_cursor, :, :] .= state_phi
        end
    end
    stats_full = Dict{Symbol, Any}(
        :saved_dt => params.forward_validation_dt * params.forward_validation_save_stride,
        :eval_clamp_fraction => eval_clamp_full / (total_steps * size(state_full, 1) * size(state_full, 2)),
        :hard_clamp_fraction => hard_clamp_full / (total_steps * size(state_full, 1) * size(state_full, 2)),
        :sym_phi_lambda_min => minimum(sym_phi_eig),
        :sym_phi_lambda_max => maximum(sym_phi_eig),
        :sym_phi_q_fro => sym_phi_q_fro,
    )
    stats_phi = Dict{Symbol, Any}(
        :saved_dt => params.forward_validation_dt * params.forward_validation_save_stride,
        :eval_clamp_fraction => eval_clamp_phi / (total_steps * size(state_phi, 1) * size(state_phi, 2)),
        :hard_clamp_fraction => hard_clamp_phi / (total_steps * size(state_phi, 1) * size(state_phi, 2)),
        :sym_phi_lambda_min => minimum(sym_phi_eig),
        :sym_phi_lambda_max => maximum(sym_phi_eig),
        :sym_phi_q_fro => sym_phi_q_fro,
    )
    return times, pred_full, pred_phi, stats_full, stats_phi
end

function pdf_rmse(ref::AbstractArray{<:Real}, pred::AbstractArray{<:Real})
    return sqrt(mean((Float64.(ref) .- Float64.(pred)) .^ 2))
end

function kl_divergence_1d(ref::Vector{Float64}, pred::Vector{Float64}, width::Float64)
    eps_val = 1.0e-12
    p = ref .* width
    q = pred .* width
    p .+= eps_val
    q .+= eps_val
    p ./= sum(p)
    q ./= sum(q)
    return sum(p .* log.(p ./ q))
end

function univariate_pdf_metrics(reference::L96PdfReference, pred::L96PdfReference)
    width = length(reference.centers) > 1 ? (reference.centers[2] - reference.centers[1]) : 1.0
    return Dict(
        :rmse => pdf_rmse(reference.density, pred.density),
        :kl => kl_divergence_1d(reference.density, pred.density, width),
    )
end

function pair_pdf_metrics(reference::L96PairPdfReference, pred::L96PairPdfReference)
    return Dict(:rmse => pdf_rmse(reference.density, pred.density))
end

function cphi_channel_mean_rmse(reference::Dict{Symbol, Matrix{Float64}}, pred::Dict{Symbol, Matrix{Float64}})
    values = Float64[]
    for key in trainable_observable_keys()
        push!(values, rmse(reference[key], pred[key]))
    end
    return mean(values)
end

function forward_summary_lines(pdf_metrics_full::Dict{Symbol, Float64}, pdf_metrics_phi::Dict{Symbol, Float64},
        corr_full::L96CorrelationSummaryFV, corr_phi::L96CorrelationSummaryFV,
        observed_corr::L96CorrelationSummaryFV, cphi_rmse_full::Float64, cphi_rmse_phi::Float64,
        current_diag::L96CurrentDiagnostics, stats_full::Dict{Symbol, Any}, stats_phi::Dict{Symbol, Any})
    return [
        @sprintf("PDF RMSE full/Phi = %.3e / %.3e", pdf_metrics_full[:rmse], pdf_metrics_phi[:rmse]),
        @sprintf("PDF KL full/Phi = %.3e / %.3e", pdf_metrics_full[:kl], pdf_metrics_phi[:kl]),
        @sprintf("t_decorr observed/full/Phi = %.3f / %.3f / %.3f", observed_corr.t_decorrelation, corr_full.t_decorrelation, corr_phi.t_decorrelation),
        @sprintf("Cphi mean RMSE full/Phi = %.3e / %.3e", cphi_rmse_full, cphi_rmse_phi),
        @sprintf("Held-out current RMSE Phi/reference/NN = %.3e / %.3e / %.3e", current_diag.rmse_phi, current_diag.rmse_ref, current_diag.rmse_nn),
        @sprintf("sym(Phi) eig min/max = %.3e / %.3e", stats_phi[:sym_phi_lambda_min], stats_phi[:sym_phi_lambda_max]),
        @sprintf("||sym(Phi)-Q||_F = %.3e", stats_phi[:sym_phi_q_fro]),
        @sprintf("Eval clamp frac full/Phi = %.3e / %.3e", stats_full[:eval_clamp_fraction], stats_phi[:eval_clamp_fraction]),
        @sprintf("Hard clamp frac full/Phi = %.3e / %.3e", stats_full[:hard_clamp_fraction], stats_phi[:hard_clamp_fraction]),
    ]
end

function heatmap_panel!(parent, x, y, z; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString,
        colorbar_label::AbstractString="", colormap=STYLE_SEQUENTIAL_BLUE)
    layout = GridLayout(parent)
    ax = Axis(layout[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)
    hm = heatmap!(ax, x, y, z; colormap=colormap)
    Colorbar(layout[1, 2], hm; label=colorbar_label)
    return ax
end

function forward_stats_figure(output_path::AbstractString, observed_ref::L96ObservedStatisticsReference,
        pred_pdf_full::L96PdfReference, pred_pdf_phi::L96PdfReference,
        pred_corr_full::L96CorrelationSummaryFV, pred_corr_phi::L96CorrelationSummaryFV,
        pair_ref::Union{Nothing, L96PairPdfReference}, pair_full::Union{Nothing, L96PairPdfReference},
        pair_phi::Union{Nothing, L96PairPdfReference}, params::FitDML96Params,
        pdf_metrics_full::Dict{Symbol, Float64}, pdf_metrics_phi::Dict{Symbol, Float64},
        cphi_rmse_full::Float64, cphi_rmse_phi::Float64, current_diag::L96CurrentDiagnostics,
        stats_full::Dict{Symbol, Any}, stats_phi::Dict{Symbol, Any})
    ensure_figure_support_loaded!()
    summary_lines = forward_summary_lines(pdf_metrics_full, pdf_metrics_phi,
        pred_corr_full, pred_corr_phi, observed_ref.corr, cphi_rmse_full, cphi_rmse_phi,
        current_diag, stats_full, stats_phi)
    return Base.invokelatest(() -> begin
        with_scaled_figure_style(params.figure_width, params.figure_height) do _
            fig = Figure(size=(params.figure_width, params.figure_height))
            figure_title!(fig, "L96 forward validation against observed statistics")

            ax_pdf = Axis(fig[1, 1]; title=@sprintf("Univariate PDF | full %.3e | Phi %.3e", pdf_metrics_full[:rmse], pdf_metrics_phi[:rmse]),
                xlabel="x_i", ylabel="density")
            lines!(ax_pdf, observed_ref.univariate.centers, observed_ref.univariate.density; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
            lines!(ax_pdf, pred_pdf_full.centers, pred_pdf_full.density; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label="learned M")
            lines!(ax_pdf, pred_pdf_phi.centers, pred_pdf_phi.density; color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label="Phi only")
            axislegend(ax_pdf; position=:rt)

            max_corr_time = min(observed_ref.corr.lags[end], params.forward_validation_correlation_max_time)
            keep_obs = findall(t -> t <= max_corr_time + 1.0e-12, observed_ref.corr.lags)
            keep_full = findall(t -> t <= max_corr_time + 1.0e-12, pred_corr_full.lags)
            keep_phi = findall(t -> t <= max_corr_time + 1.0e-12, pred_corr_phi.lags)

            ax_acf = Axis(fig[1, 2]; title="Translation-averaged ACF", xlabel="lag tau", ylabel="C_0(tau)")
            hlines!(ax_acf, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            lines!(ax_acf, observed_ref.corr.lags[keep_obs], observed_ref.corr.acf_mean[keep_obs]; color=STYLE_REFERENCE, linewidth=curve_linewidth(), label="observed")
            lines!(ax_acf, pred_corr_full.lags[keep_full], pred_corr_full.acf_mean[keep_full]; color=STYLE_PRIMARY, linestyle=:dash, linewidth=curve_linewidth(), label="learned M")
            lines!(ax_acf, pred_corr_phi.lags[keep_phi], pred_corr_phi.acf_mean[keep_phi]; color=STYLE_SECONDARY, linestyle=:dot, linewidth=curve_linewidth(), label="Phi only")
            axislegend(ax_acf; position=:rt)

            ax_cross = Axis(fig[1, 3]; title="Shifted cross-correlations", xlabel="lag tau", ylabel="C_r(tau)")
            hlines!(ax_cross, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
            colors = Makie.wong_colors()
            for (offset_idx, offset) in enumerate(observed_ref.corr.cross_offsets)
                color = colors[mod1(offset_idx, length(colors))]
                lines!(ax_cross, observed_ref.corr.lags[keep_obs], observed_ref.corr.cross_mean[keep_obs, offset_idx]; color=color, linewidth=curve_linewidth(), label=@sprintf("obs r=%d", offset))
                lines!(ax_cross, pred_corr_full.lags[keep_full], pred_corr_full.cross_mean[keep_full, offset_idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9), label=@sprintf("full r=%d", offset))
                lines!(ax_cross, pred_corr_phi.lags[keep_phi], pred_corr_phi.cross_mean[keep_phi, offset_idx]; color=color, linestyle=:dot, linewidth=curve_linewidth(emphasis=0.8), label=@sprintf("Phi r=%d", offset))
            end
            axislegend(ax_cross; position=:rb, nbanks=2)

            if pair_ref !== nothing && pair_full !== nothing && pair_phi !== nothing
                heatmap_panel!(fig[2, 1], pair_ref.xgrid, pair_ref.ygrid, pair_ref.density;
                    title=@sprintf("Observed pair PDF r=%d", pair_ref.offset), xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pair_ref.offset), colorbar_label="density")
                heatmap_panel!(fig[2, 2], pair_full.xgrid, pair_full.ygrid, pair_full.density;
                    title="Learned-M pair PDF", xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pair_ref.offset), colorbar_label="density")
                heatmap_panel!(fig[2, 3], pair_phi.xgrid, pair_phi.ygrid, pair_phi.density;
                    title="Phi-only pair PDF", xlabel="x_i", ylabel=@sprintf("x_{i+%d}", pair_ref.offset), colorbar_label="density")
            else
                text_panel!(fig[2, 1], ["No observed pair-PDF reference available for the requested offsets."]; title="Pair PDF")
                text_panel!(fig[2, 2], summary_lines[1:3]; title="Metrics")
                text_panel!(fig[2, 3], summary_lines[4:end]; title="Diagnostics")
            end

            summary_columns = split_text_lines(summary_lines, 3)
            text_panel!(fig[3, 1], summary_columns[1]; title="PDF and Cphi")
            text_panel!(fig[3, 2], summary_columns[2]; title="Dynamics")
            text_panel!(fig[3, 3], summary_columns[3]; title="Model")

            apply_publication_grid!(fig.layout, 3, 3; row_gap=26, col_gap=22)
            save_figure(output_path, fig)
        end
        nothing
    end)
end

function forward_channels_figure(output_path::AbstractString, lag_times::Vector{Float64},
        observed_channels::Dict{Symbol, Matrix{Float64}}, pred_channels_full::Dict{Symbol, Matrix{Float64}},
        pred_channels_phi::Dict{Symbol, Matrix{Float64}}, params::FitDML96Params)
    ensure_figure_support_loaded!()
    return Base.invokelatest(() -> begin
        with_scaled_figure_style(params.figure_width, max(params.figure_height - 400, 2400)) do _
            fig = Figure(size=(params.figure_width, max(params.figure_height - 400, 2400)))
            figure_title!(fig, "Forward-validation observable channels")
            panel_specs = [
                (:coord, params.plot_coordinate_separations, fig[1, 1]),
                (:var, params.plot_standard_nonlinear_separations, fig[1, 2]),
                (:nn1, params.plot_standard_nonlinear_separations, fig[2, 1]),
                (:nn2, params.plot_standard_nonlinear_separations, fig[2, 2]),
                (:adv, params.plot_adv_flux_separations, fig[3, 1]),
                (:flux, params.plot_adv_flux_separations, fig[3, 2]),
            ]
            for (key, plotted_seps, parent) in panel_specs
                sep_list = key == :coord ? params.coordinate_separations : params.nonlinear_separations
                sep_to_idx = Dict(sep => idx for (idx, sep) in enumerate(sep_list))
                rmse_full = rmse(observed_channels[key], pred_channels_full[key])
                rmse_phi = rmse(observed_channels[key], pred_channels_phi[key])
                ax = Axis(parent; xlabel="tau", ylabel="C(tau)",
                    title=@sprintf("%s | full %.3e | Phi %.3e", observable_title(key), rmse_full, rmse_phi))
                hlines!(ax, [0.0]; color=STYLE_ZERO, linestyle=:dot, linewidth=guide_linewidth())
                colors = Makie.wong_colors()
                for (local_idx, sep) in enumerate(plotted_seps)
                    haskey(sep_to_idx, sep) || continue
                    idx = sep_to_idx[sep]
                    color = colors[mod1(local_idx, length(colors))]
                    lines!(ax, lag_times, observed_channels[key][:, idx]; color=color, linewidth=curve_linewidth())
                    lines!(ax, lag_times, pred_channels_full[key][:, idx]; color=color, linestyle=:dash, linewidth=curve_linewidth(emphasis=0.9))
                    lines!(ax, lag_times, pred_channels_phi[key][:, idx]; color=color, linestyle=:dot, linewidth=curve_linewidth(emphasis=0.8))
                end
            end
            apply_publication_grid!(fig.layout, 3, 2; row_gap=26, col_gap=22)
            save_figure(output_path, fig)
        end
        nothing
    end)
end

function save_forward_trajectories(path::AbstractString, times::Vector{Float64},
        pred_full::Array{Float64, 3}, pred_phi::Array{Float64, 3})
    h5open(path, "w") do file
        write(file, "/time", times)
        write(file, "/predicted_full/states", pred_full)
        write(file, "/predicted_phi/states", pred_phi)
    end
    return nothing
end

function aligned_correlation_support(observed_corr::L96CorrelationSummaryFV,
        pred_corr::L96CorrelationSummaryFV)
    obs_keep = findall(t -> t <= pred_corr.lags[end] + 1.0e-12, observed_corr.lags)
    require_condition(!isempty(obs_keep),
        "No shared lag support remains between observed and predicted forward-validation correlations.")
    n_acf = min(length(obs_keep), length(pred_corr.acf_mean))
    n_cross = min(length(obs_keep), size(pred_corr.cross_mean, 1))
    return (
        obs_acf = observed_corr.acf_mean[obs_keep[1:n_acf]],
        pred_acf = pred_corr.acf_mean[1:n_acf],
        obs_cross = observed_corr.cross_mean[obs_keep[1:n_cross], :],
        pred_cross = pred_corr.cross_mean[1:n_cross, :],
    )
end

function forward_validation_metrics_report(path::AbstractString,
        pdf_metrics_full::Dict{Symbol, Float64}, pdf_metrics_phi::Dict{Symbol, Float64},
        pred_corr_full::L96CorrelationSummaryFV, pred_corr_phi::L96CorrelationSummaryFV,
        observed_corr::L96CorrelationSummaryFV, cphi_rmse_full::Float64, cphi_rmse_phi::Float64,
        current_diag::L96CurrentDiagnostics, stats_full::Dict{Symbol, Any}, stats_phi::Dict{Symbol, Any})
    aligned_full = aligned_correlation_support(observed_corr, pred_corr_full)
    aligned_phi = aligned_correlation_support(observed_corr, pred_corr_phi)

    open(path, "w") do io
        println(io, "L96 learned-mobility forward validation")
        println(io, @sprintf("pdf_rmse_full = %.8e", pdf_metrics_full[:rmse]))
        println(io, @sprintf("pdf_rmse_phi = %.8e", pdf_metrics_phi[:rmse]))
        println(io, @sprintf("pdf_kl_full = %.8e", pdf_metrics_full[:kl]))
        println(io, @sprintf("pdf_kl_phi = %.8e", pdf_metrics_phi[:kl]))
        println(io, @sprintf("acf_rmse_full = %.8e", rmse(aligned_full.obs_acf, aligned_full.pred_acf)))
        println(io, @sprintf("acf_rmse_phi = %.8e", rmse(aligned_phi.obs_acf, aligned_phi.pred_acf)))
        println(io, @sprintf("cross_rmse_full = %.8e", rmse(aligned_full.obs_cross, aligned_full.pred_cross)))
        println(io, @sprintf("cross_rmse_phi = %.8e", rmse(aligned_phi.obs_cross, aligned_phi.pred_cross)))
        println(io, @sprintf("cphi_mean_rmse_full = %.8e", cphi_rmse_full))
        println(io, @sprintf("cphi_mean_rmse_phi = %.8e", cphi_rmse_phi))
        println(io, @sprintf("heldout_current_rmse_phi = %.8e", current_diag.rmse_phi))
        println(io, @sprintf("heldout_current_rmse_reference = %.8e", current_diag.rmse_ref))
        println(io, @sprintf("heldout_current_rmse_nn = %.8e", current_diag.rmse_nn))
        println(io, @sprintf("sym_phi_lambda_min = %.8e", stats_phi[:sym_phi_lambda_min]))
        println(io, @sprintf("sym_phi_lambda_max = %.8e", stats_phi[:sym_phi_lambda_max]))
        println(io, @sprintf("sym_phi_q_fro = %.8e", stats_phi[:sym_phi_q_fro]))
        println(io, @sprintf("eval_clamp_fraction_full = %.8e", stats_full[:eval_clamp_fraction]))
        println(io, @sprintf("eval_clamp_fraction_phi = %.8e", stats_phi[:eval_clamp_fraction]))
        println(io, @sprintf("hard_clamp_fraction_full = %.8e", stats_full[:hard_clamp_fraction]))
        println(io, @sprintf("hard_clamp_fraction_phi = %.8e", stats_phi[:hard_clamp_fraction]))
    end
    return nothing
end

function run_forward_validation(paths::ManagedRunPaths, sampler::PairSampler, models::LoadedModels,
        params::FitDML96Params, input_hdf5::AbstractString, mobility_model, phi_raw::Matrix{Float64},
        mu::Float64, sigma_x::Float64, means::ObservableMeans, device::ExecutionDevice,
        current_diag::L96CurrentDiagnostics, fit_lag_times::Vector{Float64})
    observed_ref = load_observed_statistics_reference(input_hdf5, params.forward_validation_bivariate_offsets)
    times, pred_full, pred_phi, stats_full, stats_phi = integrate_forward_validation(
        sampler, models, params, mobility_model, phi_raw, observed_ref, mu, sigma_x, device)
    saved_dt = stats_full[:saved_dt]
    pred_pdf_full = compute_univariate_pdf_on_reference(pred_full, 1, observed_ref.univariate,
        params.forward_validation_pdf_max_samples, params.forward_validation_seed + 401)
    pred_pdf_phi = compute_univariate_pdf_on_reference(pred_phi, 1, observed_ref.univariate,
        params.forward_validation_pdf_max_samples, params.forward_validation_seed + 503)
    pdf_metrics_full = univariate_pdf_metrics(observed_ref.univariate, pred_pdf_full)
    pdf_metrics_phi = univariate_pdf_metrics(observed_ref.univariate, pred_pdf_phi)
    pair_ref = nothing
    pair_full = nothing
    pair_phi = nothing
    if !isempty(observed_ref.pair_pdfs)
        first_offset = first(sort(collect(keys(observed_ref.pair_pdfs))))
        pair_ref = observed_ref.pair_pdfs[first_offset]
        pair_full = compute_pair_pdf_on_reference(pred_full, 1, pair_ref,
            params.forward_validation_pdf_max_samples, params.forward_validation_seed + 607)
        pair_phi = compute_pair_pdf_on_reference(pred_phi, 1, pair_ref,
            params.forward_validation_pdf_max_samples, params.forward_validation_seed + 709)
    end
    lag_keep = findall(t -> t <= min(params.forward_validation_cphi_max_time, times[end]) + 1.0e-12, fit_lag_times)
    lag_times = fit_lag_times[lag_keep]
    lag_steps = lag_steps_from_times(lag_times, saved_dt; allow_zero=false)
    pred_corr_full = compute_lattice_correlations_at_steps(pred_full, saved_dt,
        lag_steps_from_times(observed_ref.corr.lags[observed_ref.corr.lags .<= min(params.forward_validation_correlation_max_time, times[end]) + 1.0e-12], saved_dt; allow_zero=true),
        params.forward_validation_correlation_threshold, observed_ref.corr.cross_offsets)
    pred_corr_phi = compute_lattice_correlations_at_steps(pred_phi, saved_dt,
        lag_steps_from_times(observed_ref.corr.lags[observed_ref.corr.lags .<= min(params.forward_validation_correlation_max_time, times[end]) + 1.0e-12], saved_dt; allow_zero=true),
        params.forward_validation_correlation_threshold, observed_ref.corr.cross_offsets)
    observed_channels = compute_rollout_channel_correlations(Float64.(sampler.states[sampler.start_idx:end, :, :]),
        lag_steps_from_times(lag_times, sampler.save_dt; allow_zero=false), mu, sigma_x, means,
        params.coordinate_separations, params.nonlinear_separations;
        start_idx=1, max_pairs=min(params.forward_validation_auxiliary_max_samples, 120_000),
        seed=params.forward_validation_seed + 811)
    pred_channels_full = compute_rollout_channel_correlations(pred_full, lag_steps, mu, sigma_x, means,
        params.coordinate_separations, params.nonlinear_separations;
        start_idx=1, max_pairs=min(params.forward_validation_auxiliary_max_samples, 120_000),
        seed=params.forward_validation_seed + 913)
    pred_channels_phi = compute_rollout_channel_correlations(pred_phi, lag_steps, mu, sigma_x, means,
        params.coordinate_separations, params.nonlinear_separations;
        start_idx=1, max_pairs=min(params.forward_validation_auxiliary_max_samples, 120_000),
        seed=params.forward_validation_seed + 1019)
    cphi_rmse_full = cphi_channel_mean_rmse(observed_channels, pred_channels_full)
    cphi_rmse_phi = cphi_channel_mean_rmse(observed_channels, pred_channels_phi)

    forward_stats_figure(paths.forward_stats_png, observed_ref, pred_pdf_full, pred_pdf_phi,
        pred_corr_full, pred_corr_phi, pair_ref, pair_full, pair_phi, params,
        pdf_metrics_full, pdf_metrics_phi, cphi_rmse_full, cphi_rmse_phi,
        current_diag, stats_full, stats_phi)
    forward_channels_figure(paths.forward_channels_png, lag_times, observed_channels,
        pred_channels_full, pred_channels_phi, params)
    forward_validation_metrics_report(paths.forward_metrics_txt, pdf_metrics_full, pdf_metrics_phi,
        pred_corr_full, pred_corr_phi, observed_ref.corr, cphi_rmse_full, cphi_rmse_phi,
        current_diag, stats_full, stats_phi)
    save_forward_trajectories(paths.forward_trajectories_hdf5, times, pred_full, pred_phi)
    aligned_full = aligned_correlation_support(observed_ref.corr, pred_corr_full)
    aligned_phi = aligned_correlation_support(observed_ref.corr, pred_corr_phi)
    BSON.bson(paths.forward_diagnostics_bson, Dict{Symbol, Any}(
        :pdf_metrics_full => pdf_metrics_full,
        :pdf_metrics_phi => pdf_metrics_phi,
        :acf_rmse_full => rmse(aligned_full.obs_acf, aligned_full.pred_acf),
        :acf_rmse_phi => rmse(aligned_phi.obs_acf, aligned_phi.pred_acf),
        :cross_rmse_full => rmse(aligned_full.obs_cross, aligned_full.pred_cross),
        :cross_rmse_phi => rmse(aligned_phi.obs_cross, aligned_phi.pred_cross),
        :cphi_mean_rmse_full => cphi_rmse_full,
        :cphi_mean_rmse_phi => cphi_rmse_phi,
        :current_rmse_phi => current_diag.rmse_phi,
        :current_rmse_reference => current_diag.rmse_ref,
        :current_rmse_nn => current_diag.rmse_nn,
        :stats_full => stats_full,
        :stats_phi => stats_phi,
        :lag_times => lag_times,
        :observed_channels => observed_channels,
        :pred_channels_full => pred_channels_full,
        :pred_channels_phi => pred_channels_phi,
    ))
    return cphi_rmse_full, cphi_rmse_phi
end

function run_learned_mobility_pipeline(paths::ManagedRunPaths, sampler::PairSampler, models::LoadedModels,
    params::FitDML96Params, input_hdf5::AbstractString, device::ExecutionDevice, mu::Float64, sigma_x::Float64,
        means::ObservableMeans, reference::ReferenceMobilityResult, taus::Vector{Float64},
        C_data::Dict{Symbol, Matrix{Float64}}, Cdot_data::Dict{Symbol, Matrix{Float64}},
        Cdot_ref_q::Dict{Symbol, Matrix{Float64}}, Cdot_ref_fit::Dict{Symbol, Matrix{Float64}},
        Phi_data_profile::Vector{Float64}, raw_config, metrics_path::AbstractString)
    channel_specs = training_channel_specs(params)
    phi_raw = Matrix{Float64}(reference.active_matrix)
    gamma_phi = evaluate_phi_gamma_channels(sampler, models, params, mu, sigma_x, means, phi_raw, device)
    cdot_positive = positive_lag_channels(Cdot_data)
    a_data = a_channels_from_gamma_and_cdot(gamma_phi, cdot_positive)
    a_ref_active = a_channels_from_gamma_and_cdot(gamma_phi,
        reference.active_mode == "local_antisymmetric_fit" && reference.fitted_matrix !== nothing ? Cdot_ref_fit : Cdot_ref_q)
    target_train = channel_tensor_from_dict(a_data, channel_specs, params)

    cache = build_mobility_training_cache(sampler, models, params, mu, sigma_x, means, device;
        pair_seed=params.seed + 5001, anchor_seed=params.seed + 5101)
    current_action_cache = params.mobility_nn_current_action_penalty > 0.0 ?
        build_current_action_cache(sampler, models, params, mu, sigma_x, reference, device;
            seed=params.seed + 5201) : nothing
    validation_caches = L96MobilityNNCache[]
    for (seed_idx, seed_value) in enumerate(params.mobility_nn_validation_pair_seeds)
        push!(validation_caches, build_mobility_training_cache(sampler, models, params, mu, sigma_x, means, device;
            pair_seed=seed_value, anchor_seed=seed_value + 10_000 + seed_idx))
    end
    mobility_model, history, target_scale = train_mobility_model(cache, validation_caches, target_train,
        channel_specs, params, sigma_x, size(sampler.states, 2), device, current_action_cache)
    train_pred = evaluate_training_model_on_cache(mobility_model, cache, channel_specs, params, sigma_x)

    a_nn, cdot_nn = evaluate_nn_a_and_cdot_channels(sampler, models, params, mu, sigma_x, means,
        phi_raw, mobility_model, device)
    a_family_rmse = channel_family_rmse(a_data, a_nn)
    cdot_family_rmse = channel_family_rmse(cdot_positive, cdot_nn)
    family_comparison_figure(paths.fit_a_png, sampler.lag_times, a_data, a_ref_active, a_nn, params,
        "Learned mobility A-channel fit", "A(tau)", a_family_rmse,
        channel_family_rmse(a_ref_active, a_nn);
        data_label="A data", ref_label="reference", nn_label="learned M")
    family_comparison_figure(paths.fit_cdot_png, sampler.lag_times, cdot_positive,
        reference.active_mode == "local_antisymmetric_fit" && reference.fitted_matrix !== nothing ? Cdot_ref_fit : Cdot_ref_q,
        cdot_nn, params,
        "Learned mobility Cdot channels", "Cdot(tau)", cdot_family_rmse,
        channel_family_rmse(reference.active_mode == "local_antisymmetric_fit" && reference.fitted_matrix !== nothing ? Cdot_ref_fit : Cdot_ref_q, cdot_nn);
        data_label="data", ref_label="reference", nn_label="learned M")

    current_diag = evaluate_current_diagnostics(mobility_model, sampler, models, params, mu, sigma_x,
        reference, phi_raw, device)
    q_profile = circulant_profile(sampler.Q)
    phi_profile = circulant_profile(phi_raw)
    ref_profile = circulant_profile(reference.active_matrix)
    mobility_summary_figure(paths.fit_mobility_png, q_profile, phi_profile, ref_profile,
        current_diag.mean_full_profile, current_diag, params, reference)

    channel_nrmse = [rmse(vec(@view(target_train[channel_idx, 1, :])), vec(@view(train_pred[channel_idx, 1, :]))) / max(target_scale[channel_idx], params.mobility_nn_scale_floor)
        for channel_idx in 1:length(channel_specs)]
    summary_lines = [
        "Training target = " * L96_TRAINING_TARGET_SOURCE,
        @sprintf("Best validation nRMSE = %.3e", minimum(history.normalized_rmse)),
        @sprintf("Final validation nRMSE = %.3e", history.normalized_rmse[end]),
        @sprintf("Held-out current RMSE Phi/reference/NN = %.3e / %.3e / %.3e",
            current_diag.rmse_phi, current_diag.rmse_ref, current_diag.rmse_nn),
        @sprintf("Phi consistency ||sym(Phi)-Q||_F = %.3e", current_diag.phi_consistency_error),
        @sprintf("Current-action penalty = %.2e", params.mobility_nn_current_action_penalty),
        "Offsets = [" * join(string.(params.mobility_nn_offsets), ", ") * "]",
        "Window offsets = [" * join(string.(params.mobility_nn_window_offsets), ", ") * "]",
        "Widths = [" * join(string.(params.mobility_nn_widths), ", ") * "]",
        @sprintf("epochs = %d", params.mobility_nn_epochs),
        @sprintf("lr = %.2e", params.mobility_nn_learning_rate),
        @sprintf("weight decay = %.2e", params.mobility_nn_weight_decay),
        @sprintf("zero-mean penalty = %.2e", params.mobility_nn_zero_mean_penalty),
        @sprintf("anchor RMS penalty = %.2e", params.mobility_nn_anchor_rms_penalty),
        "checkpoint metric = " * params.mobility_nn_checkpoint_metric,
    ]
    if params.mobility_nn_zero_mean_penalty_final_scale != 1.0
        push!(summary_lines, @sprintf("zero-mean final scale = %.2f", params.mobility_nn_zero_mean_penalty_final_scale))
    end
    if params.mobility_nn_anchor_rms_penalty_final_scale != 1.0
        push!(summary_lines, @sprintf("anchor final scale = %.2f", params.mobility_nn_anchor_rms_penalty_final_scale))
    end
    training_diagnostics_figure(paths.fit_training_png, history, channel_specs, channel_nrmse,
        params, summary_lines)
    save_mobility_model(paths.fit_model_bson, mobility_model, history, phi_raw, mu, sigma_x,
        params, channel_specs, target_scale)

    forward_cphi_rmse = nothing
    forward_phi_cphi_rmse = nothing
    if params.forward_validation_enabled
        forward_cphi_rmse, forward_phi_cphi_rmse = run_forward_validation(paths, sampler, models, params,
            input_hdf5, mobility_model, phi_raw, mu, sigma_x,
            means, device, current_diag, sampler.lag_times)
    end

    learned_summary = LearnedPipelineSummary(
        mean(channel_nrmse),
        mean(collect(values(cdot_family_rmse))),
        forward_cphi_rmse,
        forward_phi_cphi_rmse,
        current_diag.rmse_nn < current_diag.rmse_phi &&
            (forward_cphi_rmse === nothing || forward_cphi_rmse < forward_phi_cphi_rmse),
    )
    append_learned_metrics_report!(metrics_path, history, channel_specs, target_scale,
        target_train, train_pred, a_family_rmse, cdot_family_rmse, current_diag,
        params, learned_summary)
    append_learned_artifacts!(paths.output_bson, Dict{Symbol, Any}(
        :training_target_source_l96 => L96_TRAINING_TARGET_SOURCE,
        :phi_raw_learned => phi_raw,
        :gamma_phi => gamma_phi,
        :a_data => a_data,
        :a_nn => a_nn,
        :cdot_nn => cdot_nn,
        :channel_labels => [channel_spec_label(spec) for spec in channel_specs],
        :target_scale_l96 => target_scale,
        :fit_summary_l96 => Dict(
            :fit_mean_normalized_rmse => learned_summary.fit_mean_normalized_rmse,
            :fit_mean_cdot_rmse => learned_summary.fit_mean_cdot_rmse,
            :forward_cphi_mean_rmse => learned_summary.forward_cphi_mean_rmse,
            :forward_phi_cphi_mean_rmse => learned_summary.forward_phi_cphi_mean_rmse,
            :acceptable => learned_summary.acceptable,
        ),
    ))
    @printf("Saved learned mobility figures to %s, %s, %s, and %s\n",
        paths.fit_a_png, paths.fit_cdot_png, paths.fit_training_png, paths.fit_mobility_png)
    @printf("Saved learned mobility model to %s\n", paths.fit_model_bson)
    if params.forward_validation_enabled
        @printf("Saved forward-validation outputs to %s, %s, %s, %s, and %s\n",
            paths.forward_stats_png, paths.forward_channels_png, paths.forward_metrics_txt,
            paths.forward_diagnostics_bson, paths.forward_trajectories_hdf5)
    end
    return learned_summary
end
