#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using Printf
using Statistics

function subset_dm_targets(source_bson::AbstractString, channels_toml::AbstractString,
        output_bson::AbstractString; scale_source::Symbol=:retained_channel_rms,
        force_full_arrays::Bool=false)
    src = BSON.load(source_bson)
    old_names = Vector{String}(src[:names])
    old_index = Dict(name => i for (i, name) in enumerate(old_names))
    channels = load_retained_channels(channels_toml)
    names = unique_observable_names(channels)
    missing_names = setdiff(names, old_names)
    require_condition(isempty(missing_names),
        "Subset names not present in source target artifact: $(missing_names)")
    name_indices = [old_index[name] for name in names]
    N = Int(src[:N])
    D = Int(src[:D])
    obs_index = Dict(name => i for (i, name) in enumerate(names))
    selected_inds, selected_channel_ids = selected_linear_indices(channels, obs_index, N)

    target_vec = Array{Float32}(undef, length(src[:lags]), length(selected_inds))
    scale_vec = Array{Float32}(undef, length(selected_inds))
    save_full_arrays = haskey(src, :A_target) && haskey(src, :Cdot_data) && haskey(src, :Cdot_phi)
    preserve_source_selection = !force_full_arrays &&
        haskey(src, :target_vec) &&
        haskey(src, :selected_indices) &&
        haskey(src, :selected_channel_ids) &&
        haskey(src, :channels)
    source_uses_legacy = Bool(get(src, :legacy_selected_indices, true))
    if save_full_arrays && !preserve_source_selection
        Cdata = Array(src[:Cdot_data])[:, :, name_indices, :]
        Cphi = Array(src[:Cdot_phi])[:, :, name_indices, :]
        A_target = Array(src[:A_target])[:, :, name_indices, :]
        for li in axes(target_vec, 1)
            flat = vec(A_target[li, :, :, :])
            target_vec[li, :] .= Float32.(flat[selected_inds])
        end
    else
        source_channels = Vector{RetainedChannel}(src[:channels])
        source_channel_ids = Vector{Int}(src[:selected_channel_ids])
        source_selected_inds = Vector{Int}(src[:selected_indices])
        source_channel_index = Dict((ch.observable, ch.target_component) => cid
            for (cid, ch) in enumerate(source_channels))
        old_nobs = length(old_names)
        new_nobs = length(names)
        remap_index(idx::Int) = begin
            old_rows = N * old_nobs
            col = div(idx - 1, old_rows) + 1
            row = rem(idx - 1, old_rows) + 1
            if source_uses_legacy
                i = div(row - 1, old_nobs) + 1
                aold = rem(row - 1, old_nobs) + 1
                anew = obs_index[old_names[aold]]
                new_row = (i - 1) * new_nobs + anew
            else
                i = rem(row - 1, N) + 1
                aold = div(row - 1, N) + 1
                anew = obs_index[old_names[aold]]
                new_row = i + (anew - 1) * N
            end
            return new_row + (col - 1) * (N * new_nobs)
        end
        source_cols = Int[]
        new_selected_inds = Int[]
        new_selected_channel_ids = Int[]
        for (cid, ch) in enumerate(channels)
            key = (ch.observable, ch.target_component)
            require_condition(haskey(source_channel_index, key),
                "Requested channel $(key) is not present in slim source target artifact.")
            source_cid = source_channel_index[key]
            cols = findall(==(source_cid), source_channel_ids)
            append!(source_cols, cols)
            append!(new_selected_inds, [remap_index(source_selected_inds[col]) for col in cols])
            append!(new_selected_channel_ids, fill(cid, length(cols)))
        end
        selected_inds = new_selected_inds
        selected_channel_ids = new_selected_channel_ids
        target_vec = Array{Float32}(src[:target_vec])[:, source_cols]
        scale_vec = Array{Float32}(undef, length(selected_inds))
    end
    if scale_source == :retained_channel_rms
        for (k, cid) in enumerate(selected_channel_ids)
            scale_vec[k] = Float32(max(channels[cid].data_rms, 1f-6))
        end
    elseif scale_source == :data_target
        for k in axes(target_vec, 2)
            scale_vec[k] = Float32(max(sqrt(mean(abs2, @view target_vec[:, k])), 1f-6))
        end
    elseif scale_source == :data_channel
        for cid in unique(selected_channel_ids)
            inds = findall(==(cid), selected_channel_ids)
            scale = Float32(max(sqrt(mean(abs2, @view target_vec[:, inds])), 1f-6))
            scale_vec[inds] .= scale
        end
    else
        error("Unsupported scale_source=$(scale_source)")
    end

    means_old = Vector{Float64}(src[:observable_means])
    means = means_old[name_indices]
    ensure_parent_dir(output_bson)
    out = Dict(:names => names, :channels => channels,
        :observable_means => means, :lags => src[:lags], :taus => src[:taus],
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Phi => src[:Phi], :Phi_block => src[:Phi_block],
        :scale_source => scale_source,
        :save_dt => src[:save_dt], :N => N, :D => D,
        :source_target_artifact => source_bson,
        :source_channels_toml => channels_toml,
        :legacy_selected_indices => source_uses_legacy,
        :no_cheating_audit => "This target artifact is a strict channel subset of an existing data/learned-score target artifact. No analytic score or true mobility was added.")
    if save_full_arrays && !preserve_source_selection
        out[:Cdot_data] = Cdata
        out[:Cdot_phi] = Cphi
        out[:A_target] = A_target
    else
        out[:source_was_slim_training_artifact] = true
        out[:source_target_kind] = get(src, :target_kind, :unknown)
    end
    for key in (:target_kind, :score_radial_scale, :uses_learned_score_proxy_observables,
            :hybrid_short_prefix_source, :hybrid_long_suffix_source, :hybrid_prefix_lags,
            :hybrid_replaced_prefix_rel_diff, :hybrid_replaced_prefix_corr)
        haskey(src, key) && (out[key] = src[key])
    end
    BSON.bson(output_bson, out)
    @printf("Subset %d channels and %d observables from %s\n",
        length(channels), length(names), source_bson)
    @printf("Saved subset target artifact to %s\n", output_bson)
end

function main()
    root = normpath(joinpath(@__DIR__, ".."))
    source = length(ARGS) >= 1 ? ARGS[1] : joinpath(root, "models", "dM_targets_compact.bson")
    channels = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(root, "configs", "nonlinear_observable_retained_channels_top6.toml")
    output = length(ARGS) >= 3 ? ARGS[3] : joinpath(root, "models", "dM_targets_top6.bson")
    scale_source = length(ARGS) >= 4 ? Symbol(lowercase(ARGS[4])) : :retained_channel_rms
    force_full_arrays = length(ARGS) >= 5 ? lowercase(ARGS[5]) in ("full", "full_arrays", "true", "1") : false
    subset_dm_targets(source, channels, output; scale_source, force_full_arrays)
end

main()
