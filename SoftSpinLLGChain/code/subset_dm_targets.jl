#!/usr/bin/env julia

include(joinpath(@__DIR__, "fit_dM.jl"))

using Printf
using Statistics

function subset_dm_targets(source_bson::AbstractString, channels_toml::AbstractString,
        output_bson::AbstractString; scale_source::Symbol=:retained_channel_rms)
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

    Cdata = Array(src[:Cdot_data])[:, :, name_indices, :]
    Cphi = Array(src[:Cdot_phi])[:, :, name_indices, :]
    A_target = Array(src[:A_target])[:, :, name_indices, :]
    target_vec = Array{Float32}(undef, length(src[:lags]), length(selected_inds))
    scale_vec = Array{Float32}(undef, length(selected_inds))
    for li in axes(target_vec, 1)
        flat = vec(A_target[li, :, :, :])
        target_vec[li, :] .= Float32.(flat[selected_inds])
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
    BSON.bson(output_bson, Dict(:names => names, :channels => channels,
        :observable_means => means, :lags => src[:lags], :taus => src[:taus],
        :selected_indices => selected_inds, :selected_channel_ids => selected_channel_ids,
        :target_vec => target_vec, :scale_vec => scale_vec,
        :Cdot_data => Cdata, :Cdot_phi => Cphi, :A_target => A_target,
        :Phi => src[:Phi], :Phi_block => src[:Phi_block],
        :scale_source => scale_source,
        :save_dt => src[:save_dt], :N => N, :D => D,
        :source_target_artifact => source_bson,
        :source_channels_toml => channels_toml,
        :no_cheating_audit => "This target artifact is a strict channel subset of an existing data/learned-score target artifact. No analytic score or true mobility was added."))
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
    subset_dm_targets(source, channels, output; scale_source)
end

main()
