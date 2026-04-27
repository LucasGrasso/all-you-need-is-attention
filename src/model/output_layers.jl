module OutputLayers

using Lux, NNlib

export TransformerOutput

struct TransformerOutput <: Lux.AbstractLuxContainerLayer{(:projection,)}
    projection::Lux.Dense
end

function TransformerOutput(d_model::Int, vocab_size::Int)
    return TransformerOutput(Lux.Dense(d_model => vocab_size))
end

function (m::TransformerOutput)(x::AbstractArray, ps, st)
    d_model, seq_len, batch_size = size(x)

    x_flat = reshape(x, d_model, :)

    logits_flat, st_proj = Lux.apply(m.projection, x_flat, ps.projection, st.projection)

    logits = reshape(logits_flat, :, seq_len, batch_size)

    return logits, (projection=st_proj,)
end

end