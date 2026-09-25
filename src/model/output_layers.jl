module OutputLayers

using Lux, NNlib

export TransformerOutput

struct TransformerOutput <: Lux.AbstractLuxContainerLayer{(:projection,)}
    projection::Lux.Dense
end

function TransformerOutput(d_model::Int, vocab_size::Int)
    return TransformerOutput(Lux.Dense(d_model => vocab_size))
end

function (m::TransformerOutput)(x::AbstractArray{<:AbstractFloat,3}, ps, st)
    logits, st_proj = Lux.apply(m.projection, x, ps.projection, st.projection)
    new_st = (projection=st_proj,)
    return logits, new_st
end

end
