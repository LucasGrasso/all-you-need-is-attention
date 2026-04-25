module OutputLayers

using Lux, NNlib

export TransformerOutput

struct TransformerOutput <: Lux.AbstractLuxContainerLayer{(:projection,)}
    projection::Lux.Dense
end

function TransformerOutput(d_model::Int, vocab_size::Int)
    return TransformerOutput(Lux.Dense(d_model => vocab_size))
end

function (m::TransformerOutput)(x::AbstractMatrix, ps, st)
    logits, st_proj = Lux.apply(m.projection, x, ps.projection, st)
    prob = softmax(logits; dims=1) # Apply softmax along the vocab dimension
    return prob, st_proj
end

end