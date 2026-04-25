include("./positional_encoders.jl")

module InputLayers

using Lux, Random
using ..PositionalEncoders

export TransformerInput

struct TransformerInput <: Lux.AbstractLuxContainerLayer{(:embedding,)}
    embedding::Lux.Embedding
    pos_enc::PositionalEncoding
end

function TransformerInput(vocab_size::Int, d_model::Int, max_len::Int)
    return TransformerInput(
        Lux.Embedding(vocab_size => d_model),
        PositionalEncoding(d_model, max_len)
    )
end

function (m::TransformerInput)(x::AbstractMatrix{Int}, ps, st)
    emb, st_emb = Lux.apply(m.embedding, x, ps.embedding, st)

    d_model = size(emb, 1)
    scaled_emb = emb .* sqrt(Float32(d_model))

    return m.pos_enc(scaled_emb), st_emb
end

end