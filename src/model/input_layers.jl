include("./positional_encoders.jl")

module InputLayers

using Lux, Random
using ..PositionalEncoders

export TransformerInput

struct TransformerInput <: Lux.AbstractLuxContainerLayer{(:embedding, :pos_enc)}
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
    emb, st_emb = Lux.apply(m.embedding, x, ps.embedding, st.embedding)
    emb = dropdims(emb, dims=3) 
    d_model = size(emb, 1)
    scaled_emb = emb .* sqrt(Float32(d_model))
    out, st_pe = m.pos_enc(scaled_emb, ps.pos_enc, st.pos_enc)
    new_st = (embedding=st_emb, pos_enc=st_pe)

    return out, new_st
end

end