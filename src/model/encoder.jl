include("./attention.jl")
include("./ffn.jl")

module Encoder

using Lux
using ..Attention
using ..FFN

struct EncoderBlock <: Lux.AbstractLuxContainerLayer{(:multihead_attention, :ffn, :norm1, :norm2)}
    multihead_attention::MultiheadAttention
    ffn::FeedForward
    norm1::LayerNorm
    norm2::LayerNorm
end

function EncoderBlock(d_model::Int, h::Int, d_ff::Int)
    d_v = div(d_model, h)
    EncoderBlock(
        MultiheadAttention(h, d_model, d_v, d_v),
        FeedForward(d_model, d_ff),
        LayerNorm((d_model,)),
        LayerNorm((d_model,))
    )
end

function (m::EncoderBlock)(X, ps, st)
    # Self-attention
    attn_out, st_attn = m.multihead_attention((X, X, X), ps.multihead_attention, st.multihead_attention)
    X, st_n1 = m.norm1(X .+ attn_out, ps.norm1, st.norm1)  # Add & Norm

    # Feed-forward
    ffn_out, st_att = m.ffn(X, ps.ffn, st.ffn)
    X, st_n2 = m.norm2(X .+ ffn_out, ps.norm2, st.norm2)  # Add & Norm

    new_st = (
        multihead_attention=st_attn,
        ffn=st_att,
        norm1=st_n1,
        norm2=st_n2,
    )

    return X, new_st
end

end