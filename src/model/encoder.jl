include("./attention.jl")
include("./ffn.jl")
include("./layer_norm_1d.jl")

module Encoder

using Lux
using ..Attention
using ..LayerNorm1D
using ..FFN

export EncoderBlock

struct EncoderBlock <: Lux.AbstractLuxContainerLayer{(:multihead_attention, :ffn, :norm1, :norm2)}
    multihead_attention::MultiheadAttention
    ffn::FeedForward
    norm1::LayerNorm1DLayer
    norm2::LayerNorm1DLayer
end

function EncoderBlock(d_model::Int, h::Int, d_ff::Int)
    d_v = div(d_model, h)
    EncoderBlock(
        MultiheadAttention(h, d_model, d_v, d_v),
        FeedForward(d_model, d_ff),
        LayerNorm1DLayer(d_model),
        LayerNorm1DLayer(d_model),
    )
end

function (m::EncoderBlock)((X, src_mask), ps, st)
    # Self-attention
    attn_out, st_attn = m.multihead_attention((X, X, X, src_mask), ps.multihead_attention, st.multihead_attention)
    X, st_n1 = m.norm1(X .+ attn_out, ps.norm1, st.norm1)  # Add & Norm

    # Feed-forward
    ffn_out, st_ffn = m.ffn(X, ps.ffn, st.ffn)
    X, st_n2 = m.norm2(X .+ ffn_out, ps.norm2, st.norm2)  # Add & Norm

    new_st = (
        multihead_attention=st_attn,
        ffn=st_ffn,
        norm1=st_n1,
        norm2=st_n2,
    )

    return (X, src_mask), new_st
end

end