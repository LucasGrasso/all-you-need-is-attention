include("./attention.jl")
include("./ffn.jl")

module Decoder

using Lux
using ..Attention
using ..FFN

struct DecoderBlock <: Lux.AbstractLuxContainerLayer{(:masked_multihead_attention, :multihead_attention, :ffn, :norm1, :norm2, :norm3)}
    masked_multihead_attention::MultiheadAttention
    multihead_attention::MultiheadAttention
    ffn::FeedForward
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
end

function DecoderBlock(d_model::Int, h::Int, d_ff::Int)
    d_v = div(d_model, h)
    DecoderBlock(
        MultiheadAttention(h, d_model, d_v, d_v),  # masked multi-head attention
        MultiheadAttention(h, d_model, d_v, d_v),  # multi-head attention
        FeedForward(d_model, d_ff),
        LayerNorm((d_model,)),
        LayerNorm((d_model,)),
        LayerNorm((d_model,))
    )
end

function (m::DecoderBlock)((X, K_e, V_e)::Tuple, ps, st)
    n = size(X, 1)

    # Masked self-attention
    mask = triu(trues(n, n), 1)  # Upper triangular mask to prevent attending to future tokens
    masked_attn_out, st_masked_attn = m.masked_multihead_attention((X, X, X, mask), ps.masked_multihead_attention, st.masked_multihead_attention)
    X, st_n1 = m.norm1(X .+ masked_attn_out, ps.norm1, st.norm1)  # Add & Norm

    # Encoder-decoder attention
    enc_dec_attn_out, st_attn = m.multihead_attention((X, K_e, V_e), ps.multihead_attention, st.multihead_attention)
    X, st_n2 = m.norm2(X .+ enc_dec_attn_out, ps.norm2, st.norm2)  # Add & Norm

    # Feed-forward
    ffn_out, st_ffn = m.ffn(X, ps.ffn, st.ffn)
    X, st_n3 = m.norm3(X .+ ffn_out, ps.norm3, st.norm3)  # Add & Norm

    new_st = (
        masked_multihead_attention=st_masked_attn,
        multihead_attention=st_attn,
        ffn=st_ffn,
        norm1=st_n1,
        norm2=st_n2,
        norm3=st_n3
    )

    return X, new_st
end

end