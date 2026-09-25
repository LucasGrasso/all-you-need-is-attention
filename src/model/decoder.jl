include("./attention.jl")
include("./ffn.jl")

module Decoder

using Lux
using LinearAlgebra
using ..Attention
using ..FFN

export DecoderBlock

"""Apply Lux's 2-D LayerNorm independently to every sequence position and batch item."""
function layer_norm_3d(layer, x, ps, st)
    x_shape = size(x)
    normalized, new_st = layer(reshape(x, x_shape[1], :), ps, st)
    return reshape(normalized, x_shape), new_st
end

struct DecoderBlock <: Lux.AbstractLuxContainerLayer{(:masked_multihead_attention, :multihead_attention, :ffn, :norm1, :norm2, :norm3)}
    masked_multihead_attention::MultiheadAttention
    multihead_attention::MultiheadAttention
    ffn::FeedForward
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
end

function DecoderBlock(d_model::Int, h::Int, d_ff::Int)
    d_model % h == 0 || throw(ArgumentError("d_model must be divisible by h"))
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

function (m::DecoderBlock)((X, tgt_padding_mask, K_e, V_e, src_padding_mask)::Tuple, ps, st)
    _, tgt_len, batch_size = size(X)
    _, src_len, src_batch_size = size(K_e)
    batch_size == src_batch_size || throw(DimensionMismatch("encoder and decoder batch sizes must match"))
    size(tgt_padding_mask) == (tgt_len, batch_size) || throw(DimensionMismatch("target padding mask must have shape (sequence, batch)"))
    size(src_padding_mask) == (src_len, batch_size) || throw(DimensionMismatch("source padding mask must have shape (sequence, batch)"))

    # Masked self-attention
    causal_mask = reshape(triu(trues(tgt_len, tgt_len), 1) |> Lux.get_device(X), tgt_len, tgt_len, 1, 1)
    target_key_mask = reshape(tgt_padding_mask, 1, tgt_len, 1, batch_size)
    self_attention_mask = causal_mask .| target_key_mask
    masked_attn_out, st_masked_attn = m.masked_multihead_attention((X, X, X, self_attention_mask), ps.masked_multihead_attention, st.masked_multihead_attention)
    X, st_n1 = layer_norm_3d(m.norm1, X .+ masked_attn_out, ps.norm1, st.norm1)  # Add & Norm

    # Encoder-decoder attention
    source_key_mask = reshape(src_padding_mask, 1, src_len, 1, batch_size)
    enc_dec_attn_out, st_attn = m.multihead_attention((X, K_e, V_e, source_key_mask), ps.multihead_attention, st.multihead_attention)
    X, st_n2 = layer_norm_3d(m.norm2, X .+ enc_dec_attn_out, ps.norm2, st.norm2)  # Add & Norm

    # Feed-forward
    ffn_out, st_ffn = m.ffn(X, ps.ffn, st.ffn)
    X, st_n3 = layer_norm_3d(m.norm3, X .+ ffn_out, ps.norm3, st.norm3)  # Add & Norm

    new_st = (
        masked_multihead_attention=st_masked_attn,
        multihead_attention=st_attn,
        ffn=st_ffn,
        norm1=st_n1,
        norm2=st_n2,
        norm3=st_n3
    )

    return (X, tgt_padding_mask, K_e, V_e, src_padding_mask), new_st
end

end
