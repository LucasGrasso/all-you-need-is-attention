include("./attention.jl")
include("./ffn.jl")

module Encoder

using Lux
using ..Attention
using ..FFN

export EncoderBlock

"""Apply Lux's 2-D LayerNorm independently to every sequence position and batch item."""
function layer_norm_3d(layer, x, ps, st)
    x_shape = size(x)
    normalized, new_st = layer(reshape(x, x_shape[1], :), ps, st)
    return reshape(normalized, x_shape), new_st
end

struct EncoderBlock <: Lux.AbstractLuxContainerLayer{(:multihead_attention, :ffn, :norm1, :norm2)}
    multihead_attention::MultiheadAttention
    ffn::FeedForward
    norm1::LayerNorm
    norm2::LayerNorm
end

function EncoderBlock(d_model::Int, h::Int, d_ff::Int)
    d_model % h == 0 || throw(ArgumentError("d_model must be divisible by h"))
    d_v = div(d_model, h)
    EncoderBlock(
        MultiheadAttention(h, d_model, d_v, d_v),
        FeedForward(d_model, d_ff),
        LayerNorm((d_model,)),
        LayerNorm((d_model,))
    )
end

function (m::EncoderBlock)((X, padding_mask)::Tuple, ps, st)
    _, seq_len, batch_size = size(X)
    size(padding_mask) == (seq_len, batch_size) || throw(DimensionMismatch("encoder padding mask must have shape (sequence, batch)"))
    key_mask = reshape(padding_mask, 1, seq_len, 1, batch_size)

    # Self-attention
    attn_out, st_attn = m.multihead_attention((X, X, X, key_mask), ps.multihead_attention, st.multihead_attention)
    X, st_n1 = layer_norm_3d(m.norm1, X .+ attn_out, ps.norm1, st.norm1)  # Add & Norm

    # Feed-forward
    ffn_out, st_ffn = m.ffn(X, ps.ffn, st.ffn)
    X, st_n2 = layer_norm_3d(m.norm2, X .+ ffn_out, ps.norm2, st.norm2)  # Add & Norm

    new_st = (
        multihead_attention=st_attn,
        ffn=st_ffn,
        norm1=st_n1,
        norm2=st_n2,
    )

    return (X, padding_mask), new_st
end

end
