module Attention

using Lux
using NNlib
using LinearAlgebra
using Random

export MultiheadAttention

struct MultiheadAttention <: Lux.AbstractLuxLayer
    h::Int
    d_model::Int
    d_k::Int
    d_v::Int
end

Lux.initialstates(::AbstractRNG, ::MultiheadAttention) = (;)  # stateless

function Lux.initialparameters(rng::AbstractRNG, m::MultiheadAttention)
    T = Float32
    scale = T(sqrt(1 / m.d_model)) # Standard scaling factor

    return (
        WQ=randn(rng, T, m.d_model, m.d_model) .* scale,
        WK=randn(rng, T, m.d_model, m.d_model) .* scale,
        WV=randn(rng, T, m.d_model, m.d_model) .* scale,
        WO=randn(rng, T, m.d_model, m.d_model) .* scale,
    )
end

function (m::MultiheadAttention)(X::Tuple, ps, st)
    Q, K, V = X[1], X[2], X[3]
    mask = length(X) == 4 ? X[4] : nothing
    out = multihead_attention(m, Q, K, V, ps.WQ, ps.WK, ps.WV, ps.WO, mask)
    return out, st
end

function multihead_attention(
    m::MultiheadAttention,
    Q::AbstractArray{T,3},   # (d_model , seq_len, batch_size) - NOTE: Julia's convention is column-major, so the sequence length is the second dimension
    K::AbstractArray{T,3},   # (d_model , src_len, batch_size)
    V::AbstractArray{T,3},   # (d_model , src_len, batch_size)
    WQ::AbstractMatrix{T},  # (d_model , d_model)
    WK::AbstractMatrix{T},  # (d_model , d_model)
    WV::AbstractMatrix{T},  # (d_model , d_model)
    WO::AbstractMatrix{T},  # (h x d_v , d_model)
    mask::Union{Nothing,AbstractArray{Bool,3}}=nothing
) where {T<:AbstractFloat}
    _, seq_len, batch_size = size(Q)
    src_len = size(K, 2)   # may differ from seq_len

    # Project

    # Flatten seq and batch together, multiply, then restore 3D
    Q_proj = reshape(WQ * reshape(Q, m.d_model, :), m.d_model, seq_len, batch_size) # (d_model, seq_len, batch_size)
    K_proj = reshape(WK * reshape(K, m.d_model, :), m.d_model, src_len, batch_size) # (d_model, src_len, batch_size)
    V_proj = reshape(WV * reshape(V, m.d_model, :), m.d_model, src_len, batch_size) # (d_model, src_len, batch_size)

    # Split into heads

    # Split into 4D: (d_k, h, seq, batch)
    Q_h = reshape(Q_proj, m.d_k, m.h, seq_len, batch_size)
    K_h = reshape(K_proj, m.d_k, m.h, src_len, batch_size)

    # Permute so d_k and seq/src are the first two dims: (d_k, seq, h, batch)
    Q_h = permutedims(Q_h, (1, 3, 2, 4))
    K_h = permutedims(K_h, (1, 3, 2, 4))

    Q_h_adj = permutedims(Q_h, (2, 1, 3, 4))

    Q_final = reshape(Q_h_adj, seq_len, m.d_k, :)
    K_final = reshape(K_h, m.d_k, src_len, :)

    scores = batched_mul(Q_final, K_final) ./ T(sqrt(m.d_k))

    # Apply mask if provided
    if mask !== nothing
        q_len, k_len, _ = size(scores)


        m_rows = min(size(mask, 1), q_len)
        m_cols = min(size(mask, 2), k_len)

        m_curr = mask[1:m_rows, 1:m_cols, :]

        m_h = reshape(m_curr, m_rows, m_cols, 1, batch_size)
        m_h = repeat(m_h, outer=(div(q_len, m_rows), div(k_len, m_cols), m.h, 1))
        m_h_flat = reshape(m_h, q_len, k_len, :)

        scores = scores .+ ifelse.(m_h_flat, T(-1f9), T(0))
    end

    weights = softmax(scores, dims=2)

    # Prepare V_final: (src_len, d_v, total_heads)
    # We want V to be (src_len, d_v) so we can do (seq, src) * (src, d_v)
    V_h = reshape(V_proj, m.d_v, m.h, src_len, batch_size)
    V_h = permutedims(V_h, (3, 1, 2, 4)) # (src_len, d_v, h, batch)
    V_final = reshape(V_h, src_len, m.d_v, :) # (src_len, d_v, total_heads)

    # Multiply: (seq, src) * (src, d_v) -> (seq, d_v)
    # This is the standard Transformer Attention formula
    heads = batched_mul(weights, V_final) # (seq_len, d_v, total_heads)

    # Reshape back: (d_v, seq_len, h, batch_size)
    # Note: seq_len and d_v are now dims 1 and 2, so we permute
    heads_4d = reshape(heads, seq_len, m.d_v, m.h, batch_size)
    concat = permutedims(heads_4d, (2, 3, 1, 4)) # (d_v, h, seq_len, batch)

    # Final Projection
    concat_flat = reshape(concat, m.d_model, seq_len, batch_size)
    out = reshape(WO * reshape(concat_flat, m.d_model, :), m.d_model, seq_len, batch_size)

    return out
end

end