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
    scale = inv(sqrt(T(m.d_model)))
    (
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
    Q::AbstractMatrix{T},   # (d_model , seq_len) - NOTE: Julia's convention is column-major, so the sequence length is the second dimension
    K::AbstractMatrix{T},   # (d_model , src_len)
    V::AbstractMatrix{T},   # (d_model , src_len)
    WQ::AbstractMatrix{T},  # (d_model , d_model)
    WK::AbstractMatrix{T},  # (d_model , d_model)
    WV::AbstractMatrix{T},  # (d_model , d_model)
    WO::AbstractMatrix{T},  # (h x d_v , d_model)
    mask::Union{Nothing,AbstractMatrix{Bool}}=nothing
) where {T<:AbstractFloat}
    seq_len = size(Q, 2)   # 
    src_len = size(K, 2)   # may differ from seq_len

    # Project

    Q_proj = WQ * Q # (d_model, seq_len)
    K_proj = WK * K # (d_model, src_len)
    V_proj = WV * V # (d_model, src_len)

    # Split into heads

    Q_h = reshape(Q_proj, m.d_k, m.h, seq_len)  # (d_k, h, seq_len)
    K_h = reshape(K_proj, m.d_k, m.h, src_len)  # (d_k, h, src_len)
    V_h = reshape(V_proj, m.d_v, m.h, src_len)  # (d_v, h, src_len)

    Q_h = permutedims(Q_h, (1, 3, 2))  # (d_k, seq_len, h)
    K_h = permutedims(K_h, (1, 3, 2))  # (d_k, src_len, h)
    V_h = permutedims(V_h, (1, 3, 2))  # (d_v, src_len, h)

    # Now h is the batch dimention, so we can use batched matrix multiplication for attention scores

    scores = batched_mul(batched_adjoint(Q_h), K_h) ./ T(sqrt(m.d_k))  # (seq_len, src_len, m.h)

    # Apply mask if provided
    if mask !== nothing
        scores = scores .+ ifelse.(mask, T(-Inf), T(0))
    end

    weights = softmax(scores, dims=2)
    heads = batched_mul(V_h, batched_adjoint(weights))  # (d_v, seq_len, m.h)
    concat = permutedims(heads, (1, 3, 2))  # (d_v, m.h, seq_len)
    concat = reshape(concat, m.d_model, seq_len) # (d_model, seq_len)

    out = WO * concat  # (seq_len, m.d_model)

    return out
end

end