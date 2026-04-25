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

function (m::MultiheadAttention)(X::Tuple, ps, st)
    Q, K, V = X[1], X[2], X[3]
    mask = length(X) == 4 ? X[4] : nothing
    out = multihead_attention(m, Q, K, V, ps.WQ, ps.WK, ps.WV, ps.WO, mask)
    return out, st
end

function multihead_attention(
    m::MultiheadAttention,
    Q::AbstractMatrix{T},   # (seq , n)
    K::AbstractMatrix{T},   # (seq , n)
    V::AbstractMatrix{T},   # (seq , n)
    WQ::AbstractMatrix{T},  # (d_model , d_model)
    WK::AbstractMatrix{T},  # (d_model , d_model)
    WV::AbstractMatrix{T},  # (d_model , d_model)
    WO::AbstractMatrix{T},  # (h x d_v , d_model)
    mask::Union{Nothing,AbstractMatrix{Bool}}=nothing
) where {T<:AbstractFloat}
    # Project

    Q_proj = WQ * Q # (seq, d_model)
    K_proj = WK * K # (seq, d_model)
    V_proj = WV * V # (seq, d_model)

    # Split into heads

    Q_h = reshape(Q_proj, n, m.d_k, m.h)  # (seq, d_k, h)
    K_h = reshape(K_proj, n, m.d_k, m.h)  # (seq, d_k, h)
    V_h = reshape(V_proj, n, m.d_v, m.h)  # (seq, d_v, h)

    # Now h is the batch dimention, so we can use batched matrix multiplication for attention scores

    scores = batched_mul(Q_h, batched_adjoint(K_h)) ./ T(sqrt(m.d_k))  # (n, n, h)

    # Apply mask if provided
    if mask !== nothing
        scores[mask] .= -Inf
    end

    weights = softmax(scores, dims=2)
    heads = batched_mul(weights, V_h)  # (seq, d_v, h)

    concat = reshape(heads, n, m.d_v * m.h)  # (seq, h * d_v)
    out = WO * concat  # (seq, d_model)

    return out
end

end