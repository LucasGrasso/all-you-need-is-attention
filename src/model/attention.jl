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

"""
Apply multi-head attention to tensors shaped `(d_model, sequence_length, batch_size)`.

`mask` has shape `(1 or query_length, key_length, 1, batch_size)` and uses
`true` for positions that must not receive attention. Singleton query and head
dimensions let key-padding masks broadcast over all query positions and heads.
"""
function multihead_attention(
    m::MultiheadAttention,
    Q::AbstractArray{T,3},
    K::AbstractArray{T,3},
    V::AbstractArray{T,3},
    WQ::AbstractMatrix{T},  # (d_model , d_model)
    WK::AbstractMatrix{T},  # (d_model , d_model)
    WV::AbstractMatrix{T},  # (d_model , d_model)
    WO::AbstractMatrix{T},  # (h x d_v , d_model)
    mask::Union{Nothing,AbstractArray{Bool,4}}=nothing
) where {T<:AbstractFloat}
    D, Tq, B = size(Q)
    _, Tk, K_batch = size(K)
    size(V) == (D, Tk, B) || throw(DimensionMismatch("K and V must have matching sequence and batch dimensions"))
    K_batch == B || throw(DimensionMismatch("Q, K, and V must have the same batch size"))
    D == m.d_model || throw(DimensionMismatch("input feature dimension must equal d_model"))

    # Flatten only while applying independent linear projections. The original
    # `(feature, sequence, batch)` structure is restored before attention.
    Q_proj = reshape(WQ * reshape(Q, D, :), D, Tq, B)
    K_proj = reshape(WK * reshape(K, D, :), D, Tk, B)
    V_proj = reshape(WV * reshape(V, D, :), D, Tk, B)

    # Heads and examples become independent batched GEMMs, never a longer
    # sequence. The final axis of Qh/Kh/Vh is `head × batch`.
    Qh = reshape(permutedims(reshape(Q_proj, m.d_k, m.h, Tq, B), (1, 3, 2, 4)), m.d_k, Tq, :)
    Kh = reshape(permutedims(reshape(K_proj, m.d_k, m.h, Tk, B), (1, 3, 2, 4)), m.d_k, Tk, :)
    Vh = reshape(permutedims(reshape(V_proj, m.d_v, m.h, Tk, B), (1, 3, 2, 4)), m.d_v, Tk, :)

    scores = reshape(
        batched_mul(batched_adjoint(Qh), Kh) ./ sqrt(T(m.d_k)),
        Tq, Tk, m.h, B,
    )

    # Apply mask if provided
    if mask !== nothing
        (size(mask, 1) in (1, Tq) && size(mask, 2) == Tk && size(mask, 3) == 1 && size(mask, 4) == B) ||
            throw(DimensionMismatch("mask must have shape (1 or Tq, Tk, 1, B)"))
        scores = scores .+ ifelse.(mask, T(-Inf), T(0))
    end

    weights = softmax(scores; dims=2)
    heads = batched_mul(Vh, batched_adjoint(reshape(weights, Tq, Tk, :)))
    heads = reshape(heads, m.d_v, Tq, m.h, B)
    concat = reshape(permutedims(heads, (1, 3, 2, 4)), m.d_model, Tq, B)

    return reshape(WO * reshape(concat, m.d_model, :), m.d_model, Tq, B)
end

end
