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
        scores = scores .+ ifelse.(mask, T(-Inf), T(0))
    end

    weights = softmax(scores, dims=2)

    V_h = reshape(V_proj, m.d_v, m.h, src_len, batch_size)
    V_h = permutedims(V_h, (1, 3, 2, 4))
    V_final = reshape(V_h, m.d_v, src_len, :)

    # weights_adj: (src_len, seq_len, h * batch_size)
    weights_adj = permutedims(weights, (2, 1, 3))

    # heads: (d_v, seq_len, h * batch_size)
    heads = batched_mul(V_final, weights_adj)

    # Bring back to 4D to isolate heads: (d_v, seq_len, h, batch_size)
    heads = reshape(heads, m.d_v, seq_len, m.h, batch_size)

    # Permute to get d_v and h back together: (d_v, h, seq_len, batch_size)
    concat = permutedims(heads, (1, 3, 2, 4))

    concat_flat = reshape(concat, m.d_model, seq_len, batch_size)


    out = reshape(WO * reshape(concat_flat, m.d_model, :), m.d_model, seq_len, batch_size)

    return out
end

end