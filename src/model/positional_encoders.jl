"""
This module implements positional encoding for transformer models in Julia.
Positional encoding is a technique used to inject information about the position of tokens in a sequence into the model.
"""
module PositionalEncoders

using Lux, Random

export PositionalEncoding

struct PositionalEncoding <: Lux.AbstractLuxLayer
    d_model :: Int
    max_len :: Int
end

function Lux.initialparameters(rng::AbstractRNG, m::PositionalEncoding)
    pe = zeros(Float32, m.d_model, m.max_len)
    pos         = 1:m.max_len
    i_term      = 0:2:(m.d_model-1)
    denominators = 10000.0f0 .^ (i_term ./ Float32(m.d_model))
    angles      = (1 ./ denominators) .* pos'
    pe[1:2:end, :] .= sin.(angles)
    pe[2:2:end, :] .= cos.(angles)
    return (pe = pe,)
end

Lux.initialstates(::AbstractRNG, ::PositionalEncoding) = (;)

function (m::PositionalEncoding)(x::AbstractMatrix, ps, st)
    seq_len = size(x, 2)
    return x .+ @view(ps.pe[:, 1:seq_len]), st
end

end