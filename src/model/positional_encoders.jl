"""
This module implements positional encoding for transformer models in Julia.
Positional encoding is a technique used to inject information about the position of tokens in a sequence into the model.
"""
module PositionalEncoders

export PositionalEncoding

"""
A struct to hold the positional encoding matrix.
"""
struct PositionalEncoding{T<:AbstractMatrix{Float32}}
    pe::T
end

"""
Create a positional encoding matrix of size (`d_model`, `max_len`).
"""
function PositionalEncoding(d_model::Int, max_len::Int)
    pe = zeros(Float32, d_model, max_len)

    pos = 1:max_len

    i_term = 0:2:(d_model-1)

    denominators = 10000.0 .^ (i_term ./ Float32(d_model)) # . is the broadcasting operator

    angles = (1 ./ denominators) .* pos'

    # Fill odd rows with sine, even rows with cosine
    pe[1:2:end, :] .= sin.(angles)
    pe[2:2:end, :] .= cos.(angles)

    return PositionalEncoding(pe)
end

"""
Apply positional encoding to the input matrix `x`. 
The input `x` is expected to have dimensions (`d_model`, `seq_len`).
"""
function (m::PositionalEncoding)(x::AbstractMatrix)
    # size(x, 2) is the number of tokens in the current input
    seq_len = size(x, 2)

    # Slicing with 1:seq_len
    return x .+ @view m.pe[:, 1:seq_len]
end

end