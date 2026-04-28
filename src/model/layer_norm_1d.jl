module LayerNorm1D

using Lux, LinearAlgebra, Statistics, Random

export LayerNorm1DLayer

struct LayerNorm1DLayer <: Lux.AbstractLuxLayer
    d_model::Int
    ε::Float32
end
LayerNorm1DLayer(d_model::Int) = LayerNorm1DLayer(d_model, 1f-5)

Lux.initialparameters(rng::AbstractRNG, m::LayerNorm1DLayer) = (
    γ=ones(Float32, m.d_model),
    β=zeros(Float32, m.d_model),
)
Lux.initialstates(::AbstractRNG, ::LayerNorm1DLayer) = (;)

function (m::LayerNorm1DLayer)(x::AbstractArray{T,3}, ps, st) where T
    # x: (d_model, T, B) — normalize over dim 1
    μ = mean(x; dims=1)                          # (1, T, B)
    σ = std(x; dims=1, corrected=false) .+ m.ε  # (1, T, B)
    γ = reshape(ps.γ, :, 1, 1)                     # (d_model, 1, 1)
    β = reshape(ps.β, :, 1, 1)
    return γ .* ((x .- μ) ./ σ) .+ β, st
end

end