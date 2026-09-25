using Test
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "..", "src", "model", "transformer.jl"))
include(joinpath(@__DIR__, "..", "examples", "tatoeba", "data_pipeline.jl"))
using .Attention
using .AttTransformer

@testset "batched multi-head attention" begin
    m = MultiheadAttention(1, 2, 2, 2)
    W = Matrix{Float32}(I, 2, 2)

    first_example = Float32[1 0 1; 0 1 1]
    second_example = Float32[-1 2; 3 0]
    padded_second = Float32[-1 2 0; 3 0 0]
    batched = cat(reshape(first_example, 2, 3, 1), reshape(padded_second, 2, 3, 1); dims=3)
    mask = falses(3, 3, 1, 2)
    mask[:, 3, 1, 2] .= true

    output = Attention.multihead_attention(m, batched, batched, batched, W, W, W, W, mask)
    first_output = Attention.multihead_attention(m, reshape(first_example, 2, 3, 1), reshape(first_example, 2, 3, 1), reshape(first_example, 2, 3, 1), W, W, W, W)
    second_output = Attention.multihead_attention(m, reshape(second_example, 2, 2, 1), reshape(second_example, 2, 2, 1), reshape(second_example, 2, 2, 1), W, W, W, W)

    @test output[:, :, 1] ≈ first_output[:, :, 1]
    @test output[:, 1:2, 2] ≈ second_output[:, :, 1]
end

@testset "batch collation and Transformer forward pass" begin
    examples = [
        (reshape([5, 6], :, 1), reshape([2, 7, 3], :, 1)),
        (reshape([8], :, 1), reshape([2, 9, 10, 3], :, 1)),
    ]
    batches = make_batches(examples; batch_size=2, pad_id=1)
    src, tgt, src_padding_mask, tgt_padding_mask = only(batches)

    @test size(src) == (2, 2)
    @test size(tgt) == (4, 2)
    @test src_padding_mask[:, 1] == [false, false]
    @test src_padding_mask[:, 2] == [false, true]
    @test tgt_padding_mask[:, 1] == [false, false, false, true]

    model = Transformer(16, 16, 8, 8, 2, 2, 16)
    ps, st = Lux.setup(MersenneTwister(7), model)
    logits, _ = Lux.apply(model, (src, tgt[1:end-1, :], src_padding_mask, tgt_padding_mask[1:end-1, :]), ps, st)
    @test size(logits) == (16, 3, 2)
end
