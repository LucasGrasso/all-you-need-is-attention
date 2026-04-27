include("./encoder.jl")
include("./decoder.jl")
include("./input_layers.jl")
include("./output_layers.jl")

module AttTransformer

using Lux
using ..Encoder, ..Decoder, ..InputLayers, ..OutputLayers

export Transformer

struct Transformer <: Lux.AbstractLuxContainerLayer{(:encoder_input, :decoder_input, :encoder_blocks, :decoder_blocks, :output_layer)}
    encoder_input::TransformerInput
    decoder_input::TransformerInput
    encoder_blocks::Lux.Chain
    decoder_blocks::Lux.Chain
    output_layer::TransformerOutput
end

function Transformer(src_vocab_size::Int, tgt_vocab_size::Int, d_model::Int, max_len::Int, num_layers::Int, h::Int, d_ff::Int)
    return Transformer(
        TransformerInput(src_vocab_size, d_model, max_len),
        TransformerInput(tgt_vocab_size, d_model, max_len),
        Lux.Chain([EncoderBlock(d_model, h, d_ff) for _ in 1:num_layers]...),
        Lux.Chain([DecoderBlock(d_model, h, d_ff) for _ in 1:num_layers]...),
        TransformerOutput(d_model, tgt_vocab_size)
    )
end

function (t::Transformer)((src, tgt)::Tuple{AbstractMatrix,AbstractMatrix}, ps, st)
    # 1. Input processing
    src_enc, st_src = t.encoder_input(src, ps.encoder_input, st.encoder_input)
    tgt_enc, st_tgt = t.decoder_input(tgt, ps.decoder_input, st.decoder_input)

    # 2. Encoder
    out_enc, st_enc = Lux.apply(t.encoder_blocks, src_enc, ps.encoder_blocks, st.encoder_blocks)

    # 3. Decoder
    (out_dec, _, _), st_dec = Lux.apply(t.decoder_blocks, (tgt_enc, out_enc, out_enc), ps.decoder_blocks, st.decoder_blocks)

    # 4. Final output layer
    logits, st_out = t.output_layer(out_dec, ps.output_layer, st.output_layer)

    new_st = (
        encoder_input=st_src,
        decoder_input=st_tgt,
        encoder_blocks=st_enc,
        decoder_blocks=st_dec,
        output_layer=st_out
    )

    return logits, new_st
end

end