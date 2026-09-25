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

function (t::Transformer)(inputs::Tuple, ps, st)
    length(inputs) in (2, 4) || throw(ArgumentError("Transformer expects (src, tgt) or (src, tgt, src_padding_mask, tgt_padding_mask)"))
    src, tgt = inputs[1], inputs[2]
    src_padding_mask, tgt_padding_mask = if length(inputs) == 4
        inputs[3], inputs[4]
    else
        # Retain the original API for unpadded inputs.
        fill!(similar(src, Bool, size(src)), false), fill!(similar(tgt, Bool, size(tgt)), false)
    end

    # 1. Input processing
    src_enc, st_src = t.encoder_input(src, ps.encoder_input, st.encoder_input)
    tgt_enc, st_tgt = t.decoder_input(tgt, ps.decoder_input, st.decoder_input)

    # 2. Encoder
    (out_enc, _), st_enc = Lux.apply(
        t.encoder_blocks, (src_enc, src_padding_mask), ps.encoder_blocks, st.encoder_blocks
    )

    # 3. Decoder
    (out_dec, _, _, _, _), st_dec = Lux.apply(
        t.decoder_blocks,
        (tgt_enc, tgt_padding_mask, out_enc, out_enc, src_padding_mask),
        ps.decoder_blocks,
        st.decoder_blocks,
    )

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
