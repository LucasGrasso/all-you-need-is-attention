include("./transformer.jl")

function infer(model, ps, st, src, sos_id::Int, eos_id::Int, max_len::Int)
    src_enc, _ = Lux.apply(
        model.encoder_input, src, ps.encoder_input, st.encoder_input
    )
    memory, _ = Lux.apply(
        model.encoder_blocks, src_enc, ps.encoder_blocks, st.encoder_blocks
    )

    # Start with <sos> token
    tgt_seq = fill(sos_id, 1, 1)  # (1, 1)
    for _ in 1:max_len
        tgt_enc, _ = Lux.apply(
            model.decoder_input, tgt_seq, ps.decoder_input, st.decoder_input
        )
        (out_dec, _, _), _ = Lux.apply(
            model.decoder_blocks, (tgt_enc, memory, memory), ps.decoder_blocks, st.decoder_blocks
        )
        logits, _ = Lux.apply(
            model.output_layer, out_dec, ps.output_layer, st.output_layer
        )
        next_token = argmax(logits[:, end, 1])  # Get the last token's logits
        next_token == eos_id && break  # Stop if <eos> is predicted

        tgt_seq = vcat(tgt_seq, fill(Int(next_token), 1, 1))

    end

    return vec(Array(tgt_seq[2:end, :]))  # Remove <sos> and convert to CPU array
end