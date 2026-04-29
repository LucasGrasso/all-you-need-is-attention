include("./transformer.jl")

using Lux

function infer(model, ps, st, src, sos_id::Int, eos_id::Int, max_len::Int; debug=false)
    src_enc, _ = Lux.apply(
        model.encoder_input, src, ps.encoder_input, st.encoder_input
    )
    memory, _ = Lux.apply(
        model.encoder_blocks, src_enc, ps.encoder_blocks, st.encoder_blocks
    )

    # Start with <sos> token
    tgt_seq = fill(sos_id, 1, 1)  # (seq_len=1, batch=1)
    debug && println("[DEBUG] Started with SOS token: $sos_id")
    
    for step in 1:max_len
        tgt_enc, _ = Lux.apply(
            model.decoder_input, tgt_seq, ps.decoder_input, st.decoder_input
        )
        (out_dec, _, _), _ = Lux.apply(
            model.decoder_blocks, (tgt_enc, memory, memory), ps.decoder_blocks, st.decoder_blocks
        )
        logits, _ = Lux.apply(
            model.output_layer, out_dec, ps.output_layer, st.output_layer
        )
        next_token = argmax(logits[:, end])  # Get the last token's logits
        next_logits = logits[:, end]
        max_logit = maximum(next_logits)
        debug && println("[DEBUG] Step $step: token=$next_token, max_logit=$max_logit, EOS=$eos_id")
        
        next_token == eos_id && (debug && println("[DEBUG] Generated EOS, stopping."); break)

        tgt_seq = vcat(tgt_seq, fill(next_token, 1, 1))
    end

    result = vec(tgt_seq[2:end, :])  # Exclude <sos> token
    debug && println("[DEBUG] Final output tokens: $result (length=$(length(result)))")
    return result
end