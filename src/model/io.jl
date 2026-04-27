include("./transformer.jl")

module ModelIO

import ..AttTransformer

using JLD2

function save_model(path::String, ps, st, model_config::NamedTuple)
    jldsave(path;
        ps=ps,
        st=st,
        model_config=model_config
    )
    println("Model saved to $path")
end

function load_model(path::String)
    data = JLD2.load(path)
    cfg = data["model_config"]

    # reconstruct model from config
    model = AttTransformer.Transformer(
        cfg.src_vocab_size,
        cfg.tgt_vocab_size,
        cfg.d_model,
        cfg.max_len,
        cfg.num_layers,
        cfg.h,
        cfg.d_ff
    )

    return model, data["ps"], data["st"]
end

end