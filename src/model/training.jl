include("./transformer.jl")

using Lux, Zygote, Optimisers, Random

function loss_fn(model, ps, st, src, tgt, pad_id, eps=1f-7)
    tgt_in = tgt[:, 1:end-1]
    tgt_out = vec(tgt[:, 2:end])
    pad_mask = tgt_out .!= pad_id

    logits, new_st = Lux.apply(model, (src, tgt_in), ps, st)

    loss = Lux.CrossEntropyLoss(; dims=1, logits=Val(false), epsilon=eps)(
        logits[:, pad_mask],
        tgt_out[pad_mask]
    )

    return loss, new_st
end

function train!(model::AttTransformer.Transformer, ps, st, data; epochs=10, lr=1e-3, eps=1f-7, rng=Random.default_rng(), pad_id=0)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    losses    = Float32[] 

    for epoch in 1:epochs
        total_loss = 0.0
        for (src, tgt) in data
            (loss, new_st), grads = Zygote.withgradient(ps) do ps
                loss_fn(model, ps, st, src, tgt, pad_id, eps)
            end
            # Backward pass
            opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
            st = new_st  # Update state if needed (for stateful layers)
            total_loss += loss
        end
        epoch_loss = total_loss / length(data)
        losses = push!(losses, epoch_loss)
        println("Epoch $epoch, Loss: $epoch_loss")
    end

    return ps, st, losses
end