include("./transformer.jl")

using Lux, Zygote, Optimisers, Random, NNlib, ProgressMeter, OneHotArrays, Statistics


function loss_fn(model, ps, st, src, tgt, pad_id)
    tgt_in = tgt[1:end-1, :]   # (T-1, B)
    tgt_out = tgt[2:end, :]   # (T-1, B)

    logits, new_st = Lux.apply(model, (src, tgt_in, pad_id), ps, st)
    # logits: (V, T-1, B)

    V = size(logits, 1)
    logits_flat = reshape(logits, V, :)
    targets_vec = vec(tgt_out)  # (N,) where N = (T-1)*B
    targets_oh = onehotbatch(targets_vec, 1:V)

    all_losses = Lux.CrossEntropyLoss(; agg=nothing, logits=Val(true))(logits_flat, targets_oh)

    mask = targets_vec .!= pad_id                      # (N,) — ignora <PAD>
    loss = sum(all_losses .* mask) / (sum(mask) + 1f-7)


    return loss, new_st
end


function train!(model, ps, st, data; epochs=10, lr=1e-3, rng=Random.default_rng(), pad_id=1)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    losses = Float32[]
    n_samples = length(data)
    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        total_loss = 0.0f0
        pbar = Progress(n_samples; dt=0.1, desc="Optimizing: ")
        for (src, tgt) in data
            ((loss, new_st), grads) = Zygote.withgradient(ps) do p
                loss_fn(model, p, st, src, tgt, pad_id)
            end
            opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
            st = new_st
            total_loss += loss

            next!(pbar; showvalues=[(:loss, loss)])
        end
        epoch_loss = total_loss / n_samples
        push!(losses, epoch_loss)
        println("Epoch $epoch, Loss: $epoch_loss")
    end

    return ps, st, losses
end