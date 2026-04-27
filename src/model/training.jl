include("./transformer.jl")

using Lux, Zygote, Optimisers, Random, NNlib, ProgressMeter, OneHotArrays

function loss_fn(model, ps, st, src, tgt, pad_id, eps=1f-7)
    tgt_in = tgt[1:end-1, :]
    tgt_out = tgt[2:end, :]

    Z, new_st = Lux.apply(model, (src, tgt_in), ps, st)
    V = size(Z, 1)

    Y_onehot = OneHotArrays.onehotbatch(tgt_out, 1:V)

    ce_loss = Lux.CrossEntropyLoss(; logits=Val(true), agg=nothing)

    elementwise_loss = ce_loss(Z, Y_onehot)

    mask = Float32.(tgt_out .!= pad_id)
    mask = reshape(mask, 1, size(mask, 1), size(mask, 2))

    loss = sum(elementwise_loss .* mask) / (sum(mask) + eps)

    return loss, new_st
end

function train!(model, ps, st, data; epochs=10, lr=1e-3, eps=1f-7, rng=Random.default_rng(), pad_id=1)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    losses = Float32[]
    n_samples = length(data)
    pbar = Progress(n_samples; dt=0.1, desc="Optimizing: ")
    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        total_loss = 0.0f0
        ProgressMeter.update!(pbar, 0)
        for (src, tgt) in data
            ((loss, new_st), grads) = Zygote.withgradient(ps) do p
                loss_fn(model, p, st, src, tgt, pad_id, eps)
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