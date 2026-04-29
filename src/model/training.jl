include("./transformer.jl")

using Lux, Zygote, Optimisers, Random, NNlib, Statistics, ProgressMeter

function loss_fn(model, ps, st, src, tgt)
    tgt_in = tgt[1:end-1, :]
    tgt_out = tgt[2:end, :]

    logits, new_st = Lux.apply(model, (src, tgt_in), ps, st)

    logits_flat = reshape(logits, size(logits, 1), :)
    y = vec(tgt_out)
    y_hat = logits_flat

    lse = logsumexp(y_hat; dims=1)
    vocab_size = size(y_hat, 1)
    len = length(y)

    offsets = (0:len-1)
    indices = y .+ (offsets .* vocab_size)

    loss = mean(vec(lse) .- y_hat[indices])

    return loss, new_st
end

function train!(model, ps, st, data; epochs=10, lr=1e-3, rng=Random.default_rng())
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    losses = Float32[]
    n_samples = length(data)
    epoch_data = collect(data)

    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        Random.shuffle!(rng, epoch_data)
        total_loss = 0f0
        pbar = Progress(n_samples; desc="epoch $epoch/$epochs")
        for (src, tgt) in epoch_data
            (loss, new_st), grads = Zygote.withgradient(ps) do ps
                loss_fn(model, ps, st, src, tgt)
            end
            opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
            st = new_st
            total_loss += loss
            next!(pbar; showvalues=[(:loss, round(loss; digits=4))])
        end
        finish!(pbar)
        epoch_loss = total_loss / n_samples
        push!(losses, epoch_loss)
        println("Epoch $epoch, Loss: $epoch_loss")
    end

    return ps, st, losses
end