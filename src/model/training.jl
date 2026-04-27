include("./transformer.jl")

using Lux, Zygote, Optimisers, Random, NNlib, Statistics

function loss_fn(model, ps, st, src, tgt, pad_id, eps=1f-7)
    # 1. Prepare shifted inputs
    tgt_in = tgt[1:end-1, :]
    tgt_out = tgt[2:end, :]

    # 2. Forward pass
    logits, new_st = Lux.apply(model, (src, tgt_in), ps, st)

    # 3. Flatten
    logits_flat = reshape(logits, size(logits, 1), :)
    targets_flat = vec(tgt_out)

    # 4. Masking
    pad_mask = targets_flat .!= pad_id
    if !any(pad_mask)
        return 0f0, new_st
    end

    y_hat = logits_flat[:, pad_mask]
    y = targets_flat[pad_mask]

    # --- THE UNIVERSAL FIX ---
    # logsumexp is defined for both CPU and GPU (via NNlib)
    lse = logsumexp(y_hat; dims=1)

    # We calculate the indices using the same type as 'y' 
    # This automatically puts the result on the correct device.
    vocab_size = size(y_hat, 1)
    len = length(y)

    # By using 'zero(y)', we ensure 'offsets' is on the same device as 'y'
    # '0:len-1' is a range (isbits), so it won't crash the GPU kernel.
    offsets = (0:len-1)
    indices = y .+ (offsets .* vocab_size)

    loss = mean(vec(lse) .- y_hat[indices])

    return loss, new_st
end

function train!(model, ps, st, data; epochs=10, lr=1e-3, eps=1f-7, rng=Random.default_rng(), pad_id=1)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    losses = Float32[]
    n_samples = length(data)

    for epoch in 1:epochs
        println("Epoch $epoch/$epochs")
        total_loss = 0.0
        for (src, tgt) in data
            (loss, new_st), grads = Zygote.withgradient(ps) do ps
                loss_fn(model, ps, st, src, tgt, pad_id, eps)
            end
            opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
            st = new_st
            total_loss += loss
        end
        epoch_loss = total_loss / n_samples
        push!(losses, epoch_loss)
        println("Epoch $epoch, Loss: $epoch_loss")
    end

    return ps, st, losses
end