module FFN

using Lux, NNlib

export FeedForward

struct FeedForward <: Lux.AbstractLuxContainerLayer{(:layer1, :layer2)}
    layer1::Dense
    layer2::Dense
end

# Constructor — only hyperparameters
function FeedForward(d_model::Int, d_ff::Int, activation=relu)
    FeedForward(
        Dense(d_model => d_ff, activation),
        Dense(d_ff => d_model)
    )
end

function (m::FeedForward)(X, ps, st)
    d_model, seq_len, batch_size = size(X)
    x_flat = reshape(X, d_model, :)
    x, st1 = m.layer1(x_flat, ps.layer1, st.layer1)
    x, st2 = m.layer2(x, ps.layer2, st.layer2)
    out = reshape(x, d_model, seq_len, batch_size)
    new_st = (layer1=st1, layer2=st2)

    return out, new_st
end

end