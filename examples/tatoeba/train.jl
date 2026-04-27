include(joinpath(@__DIR__, "./data_pipeline.jl"))
include(joinpath(@__DIR__, "../../src/model/transformer.jl"))
include(joinpath(@__DIR__, "../../src/model/training.jl"))
include(joinpath(@__DIR__, "../../src/model/io.jl"))

using Lux, Random, Plots
using CUDA, LuxCUDA

# ── DATA ───────────────────────────────────────────────

es_sentences, it_sentences = load_tsv(joinpath(@__DIR__, "es-it.tsv"))

es_vocab = build_vocab(es_sentences)
it_vocab = build_vocab(it_sentences)

src_vocab_size = length(es_vocab)
tgt_vocab_size = length(it_vocab)

pad_id = 1
sos_id = 2
eos_id = 3

data = make_batch_dataset(
    es_sentences, it_sentences,
    es_vocab, it_vocab;
    batch_size=32,      # Choose a size your GPU can handle (16, 32, 64)
    max_len=64,
    pad_id=pad_id,      # Critical for the loss_fn to ignore padding
    sos_id=sos_id,
    eos_id=eos_id,
)

println("Loaded $(length(data)) batches.")

# ── MODEL ──────────────────────────────────────────────

config = (
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=128,
    max_len=64,
    num_layers=2,
    h=4,
    d_ff=206
)

model = AttTransformer.Transformer(
    config.src_vocab_size,
    config.tgt_vocab_size,
    config.d_model,
    config.max_len,
    config.num_layers,
    config.h,
    config.d_ff
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Move to GPU if available
device = Lux.gpu_device()

if device != Lux.cpu_device()
    println("CUDA available, moving to GPU...")
    ps = ps |> device
    st = st |> device
    data = [(src |> device, tgt |> device) for (src, tgt) in data]
else
    println("No GPU found, training on CPU...")
end

# ── TRAINING ───────────────────────────────────────────

ps, st, losses = train!(model, ps, st, data; epochs=20, lr=1e-3, rng=rng, pad_id=pad_id)
plot(losses, xlabel="Epoch", ylabel="Loss", title="Training Loss")
savefig(joinpath(@__DIR__, "loss_curve.png"))

# ── SAVE MODEL ─────────────────────────────────────────

ModelIO.save_model(joinpath(@__DIR__, "tatoeba_transformer.jld2"), ps, st, config)