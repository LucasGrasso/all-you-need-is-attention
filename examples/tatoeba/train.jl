
using Lux, Random, Plots

# ── DATA ───────────────────────────────────────────────
include(joinpath(@__DIR__,"./data_pipeline.jl"))
es_sentences, it_sentences = load_tsv(joinpath(@__DIR__, "es-it.tsv"))

es_vocab      = build_vocab(es_sentences)   # Spanish only
it_vocab      = build_vocab(it_sentences)   # Italian only

src_vocab_size = length(es_vocab)
tgt_vocab_size = length(it_vocab)

println("Spanish vocab size: $src_vocab_size")
println("Italian vocab size: $tgt_vocab_size")

pad_id = 1
sos_id = 2
eos_id = 3

data = make_dataset(
    es_sentences, it_sentences,
    es_vocab, it_vocab;
    max_len     = 64,
    sos_id      = sos_id,
    eos_id      = eos_id,
    max_samples = 5000
)

# ── MODEL ──────────────────────────────────────────────
include(joinpath(@__DIR__, "../../src/model/transformer.jl"))

config = (
    src_vocab_size = src_vocab_size,
    tgt_vocab_size = tgt_vocab_size,
    d_model    = 128,
    max_len    = 64,
    num_layers = 2,
    h          = 4,
    d_ff       = 512
)

model  = AttTransformer.Transformer(
    config.src_vocab_size,
    config.tgt_vocab_size,
    config.d_model,
    config.max_len,
    config.num_layers,
    config.h,
    config.d_ff
)

rng    = Random.default_rng()
ps, st = Lux.setup(rng, model)

# ── TRAINING ───────────────────────────────────────────
include(joinpath(@__DIR__, "../../src/model/training.jl"))

ps, st, losses = train!(model, ps, st, data; epochs=20, lr=1e-3, pad_id=pad_id)
plot(losses, xlabel="Epoch", ylabel="Loss", title="Training Loss")
savefig(joinpath(@__DIR__, "loss_curve.png"))

# ── SAVE MODEL ─────────────────────────────────────────
include(joinpath(@__DIR__, "../../src/model/io.jl"))
ModelIO.save_model(joinpath(@__DIR__, "tatoeba_transformer.jld2"), ps, st, config)