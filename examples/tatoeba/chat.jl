include(joinpath(@__DIR__, "./data_pipeline.jl"))
include(joinpath(@__DIR__, "../../src/model/transformer.jl"))
include(joinpath(@__DIR__, "../../src/model/inference.jl"))
include(joinpath(@__DIR__, "../../src/model/io.jl"))

using Lux, Random, Plots
using CUDA, LuxCUDA

es_sentences, it_sentences = load_tsv(joinpath(@__DIR__, "es-it.tsv"))

es_vocab = build_vocab(es_sentences)   # Spanish only
it_vocab = build_vocab(it_sentences)   # Italian only

pad_id = 1
sos_id = 2
eos_id = 3

max_len = 64

src_vocab_size = length(es_vocab)
tgt_vocab_size = length(it_vocab)

model, ps, st = ModelIO.load_model(joinpath(@__DIR__, "tatoeba_transformer.jld2"))

# move to GPU if available
device = Lux.gpu_device()

if device != Lux.cpu_device()
    ps = ps |> device
    st = st |> device
end

while (true)
    print("Enter a Spanish sentence (or 'exit' to quit): ")
    input_sentence = readline()
    input_sentence == "exit" && break

    input_ids = encode_sentence(es_vocab, input_sentence, sos_id, eos_id)
    input_ids = reshape(input_ids, :, 1)  # (seq_len, 1)

    if CUDA.has_cuda()
        input_ids = input_ids |> device
    end

    output_ids = infer(model, ps, st, input_ids, sos_id, eos_id, max_len)
    output_sentence = decode(it_vocab, output_ids)

    println("Translated to Italian: $output_sentence")
end
