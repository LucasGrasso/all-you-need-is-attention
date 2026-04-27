function load_tsv(filepath::String)
    es_sentences = String[]
    it_sentences = String[]

    for line in eachline(filepath)
        parts = split(line, '\t')
        length(parts) == 4 || continue     # skip malformed lines
        push!(es_sentences, parts[2])
        push!(it_sentences, parts[4])
    end

    return es_sentences, it_sentences
end


function build_vocab(sentences::Vector{String})
    vocab = Dict{String,Int}()
    vocab["<PAD>"] = 1
    vocab["<SOS>"] = 2
    vocab["<EOS>"] = 3
    vocab["<UNK>"] = 4

    for sentence in sentences
        for word in split(sentence)
            if !haskey(vocab, word)
                vocab[word] = length(vocab) + 1
            end
        end
    end

    return vocab
end

function encode(vocab::Dict{String,Int}, sentence::String)
    return [get(vocab, word, vocab["<UNK>"]) for word in split(sentence)]
end

function encode_sentence(vocab, sentence, sos_id, eos_id)
    return [sos_id; encode(vocab, sentence); eos_id]
end

function decode(vocab::Dict{String,Int}, ids::Vector{Int})::String
    inv_vocab = Vector{String}(undef, length(vocab))
    for (word, id) in vocab
        inv_vocab[id] = word
    end
    words = [inv_vocab[id] for id in ids if id > 3]
    return join(words, " ")
end

function make_dataset(
    es_sentences::Vector{String},
    it_sentences::Vector{String},
    src_vocab::Dict{String,Int},
    tgt_vocab::Dict{String,Int};
    max_len::Int=64,
    sos_id::Int=2,
    eos_id::Int=3,
    max_samples::Int=typemax(Int)
)
    data = Vector{Tuple{Matrix{Int},Matrix{Int}}}()

    for (es, it) in Iterators.take(zip(es_sentences, it_sentences), max_samples)
        src = encode(src_vocab, es)
        tgt = encode_sentence(tgt_vocab, it, sos_id, eos_id)

        (length(src) > max_len || length(tgt) > max_len) && continue

        # FIX: Reshape to (Sequence, Batch) 
        # Before: (1, seq) -> After: (seq, 1)
        src_mat = reshape(src, :, 1)
        tgt_mat = reshape(tgt, :, 1)

        push!(data, (src_mat, tgt_mat))
    end

    println("Loaded $(length(data)) sentence pairs")
    return data
end