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



function tokenize(sentence::String)
    s = lowercase(sentence)
    s = replace(s, r"([.,!?;:])" => s" \1 ")
    return split(s)
end

function build_vocab(sentences::Vector{String})
    vocab = Dict{String,Int}("<PAD>" => 1, "<SOS>" => 2, "<EOS>" => 3, "<UNK>" => 4)

    for sentence in sentences
        for word in tokenize(sentence)
            if !haskey(vocab, word)
                vocab[word] = length(vocab) + 1
            end
        end
    end
    return vocab
end

function encode(vocab::Dict{String,Int}, sentence::String)
    return [get(vocab, word, vocab["<UNK>"]) for word in tokenize(sentence)]
end

function encode_sentence(vocab, sentence, sos_id, eos_id)
    return [sos_id; encode(vocab, sentence); eos_id]
end


function decode(vocab::Dict{String,Int}, ids::AbstractVector{Int})::String
    inv_vocab = Vector{String}(undef, length(vocab))
    for (word, id) in vocab
        inv_vocab[id] = word
    end

    # We ignore <PAD> (1), <SOS> (2), and <EOS> (3)
    words = String[]
    for id in ids
        if id == 3 # <EOS>
            break
        elseif id > 3
            push!(words, inv_vocab[id])
        end
    end

    if isempty(words)
        return ""
    end

    sentence = join(words, " ")
    sentence = replace(sentence, r"\s+([.,!?;:])" => s"\1") # "ciao ." -> "ciao."

    return uppercasefirst(sentence)
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

"""Pad examples into fixed-size `sequence × batch` matrices and padding masks."""
function make_batches(data; batch_size::Int=32, pad_id::Int=1)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    batches = Vector{Tuple{Matrix{Int},Matrix{Int},Matrix{Bool},Matrix{Bool}}}()

    for first_index in 1:batch_size:length(data)
        examples = @view data[first_index:min(first_index + batch_size - 1, length(data))]
        batch_len = length(examples)
        src_len = maximum(length(vec(src)) for (src, _) in examples)
        tgt_len = maximum(length(vec(tgt)) for (_, tgt) in examples)

        src_batch = fill(pad_id, src_len, batch_len)
        tgt_batch = fill(pad_id, tgt_len, batch_len)
        for (column, (src, tgt)) in enumerate(examples)
            src_ids = vec(src)
            tgt_ids = vec(tgt)
            src_batch[1:length(src_ids), column] .= src_ids
            tgt_batch[1:length(tgt_ids), column] .= tgt_ids
        end

        push!(batches, (src_batch, tgt_batch, src_batch .== pad_id, tgt_batch .== pad_id))
    end

    return batches
end
