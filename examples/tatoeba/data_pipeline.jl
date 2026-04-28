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
    return [get(vocab, String(word), vocab["<UNK>"]) for word in tokenize(sentence)]
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

function pad_sequence(seq::Vector{Int}, max_len::Int, pad_id::Int)
    new_seq = fill(pad_id, max_len)
    len = min(length(seq), max_len)
    new_seq[1:len] .= seq[1:len]
    return new_seq
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

function make_batch_dataset(
    es_sentences, it_sentences, src_vocab, tgt_vocab;
    batch_size::Int=32,
    max_len::Int=64,
    pad_id::Int=1,
    sos_id::Int=2,
    eos_id::Int=3,
    max_samples::Int=typemax(Int)
)
    # 1. Filter and Encode
    # We use encode_sentence for both to ensure <SOS> and <EOS> are present
    valid_pairs = []
    for (es, it) in Iterators.take(zip(es_sentences, it_sentences), max_samples)
        src = encode_sentence(src_vocab, es, sos_id, eos_id)
        tgt = encode_sentence(tgt_vocab, it, sos_id, eos_id)

        if length(src) <= max_len && length(tgt) <= max_len
            push!(valid_pairs, (src, tgt))
        end
    end

    # 2. Shuffle data 
    # Critical so the model doesn't learn based on the order of the TSV
    Random.shuffle!(valid_pairs)

    # 3. Batching Logic
    dataset = []
    n_total = length(valid_pairs)
    n_batches = div(n_total, batch_size)

    for b in 0:(n_batches-1)
        start_idx = b * batch_size + 1

        # Pre-allocate matrices (SequenceLength, BatchSize)
        # Using Int64 ensures no conversion overhead on GPU embeddings
        batch_src = Matrix{Int64}(undef, max_len, batch_size)
        batch_tgt = Matrix{Int64}(undef, max_len, batch_size)

        for local_idx in 1:batch_size
            global_idx = start_idx + local_idx - 1
            src_vec, tgt_vec = valid_pairs[global_idx]

            # Use your existing pad_sequence helper
            batch_src[:, local_idx] .= pad_sequence(src_vec, max_len, pad_id)
            batch_tgt[:, local_idx] .= pad_sequence(tgt_vec, max_len, pad_id)
        end

        push!(dataset, (batch_src, batch_tgt))
    end

    println("Total valid pairs: $n_total")
    println("Created $n_batches batches of size $batch_size")

    return dataset
end