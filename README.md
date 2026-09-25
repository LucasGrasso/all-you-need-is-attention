# Attention is all you need 🥺

This is a simple implementation of the attention mechanism in Julia. It is based on the paper "Attention is all you need" by Vaswani et al. (2017).

## Install Deps

```bash
] activate .
] instantiate
```

## Batching

Training inputs use `sequence × batch` token matrices. The model internally
uses `d_model × sequence × batch` tensors, and accepts source and target
padding masks alongside the token matrices:

```julia
logits, state = Lux.apply(model, (src, tgt, src_padding_mask, tgt_padding_mask), ps, state)
```

`examples/tatoeba/data_pipeline.jl` provides `make_batches`, which pads a
collection of examples and constructs those masks. The original two-input form
`(src, tgt)` remains available for unpadded inputs.

## Tests

```bash
julia --project=. test/runtests.jl
```

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762
