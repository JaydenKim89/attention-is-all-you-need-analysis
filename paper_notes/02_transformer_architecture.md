# Transformer Architecture

## Encoder

An encoder layer consists of two main sub-layers:

1. Multi-head self-attention

2. Position-wise feed-forward network

Self-attention combines information across sequence positions, while the feed-forward network transforms each token representation independently along the feature dimension. The FFN projects each representation to a higher-dimensional space, applies a nonlinear activation, and projects it back to the model dimension.

## Decoder

The decoder consists of a stack of identical layers, each containing three sub-layers. First, it applies **masked self-attention** with a causal mask to preserve the autoregressive property by preventing positions from attending to subsequent tokens. Second, it performs **encoder-decoder attention**, where queries come from the decoder and keys and values come from the encoder output. Finally, a **position-wise feed-forward network** processes each position independently.

### Encoder-Decoder Attention

In encoder-decoder attention, queries come from the previous decoder sub-layer, while keys and values come from the encoder output.

This allows each decoder position to attend to all positions in the input sequence. Attention weights are computed between the decoder query and the encoder keys, and these weights are used to form a weighted sum of the encoder values.

## Encoder vs. Decoder

The encoder can attend to all positions in the input sequence.

The decoder uses causal masking so that each position can attend only to itself and earlier output positions.

In addition to self-attention and the feed-forward network, each decoder layer includes encoder-decoder attention to access the encoder representations.

## Residual Connections and Layer Normalization

Each sub-layer is wrapped with a residual connection followed by layer normalization:

$$

\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))

$$

The residual connection provides a direct path for the input representation and helps gradient flow through the network. It also allows each sub-layer to learn a modification of the existing representation rather than replacing it entirely.

Layer normalization stabilizes training by normalizing the mean and variance of activations across the feature dimension for each token independently. Because it does not depend on batch statistics, it operates consistently across varying batch sizes and variable-length sequences.

## Questions to Answer

- Positional Encoding: How does the Transformer recover sequence-order information after removing recurrence?

- Scaled Dot-Product Attention: Why are the dot products scaled by $\sqrt{d_k}$?

- Multi-Head Attention: Why use multiple attention heads instead of a single head?
Where these are addressed in this repository:

- Positional encoding: [03 §8](03_attention_math.md#8-positional-information-and-a-permutation-symmetry) (what removing it does) and the `no_enc_pos` experiment in [04](04_training_and_results.md).
- Scaling by $\sqrt{d_k}$: [03 §2](03_attention_math.md#2-scaled-dot-product-attention-paper-eq-1-sec-321), including the assumptions behind the variance argument. The scale is not ablated here.
- Multiple heads: shapes in [03 §4](03_attention_math.md#4-multi-head-attention-sec-322). This project uses 4 heads and does not vary the count; the paper's head-count comparison is Table 3, rows (A).

## Implementation map

| Component | Paper | Implementation |
|---|---|---|
| Token embeddings, shared and scaled by $\sqrt{d_{model}}$ | Sec. 3.4 | `Transformer._embed`, tied output projection in `Transformer.decode` ([src/model.py](../src/model.py)) |
| Sinusoidal positional encoding | Sec. 3.5 | `sinusoidal_encoding`; switches `enc_pos`, `dec_pos` |
| Multi-head attention (no projection bias) | Sec. 3.2.2 | `MultiHeadAttention` |
| Source padding mask; causal + padding target mask | Sec. 3.2.3 | `Transformer.src_mask`, `Transformer.tgt_mask` |
| Position-wise FFN with ReLU | Eq. 2 | `FeedForward` |
| Residual, dropout, post-norm LayerNorm | Sec. 3.1, 5.4 | `EncoderLayer`, `DecoderLayer` |
| Token cross-entropy, PAD ignored | — | `token_cross_entropy` |
| Teacher-forced training / greedy decoding | Sec. 3.1, 6.1 (paper uses beam search) | `Transformer.forward` / `Transformer.greedy_decode` |
