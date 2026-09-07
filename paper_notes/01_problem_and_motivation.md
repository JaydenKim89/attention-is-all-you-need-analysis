# # Problem and Motivation

## Research Question

Can we perform sequence transduction using attention alone—without recurrence or convolution—while enabling greater parallelization?

## Existing Approach

Traditionally, sequence transduction models relied on recurrent or convolutional architectures within an encoder-decoder framework. Recurrent models such as RNNs, LSTMs, and GRUs were standard for sequence modeling. While attention mechanisms were also used, they were almost always combined with recurrent networks.

## Main Limitation

Recurrent models process sequences step by step because each hidden state depends on the previous state. This sequential nature prevents parallelization across token positions during training, and the limitation becomes more significant for longer sequences.



## Proposed Approach

The Transformer replaces recurrence and convolution with self-attention to model dependencies between sequence positions. This allows sequence representations to be computed in parallel across positions and reduces the path length between distant positions.