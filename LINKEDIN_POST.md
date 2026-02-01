# LinkedIn Post - Transformers From Scratch

Built a complete Transformer encoder-decoder from first principles in PyTorch.

The implementation includes multi-head attention with parallel head computation, residual streams with pre-norm architecture, and position-wise feedforward networks. Both encoder and decoder stacks handle proper masking—padding masks for variable-length sequences and causal masks for autoregressive decoding.

Key architectural choices: pre-norm for training stability, learned positional embeddings up to max sequence length, and standard scaled dot-product attention with dropout regularization. The decoder uses both masked self-attention and cross-attention to the encoder's output, enabling proper sequence-to-sequence generation.

Validated correctness through a 100% token-level accuracy overfit test on a fixed sequence copying task. The model consistently reaches perfect token-level predictions on the training set, confirming that attention routing, residual connections, and gradient flow all work as expected.

The codebase is ~400 lines of model code. Clean enough to understand every operation, complete enough to train and generate sequences.

Repository: github.com/DulithAthukorala/Transformers-From-Scratch
