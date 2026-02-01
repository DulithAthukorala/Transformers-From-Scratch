# Transformer From Scratch (PyTorch)

A clean, educational implementation of the Transformer architecture built from the ground up. This project demystifies the inner workings of Transformers by implementing each component explicitly, making it ideal for learning and understanding how modern sequence-to-sequence models work.

## 🎯 What This Project Is About

The Transformer architecture powers today's most advanced language models, yet its mechanisms are often hidden behind high-level APIs. This implementation:

- **Builds everything from scratch** – Every component is implemented explicitly using PyTorch fundamentals
- **Prioritizes understanding** – Clear, well-commented code over production optimizations
- **Demonstrates key concepts** – Multi-head attention, positional encoding, encoder-decoder architecture
- **Includes a working training script** – See the model learn on a simple sequence copying task

This is a learning tool, not a production library. If you want to understand what's really happening inside a Transformer, you're in the right place.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/DulithAthukorala/Transformers-From-Scratch.git
cd Transformers-From-Scratch

# Install dependencies (optional but recommended: create a virtual environment first)
pip install -r requirements.txt

# Run the training script
python scripts/transformer_overfit_proof.py
```

The training script will:
- Create a simple sequence copying task
- Train a Transformer model from scratch
- Save checkpoints and training metrics
- Demonstrate greedy decoding on sample sequences

---

## 📚 Architecture Overview

The Transformer architecture is built in 5 modular phases, each building on the previous one:

### Phase 1: Multi-Head Attention (`src/model/attention.py`)

The heart of the Transformer - the attention mechanism that allows the model to focus on different parts of the input sequence.

**Key Components:**
- **Scaled Dot-Product Attention**: Computes attention scores using queries, keys, and values
- **Multi-Head Mechanism**: Splits embeddings across multiple attention heads for diverse representations
- **Masking Support**: Handles both padding masks and causal (look-ahead) masks

**How It Works:**
```
Input (512d) → Split into 8 heads (64d each)
             → Q, K, V projections
             → Attention(Q, K, V) = softmax(QK^T / √d_k) * V
             → Concatenate heads
             → Output projection → (512d)
```

**Key Features:**
- Parallel attention across multiple heads (default: 8 heads)
- Dropout regularization for attention weights
- Optional attention weight visualization for debugging

---

### Phase 2: Transformer Blocks (`src/model/transformer_blocks.py`)

Building blocks that combine attention with feedforward layers and residual connections.

**Components:**

**A) TransformerBlock (Encoder Block):**
- Multi-head self-attention layer
- Add & Layer Normalization
- Position-wise feedforward network (2-layer MLP with ReLU)
- Add & Layer Normalization
- Supports both Pre-Norm (more stable) and Post-Norm (original paper) variants

**B) DecoderBlock:**
- Masked multi-head self-attention (prevents looking ahead)
- Add & Layer Normalization
- Cross-attention with encoder outputs
- Add & Layer Normalization
- Position-wise feedforward network
- Add & Layer Normalization

**Architecture Pattern:**
```
Pre-Norm (recommended):
  LayerNorm → Sublayer → Residual Connection

Post-Norm (original paper):
  Sublayer → Residual Connection → LayerNorm
```

---

### Phase 3: Encoder (`src/model/encoder.py`)

Processes the source sequence and creates contextualized representations.

**Structure:**
```
Input tokens
  ↓
Word Embeddings (lookup table)
  ↓
Positional Embeddings (position info)
  ↓
Dropout
  ↓
Stack of N Transformer Blocks (default: 6 layers)
  ↓
Contextualized output
```

**Key Features:**
- Token embeddings scaled by √(embed_size) to balance with positional encodings
- Learned positional embeddings (up to max_length tokens)
- Stacked transformer blocks for deep contextual understanding
- Padding mask support to ignore pad tokens

---

### Phase 4: Decoder (`src/model/decoder.py`)

Generates the target sequence one token at a time, attending to both its own previous outputs and the encoder's representations.

**Structure:**
```
Target tokens (shifted right)
  ↓
Word Embeddings
  ↓
Positional Embeddings
  ↓
Dropout
  ↓
Stack of N Decoder Blocks (default: 6 layers)
  Each block performs:
    1. Masked self-attention (causal)
    2. Cross-attention with encoder output
    3. Feedforward network
  ↓
Linear projection → Vocabulary logits
```

**Key Features:**
- Causal (autoregressive) masking prevents future token leakage
- Cross-attention allows decoder to access full encoder context
- Final linear layer projects to vocabulary size for token prediction

---

### Phase 5: Full Transformer Model (`src/model/transformer.py`)

The complete sequence-to-sequence model that ties everything together.

**Architecture:**
```
Source Sequence → Encoder → Contextualized representations
                              ↓
Target Sequence → Decoder → Output logits
```

**Key Operations:**
1. **Mask Generation**:
   - Source mask: Hide padding tokens
   - Target mask: Hide padding tokens + prevent future peeking (causal mask)

2. **Forward Pass**:
   - Encode source sequence with padding mask
   - Decode target sequence with both source and causal target masks
   - Output vocabulary-sized logits for each position

**Configuration:**
- Vocabulary sizes for source and target languages
- Embedding dimension (default: 512)
- Number of layers (default: 6)
- Number of attention heads (default: 8)
- Feedforward expansion factor (default: 4x)
- Dropout rate for regularization

---

## 🧪 Training Script

The `scripts/transformer_overfit_proof.py` demonstrates the model on a **sequence copying task**:

**Task:** Given a random sequence of tokens, learn to reproduce it exactly.

**Why This Task?**
- Simple enough to debug quickly
- Complex enough to verify all components work
- Perfect for validating attention mechanisms
- Shows clear overfitting behavior (useful for testing)

**Training Features:**
- Fixed dataset for reproducible results
- Checkpoint saving and resuming
- TensorBoard logging for loss and accuracy
- Greedy decoding demonstration
- Attention heatmap visualization (for supported configurations)

---

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP

---

## License

MIT