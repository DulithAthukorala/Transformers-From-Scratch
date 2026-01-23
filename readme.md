# Transformer From Scratch (PyTorch)

A pedagogical implementation of the Transformer architecture built from first principles, with a focus on understanding over abstraction.

> **⚠️ Work in Progress**  
> This project is in early development. Core components are being built incrementally.

---

## Motivation

The Transformer architecture is foundational to modern deep learning, yet its internal mechanics are often hidden behind high-level APIs. This project:

- Implements each component explicitly to understand what happens under the hood
- Starts with raw tensor operations before introducing PyTorch conveniences
- Prioritizes clarity and learning over performance or production-readiness
- Serves as a reference for the core ideas in *Attention Is All You Need* (Vaswani et al., 2017)

This is a learning artifact, not a library.

---

## Scope

### What this project includes:
- Manual implementation of multi-head self-attention
- Positional encoding, feedforward layers, layer normalization
- Encoder and decoder stacks
- Training loop for simple sequence-to-sequence tasks

### What this project avoids:
- Pretrained models or large-scale datasets
- Production optimizations (e.g., flash attention, kernel fusion)
- High-level abstraction layers (at least initially)
- Tokenization pipelines or deployment utilities

---

## Project Structure

```
.
├── src/
│   ├── attention.py          # Multi-head self-attention
│   ├── encoder.py            # Encoder block and stack
│   ├── decoder.py            # Decoder block and stack
│   ├── positional_encoding.py
│   ├── feedforward.py
│   └── transformer.py        # Full model
├── notebooks/
│   └── exploration.ipynb     # Step-by-step walkthroughs
├── tests/
│   └── test_attention.py     # Unit tests for components
├── requirements.txt
└── README.md
```

*(Structure is tentative and will evolve)*

---

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.x** – tensor operations and autograd
- **NumPy** – for manual implementations and comparisons
- **Jupyter** – for exploratory notebooks
- **pytest** – for component testing

---

## Setup

```bash
# Clone the repository
git clone https://github.com/DulithAthukorala/transformer-from-scratch.git
cd transformer-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Roadmap

- [ ] Scaled dot-product attention (manual)
- [ ] Multi-head attention mechanism
- [ ] Positional encoding
- [ ] Feedforward network + layer norm
- [ ] Encoder block
- [ ] Decoder block with masked attention
- [ ] Full Transformer model
- [ ] Training loop (toy task)
- [ ] Notebook: attention visualization
- [ ] Notebook: comparing manual vs PyTorch implementations

---

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP

---

## License

MIT