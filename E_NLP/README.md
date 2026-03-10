# 🧠 Building Large Language Models from Scratch

> A comprehensive, visualization-rich journey through the foundations of modern LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This project implements a GPT-style language model **from scratch**, focusing on deep understanding through extensive visualizations and step-by-step explanations. Based on the "Building LLMs from Scratch" lecture series by Vizuara AI and Sebastian Raschka's book, this repository takes you from raw text to a working language model.

### 🎯 What You'll Learn

- **Tokenization**: How text becomes numbers (Character-level & Byte Pair Encoding)
- **Data Pipeline**: Creating training batches with the sliding window technique
- **Embeddings**: Converting tokens to rich vector representations
- **Positional Encoding**: Teaching the model about word order
- *(Future)* Attention mechanisms, Transformer blocks, and training

### 🚫 What We Skip

This is **not** a basic deep learning tutorial. We assume you understand:
- Forward/backward propagation
- Gradient descent optimization
- Basic neural network concepts

We focus exclusively on **LLM-specific architecture** and the innovations that make Transformers work.

---

## 🗺️ Learning Path

```
📝 Raw Text
    ↓
🔤 Tokenization (Lectures 7-8)
    ↓
📦 Batching & Data Pipeline (Lecture 9)
    ↓
🎯 Token Embeddings (Lecture 10)
    ↓
📍 Positional Embeddings (Lecture 11)
    ↓
⚡ [Future: Attention Mechanism]
    ↓
🏗️ [Future: Transformer Blocks]
    ↓
🎓 [Future: Training & Generation]
```

---

## 📁 Project Structure

```
E_NLP/
├── README.md                          # You are here
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Interactive Jupyter notebooks
│   ├── 00_llm_fundamentals.ipynb     # What is an LLM? (Lectures 1-4)
│   ├── 01_char_tokenization.ipynb    # Character-level tokenizer (Lecture 7)
│   ├── 02_bpe_tokenization.ipynb     # Byte Pair Encoding (Lecture 8)
│   ├── 03_data_pipeline.ipynb        # Sliding window & batching (Lecture 9)
│   ├── 04_token_embeddings.ipynb     # Token embedding layer (Lecture 10)
│   └── 05_positional_embeddings.ipynb # Positional encoding (Lecture 11)
│
├── src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── tokenizer.py                   # CharTokenizer & BPETokenizer classes
│   ├── data.py                        # Dataset & DataLoader utilities
│   ├── embeddings.py                  # Embedding layer implementations
│   └── visualization.py               # Plotting & diagram utilities
│
├── data/                              # Training corpora
│   ├── shakespeare.txt                # Complete works of Shakespeare
│   └── sample.txt                     # Small test file
│
└── visualizations/                    # Generated plots & diagrams
    ├── tokenization/
    ├── embeddings/
    └── data_pipeline/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic understanding of PyTorch
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd E_NLP

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### First Steps

1. Start with `notebooks/00_llm_fundamentals.ipynb` to understand what LLMs are
2. Progress sequentially through the notebooks
3. Run all cells and interact with visualizations
4. Experiment by modifying hyperparameters

---

## 📚 Notebook Guide

### 🌟 Notebook 0: LLM Fundamentals
**Time:** ~30 minutes | **Lectures:** 1-4

**What you'll learn:**
- What is next-token prediction?
- Why Transformers replaced RNNs
- Encoder vs Decoder architecture
- Why GPT is "Decoder-only"

**Key visualizations:**
- Probability distribution over vocabulary
- Sequential (RNN) vs Parallel (Transformer) processing
- Bidirectional vs Unidirectional attention patterns

---

### 🔤 Notebook 1: Character-Level Tokenization
**Time:** ~45 minutes | **Lecture:** 7

**What you'll learn:**
- Building vocabulary from text
- String-to-Integer (stoi) and Integer-to-String (itos) mappings
- Encode/decode functions
- Trade-offs: small vocab, long sequences

**Key visualizations:**
- Character frequency histogram
- Encoding/decoding flow diagram
- Sequence length comparison

**Code preview:**
```python
# Simple character tokenizer
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

---

### 🧩 Notebook 2: Byte Pair Encoding (BPE)
**Time:** ~60 minutes | **Lecture:** 8

**What you'll learn:**
- The "Goldilocks" solution (not too small, not too large)
- Iterative pair merging algorithm
- How GPT-4's tokenizer works
- Unicode and emoji handling

**Key visualizations:**
- **Animated BPE merging** (step-by-step compression)
- Token count reduction over iterations
- Vocabulary growth curve
- Comparison with tiktoken (OpenAI's library)

**Algorithm overview:**
```
1. Start with byte-level vocabulary (256 tokens)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Replace all occurrences in the text
5. Repeat until target vocabulary size (e.g., 50,000)
```

---

### 📦 Notebook 3: Data Pipeline & Batching
**Time:** ~45 minutes | **Lecture:** 9

**What you'll learn:**
- The sliding window technique
- Input-target offset relationship
- Why one sequence creates multiple training examples
- Batch tensor construction

**Key visualizations:**
- **3D tensor visualization** (Batch × Sequence × Features)
- Sliding window animation
- Multiple (x, y) pairs from single sequence
- DataLoader pipeline flowchart

**Core concept:**
```python
# Input:  [18, 47, 56, 1, 12]
# Target: [47, 56, 1, 12, 33]  # Shifted by 1

# This creates 5 training examples:
# Given [18]           → predict 47
# Given [18, 47]       → predict 56
# Given [18, 47, 56]   → predict 1
# ...and so on
```

---

### 🎯 Notebook 4: Token Embeddings
**Time:** ~40 minutes | **Lecture:** 10

**What you'll learn:**
- Why integers can't be fed directly to neural networks
- Embedding as a learnable lookup table
- Semantic clustering in high-dimensional space
- Shape transformation: (B, T) → (B, T, C)

**Key visualizations:**
- Embedding matrix heatmap
- **t-SNE projection** (2D visualization of learned embeddings)
- Distance matrix between similar words
- Lookup operation diagram

**Implementation:**
```python
import torch.nn as nn

# Embedding layer
token_embedding = nn.Embedding(vocab_size, n_embd)

# Input: (Batch=4, Sequence=8) of integers
# Output: (Batch=4, Sequence=8, Embedding=384) of floats
```

---

### 📍 Notebook 5: Positional Embeddings
**Time:** ~40 minutes | **Lecture:** 11

**What you'll learn:**
- The permutation invariance problem
- Absolute vs relative positional encoding
- Why we add (not concatenate) embeddings
- Sinusoidal vs learned positional embeddings

**Key visualizations:**
- Positional encoding patterns (sine/cosine waves)
- **Before/after addition** comparison
- "The cat ate the mouse" vs "The mouse ate the cat"
- Position embedding matrix heatmap

**The magic formula:**
```python
# Token embedding: "What is this word?"
tok_emb = token_embedding_table(idx)  # (B, T, C)

# Position embedding: "Where is this word?"
pos_emb = position_embedding_table(torch.arange(T))  # (T, C)

# Final representation: Content + Location
x = tok_emb + pos_emb  # (B, T, C)
```

---

## 🎨 Visualization Gallery

### Tokenization Comparison
![Tokenization](visualizations/tokenization/comparison.png)
*Same text encoded with character-level vs BPE tokenization*

### Sliding Window Mechanism
![Sliding Window](visualizations/data_pipeline/sliding_window.png)
*How multiple training examples are extracted from a single sequence*

### Embedding Addition
![Embeddings](visualizations/embeddings/addition.png)
*Token vector + Position vector = Final representation*

### Complete Pipeline
![Pipeline](visualizations/pipeline_overview.png)
*End-to-end data flow from raw text to embedded tensors*

---

## 🔧 Technical Specifications

### Model Configuration (Educational Scale)

```python
config = {
    'vocab_size': 65,          # Character-level (small)
    'block_size': 256,         # Context window (sequence length)
    'n_embd': 384,            # Embedding dimension
    'batch_size': 32,         # Training batch size
    'learning_rate': 3e-4,    # AdamW learning rate
    'max_iters': 5000,        # Training iterations
    'eval_interval': 500,     # Validation frequency
}
```

### Hardware Requirements

- **Minimum:** CPU with 8GB RAM (for character-level models)
- **Recommended:** GPU with 6GB+ VRAM (for BPE models)
- **Training time:** ~10-30 minutes on GPU for educational models

---

## 📖 Key Concepts Explained

### 🔍 What is Next-Token Prediction?

An LLM's core task is simple: given a sequence of words, predict the most likely next word.

```
Input:  "The cat sat on the"
Output: Probability distribution over all words
        - "mat"   → 15%
        - "floor" → 10%
        - "moon"  → 0.001%
```

The model doesn't "understand" language—it learns statistical patterns from massive text corpora.

### 🔄 Why Transformers?

**Before (RNNs):**
- Process text sequentially (word-by-word)
- Forget long-range dependencies
- Cannot parallelize training

**After (Transformers):**
- Process entire sequence simultaneously
- Attention mechanism captures long-range relationships
- Massively parallelizable (faster training)

### 🎭 Encoder vs Decoder

| Component | Direction | Use Case | Example Model |
|-----------|-----------|----------|---------------|
| **Encoder** | Bidirectional | Understanding text | BERT (classification, Q&A) |
| **Decoder** | Unidirectional | Generating text | GPT (chat, completion) |
| **Both** | Hybrid | Translation | T5, BART |

**This project builds a Decoder-only model (GPT-style)** because our goal is text generation.

### 🧮 Why Embeddings?

**Problem:** Neural networks need continuous values, but tokens are discrete integers.

**Bad idea:** Feed integers directly
- Token 100 ("cat") vs Token 1000 ("dog") → model thinks "dog" is "10x more" than "cat"

**Solution:** Embedding lookup table
- Each token gets a unique vector of floats
- Similar words cluster together in high-dimensional space
- Learned during training via backpropagation

### 📍 Why Positional Embeddings?

**Problem:** Transformers process all tokens simultaneously (parallel), so they lose word order.

```python
# Without positional info, these are identical:
"The cat ate the mouse"
"The mouse ate the cat"
```

**Solution:** Add position information
- Token embedding: "What is this word?"
- Position embedding: "Where is this word in the sequence?"
- Final = Token + Position (element-wise addition)

---

## 🎓 Learning Resources

### Primary References

1. **Lecture Series:** [Building LLMs from Scratch - Vizuara AI](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)
   - Lectures 1-11 covered in this project
   
2. **Book:** [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
   - Chapters 1-4 align with our notebooks

3. **Original Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
   - The paper that introduced Transformers

### Supplementary Materials

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by OpenAI
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's BPE tokenizer

---

## 🛣️ Roadmap

### ✅ Phase 1: Preprocessing & Embeddings (Current)
- [x] Project structure
- [x] Character-level tokenization
- [x] Byte Pair Encoding
- [x] Data pipeline & batching
- [x] Token embeddings
- [x] Positional embeddings

### 🔜 Phase 2: Attention Mechanism (Planned)
- [ ] Self-attention (single head)
- [ ] Causal masking
- [ ] Multi-head attention
- [ ] Attention visualization tools

### 🔮 Phase 3: Transformer Architecture (Future)
- [ ] Feed-forward networks
- [ ] Layer normalization
- [ ] Residual connections
- [ ] Complete transformer block

### 🎯 Phase 4: Training & Generation (Future)
- [ ] Cross-entropy loss
- [ ] Training loop
- [ ] Text generation strategies
- [ ] Model evaluation

---

## 🤝 Contributing

This is an educational project. Contributions that improve clarity, add visualizations, or fix bugs are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-visualization`)
3. Commit your changes (`git commit -m 'Add amazing visualization'`)
4. Push to the branch (`git push origin feature/amazing-visualization`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Vizuara AI Team** (Dr. Raj Dandekar, Dr. Rajat Dandekar, Dr. Sreedath Panat) for the excellent lecture series
- **Sebastian Raschka** for the comprehensive book and educational materials
- **OpenAI** for pioneering GPT architecture and releasing tiktoken
- **The Transformer authors** (Vaswani et al.) for revolutionizing NLP

---

## 📧 Contact

Questions? Suggestions? Open an issue or reach out!

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Built with ❤️ for the AI learning community

</div>
