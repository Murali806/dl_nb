# Layer Normalization — Visual Walkthrough

> **Context:** Layer Norm appears twice inside every Transformer block,
> wrapping the attention and feed-forward sublayers.
> This file explains exactly what it does and why.

---

## The Problem — Why Normalize at All?

After embeddings, attention, or feed-forward layers, the activation values
can have wildly different scales:

```
Token embeddings after one attention layer (d_model=4):

         d0       d1       d2       d3
"the"  [  0.3,   -0.1,    0.4,    0.2 ]   ← small, well-behaved
"cat"  [  8.7,  -12.4,   15.3,    9.1 ]   ← large values!
"sat"  [ -0.02,   0.01,  -0.03,   0.02]   ← tiny values!
```

**Problems this causes:**
- Large values → softmax saturates → gradients vanish
- Tiny values → no signal flows through the network
- Different scales per token → training is unstable and slow

**Solution:** Normalize each token's vector independently so every token
has mean ≈ 0 and std ≈ 1 before the next operation.

---

## Step 1: What Layer Norm Normalizes

**Key distinction — three types of normalization:**

```
Input tensor shape: (B=2, T=3, d_model=4)

Batch Norm  → normalizes across the BATCH dimension (B)
              "same feature, different samples"
              ❌ bad for variable-length sequences

Layer Norm  → normalizes across the FEATURE dimension (d_model)
              "all features of ONE token"
              ✅ used in Transformers

              For each token independently:
              normalize over [d0, d1, d2, d3]
```

```
Visualization — what gets normalized together:

         d0      d1      d2      d3
"the"  [ 0.3,  -0.1,   0.4,   0.2 ]  ← LayerNorm normalizes THIS row
"cat"  [ 8.7, -12.4,  15.3,   9.1 ]  ← LayerNorm normalizes THIS row
"sat"  [-0.02,  0.01, -0.03,  0.02]  ← LayerNorm normalizes THIS row

Each token is normalized independently.
No information flows between tokens during LayerNorm.
```

---

## Step 2: The Formula

```
For a single token vector x = [x0, x1, x2, x3]:

Step A: Compute mean
  μ = (x0 + x1 + x2 + x3) / d_model

Step B: Compute variance
  σ² = ((x0-μ)² + (x1-μ)² + (x2-μ)² + (x3-μ)²) / d_model

Step C: Normalize
  x̂_i = (x_i - μ) / sqrt(σ² + ε)       ε = 1e-5 (numerical stability)

Step D: Scale and shift (learnable)
  y_i = γ_i * x̂_i + β_i

Where:
  γ (gamma) = learnable scale parameter,  initialized to 1
  β (beta)  = learnable shift parameter,  initialized to 0
  ε         = small constant to avoid division by zero
```

---

## Step 3: Concrete Walkthrough — Token "cat"

Let's trace through `"cat"` with embedding `[8.7, -12.4, 15.3, 9.1]`:

### A: Compute Mean

```
x = [8.7, -12.4, 15.3, 9.1]

μ = (8.7 + (-12.4) + 15.3 + 9.1) / 4
  = 20.7 / 4
  = 5.175
```

### B: Compute Variance

```
Deviations from mean:
  x0 - μ =  8.7  - 5.175 =  3.525
  x1 - μ = -12.4 - 5.175 = -17.575
  x2 - μ = 15.3  - 5.175 = 10.125
  x3 - μ =  9.1  - 5.175 =  3.925

Squared deviations:
  (3.525)²  =  12.43
  (-17.575)² = 308.88
  (10.125)² = 102.52
  (3.925)²  =  15.41

σ² = (12.43 + 308.88 + 102.52 + 15.41) / 4
   = 439.24 / 4
   = 109.81

σ  = sqrt(109.81) = 10.48
```

### C: Normalize

```
x̂_i = (x_i - μ) / σ

x̂0 = (8.7  - 5.175) / 10.48 =  3.525 / 10.48 =  0.336
x̂1 = (-12.4 - 5.175) / 10.48 = -17.575 / 10.48 = -1.677
x̂2 = (15.3  - 5.175) / 10.48 = 10.125 / 10.48 =  0.966
x̂3 = (9.1  - 5.175) / 10.48 =  3.925 / 10.48 =  0.375

x̂ = [0.336, -1.677, 0.966, 0.375]

Verify: mean(x̂) ≈ 0.0,  std(x̂) ≈ 1.0  ✓
```

### D: Scale and Shift (γ and β)

```
At initialization:  γ = [1, 1, 1, 1],  β = [0, 0, 0, 0]

y_i = γ_i * x̂_i + β_i

y0 = 1 * 0.336  + 0 =  0.336
y1 = 1 * (-1.677) + 0 = -1.677
y2 = 1 * 0.966  + 0 =  0.966
y3 = 1 * 0.375  + 0 =  0.375

y = [0.336, -1.677, 0.966, 0.375]   ← same as x̂ at init

After training, γ and β are learned:
  γ = [0.8, 1.2, 0.9, 1.1]   ← model decides how much to scale each dim
  β = [0.1, -0.2, 0.0, 0.3]  ← model decides how much to shift each dim

y0 = 0.8 * 0.336  + 0.1  =  0.369
y1 = 1.2 * (-1.677) + (-0.2) = -2.212
y2 = 0.9 * 0.966  + 0.0  =  0.869
y3 = 1.1 * 0.375  + 0.3  =  0.713
```

> **Why γ and β?**
> Pure normalization forces mean=0, std=1 — too rigid.
> γ and β let the model **undo** the normalization if needed.
> They give the model the flexibility to find the best scale/shift
> for each feature dimension.

---

## Step 4: All Three Tokens Together

```
Before LayerNorm:

         d0       d1       d2       d3      mean     std
"the"  [  0.3,   -0.1,    0.4,    0.2 ]    0.20    0.19
"cat"  [  8.7,  -12.4,   15.3,    9.1 ]    5.18   10.48
"sat"  [ -0.02,   0.01,  -0.03,   0.02]   -0.005   0.02


After LayerNorm (γ=1, β=0):

         d0       d1       d2       d3      mean     std
"the"  [ 0.52,  -1.57,   1.05,    0.00]    0.00    1.00
"cat"  [ 0.34,  -1.68,   0.97,    0.38]    0.00    1.00
"sat"  [ 0.52,   1.57,  -1.57,    1.57]    0.00    1.00

Every token now has mean=0, std=1 regardless of its original scale.
```

---

## Step 5: Where Layer Norm Sits in the Transformer Block

```
TRANSFORMER BLOCK (Pre-Norm style — used in GPT):

Input x   shape (B, T, d_model)
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              │ (residual)
LayerNorm(x)   ← normalize BEFORE attention        │
    │                                              │
    ▼                                              │
MultiHeadAttention(...)                            │
    │                                              │
    ▼                                              │
    +──────────────────────────────────────────────┘
    │
    x = x + Attention(LayerNorm(x))   ← residual added back
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              │ (residual)
LayerNorm(x)   ← normalize BEFORE feed-forward    │
    │                                              │
    ▼                                              │
FeedForward(...)                                   │
    │                                              │
    ▼                                              │
    +──────────────────────────────────────────────┘
    │
    x = x + FFN(LayerNorm(x))         ← residual added back
    │
    ▼
Output x   shape (B, T, d_model)   ← same shape as input!
```

**In code:**
```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))   # ln1 = LayerNorm(n_embd)
    x = x + self.ffn(self.ln2(x))  # ln2 = LayerNorm(n_embd)
    return x
```

> **Pre-Norm vs Post-Norm:**
> Original "Attention is All You Need" paper used Post-Norm:
>   `x = LayerNorm(x + sublayer(x))`
> GPT and most modern models use Pre-Norm:
>   `x = x + sublayer(LayerNorm(x))`
> Pre-Norm trains more stably for deep networks.

---

## Step 6: LayerNorm Parameters

```
nn.LayerNorm(d_model=4)

Learnable parameters:
  γ (weight): shape (4,)   initialized to [1, 1, 1, 1]
  β (bias):   shape (4,)   initialized to [0, 0, 0, 0]

Total parameters: 2 × d_model = 2 × 4 = 8

In a full GPT model with n_layer=6, d_model=32:
  Each block has 2 LayerNorms → 2 × 2 × 32 = 128 params per block
  6 blocks → 768 params
  Final LayerNorm → 64 params
  Total LayerNorm params: 832

  Compare to total model params: ~50,000+
  LayerNorm is < 2% of parameters — tiny but critical!
```

---

## Step 7: Full Picture in One Place

```
Single token "cat" = [8.7, -12.4, 15.3, 9.1]   (d_model=4)

STEP A: Mean
  μ = (8.7 + (-12.4) + 15.3 + 9.1) / 4 = 5.175

STEP B: Std
  σ = sqrt(mean of squared deviations) = 10.48

STEP C: Normalize
  x̂ = (x - μ) / σ = [0.336, -1.677, 0.966, 0.375]
  Now: mean(x̂) = 0,  std(x̂) = 1

STEP D: Scale & Shift (learned γ, β)
  y = γ ⊙ x̂ + β
  At init (γ=1, β=0): y = x̂ = [0.336, -1.677, 0.966, 0.375]
  After training:      y = learned rescaling of x̂


SHAPE TRACE through LayerNorm:

  Input:   (B, T, d_model)  e.g. (4, 8, 32)
  μ:       (B, T, 1)        one mean per token
  σ:       (B, T, 1)        one std per token
  x̂:       (B, T, d_model)  normalized
  γ, β:    (d_model,)       broadcast across B and T
  Output:  (B, T, d_model)  ← same shape as input!
```

---

## Step 8: Runnable Code — All Steps Together

```python
import torch
import torch.nn as nn
import math

# ── Config ──────────────────────────────────────────────────────
d_model = 4
eps     = 1e-5

# ── Manual LayerNorm (to see every step) ────────────────────────
def manual_layer_norm(x, gamma, beta, eps=1e-5):
    """
    x:     shape (..., d_model)
    gamma: shape (d_model,)
    beta:  shape (d_model,)
    """
    # Step A: mean over last dimension (features)
    mean = x.mean(dim=-1, keepdim=True)          # (..., 1)

    # Step B: variance over last dimension
    var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # (..., 1)

    # Step C: normalize
    x_hat = (x - mean) / torch.sqrt(var + eps)   # (..., d_model)

    # Step D: scale and shift
    out = gamma * x_hat + beta                   # (..., d_model)

    return out, mean, var, x_hat


# ── Test with "cat" embedding ────────────────────────────────────
x_cat = torch.tensor([[8.7, -12.4, 15.3, 9.1]])   # shape (1, 4)

gamma = torch.ones(d_model)    # initialized to 1
beta  = torch.zeros(d_model)   # initialized to 0

out, mean, var, x_hat = manual_layer_norm(x_cat, gamma, beta)

print("Input x:        ", x_cat.numpy())
print(f"Mean μ:          {mean.item():.4f}")
print(f"Std  σ:          {math.sqrt(var.item()):.4f}")
print("Normalized x̂:   ", x_hat.detach().numpy())
print(f"x̂ mean:          {x_hat.mean().item():.6f}  (should be ~0)")
print(f"x̂ std:           {x_hat.std().item():.6f}   (should be ~1)")
print("Output y:        ", out.detach().numpy())


# ── PyTorch built-in (same result) ──────────────────────────────
ln = nn.LayerNorm(d_model)   # gamma=1, beta=0 by default

out_torch = ln(x_cat)
print("\nPyTorch LayerNorm output:", out_torch.detach().numpy())
print("Match:", torch.allclose(out, out_torch, atol=1e-5))


# ── All three tokens ─────────────────────────────────────────────
x_all = torch.tensor([
    [ 0.3,  -0.1,   0.4,   0.2],   # "the"
    [ 8.7, -12.4,  15.3,   9.1],   # "cat"
    [-0.02,  0.01, -0.03,  0.02],  # "sat"
])

out_all = ln(x_all)

print("\nBefore LayerNorm:")
print(x_all.numpy())
print(f"  means: {x_all.mean(dim=-1).numpy()}")
print(f"  stds:  {x_all.std(dim=-1).numpy()}")

print("\nAfter LayerNorm:")
print(out_all.detach().numpy())
print(f"  means: {out_all.mean(dim=-1).detach().numpy()}")
print(f"  stds:  {out_all.std(dim=-1).detach().numpy()}")


# ── In a Transformer Block ───────────────────────────────────────
class TransformerBlockWithLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)   # before attention
        self.ln2 = nn.LayerNorm(d_model)   # before feed-forward
        self.attn = nn.Linear(d_model, d_model)   # simplified
        self.ffn  = nn.Linear(d_model, d_model)   # simplified

    def forward(self, x):
        # Pre-norm + residual (attention)
        x = x + self.attn(self.ln1(x))

        # Pre-norm + residual (feed-forward)
        x = x + self.ffn(self.ln2(x))

        return x

block = TransformerBlockWithLayerNorm(d_model)
x_in  = torch.randn(2, 3, d_model)   # (B=2, T=3, d_model=4)
x_out = block(x_in)

print(f"\nTransformer Block:")
print(f"  Input  shape: {x_in.shape}")
print(f"  Output shape: {x_out.shape}")   # same!
```

---

## Intuition Summary

| Step | Operation | Input | Output | Meaning |
|------|-----------|-------|--------|---------|
| A | `mean(x, dim=-1)` | `(B,T,d)` | `(B,T,1)` | Average feature value per token |
| B | `var(x, dim=-1)` | `(B,T,d)` | `(B,T,1)` | Spread of features per token |
| C | `(x - μ) / σ` | `(B,T,d)` | `(B,T,d)` | Normalize to mean=0, std=1 |
| D | `γ * x̂ + β` | `(B,T,d)` | `(B,T,d)` | Learned rescale & shift |

---

## Key Insights

> **LayerNorm normalizes each token independently.**
> It looks at all `d_model` features of one token and rescales them.
> No information flows between tokens — it's purely per-token.

> **γ and β give the model an escape hatch.**
> If the model needs a different scale/shift for a particular feature,
> it can learn that through γ and β via backprop.
> Without them, LayerNorm would be too rigid.

> **Pre-Norm (used in GPT) is more stable than Post-Norm.**
> `x = x + sublayer(LayerNorm(x))` ensures the residual stream
> always has a clean, unnormalized signal flowing through it.
> The sublayer only needs to learn a small correction.

> **ε prevents division by zero.**
> If all features of a token happen to be identical (std=0),
> `sqrt(0 + 1e-5)` keeps the computation numerically safe.

> **LayerNorm has almost no parameters.**
> Just `2 × d_model` (γ and β). For `d_model=32`, that's 64 numbers.
> Yet it's one of the most important stabilizers in the whole model.
