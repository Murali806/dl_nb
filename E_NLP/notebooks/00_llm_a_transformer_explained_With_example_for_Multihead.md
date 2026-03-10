# Multi-Head Attention — Visual Walkthrough

> **Prerequisite:** You understand single-head self-attention (Q, K, V).
> This file picks up exactly where that left off.

---

## Where We Left Off — Single Head Output

After one self-attention head, each token has a new vector that is a
**weighted mix of Value vectors** from all (past) tokens:

```
Input x:   (B, T, d_model)   e.g. (1, 3, 4)
                ↓
  Single SelfAttentionHead
  (W_q, W_k, W_v each of shape d_model × head_size)
                ↓
Output:    (B, T, head_size)  e.g. (1, 3, 2)
```

**The problem:** one head learns **one way** to attend.

```
"the cat sat"

Head might learn:
  "sat" attends strongly to "cat"  (subject-verb)

But it CANNOT simultaneously learn:
  "sat" attends strongly to "the"  (article-noun)
  "sat" attends strongly to "sat"  (self-reference / position)
```

One head = one lens. We need **multiple lenses**.

---

## Step 1: The Key Idea — Split the Embedding Dimension

Instead of one big head of size `d_model`, we run **h small heads** each of
size `head_size = d_model // h`.

```
d_model = 4,   num_heads = 2,   head_size = 2

                  d_model = 4
         ┌────────────────────────┐
         │  dim0  dim1  dim2  dim3│
"the"    │  0.21 -0.54  0.83  0.12│
"cat"    │  0.67  0.91 -0.34  0.55│
"sat"    │ -0.43  0.22  0.71 -0.88│
         └────────────────────────┘
                    ↓ split responsibility (NOT the data itself)

Head 1 gets its own W_q1, W_k1, W_v1  shape (4, 2)  → projects to 2 dims
Head 2 gets its own W_q2, W_k2, W_v2  shape (4, 2)  → projects to 2 dims

BOTH heads receive the FULL input x (shape 3×4).
The split is in the weight matrices, not in x.
```

> **Key insight:** `x` is NOT split. Each head sees the whole embedding.
> What differs is each head's own `W_q`, `W_k`, `W_v` — they learn
> different projections, so they attend to different things.

---

## Step 2: Each Head Computes Its Own Q, K, V

Using the same sentence `"the cat sat"` with `d_model=4`, `head_size=2`:

```
x  (shape 3×4):

         d0      d1      d2      d3
"the"  [ 0.21,  -0.54,   0.83,   0.12 ]
"cat"  [ 0.67,   0.91,  -0.34,   0.55 ]
"sat"  [-0.43,   0.22,   0.71,  -0.88 ]
```

### Head 1 weight matrices (shape 4×2):

```
W_q1:              W_k1:              W_v1:
[ 0.5,  0.1 ]     [ 0.3,  0.2 ]     [ 0.4,  0.1 ]
[ 0.2, -0.3 ]     [-0.1,  0.5 ]     [ 0.2,  0.3 ]
[-0.4,  0.6 ]     [ 0.4, -0.3 ]     [-0.1,  0.5 ]
[ 0.3,  0.2 ]     [ 0.2,  0.4 ]     [ 0.3, -0.2 ]
```

```
Q1 = x @ W_q1    shape (3, 2)        K1 = x @ W_k1    shape (3, 2)

         q0      q1                           k0      k1
"the"  [ 0.12,   0.71 ]              "the"  [ 0.44,   0.31 ]
"cat"  [ 0.38,  -0.15 ]              "cat"  [-0.22,   0.63 ]
"sat"  [-0.51,   0.29 ]              "sat"  [ 0.37,  -0.48 ]

  ↑ "What is Head 1 looking for?"      ↑ "What does Head 1 offer?"
```

### Head 2 weight matrices (shape 4×2) — different learned values:

```
W_q2:              W_k2:              W_v2:
[-0.2,  0.4 ]     [ 0.1, -0.3 ]     [ 0.3,  0.2 ]
[ 0.6, -0.1 ]     [ 0.4,  0.2 ]     [-0.4,  0.5 ]
[ 0.1,  0.5 ]     [-0.2,  0.6 ]     [ 0.2, -0.1 ]
[-0.3,  0.2 ]     [ 0.5, -0.1 ]     [ 0.1,  0.4 ]
```

```
Q2 = x @ W_q2    shape (3, 2)        K2 = x @ W_k2    shape (3, 2)

         q0      q1                           k0      k1
"the"  [ 0.33,  -0.22 ]              "the"  [-0.11,   0.52 ]
"cat"  [ 0.47,   0.18 ]              "cat"  [ 0.61,  -0.34 ]
"sat"  [-0.28,   0.55 ]              "sat"  [ 0.43,   0.27 ]

  ↑ "What is Head 2 looking for?"      ↑ "What does Head 2 offer?"
```

---

## Step 3: Each Head Computes Its Own Attention Scores

### Head 1 attention (with causal mask for autoregressive):

```
scores1 = Q1 @ K1.T / sqrt(head_size=2)

           "the"   "cat"   "sat"
"the"  [   0.81,   -inf,   -inf ]   ← can only see itself
"cat"  [   0.34,   0.92,   -inf ]   ← can see "the" and "cat"
"sat"  [  -0.22,   0.55,   0.74 ]   ← can see all three

After softmax:
           "the"   "cat"   "sat"
"the"  [   1.00,   0.00,   0.00 ]
"cat"  [   0.35,   0.65,   0.00 ]
"sat"  [   0.17,   0.38,   0.45 ]

Head 1 output = attn_weights1 @ V1    shape (3, 2)

         v0      v1
"the"  [ 0.31,   0.22 ]   ← only "the"'s value
"cat"  [ 0.28,   0.35 ]   ← mix of "the" and "cat" values
"sat"  [ 0.19,   0.41 ]   ← mix of all three values
```

### Head 2 attention (different pattern — different W matrices):

```
scores2 = Q2 @ K2.T / sqrt(head_size=2)

           "the"   "cat"   "sat"
"the"  [   0.44,   -inf,   -inf ]
"cat"  [  -0.18,   1.21,   -inf ]
"sat"  [   0.63,  -0.31,   0.88 ]

After softmax:
           "the"   "cat"   "sat"
"the"  [   1.00,   0.00,   0.00 ]
"cat"  [   0.18,   0.82,   0.00 ]   ← Head 2 focuses more on "cat" itself
"sat"  [   0.44,   0.14,   0.42 ]   ← Head 2 splits between "the" and "sat"

Head 2 output = attn_weights2 @ V2    shape (3, 2)

         v0      v1
"the"  [ 0.44,   0.18 ]
"cat"  [ 0.37,   0.52 ]
"sat"  [ 0.29,   0.33 ]
```

> **Notice:** Head 1 and Head 2 produce **different attention patterns**
> from the same input. Head 1 might focus on local/syntactic patterns,
> Head 2 might focus on semantic/long-range patterns.

---

## Step 4: Concatenation — Joining All Head Outputs

```
Head 1 output:   shape (3, 2)          Head 2 output:   shape (3, 2)

         h1_0    h1_1                           h2_0    h2_1
"the"  [ 0.31,   0.22 ]                "the"  [ 0.44,   0.18 ]
"cat"  [ 0.28,   0.35 ]                "cat"  [ 0.37,   0.52 ]
"sat"  [ 0.19,   0.41 ]                "sat"  [ 0.29,   0.33 ]


torch.cat([out1, out2], dim=-1)   ← concatenate along LAST dimension

Result:   shape (3, 4)   ← back to d_model!

         h1_0    h1_1    h2_0    h2_1
"the"  [ 0.31,   0.22,   0.44,   0.18 ]
"cat"  [ 0.28,   0.35,   0.37,   0.52 ]
"sat"  [ 0.19,   0.41,   0.29,   0.33 ]

         └── Head 1 ──┘  └── Head 2 ──┘
```

**Why concatenate?**
- Each head contributes `head_size` dimensions to the final representation
- `num_heads × head_size = d_model` → shape is restored to `(3, 4)`
- The first `head_size` dims carry Head 1's perspective
- The next `head_size` dims carry Head 2's perspective

---

## Step 5: Output Projection W_o — Mixing the Heads

The concatenated output has heads **side by side** but not yet **mixed**.
The output projection `W_o` (shape `d_model × d_model`) allows every output
dimension to draw from every head's contribution.

```
concat  (3, 4)   @   W_o  (4, 4)   =   final_output  (3, 4)


W_o (shape 4×4):
[ 0.3,  0.1, -0.2,  0.4 ]
[ 0.2, -0.3,  0.5,  0.1 ]
[-0.1,  0.4,  0.3, -0.2 ]
[ 0.4,  0.2, -0.1,  0.3 ]


For "cat" (row 1 of concat = [0.28, 0.35, 0.37, 0.52]):

final_output["cat"] = [0.28, 0.35, 0.37, 0.52] @ W_o

  = 0.28 * [0.3,  0.1, -0.2,  0.4]   ← scale row 0 of W_o
  + 0.35 * [0.2, -0.3,  0.5,  0.1]   ← scale row 1 of W_o
  + 0.37 * [-0.1, 0.4,  0.3, -0.2]   ← scale row 2 of W_o
  + 0.52 * [0.4,  0.2, -0.1,  0.3]   ← scale row 3 of W_o

  = [0.32, 0.21, 0.18, 0.37]          ← "cat" now has info from BOTH heads


Final output:   shape (3, 4)   ← same shape as input x!

         d0      d1      d2      d3
"the"  [ 0.28,   0.15,   0.22,   0.31 ]
"cat"  [ 0.32,   0.21,   0.18,   0.37 ]
"sat"  [ 0.25,   0.33,   0.29,   0.24 ]
```

**Why W_o?**
- Without it, Head 1's dims and Head 2's dims never interact
- W_o lets each output dimension be a **linear combination of all heads**
- It's the "integration" step — turning parallel perspectives into one view

---

## Step 6: Full Picture in One Place

```
INPUT x   shape (3, 4)
"the cat sat" as embeddings
         │
         ├──────────────────────────────────────────────────────┐
         │                                                      │
         ▼                                                      ▼
   ┌─────────────┐                                      ┌─────────────┐
   │   HEAD 1    │                                      │   HEAD 2    │
   │             │                                      │             │
   │ W_q1 (4×2)  │                                      │ W_q2 (4×2)  │
   │ W_k1 (4×2)  │                                      │ W_k2 (4×2)  │
   │ W_v1 (4×2)  │                                      │ W_v2 (4×2)  │
   │             │                                      │             │
   │ Q1=x@W_q1   │                                      │ Q2=x@W_q2   │
   │ K1=x@W_k1   │                                      │ K2=x@W_k2   │
   │ V1=x@W_v1   │                                      │ V2=x@W_v2   │
   │             │                                      │             │
   │ scores1=    │                                      │ scores2=    │
   │  Q1@K1.T/√2 │                                      │  Q2@K2.T/√2 │
   │             │                                      │             │
   │ mask+softmax│                                      │ mask+softmax│
   │             │                                      │             │
   │ out1=attn@V1│                                      │ out2=attn@V2│
   │ shape (3,2) │                                      │ shape (3,2) │
   └──────┬──────┘                                      └──────┬──────┘
          │                                                    │
          └──────────────────┬─────────────────────────────────┘
                             │
                    torch.cat(dim=-1)
                             │
                             ▼
                    concat  shape (3, 4)
                    [head1_dims | head2_dims]
                             │
                          @ W_o  (4×4)
                             │
                             ▼
                  final_output  shape (3, 4)
                  ← same shape as input x! ←
```

---

## Step 7: Shape Summary — Every Transformation

```
n_embd=4, num_heads=2, head_size=2, seq_len=3 ("the cat sat")

Operation                    Shape           Notes
─────────────────────────────────────────────────────────────────
x  (input embeddings)        (3, 4)          seq × d_model

── Per Head (×2 heads) ──────────────────────────────────────────
W_q_h, W_k_h, W_v_h         (4, 2)          d_model × head_size
Q_h = x @ W_q_h             (3, 2)          seq × head_size
K_h = x @ W_k_h             (3, 2)          seq × head_size
V_h = x @ W_v_h             (3, 2)          seq × head_size
scores_h = Q_h @ K_h.T      (3, 3)          seq × seq
scores_h / sqrt(head_size)   (3, 3)          scaled
mask + softmax               (3, 3)          attention weights
out_h = attn_h @ V_h         (3, 2)          seq × head_size

── Combining ────────────────────────────────────────────────────
cat([out_1, out_2], dim=-1)  (3, 4)          seq × d_model  ✓
W_o                          (4, 4)          d_model × d_model
final = concat @ W_o         (3, 4)          seq × d_model  ✓
─────────────────────────────────────────────────────────────────

Input shape  = Output shape = (seq, d_model)   ← always preserved!
```

---

## Step 8: Runnable Code — All Steps Together

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Config ──────────────────────────────────────────────────────
vocab    = {"the": 0, "cat": 1, "sat": 2}
d_model  = 4
num_heads = 2
head_size = d_model // num_heads   # = 2
block_size = 8                     # max sequence length (for causal mask)

# ── Input ────────────────────────────────────────────────────────
embedding = nn.Embedding(len(vocab), d_model)
tokens    = torch.tensor([0, 1, 2])   # "the cat sat"
x         = embedding(tokens)         # shape (3, 4)

print("x (embeddings):", x.shape)
print(x)


# ── Single Attention Head ────────────────────────────────────────
class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape if x.dim() == 3 else (1, *x.shape)
        x = x.unsqueeze(0) if x.dim() == 2 else x

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Attention scores + causal mask
        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v   # (B, T, head_size)
        return out.squeeze(0) if B == 1 else out


# ── Multi-Head Attention ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        self.head_size = n_embd // num_heads

        # num_heads independent attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(n_embd, self.head_size, block_size)
            for _ in range(num_heads)
        ])

        # Output projection W_o: mixes information from all heads
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Step 1: Run all heads in parallel
        head_outputs = [head(x) for head in self.heads]
        # head_outputs[i] shape: (T, head_size)

        # Step 2: Concatenate along last dimension
        # (T, head_size) × num_heads  →  (T, d_model)
        concat = torch.cat(head_outputs, dim=-1)
        print(f"\nAfter cat: {concat.shape}")
        # e.g. (3, 4) — head1_dims | head2_dims

        # Step 3: Output projection — mix across heads
        out = self.proj(concat)   # (T, d_model)
        print(f"After W_o: {out.shape}")

        return out


# ── Run it ───────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model, num_heads, block_size)

print("\n── Head outputs (before concat) ──")
for i, head in enumerate(mha.heads):
    out_i = head(x)
    print(f"Head {i+1} output shape: {out_i.shape}")   # (3, 2)

print("\n── Full Multi-Head Attention ──")
final_output = mha(x)
print(f"\nInput  shape: {x.shape}")           # (3, 4)
print(f"Output shape: {final_output.shape}")  # (3, 4)  ← same!
print(f"\nFinal output:\n{final_output}")
```

---

## Intuition Summary

| Step | Shape | Operation | Meaning |
|------|-------|-----------|---------|
| `x` | `[3, 4]` | Input embeddings | "the cat sat" as vectors |
| `x @ W_q_h` | `[3, 2]` per head | Linear project | Each head's queries |
| `x @ W_k_h` | `[3, 2]` per head | Linear project | Each head's keys |
| `x @ W_v_h` | `[3, 2]` per head | Linear project | Each head's values |
| `Q_h @ K_h.T / √2` | `[3, 3]` per head | Dot product + scale | Each head's attention scores |
| `mask + softmax` | `[3, 3]` per head | Normalize | Causal attention weights |
| `attn_h @ V_h` | `[3, 2]` per head | Weighted sum | Each head's context output |
| `cat(heads, dim=-1)` | `[3, 4]` | Concatenate | All perspectives side by side |
| `concat @ W_o` | `[3, 4]` | Linear project | Mix all heads → final output |

---

## Key Insights

> **Why not just use one big head of size d_model?**
> One head of size 4 has the same parameter count as 2 heads of size 2.
> But multiple heads learn **diverse** attention patterns because each head
> has its own independent W_q, W_k, W_v — they are free to specialize.

> **Why does concatenation work?**
> Each head "owns" `head_size` output dimensions. Concatenating them
> gives every head its own slice of the final representation vector.
> `num_heads × head_size = d_model` — the shape is always restored.

> **What does W_o actually do?**
> After concatenation, Head 1's dims and Head 2's dims are isolated.
> W_o is a `d_model × d_model` matrix that lets every output dimension
> be a linear combination of **all** heads' outputs — true integration.

> **Same complexity, more power.**
> `h` heads of size `d/h` costs the same FLOPs as 1 head of size `d`:
> `h × (T² × d/h) = T² × d`
> But you get `h` independent attention patterns for free.
