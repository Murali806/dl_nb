# Query, Key, Value & Embeddings — Visual Walkthrough
---

Full Picture in One Place
```
STEP 1: Words → Embeddings
──────────────────────────
"the cat sat"
      ↓  nn.Embedding
x = [[ 0.21, -0.54,  0.83,  0.12],   # "the"
     [ 0.67,  0.91, -0.34,  0.55],   # "cat"
     [-0.43,  0.22,  0.71, -0.88]]   # "sat"
     shape: [3, 4]


STEP 2: Embeddings → Q and K
─────────────────────────────
Q = x @ W_q    shape: [3, 4] @ [4, 3] = [3, 3]
K = x @ W_k    shape: [3, 4] @ [4, 3] = [3, 3]

Q = [[-0.21,  0.88,  0.41],   # "the" query — what "the" is looking for
     [ 0.54, -0.32,  0.77],   # "cat" query — what "cat" is looking for
     [ 0.13,  0.61, -0.55]]   # "sat" query — what "sat" is looking for

K = [[ 0.33,  0.71,  0.22],   # "the" key — what "the" is advertising
     [-0.44,  0.55,  0.81],   # "cat" key — what "cat" is advertising
     [ 0.62, -0.28,  0.43]]   # "sat" key — what "sat" is advertising


STEP 3: Q @ K.T → Who attends to whom
───────────────────────────────────────
scores = Q @ K.T / sqrt(3)

scores = [[ 1.2,  0.3, -0.5],   # "the" → [the=high, cat=low,  sat=none]
          [ 0.4,  1.8,  0.2],   # "cat" → [the=low,  cat=high, sat=low ]
          [-0.3,  0.6,  1.4]]   # "sat" → [the=none, cat=low,  sat=high]

attn_weights = softmax(scores)
             = [[0.6, 0.3, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6]]


STEP 4: attn_weights @ V → Final Output
─────────────────────────────────────────
output = attn_weights @ V   # weighted mix of Value vectors
```


---

## Intuition Summary
 ________________________________________________________________________________________
| Step      | Shape            | Operation             | Meaning                         |
|-----------|------------------|-----------------------|---------------------------------|
| `x`       | `[seq, d_model]` | Embedding lookup      | Word → dense vector             |
| `x @ W_q` | `[seq, d_k]`     | Linear projection     | "What am I looking for?"        |
| `x @ W_k` | `[seq, d_k]`     | Linear projection     | "What do I contain?"            |
| `x @ W_v` | `[seq, d_k]`     | Linear projection     | "What will I give out?"         |
| `Q @ K.T` | `[seq, seq]`     | Dot product           | "How similar is Q to K?"        |
| `softmax` | `[seq, seq]`     | Normalize rows        | "How much to attend each word?" |
| `attn @ V`| `[seq, d_k]`     | Weighted sum          | Final context-aware output      |
|________________________________________________________________________________________|


## Key Insight

> **Q and K are the same input embeddings viewed through two different lenses.**
> The dot product `Q @ K.T` measures **compatibility** between them.
> High score = "this word should pay attention to that word."