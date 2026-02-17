# LSTM Forward Propagation, Loss, and Backpropagation Formulae

## LSTM Architecture Overview

LSTM has 4 gates + cell state + hidden state:
1. **Forget Gate (f_t)**: What to forget from cell state
2. **Input Gate (i_t)**: What new information to add
3. **Cell Gate (C̃_t)**: Candidate values to add
4. **Output Gate (o_t)**: What to output from cell state
5. **Cell State (C_t)**: Long-term memory
6. **Hidden State (h_t)**: Short-term memory / output

---

## Forward Propagation

### Notation
- `F` = input dimension (features)
- `hd` = hidden dimension
- `od` = output dimension (number of classes)
- `@` = matrix multiplication
- `*` = element-wise multiplication (Hadamard product)
- `σ` = sigmoid function
- `tanh` = hyperbolic tangent

### Input at time t
```
x_t: (F × 1)      - Current input
h_{t-1}: (hd × 1) - Previous hidden state
C_{t-1}: (hd × 1) - Previous cell state
```

---

### Step 1: Forget Gate (What to forget from cell state)

```
Matrix Dimensions:
hdx1  =  hdx(hd+F)  @  (hd+F)x1  +  hdx1

concat_t = [h_{t-1}; x_t]  (concatenate vertically)
         = (hd+F) × 1

f_t = σ(W_f @ concat_t + b_f)
    = σ(W_f @ [h_{t-1}; x_t] + b_f)
```

**Expanded form:**
```
f_t = σ(W_fh @ h_{t-1} + W_fx @ x_t + b_f)

where:
  W_f = [W_fh | W_fx]  (concatenated horizontally)
  W_fh: (hd × hd)
  W_fx: (hd × F)
  b_f: (hd × 1)
  f_t: (hd × 1)  - values in [0, 1]
```

**Sigmoid function:**
```
σ(z) = 1 / (1 + exp(-z))
σ'(z) = σ(z) * (1 - σ(z))
```

---

### Step 2: Input Gate (What new information to add)

```
Matrix Dimensions:
hdx1  =  hdx(hd+F)  @  (hd+F)x1  +  hdx1

i_t = σ(W_i @ concat_t + b_i)
    = σ(W_i @ [h_{t-1}; x_t] + b_i)
```

**Expanded form:**
```
i_t = σ(W_ih @ h_{t-1} + W_ix @ x_t + b_i)

where:
  W_i = [W_ih | W_ix]
  W_ih: (hd × hd)
  W_ix: (hd × F)
  b_i: (hd × 1)
  i_t: (hd × 1)  - values in [0, 1]
```

---

### Step 3: Cell Gate / Candidate Values (New information to consider)

```
Matrix Dimensions:
hdx1  =  hdx(hd+F)  @  (hd+F)x1  +  hdx1

C̃_t = tanh(W_C @ concat_t + b_C)
    = tanh(W_C @ [h_{t-1}; x_t] + b_C)
```

**Expanded form:**
```
C̃_t = tanh(W_Ch @ h_{t-1} + W_Cx @ x_t + b_C)

where:
  W_C = [W_Ch | W_Cx]
  W_Ch: (hd × hd)
  W_Cx: (hd × F)
  b_C: (hd × 1)
  C̃_t: (hd × 1)  - values in [-1, 1]
```

**Tanh function:**
```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
tanh'(z) = 1 - tanh²(z)
```

---

### Step 4: Update Cell State (Forget old + Add new)

```
Matrix Dimensions:
hdx1  =  hdx1  *  hdx1  +  hdx1  *  hdx1

C_t = f_t * C_{t-1} + i_t * C̃_t
```

**Interpretation:**
- `f_t * C_{t-1}`: Keep relevant parts of old memory
- `i_t * C̃_t`: Add relevant parts of new information
- Element-wise operations (Hadamard product)

```
where:
  C_t: (hd × 1)  - Updated cell state
```

---

### Step 5: Output Gate (What to output from cell state)

```
Matrix Dimensions:
hdx1  =  hdx(hd+F)  @  (hd+F)x1  +  hdx1

o_t = σ(W_o @ concat_t + b_o)
    = σ(W_o @ [h_{t-1}; x_t] + b_o)
```

**Expanded form:**
```
o_t = σ(W_oh @ h_{t-1} + W_ox @ x_t + b_o)

where:
  W_o = [W_oh | W_ox]
  W_oh: (hd × hd)
  W_ox: (hd × F)
  b_o: (hd × 1)
  o_t: (hd × 1)  - values in [0, 1]
```

---

### Step 6: Update Hidden State (Output filtered cell state)

```
Matrix Dimensions:
hdx1  =  hdx1  *  hdx1

h_t = o_t * tanh(C_t)
```

**Interpretation:**
- `tanh(C_t)`: Squash cell state to [-1, 1]
- `o_t * tanh(C_t)`: Filter what to output

```
where:
  h_t: (hd × 1)  - Updated hidden state
```

---

### Step 7: Output Layer (Classification)

```
Matrix Dimensions:
odx1  =  odxhd  @  hdx1  +  odx1

logits_t = W_y @ h_t + b_y
y_t = softmax(logits_t)
```

**Softmax:**
```
y_t[i] = exp(logits_t[i]) / Σⱼ exp(logits_t[j])

where:
  W_y: (od × hd)
  b_y: (od × 1)
  logits_t: (od × 1)
  y_t: (od × 1)  - probability distribution
```

---

## Loss Function

```
L_t = -log(y_t[c])    <---- cross entropy loss

where c is the correct class index
```

**Full form:**
```
L_t = -Σᵢ target_t[i] × log(y_t[i])

Since target_t[i] = 0 for all i ≠ c, this simplifies to:
L_t = -log(y_t[c])
```

**Total loss over sequence:**
```
L = (1/T) × Σₜ L_t
```

---

## Backpropagation Through Time (BPTT)

### Gradient of Loss w.r.t. Output

```
Matrix Dimensions: odx1

∂L_t/∂y_t[i] = { -1/y_t[c]   if i = c (correct class)
               {  0          if i ≠ c (other classes)
```

### Gradient w.r.t. Logits (Softmax + Cross-Entropy Combined)

```
Matrix Dimensions: odx1

∂L_t/∂logits_t = y_t - one_hot(c)

∂L_t/∂logits_t[i] = { y_t[i] - 1   if i = c (correct class)
                     { y_t[i]       if i ≠ c (other classes)
```

---

### Gradient w.r.t. Output Layer Parameters

```
Matrix Dimensions:
∂L_t/∂W_y: (od × hd)
∂L_t/∂b_y: (od × 1)

∂L_t/∂W_y = ∂L_t/∂logits_t @ h_t.T
          = (y_t - one_hot(c)) @ h_t.T

∂L_t/∂b_y = ∂L_t/∂logits_t
          = (y_t - one_hot(c))
```

---

### Gradient w.r.t. Hidden State

```
Matrix Dimensions: hdx1

∂L_t/∂h_t = W_y.T @ ∂L_t/∂logits_t + ∂L_{t+1}/∂h_t (from next timestep)
          = W_y.T @ (y_t - one_hot(c)) + dh_next

where dh_next comes from backprop through time
```

**Note:** At the last timestep T, `dh_next = 0`

---

### Gradient w.r.t. Output Gate

```
Step 1: Gradient w.r.t. o_t (before sigmoid)

∂L_t/∂o_t = ∂L_t/∂h_t * tanh(C_t)
          = dh_t * tanh(C_t)

where * is element-wise multiplication
```

```
Step 2: Gradient through sigmoid activation

∂L_t/∂(W_o @ concat_t + b_o) = ∂L_t/∂o_t * σ'(W_o @ concat_t + b_o)
                               = ∂L_t/∂o_t * o_t * (1 - o_t)

Let's call this: do_raw_t = ∂L_t/∂o_t * o_t * (1 - o_t)
```

```
Matrix Dimensions:
do_raw_t: (hd × 1)

∂L_t/∂W_o = do_raw_t @ concat_t.T
          = do_raw_t @ [h_{t-1}; x_t].T

∂L_t/∂b_o = do_raw_t
```

---

### Gradient w.r.t. Cell State

```
Matrix Dimensions: hdx1

∂L_t/∂C_t = ∂L_t/∂h_t * o_t * (1 - tanh²(C_t)) + ∂L_{t+1}/∂C_t
          = dh_t * o_t * (1 - tanh²(C_t)) + dC_next

where:
  - First term: gradient from current hidden state
  - Second term: gradient from next timestep (BPTT)
  - dC_next comes from backprop through time
```

**Note:** At the last timestep T, `dC_next = 0`

---

### Gradient w.r.t. Cell Gate (Candidate Values)

```
Step 1: Gradient w.r.t. C̃_t

∂L_t/∂C̃_t = ∂L_t/∂C_t * i_t
          = dC_t * i_t
```

```
Step 2: Gradient through tanh activation

∂L_t/∂(W_C @ concat_t + b_C) = ∂L_t/∂C̃_t * (1 - C̃_t²)

Let's call this: dC_tilde_raw_t = ∂L_t/∂C̃_t * (1 - C̃_t²)
```

```
Matrix Dimensions:
dC_tilde_raw_t: (hd × 1)

∂L_t/∂W_C = dC_tilde_raw_t @ concat_t.T
          = dC_tilde_raw_t @ [h_{t-1}; x_t].T

∂L_t/∂b_C = dC_tilde_raw_t
```

---

### Gradient w.r.t. Input Gate

```
Step 1: Gradient w.r.t. i_t (before sigmoid)

∂L_t/∂i_t = ∂L_t/∂C_t * C̃_t
          = dC_t * C̃_t
```

```
Step 2: Gradient through sigmoid activation

∂L_t/∂(W_i @ concat_t + b_i) = ∂L_t/∂i_t * σ'(W_i @ concat_t + b_i)
                               = ∂L_t/∂i_t * i_t * (1 - i_t)

Let's call this: di_raw_t = ∂L_t/∂i_t * i_t * (1 - i_t)
```

```
Matrix Dimensions:
di_raw_t: (hd × 1)

∂L_t/∂W_i = di_raw_t @ concat_t.T
          = di_raw_t @ [h_{t-1}; x_t].T

∂L_t/∂b_i = di_raw_t
```

---

### Gradient w.r.t. Forget Gate

```
Step 1: Gradient w.r.t. f_t (before sigmoid)

∂L_t/∂f_t = ∂L_t/∂C_t * C_{t-1}
          = dC_t * C_{t-1}
```

```
Step 2: Gradient through sigmoid activation

∂L_t/∂(W_f @ concat_t + b_f) = ∂L_t/∂f_t * σ'(W_f @ concat_t + b_f)
                               = ∂L_t/∂f_t * f_t * (1 - f_t)

Let's call this: df_raw_t = ∂L_t/∂f_t * f_t * (1 - f_t)
```

```
Matrix Dimensions:
df_raw_t: (hd × 1)

∂L_t/∂W_f = df_raw_t @ concat_t.T
          = df_raw_t @ [h_{t-1}; x_t].T

∂L_t/∂b_f = df_raw_t
```

---

### Gradient w.r.t. Previous Hidden State (for BPTT)

```
Matrix Dimensions: hdx1

∂L_t/∂h_{t-1} = W_fh.T @ df_raw_t + 
                W_ih.T @ di_raw_t + 
                W_Ch.T @ dC_tilde_raw_t + 
                W_oh.T @ do_raw_t

This gradient flows back to the previous timestep
```

---

### Gradient w.r.t. Previous Cell State (for BPTT)

```
Matrix Dimensions: hdx1

∂L_t/∂C_{t-1} = ∂L_t/∂C_t * f_t

This gradient flows back to the previous timestep
```

---

## Summary: Complete BPTT Algorithm

### Forward Pass (for all t = 1 to T)
```
1. concat_t = [h_{t-1}; x_t]
2. f_t = σ(W_f @ concat_t + b_f)
3. i_t = σ(W_i @ concat_t + b_i)
4. C̃_t = tanh(W_C @ concat_t + b_C)
5. C_t = f_t * C_{t-1} + i_t * C̃_t
6. o_t = σ(W_o @ concat_t + b_o)
7. h_t = o_t * tanh(C_t)
8. logits_t = W_y @ h_t + b_y
9. y_t = softmax(logits_t)
10. L_t = -log(y_t[c])
```

### Backward Pass (for all t = T to 1)
```
Initialize:
  dh_next = 0
  dC_next = 0

For each timestep t (from T to 1):

1. Output layer gradients:
   dlogits_t = y_t - one_hot(c)
   dW_y += dlogits_t @ h_t.T
   db_y += dlogits_t

2. Hidden state gradient:
   dh_t = W_y.T @ dlogits_t + dh_next

3. Output gate gradients:
   do_t = dh_t * tanh(C_t)
   do_raw_t = do_t * o_t * (1 - o_t)
   dW_o += do_raw_t @ concat_t.T
   db_o += do_raw_t

4. Cell state gradient:
   dC_t = dh_t * o_t * (1 - tanh²(C_t)) + dC_next

5. Cell gate gradients:
   dC_tilde_t = dC_t * i_t
   dC_tilde_raw_t = dC_tilde_t * (1 - C̃_t²)
   dW_C += dC_tilde_raw_t @ concat_t.T
   db_C += dC_tilde_raw_t

6. Input gate gradients:
   di_t = dC_t * C̃_t
   di_raw_t = di_t * i_t * (1 - i_t)
   dW_i += di_raw_t @ concat_t.T
   db_i += di_raw_t

7. Forget gate gradients:
   df_t = dC_t * C_{t-1}
   df_raw_t = df_t * f_t * (1 - f_t)
   dW_f += df_raw_t @ concat_t.T
   db_f += df_raw_t

8. Gradients for next iteration (BPTT):
   dh_next = W_fh.T @ df_raw_t + W_ih.T @ di_raw_t + 
             W_Ch.T @ dC_tilde_raw_t + W_oh.T @ do_raw_t
   dC_next = dC_t * f_t
```

---

## Key Differences from Vanilla RNN

### Vanilla RNN:
```
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)

Gradient flows through:
  ∂h_t/∂h_{t-1} = W_hh.T * (1 - h_t²)
  
Problem: Repeated multiplication causes vanishing gradient
```

### LSTM:
```
C_t = f_t * C_{t-1} + i_t * C̃_t

Gradient flows through:
  ∂C_t/∂C_{t-1} = f_t
  
Solution: Additive path (+ instead of *) preserves gradient!
```

**Why LSTM solves vanishing gradient:**
- Cell state uses **addition** (C_t = f_t * C_{t-1} + ...)
- Gradient flows through **multiplication by f_t** (not repeated matrix multiplication)
- When f_t ≈ 1, gradient flows unchanged (gradient highway!)
- Gates learn when to preserve vs. update information

---

## Gradient Clipping (Important for LSTM)

```
After computing all gradients, clip them to prevent exploding gradients:

For each gradient g in [dW_f, db_f, dW_i, db_i, dW_C, db_C, dW_o, db_o, dW_y, db_y]:
    g = clip(g, -threshold, threshold)
    
Common threshold: 5.0
```

---

## Parameter Count Comparison

### Vanilla RNN:
```
Parameters = hd × hd + hd × F + hd + od × hd + od
           = hd² + hd×F + hd + od×hd + od
```

### LSTM:
```
Parameters = 4 × (hd × (hd + F) + hd) + od × hd + od
           = 4 × (hd² + hd×F + hd) + od×hd + od
           ≈ 4 times more than vanilla RNN
```

**Why 4×?** Four gates (forget, input, cell, output) each with their own weights!

---

## Implementation Tips

1. **Concatenation vs Separate Matrices:**
   - Can use `concat_t = [h_{t-1}; x_t]` with single weight matrix
   - Or use separate `W_fh, W_fx` matrices
   - Both are equivalent, concatenation is more efficient

2. **Numerical Stability:**
   - Clip gradients to prevent explosion
   - Use stable softmax: `softmax(x - max(x))`
   - Initialize forget gate bias to 1.0 (helps learning)

3. **Initialization:**
   - Xavier/Glorot initialization for weights
   - Forget gate bias: 1.0 (default to remembering)
   - Other biases: 0.0

4. **Gradient Checking:**
   - Verify gradients numerically for small examples
   - Check each gate separately
   - Ensure dimensions match throughout

---

## Computational Complexity

### Forward Pass:
```
Time: O(T × (hd² + hd×F + od×hd))
Space: O(T × hd)  (store all hidden states for backprop)
```

### Backward Pass:
```
Time: O(T × (hd² + hd×F + od×hd))
Space: O(hd)  (only need current gradients)
```

**Note:** LSTM is ~4× slower than vanilla RNN due to 4 gates

---

## Visualization of Gradient Flow

```
Vanilla RNN:
  L → h_T → h_{T-1} → ... → h_1 → h_0
      ↓      ↓              ↓
    Gradient decays exponentially (×W_hh each step)

LSTM:
  L → C_T → C_{T-1} → ... → C_1 → C_0
      ↓      ↓              ↓
    Gradient preserved (×f_t ≈ 1 each step)
    
The cell state C_t acts as a "gradient highway"!
```

---

**End of LSTM Formulae Document**
