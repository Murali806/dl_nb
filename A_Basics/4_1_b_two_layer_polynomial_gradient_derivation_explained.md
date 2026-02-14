# Gradient Derivation for Two-Layer Neural Network: Polynomial Approximation

This document provides a detailed mathematical derivation of the gradients used in backpropagation for the **two-layer neural network** that approximates polynomial functions (y = x²).

## 📐 Network Architecture

```
Input (x) → Hidden Layer 1 (1 neuron + tanh) → Hidden Layer 2 (1 neuron + tanh) → Output (y)
```

### Mathematical Notation

- **Input**: x (scalar)
- **Layer 1**: 
  - Weight: W₁, Bias: b₁
  - Linear: z₁ = W₁·x + b₁
  - Activation: h₁ = tanh(z₁)
- **Layer 2**: 
  - Weight: W₂, Bias: b₂
  - Linear: z₂ = W₂·h₁ + b₂
  - Activation: h₂ = tanh(z₂)
- **Output Layer**: 
  - Weight: W₃, Bias: b₃
  - Linear: z₃ = W₃·h₂ + b₃
  - Output: ŷ = z₃ (no activation)

---

## 🎯 Forward Propagation

### Step-by-Step Computation

1. **Layer 1 (Hidden Layer 1)**:
   ```
   z₁ = W₁·x + b₁
   h₁ = tanh(z₁)
   ```

2. **Layer 2 (Hidden Layer 2)**:
   ```
   z₂ = W₂·h₁ + b₂
   h₂ = tanh(z₂)
   ```

3. **Output Layer**:
   ```
   z₃ = W₃·h₂ + b₃
   ŷ = z₃
   ```

### Example with Numbers

Let's say:
- x = 2.0
- W₁ = 0.5, b₁ = 0.1
- W₂ = 0.8, b₂ = -0.2
- W₃ = 1.2, b₃ = 0.3

**Forward Pass**:
```
z₁ = 0.5 × 2.0 + 0.1 = 1.1
h₁ = tanh(1.1) ≈ 0.8005

z₂ = 0.8 × 0.8005 + (-0.2) = 0.4404
h₂ = tanh(0.4404) ≈ 0.4139

z₃ = 1.2 × 0.4139 + 0.3 = 0.7967
ŷ = 0.7967
```

If true value y = 4.0 (since x² = 4), then error = 0.7967 - 4.0 = -3.2033

---

## 📉 Loss Function

We use **Mean Squared Error (MSE)**:

```
L = (1/n) × Σᵢ (ŷᵢ - yᵢ)²
```

For a single sample:
```
L = (ŷ - y)²
```

### Derivative of Loss

```
∂L/∂ŷ = 2(ŷ - y)
```

For batch training with n samples:
```
∂L/∂ŷ = (2/n) × Σᵢ (ŷᵢ - yᵢ)
```

---

## 🔄 Backpropagation: Chain Rule Application

The key to backpropagation is the **chain rule** from calculus. We compute gradients layer by layer, moving backward from output to input.

### General Chain Rule

For a composite function f(g(x)):
```
df/dx = (df/dg) × (dg/dx)
```

---

## 🎓 Layer 3 Gradients (Output Layer)

### Goal: Compute ∂L/∂W₃ and ∂L/∂b₃

**Step 1**: Derivative of loss with respect to output
```
∂L/∂ŷ = 2(ŷ - y)
```

**Step 2**: Since ŷ = z₃ (no activation):
```
∂ŷ/∂z₃ = 1
```

**Step 3**: Chain rule gives us:
```
∂L/∂z₃ = ∂L/∂ŷ × ∂ŷ/∂z₃ = 2(ŷ - y) × 1 = 2(ŷ - y)
```

**Step 4**: Now compute weight gradient. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂W₃ = h₂
```

**Step 5**: Apply chain rule:
```
∂L/∂W₃ = ∂L/∂z₃ × ∂z₃/∂W₃ = 2(ŷ - y) × h₂
```

**Step 6**: Compute bias gradient. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂b₃ = 1
```

**Step 7**: Apply chain rule:
```
∂L/∂b₃ = ∂L/∂z₃ × ∂z₃/∂b₃ = 2(ŷ - y) × 1 = 2(ŷ - y)
```

### Summary for Layer 3:
```
∂L/∂W₃ = 2(ŷ - y) × h₂
∂L/∂b₃ = 2(ŷ - y)
```

### Numerical Example:
Using our example where ŷ = 0.7967, y = 4.0, h₂ = 0.4139:
```
∂L/∂W₃ = 2(0.7967 - 4.0) × 0.4139 = 2(-3.2033) × 0.4139 ≈ -2.652
∂L/∂b₃ = 2(0.7967 - 4.0) = -6.407
```

---

## 🎓 Layer 2 Gradients (Hidden Layer 2)

### Goal: Compute ∂L/∂W₂ and ∂L/∂b₂

**Step 1**: We already have ∂L/∂z₃ = 2(ŷ - y)

**Step 2**: Compute how z₃ depends on h₂. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂h₂ = W₃
```

**Step 3**: Chain rule to get gradient at h₂:
```
∂L/∂h₂ = ∂L/∂z₃ × ∂z₃/∂h₂ = 2(ŷ - y) × W₃
```

**Step 4**: Now we need to go through the activation. Since h₂ = tanh(z₂):
```
∂h₂/∂z₂ = tanh'(z₂) = 1 - tanh²(z₂) = 1 - h₂²
```

**Step 5**: Chain rule to get gradient at z₂:
```
∂L/∂z₂ = ∂L/∂h₂ × ∂h₂/∂z₂ = 2(ŷ - y) × W₃ × (1 - h₂²)
```

**Step 6**: Compute weight gradient. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂W₂ = h₁
```

**Step 7**: Apply chain rule:
```
∂L/∂W₂ = ∂L/∂z₂ × ∂z₂/∂W₂ = 2(ŷ - y) × W₃ × (1 - h₂²) × h₁
```

**Step 8**: Compute bias gradient. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂b₂ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₂ = ∂L/∂z₂ × ∂z₂/∂b₂ = 2(ŷ - y) × W₃ × (1 - h₂²)
```

### Summary for Layer 2:
```
∂L/∂W₂ = 2(ŷ - y) × W₃ × (1 - h₂²) × h₁
∂L/∂b₂ = 2(ŷ - y) × W₃ × (1 - h₂²)
```

### Numerical Example:
Using ŷ = 0.7967, y = 4.0, W₃ = 1.2, h₂ = 0.4139, h₁ = 0.8005:
```
1 - h₂² = 1 - 0.4139² ≈ 0.8287

∂L/∂W₂ = 2(-3.2033) × 1.2 × 0.8287 × 0.8005 ≈ -5.093
∂L/∂b₂ = 2(-3.2033) × 1.2 × 0.8287 ≈ -6.361
```

---

## 🎓 Layer 1 Gradients (Hidden Layer 1)

### Goal: Compute ∂L/∂W₁ and ∂L/∂b₁

**Step 1**: We already have ∂L/∂z₂ = 2(ŷ - y) × W₃ × (1 - h₂²)

**Step 2**: Compute how z₂ depends on h₁. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂h₁ = W₂
```

**Step 3**: Chain rule to get gradient at h₁:
```
∂L/∂h₁ = ∂L/∂z₂ × ∂z₂/∂h₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂
```

**Step 4**: Go through the activation. Since h₁ = tanh(z₁):
```
∂h₁/∂z₁ = tanh'(z₁) = 1 - tanh²(z₁) = 1 - h₁²
```

**Step 5**: Chain rule to get gradient at z₁:
```
∂L/∂z₁ = ∂L/∂h₁ × ∂h₁/∂z₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

**Step 6**: Compute weight gradient. Since z₁ = W₁·x + b₁:
```
∂z₁/∂W₁ = x
```

**Step 7**: Apply chain rule:
```
∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × x
```

**Step 8**: Compute bias gradient. Since z₁ = W₁·x + b₁:
```
∂z₁/∂b₁ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₁ = ∂L/∂z₁ × ∂z₁/∂b₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

### Summary for Layer 1:
```
∂L/∂W₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × x
∂L/∂b₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

### Numerical Example:
Using previous values plus W₂ = 0.8, h₁ = 0.8005, x = 2.0:
```
1 - h₁² = 1 - 0.8005² ≈ 0.3592

∂L/∂W₁ = 2(-3.2033) × 1.2 × 0.8287 × 0.8 × 0.3592 × 2.0 ≈ -3.656
∂L/∂b₁ = 2(-3.2033) × 1.2 × 0.8287 × 0.8 × 0.3592 ≈ -1.828
```

---

## 📊 Complete Gradient Summary

For a two-layer network approximating y = x²:

### Output Layer (Layer 3):
```
∂L/∂W₃ = 2(ŷ - y) × h₂
∂L/∂b₃ = 2(ŷ - y)
```

### Hidden Layer 2:
```
∂L/∂W₂ = 2(ŷ - y) × W₃ × (1 - h₂²) × h₁
∂L/∂b₂ = 2(ŷ - y) × W₃ × (1 - h₂²)
```

### Hidden Layer 1:
```
∂L/∂W₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × x
∂L/∂b₁ = 2(ŷ - y) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

---

## 🔄 Gradient Descent Update

Once we have all gradients, we update parameters:

```
W₃ ← W₃ - α × ∂L/∂W₃
b₃ ← b₃ - α × ∂L/∂b₃

W₂ ← W₂ - α × ∂L/∂W₂
b₂ ← b₂ - α × ∂L/∂b₂

W₁ ← W₁ - α × ∂L/∂W₁
b₁ ← b₁ - α × ∂L/∂b₁
```

Where α is the learning rate (e.g., 0.01).

### Numerical Example (α = 0.01):
```
W₃ ← 1.2 - 0.01 × (-2.652) = 1.2 + 0.02652 = 1.227
b₃ ← 0.3 - 0.01 × (-6.407) = 0.3 + 0.06407 = 0.364

W₂ ← 0.8 - 0.01 × (-5.093) = 0.8 + 0.05093 = 0.851
b₂ ← -0.2 - 0.01 × (-6.361) = -0.2 + 0.06361 = -0.136

W₁ ← 0.5 - 0.01 × (-3.656) = 0.5 + 0.03656 = 0.537
b₁ ← 0.1 - 0.01 × (-1.828) = 0.1 + 0.01828 = 0.118
```

---

## 🧮 Batch Training

For multiple samples (batch size n), we average the gradients:

```
∂L/∂W₃ = (2/n) × Σᵢ (ŷᵢ - yᵢ) × h₂ᵢ
∂L/∂b₃ = (2/n) × Σᵢ (ŷᵢ - yᵢ)

∂L/∂W₂ = (2/n) × Σᵢ (ŷᵢ - yᵢ) × W₃ × (1 - h₂ᵢ²) × h₁ᵢ
∂L/∂b₂ = (2/n) × Σᵢ (ŷᵢ - yᵢ) × W₃ × (1 - h₂ᵢ²)

∂L/∂W₁ = (2/n) × Σᵢ (ŷᵢ - yᵢ) × W₃ × (1 - h₂ᵢ²) × W₂ × (1 - h₁ᵢ²) × xᵢ
∂L/∂b₁ = (2/n) × Σᵢ (ŷᵢ - yᵢ) × W₃ × (1 - h₂ᵢ²) × W₂ × (1 - h₁ᵢ²)
```

---

## 🎯 Key Insights

### 1. **Chain Rule is Essential**
Each gradient is computed by multiplying derivatives along the path from loss to parameter.

### 2. **Gradient Flow**
Gradients flow backward through:
- Loss → Output → Hidden2 → Hidden1 → Input
- Each layer multiplies by its local gradient

### 3. **Activation Derivatives**
The tanh derivative (1 - tanh²(x)) appears in hidden layer gradients, enabling non-linear learning.

### 4. **Vanishing Gradients**
Notice how gradients for W₁ and b₁ involve products of many terms. If these terms are small (<1), gradients can vanish, making learning slow for early layers.

### 5. **Why Two Layers Work**
- Layer 1 transforms input non-linearly
- Layer 2 combines features non-linearly
- Together they can approximate quadratic functions

---

## 📐 Mathematical Properties

### Tanh Activation
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Properties:
- Range: (-1, 1)
- tanh(0) = 0
- tanh'(x) = 1 - tanh²(x)
- Maximum derivative: 1 (at x=0)
```

### Why Tanh for Polynomial Approximation?
1. **Non-linearity**: Essential for learning curves
2. **Zero-centered**: Helps with gradient flow
3. **Smooth derivative**: Enables stable learning
4. **Bounded output**: Prevents exploding activations

---

## 🔍 Verification

To verify your gradient implementation:

1. **Numerical Gradient Check**:
   ```
   ∂L/∂W ≈ [L(W + ε) - L(W - ε)] / (2ε)
   ```
   where ε is a small value (e.g., 1e-7)

2. **Compare with Analytical Gradient**:
   The difference should be < 1e-7

3. **Gradient Descent Test**:
   Loss should decrease over iterations

---

## 💡 Practical Tips

1. **Initialize weights carefully**: Use Xavier/He initialization
2. **Normalize inputs**: Helps with gradient stability
3. **Monitor gradients**: Watch for vanishing/exploding gradients
4. **Learning rate**: Start with 0.01, adjust based on loss curve
5. **Batch size**: Larger batches give more stable gradients

---

**This derivation shows how backpropagation computes gradients through multiple layers, enabling neural networks to learn complex non-linear functions like y = x²!**
