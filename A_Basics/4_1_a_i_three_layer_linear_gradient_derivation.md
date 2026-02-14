# Mathematical Derivation of Gradients for Three-Layer Linear Neural Network

This document provides a step-by-step mathematical derivation of the backpropagation gradients for a **three-layer purely linear neural network** (no activation functions).

---

## ✅ Problem Setup

### Network Architecture (All Linear - No Activations)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input (x) → Layer 1 → Layer 2 → Layer 3 → Output (ŷ)         │
│                                                                 │
│  x → [W₁·x + b₁] → [W₂·h₁ + b₂] → [W₃·h₂ + b₃] → ŷ           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mathematical Notation

**Layer 1 (Hidden Layer 1):**
- Weight: W₁
- Bias: b₁
- Linear output: z₁ = W₁·x + b₁
- Output: h₁ = z₁ (no activation, just pass through)

**Layer 2 (Hidden Layer 2):**
- Weight: W₂
- Bias: b₂
- Linear output: z₂ = W₂·h₁ + b₂
- Output: h₂ = z₂ (no activation, just pass through)

**Layer 3 (Output Layer):**
- Weight: W₃
- Bias: b₃
- Linear output: z₃ = W₃·h₂ + b₃
- Output: ŷ = z₃ (no activation)

---

## 🎯 Forward Propagation

### Step-by-Step Computation

```
1. Layer 1:  z₁ = W₁·x + b₁
             h₁ = z₁

2. Layer 2:  z₂ = W₂·h₁ + b₂
             h₂ = z₂

3. Layer 3:  z₃ = W₃·h₂ + b₃
             ŷ = z₃
```

### Simplified View (Since h₁ = z₁ and h₂ = z₂):

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  h₁ = W₁·x + b₁                                │
│                                                 │
│  h₂ = W₂·h₁ + b₂ = W₂·(W₁·x + b₁) + b₂        │
│                                                 │
│  ŷ = W₃·h₂ + b₃                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 📉 Loss Function

We use **Mean Squared Error (MSE)**:

```
┌─────────────────────────────────────┐
│                                     │
│   L = (1/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²    │
│                                     │
└─────────────────────────────────────┘
```

For a single sample:
```
L = (ŷ - y)²
```

### Derivative of Loss with Respect to Output

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂ŷ = 2(ŷ - y)                 │
│                                     │
└─────────────────────────────────────┘
```

For batch training with n samples:
```
∂L/∂ŷ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)
```

## 🔄 Backpropagation: The Chain Rule

The key to backpropagation is the **chain rule** from calculus:

```
┌─────────────────────────────────────┐
│                                     │
│   For f(g(x)):                     │
│                                     │
│   df/dx = (df/dg) × (dg/dx)        │
│                                     │
└─────────────────────────────────────┘
```

We compute gradients **layer by layer**, moving **backward** from output to input.

---

## 🎓 Layer 3 Gradients (Output Layer)

### Goal: Compute ∂L/∂W₃ and ∂L/∂b₃

**Step 1**: We already have the derivative of loss with respect to output:
```
∂L/∂ŷ = 2(ŷ - y)
```

**Step 2**: Since ŷ = z₃ (no activation function):
```
∂ŷ/∂z₃ = 1
```

**Step 3**: By chain rule:
```
∂L/∂z₃ = ∂L/∂ŷ × ∂ŷ/∂z₃ = 2(ŷ - y) × 1 = 2(ŷ - y)
```

---

### Computing ∂L/∂W₃

**Step 4**: Since z₃ = W₃·h₂ + b₃, the derivative with respect to W₃ is:
```
∂z₃/∂W₃ = h₂
```

**Why?** Treating b₃ as a constant, the derivative of W₃·h₂ with respect to W₃ is h₂.

**Step 5**: Apply chain rule:
```
∂L/∂W₃ = ∂L/∂z₃ × ∂z₃/∂W₃
```

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂W₃ = 2(ŷ - y) × h₂           │
│                                     │
└─────────────────────────────────────┘
```


---

### Computing ∂L/∂b₃

**Step 6**: Since z₃ = W₃·h₂ + b₃, the derivative with respect to b₃ is:
```
∂z₃/∂b₃ = 1
```

**Step 7**: Apply chain rule:
```
∂L/∂b₃ = ∂L/∂z₃ × ∂z₃/∂b₃
```

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂b₃ = 2(ŷ - y) × 1            │
│          = 2(ŷ - y)                │
│                                     │
└─────────────────────────────────────┘
```

---

### Summary for Layer 3:

```
╔═════════════════════════════════════╗
║                                     ║
║   ∂L/∂W₃ = 2(ŷ - y) × h₂           ║
║                                     ║
║   ∂L/∂b₃ = 2(ŷ - y)                ║
║                                     ║
╚═════════════════════════════════════╝
```

**Intuition:**
- Weight gradient depends on the previous layer's output (h₂)
- Bias gradient is just the error signal

---

## 🎓 Layer 2 Gradients (Hidden Layer 2)

### Goal: Compute ∂L/∂W₂ and ∂L/∂b₂

**Step 1**: We already have:
```
∂L/∂z₃ = 2(ŷ - y)
```

**Step 2**: We need to propagate this gradient back to h₂. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂h₂ = W₃
```

**Step 3**: By chain rule:
```
∂L/∂h₂ = ∂L/∂z₃ × ∂z₃/∂h₂ = 2(ŷ - y) × W₃
```

**Step 4**: Since h₂ = z₂ (no activation):
```
∂h₂/∂z₂ = 1
```

**Step 5**: By chain rule:
```
∂L/∂z₂ = ∂L/∂h₂ × ∂h₂/∂z₂ = 2(ŷ - y) × W₃ × 1 = 2(ŷ - y) × W₃
```

---

### Computing ∂L/∂W₂

**Step 6**: Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂W₂ = h₁
```

**Step 7**: Apply chain rule:
```
∂L/∂W₂ = ∂L/∂z₂ × ∂z₂/∂W₂
```

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂W₂ = 2(ŷ - y) × W₃ × h₁     │
│                                     │
└─────────────────────────────────────┘
```

**Numerical Example:**
```
∂L/∂W₂ = 2(1.116 - 4.0) × 1.2 × 1.1
       = -5.768 × 1.2 × 1.1
       = -7.614
```

---

### Computing ∂L/∂b₂

**Step 8**: Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂b₂ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₂ = ∂L/∂z₂ × ∂z₂/∂b₂
```

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂b₂ = 2(ŷ - y) × W₃ × 1      │
│          = 2(ŷ - y) × W₃           │
│                                     │
└─────────────────────────────────────┘
```

**Numerical Example:**
```
∂L/∂b₂ = 2(1.116 - 4.0) × 1.2
       = -5.768 × 1.2
       = -6.922
```

---

### Summary for Layer 2:

```
╔═════════════════════════════════════╗
║                                     ║
║   ∂L/∂W₂ = 2(ŷ - y) × W₃ × h₁     ║
║                                     ║
║   ∂L/∂b₂ = 2(ŷ - y) × W₃           ║
║                                     ║
╚═════════════════════════════════════╝
```

**Intuition:**
- Gradient flows backward through W₃
- Weight gradient also depends on h₁ (previous layer output)

---

## 🎓 Layer 1 Gradients (Hidden Layer 1)

### Goal: Compute ∂L/∂W₁ and ∂L/∂b₁

**Step 1**: We already have:
```
∂L/∂z₂ = 2(ŷ - y) × W₃
```

**Step 2**: We need to propagate this gradient back to h₁. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂h₁ = W₂
```

**Step 3**: By chain rule:
```
∂L/∂h₁ = ∂L/∂z₂ × ∂z₂/∂h₁ = 2(ŷ - y) × W₃ × W₂
```

**Step 4**: Since h₁ = z₁ (no activation):
```
∂h₁/∂z₁ = 1
```

**Step 5**: By chain rule:
```
∂L/∂z₁ = ∂L/∂h₁ × ∂h₁/∂z₁ = 2(ŷ - y) × W₃ × W₂ × 1 = 2(ŷ - y) × W₃ × W₂
```

---

### Computing ∂L/∂W₁

**Step 6**: Since z₁ = W₁·x + b₁:
```
∂z₁/∂W₁ = x
```

**Step 7**: Apply chain rule:
```
∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁
```

```
┌─────────────────────────────────────────┐
│                                         │
│   ∂L/∂W₁ = 2(ŷ - y) × W₃ × W₂ × x     │
│                                         │
└─────────────────────────────────────────┘
```

**Numerical Example:**
```
∂L/∂W₁ = 2(1.116 - 4.0) × 1.2 × 0.8 × 2.0
       = -5.768 × 1.2 × 0.8 × 2.0
       = -11.078
```

---

### Computing ∂L/∂b₁

**Step 8**: Since z₁ = W₁·x + b₁:
```
∂z₁/∂b₁ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₁ = ∂L/∂z₁ × ∂z₁/∂b₁
```

```
┌─────────────────────────────────────────┐
│                                         │
│   ∂L/∂b₁ = 2(ŷ - y) × W₃ × W₂ × 1     │
│          = 2(ŷ - y) × W₃ × W₂         │
│                                         │
└─────────────────────────────────────────┘
```

**Numerical Example:**
```
∂L/∂b₁ = 2(1.116 - 4.0) × 1.2 × 0.8
       = -5.768 × 1.2 × 0.8
       = -5.539
```

---

### Summary for Layer 1:

```
╔═════════════════════════════════════════╗
║                                         ║
║   ∂L/∂W₁ = 2(ŷ - y) × W₃ × W₂ × x     ║
║                                         ║
║   ∂L/∂b₁ = 2(ŷ - y) × W₃ × W₂         ║
║                                         ║
╚═════════════════════════════════════════╝
```

**Intuition:**
- Gradient flows backward through both W₃ and W₂
- Weight gradient also depends on input x
- Notice the pattern: each layer multiplies by the next layer's weight

---

## 📊 Complete Gradient Summary

For a three-layer **linear** neural network:

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  OUTPUT LAYER (Layer 3):                                 ║
║  ∂L/∂W₃ = 2(ŷ - y) × h₂                                 ║
║  ∂L/∂b₃ = 2(ŷ - y)                                      ║
║                                                           ║
║  HIDDEN LAYER 2:                                         ║
║  ∂L/∂W₂ = 2(ŷ - y) × W₃ × h₁                           ║
║  ∂L/∂b₂ = 2(ŷ - y) × W₃                                ║
║                                                           ║
║  HIDDEN LAYER 1:                                         ║
║  ∂L/∂W₁ = 2(ŷ - y) × W₃ × W₂ × x                       ║
║  ∂L/∂b₁ = 2(ŷ - y) × W₃ × W₂                           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

### Pattern Recognition:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Notice the pattern:                                    │
│                                                         │
│  • Each layer's gradient includes 2(ŷ - y)             │
│  • Gradients accumulate weights as we go backward      │
│  • Weight gradients multiply by previous layer output  │
│  • Bias gradients don't depend on previous output     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Gradient Descent Update Rules

Once we have all gradients, we update parameters using:

```
┌─────────────────────────────────────┐
│                                     │
│   W₃ ← W₃ - α × (∂L/∂W₃)          │
│   b₃ ← b₃ - α × (∂L/∂b₃)          │
│                                     │
│   W₂ ← W₂ - α × (∂L/∂W₂)          │
│   b₂ ← b₂ - α × (∂L/∂b₂)          │
│                                     │
│   W₁ ← W₁ - α × (∂L/∂W₁)          │
│   b₁ ← b₁ - α × (∂L/∂b₁)          │
│                                     │
└─────────────────────────────────────┘
```

Where α is the **learning rate** (e.g., 0.01).

### Numerical Example (α = 0.01):

Using our computed gradients:

```
W₃ ← 1.2 - 0.01 × (-3.922) = 1.2 + 0.039 = 1.239
b₃ ← 0.3 - 0.01 × (-5.768) = 0.3 + 0.058 = 0.358

W₂ ← 0.8 - 0.01 × (-7.614) = 0.8 + 0.076 = 0.876
b₂ ← -0.2 - 0.01 × (-6.922) = -0.2 + 0.069 = -0.131

W₁ ← 0.5 - 0.01 × (-11.078) = 0.5 + 0.111 = 0.611
b₁ ← 0.1 - 0.01 × (-5.539) = 0.1 + 0.055 = 0.155
```

**Notice:** All parameters **increased** because all gradients were **negative**, meaning we need to move in the positive direction to reduce the loss!

---

## 📝 Complete Numerical Example

Let's verify our gradients with a full forward and backward pass.

### Given:
- Input: x = 2.0
- True output: y = 4.0
- Initial parameters:
  - W₁ = 0.5, b₁ = 0.1
  - W₂ = 0.8, b₂ = -0.2
  - W₃ = 1.2, b₃ = 0.3

### Forward Pass:

```
h₁ = W₁·x + b₁ = 0.5 × 2.0 + 0.1 = 1.1

h₂ = W₂·h₁ + b₂ = 0.8 × 1.1 + (-0.2) = 0.68

ŷ = W₃·h₂ + b₃ = 1.2 × 0.68 + 0.3 = 1.116
```

### Loss:

```
L = (ŷ - y)² = (1.116 - 4.0)² = (-2.884)² = 8.318
```

### Backward Pass:

```
Error signal: 2(ŷ - y) = 2(1.116 - 4.0) = -5.768

Layer 3:
∂L/∂W₃ = -5.768 × 0.68 = -3.922
∂L/∂b₃ = -5.768

Layer 2:
∂L/∂W₂ = -5.768 × 1.2 × 1.1 = -7.614
∂L/∂b₂ = -5.768 × 1.2 = -6.922

Layer 1:
∂L/∂W₁ = -5.768 × 1.2 × 0.8 × 2.0 = -11.078
∂L/∂b₁ = -5.768 × 1.2 × 0.8 = -5.539
```

### Update (α = 0.01):

```
W₃ = 1.2 - 0.01(-3.922) = 1.239
b₃ = 0.3 - 0.01(-5.768) = 0.358

W₂ = 0.8 - 0.01(-7.614) = 0.876
b₂ = -0.2 - 0.01(-6.922) = -0.131

W₁ = 0.5 - 0.01(-11.078) = 0.611
b₁ = 0.1 - 0.01(-5.539) = 0.155
```

### Verify: Forward Pass with Updated Parameters

```
h₁ = 0.611 × 2.0 + 0.155 = 1.377

h₂ = 0.876 × 1.377 + (-0.131) = 1.075

ŷ = 1.239 × 1.075 + 0.358 = 1.690
```

**New Loss:**
```
L = (1.690 - 4.0)² = (-2.310)² = 5.336
```

**Loss decreased from 8.318 to 5.336!** ✅ Gradient descent is working!

---

## 🧮 Batch Training

For multiple samples (batch size n), we average the gradients:

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  ∂L/∂W₃ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × h₂ᵢ                ║
║  ∂L/∂b₃ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)                       ║
║                                                           ║
║  ∂L/∂W₂ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × W₃ × h₁ᵢ           ║
║  ∂L/∂b₂ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × W₃                  ║
║                                                           ║
║  ∂L/∂W₁ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × W₃ × W₂ × xᵢ       ║
║  ∂L/∂b₁ = (2/n) × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) × W₃ × W₂            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 🎯 Key Insights

### 1. **Linear Networks are Simple**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  No activation functions means:                         │
│  • No activation derivatives (no tanh', sigmoid', etc.) │
│  • Cleaner chain rule application                      │
│  • Easier to understand backpropagation               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2. **Gradient Flow Pattern**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Gradients flow backward by multiplying weights:        │
│                                                         │
│  Layer 3: Just the error signal                        │
│  Layer 2: Error × W₃                                   │
│  Layer 1: Error × W₃ × W₂                             │
│                                                         │
│  Each layer adds one more weight to the product!       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3. **Weight vs Bias Gradients**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Weight gradients: Multiply by previous layer output   │
│  Bias gradients: Don't depend on previous output      │
│                                                         │
│  This is because:                                      │
│  • ∂(W·h)/∂W = h                                      │
│  • ∂(b)/∂b = 1                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4. **Why Linear Networks are Limited**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  A 3-layer linear network can be collapsed:            │
│                                                         │
│  ŷ = W₃(W₂(W₁·x + b₁) + b₂) + b₃                     │
│    = (W₃·W₂·W₁)·x + (W₃·W₂·b₁ + W₃·b₂ + b₃)         │
│    = W_effective·x + b_effective                       │
│                                                         │
│  It's equivalent to a single linear layer!             │
│  Can only learn linear relationships!                  │
│                                                         │
│  This is why we need activation functions in practice! │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5. **Vanishing/Exploding Gradients**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Notice how gradients multiply weights:                │
│                                                         │
│  If |W₃| < 1 and |W₂| < 1:                            │
│  → Gradients get smaller (vanishing)                   │
│                                                         │
│  If |W₃| > 1 and |W₂| > 1:                            │
│  → Gradients get larger (exploding)                    │
│                                                         │
│  This is why weight initialization matters!            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Verification: Numerical Gradient Check

To verify your gradient implementation, use **numerical gradients**:

```
∂L/∂W ≈ [L(W + ε) - L(W - ε)] / (2ε)
```

Where ε is a small value (e.g., 1e-7).

### Example for W₃:

```python
epsilon = 1e-7

# Compute loss with W₃ + ε
W3_plus = W3 + epsilon
y_pred_plus = forward_pass(x, W1, b1, W2, b2, W3_plus, b3)
loss_plus = (y_pred_plus - y)**2

# Compute loss with W₃ - ε
W3_minus = W3 - epsilon
y_pred_minus = forward_pass(x, W1, b1, W2, b2, W3_minus, b3)
loss_minus = (y_pred_minus - y)**2

# Numerical gradient
numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

# Compare with analytical gradient
analytical_grad = 2 * (y_pred - y) * h2

# Should be very close (difference < 1e-7)
print(f"Numerical: {numerical_grad}")
print(f"Analytical: {analytical_grad}")
print(f"Difference: {abs(numerical_grad - analytical_grad)}")
```

---

## 💡 Practical Tips

### 1. **Weight Initialization**
```
• Use small random values (e.g., uniform(-0.5, 0.5))
• Or use Xavier initialization: W ~ N(0, 1/√n_in)
• Avoid initializing all weights to zero!
```

### 2. **Learning Rate Selection**
```
• Start with α = 0.01
• If loss oscillates: decrease α
• If loss decreases too slowly: increase α
• Consider learning rate schedules
```

### 3. **Monitoring Training**
```
• Plot loss vs iterations
• Loss should decrease monotonically
• If loss increases: learning rate too high
• If loss plateaus: might need activation functions!
```

### 4. **Debugging Gradients**
```
• Use numerical gradient checking
• Print gradient magnitudes
• Check for NaN or Inf values
• Verify gradients sum correctly for batches
```

### 5. **When to Use Linear Networks**
```
• Linear regression problems
• As a baseline model
• For understanding backpropagation
• NOT for complex non-linear problems!
```

---

## 🎓 Summary

### The Complete Backpropagation Algorithm:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. FORWARD PASS:                                      │
│     • Compute h₁ = W₁·x + b₁                          │
│     • Compute h₂ = W₂·h₁ + b₂                         │
│     • Compute ŷ = W₃·h₂ + b₃                          │
│                                                         │
│  2. COMPUTE LOSS:                                      │
│     • L = (ŷ - y)²                                    │
│                                                         │
│  3. BACKWARD PASS (Compute Gradients):                  |
|     Generalized = ∂L/∂Wn = 2 * error * Wn+1 * hn-1      |
|                   ∂L/∂Wn = 2 * error * Wn+1             |
│     • ∂L/∂W₃ = 2(ŷ - y) × h₂                        │
│     • ∂L/∂b₃ = 2(ŷ - y)                              │
│     • ∂L/∂W₂ = 2(ŷ - y) × W₃ × h₁                   │
│     • ∂L/∂b₂ = 2(ŷ - y) × W₃                         │
│     • ∂L/∂W₁ = 2(ŷ - y) × W₃ × W₂ × x               │
│     • ∂L/∂b₁ = 2(ŷ - y) × W₃ × W₂                   │
│                                                        │
│  4. UPDATE PARAMETERS:                                 │
│     • W₃ ← W₃ - α × (∂L/∂W₃)                         │
│     • b₃ ← b₃ - α × (∂L/∂b₃)                         │
│     • W₂ ← W₂ - α × (∂L/∂W₂)                         │
│     • b₂ ← b₂ - α × (∂L/∂b₂)                         │
│     • W₁ ← W₁ - α × (∂L/∂W₁)                         │
│     • b₁ ← b₁ - α × (∂L/∂b₁)                         │
│                                                         │
│  5. REPEAT until convergence                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔗 Connection to Code

In Python/NumPy, the backward pass would look like:

```python
def backward(x, y, y_pred, h1, h2, W3, W2, learning_rate=0.01):
    """
    Compute gradients and update parameters for 3-layer linear network.
    """
    n = len(x)  # batch size
    
    # Error signal
    error = y_pred - y
    
    # Layer 3 gradients
    dW3 = (2.0 / n) * np.sum(error * h2)
    db3 = (2.0 / n) * np.sum(error)
    
    # Layer 2 gradients
    dW2 = (2.0 / n) * np.sum(error * W3 * h1)
    db2 = (2.0 / n) * np.sum(error * W3)
    
    # Layer 1 gradients
    dW1 = (2.0 / n) * np.sum(error * W3 * W2 * x)
    db1 = (2.0 / n) * np.sum(error * W3 * W2)
    
    # Update parameters
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    return W3, b3, W2, b2, W1, b1
```

---

**This is how backpropagation works in a three-layer linear neural network!** 🎉

The same principles extend to:
- Networks with more layers (just keep multiplying weights backward)
- Networks with activation functions (add activation derivatives)
- Different loss functions (change ∂L/∂ŷ)
- Different architectures (CNNs, RNNs, etc.)

**Understanding this linear case is the foundation for understanding all of deep learning!** 🚀
