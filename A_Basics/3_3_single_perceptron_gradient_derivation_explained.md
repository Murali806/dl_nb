# Mathematical Derivation of Gradients for Single Perceptron

This document provides a step-by-step mathematical derivation of the backpropagation gradients used in our single perceptron implementation.

---

## ✅ Problem Setup

### Model (Forward Propagation)

```
┌─────────────────────────────────────┐
│                                     │
│         ŷᵢ = W·xᵢ + B              │
│                                     │
└─────────────────────────────────────┘
```

**where:**
- `xᵢ` = input feature for sample i (house size)
- `W` = weight (parameter to learn)
- `B` = bias (parameter to learn)
- `ŷᵢ` = predicted output for sample i (house price)

---

### Loss Function (Mean Squared Error)

```
┌─────────────────────────────────────┐
│                                     │
│      L = 1/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²   │
│                                     │
└─────────────────────────────────────┘
```

**where:**
- `yᵢ` = true label for sample i
- `ŷᵢ` = predicted label for sample i
- `n` = number of training samples

---

## 🎯 Goal: Find ∂L/∂W and ∂L/∂B

We need to find how the loss changes with respect to our parameters (W and B) so we can update them using gradient descent.

---

## 🔍 Derivation 1: Gradient with respect to Weight (∂L/∂W)

### Step 1: Write the Full Loss Function

```
L = 1/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²
```

### Step 2: Substitute Forward Propagation

Since `ŷᵢ = W·xᵢ + B`, we can write:

```
L = 1/n × Σᵢ₌₁ⁿ (W·xᵢ + B - yᵢ)²
```

### Step 3: Define Error Term

**Define:**

```
Aᵢ = ŷᵢ - yᵢ = W·xᵢ + B - yᵢ
```

**So:**

```
L = 1/n × Σᵢ₌₁ⁿ Aᵢ²
```

---

### 🔬 Step-by-Step Derivative

#### 1. Derivative of L with respect to Aᵢ

```
∂L/∂Aᵢ = 2/n · Aᵢ
```

**Why?** Using the power rule: d/dx(x²) = 2x, and the 1/n factor stays constant.

---

#### 2. Derivative of Aᵢ with respect to W

```
Aᵢ = W·xᵢ + B - yᵢ
```

```
∂Aᵢ/∂W = xᵢ
```

**Why?** The derivative of `W·xᵢ` with respect to W is `xᵢ` (treating xᵢ as constant), and derivatives of B and yᵢ with respect to W are 0.

---

#### 3. Apply Chain Rule for Each Sample

```
∂L/∂W = Σᵢ₌₁ⁿ (∂L/∂Aᵢ · ∂Aᵢ/∂W)
```

**Substitute:**

```
∂L/∂W = Σᵢ₌₁ⁿ (2/n · Aᵢ · xᵢ)
```

**Replace Aᵢ:**

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   ∂L/∂W = 2/n × Σᵢ₌₁ⁿ (W·xᵢ + B - yᵢ) · xᵢ   │
│                                                 │
└─────────────────────────────────────────────────┘
```

Or equivalently:

```
┌─────────────────────────────────────────────┐
│                                             │
│   ∂L/∂W = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) · xᵢ      │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🎨 Final Gradient for Weight

```
╔═════════════════════════════════════════════╗
║                                             ║
║   ∂L/∂W = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) · xᵢ      ║
║                                             ║
╚═════════════════════════════════════════════╝
```

**Intuition:** "How much does changing the weight affect the loss, weighted by the input values?"

**In code notation:**
```python
∂Loss/∂weight = (2/n) × Σ(y_pred - y_true) × input
```

---

## 🔍 Derivation 2: Gradient with respect to Bias (∂L/∂B)

### Step 1: Start with the Same Loss Function

```
L = 1/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)²
  = 1/n × Σᵢ₌₁ⁿ (W·xᵢ + B - yᵢ)²
```

### Step 2: Use the Same Error Term

**Define:**

```
Aᵢ = W·xᵢ + B - yᵢ
```

**So:**

```
L = 1/n × Σᵢ₌₁ⁿ Aᵢ²
```

---

### 🔬 Step-by-Step Derivative

#### 1. Derivative of L with respect to Aᵢ (same as before)

```
∂L/∂Aᵢ = 2/n · Aᵢ
```

---

#### 2. Derivative of Aᵢ with respect to B

```
Aᵢ = W·xᵢ + B - yᵢ
```

```
∂Aᵢ/∂B = 1
```

**Why?** The derivative of `W·xᵢ` with respect to B is 0 (doesn't contain B), the derivative of B with respect to B is 1, and the derivative of yᵢ with respect to B is 0.

---

#### 3. Apply Chain Rule for Each Sample

```
∂L/∂B = Σᵢ₌₁ⁿ (∂L/∂Aᵢ · ∂Aᵢ/∂B)
```

**Substitute:**

```
∂L/∂B = Σᵢ₌₁ⁿ (2/n · Aᵢ · 1)
```

**Replace Aᵢ:**

```
┌─────────────────────────────────────────┐
│                                         │
│   ∂L/∂B = 2/n × Σᵢ₌₁ⁿ (W·xᵢ + B - yᵢ) │
│                                         │
└─────────────────────────────────────────┘
```

Or equivalently:

```
┌─────────────────────────────────────┐
│                                     │
│   ∂L/∂B = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)    │
│                                     │
└─────────────────────────────────────┘
```

---

## 🎨 Final Gradient for Bias

```
╔═════════════════════════════════════╗
║                                     ║
║   ∂L/∂B = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)    ║
║                                     ║
╚═════════════════════════════════════╝
```

**Intuition:** "How much does changing the bias affect the loss, averaged across all samples?"

**In code notation:**
```python
∂Loss/∂bias = (2/n) × Σ(y_pred - y_true)
```

---

## 🔄 Gradient Descent Update Rules

Once we have the gradients, we update the parameters:

```
┌─────────────────────────────────────┐
│                                     │
│   W_new = W_old - α × (∂L/∂W)      │
│                                     │
│   B_new = B_old - α × (∂L/∂B)      │
│                                     │
└─────────────────────────────────────┘
```

**where:**
- `α` = learning rate (e.g., 0.01)
- We move in the **opposite direction** of the gradient to minimize loss

---

## 📝 Concrete Numerical Example

Let's work through a complete example with 3 data points to see the gradients in action.

### Given Data:

```
Sample 1: x₁ = 2, y₁ = 5
Sample 2: x₂ = 3, y₂ = 7
Sample 3: x₃ = 4, y₃ = 9
```

### Current Parameters:

```
W = 1.5
B = 1.0
```

---

### Step 1: Forward Propagation

```
ŷ₁ = W·x₁ + B = 1.5 × 2 + 1.0 = 4.0
ŷ₂ = W·x₂ + B = 1.5 × 3 + 1.0 = 5.5
ŷ₃ = W·x₃ + B = 1.5 × 4 + 1.0 = 7.0
```

---

### Step 2: Compute Errors

```
A₁ = ŷ₁ - y₁ = 4.0 - 5.0 = -1.0
A₂ = ŷ₂ - y₂ = 5.5 - 7.0 = -1.5
A₃ = ŷ₃ - y₃ = 7.0 - 9.0 = -2.0
```

---

### Step 3: Compute Loss

```
L = 1/3 × [(−1.0)² + (−1.5)² + (−2.0)²]
  = 1/3 × [1.0 + 2.25 + 4.0]
  = 1/3 × 7.25
  = 2.417
```

---

### Step 4: Compute Weight Gradient

```
∂L/∂W = 2/3 × [A₁·x₁ + A₂·x₂ + A₃·x₃]
      = 2/3 × [(−1.0)×2 + (−1.5)×3 + (−2.0)×4]
      = 2/3 × [−2.0 − 4.5 − 8.0]
      = 2/3 × (−14.5)
      = −9.667
```

---

### Step 5: Compute Bias Gradient

```
∂L/∂B = 2/3 × [A₁ + A₂ + A₃]
      = 2/3 × [−1.0 − 1.5 − 2.0]
      = 2/3 × (−4.5)
      = −3.0
```

---

### Step 6: Update Parameters (with α = 0.01)

```
W_new = W − α × (∂L/∂W)
      = 1.5 − 0.01 × (−9.667)
      = 1.5 + 0.097
      = 1.597

B_new = B − α × (∂L/∂B)
      = 1.0 − 0.01 × (−3.0)
      = 1.0 + 0.03
      = 1.03
```

**Notice:** Both parameters **increased** because the gradients were **negative**, meaning we need to move in the positive direction to reduce the loss!

---

## 🧮 Why the Factor of 2?

You might wonder why we have `(2/n)` instead of just `(1/n)`.

### The Mathematical Reason:

When we take the derivative of the squared term:

```
┌─────────────────────────────────────────┐
│                                         │
│   d/dx[(ŷ - y)²] = 2(ŷ - y) · 1       │
│                  = 2(ŷ - y)            │
│                                         │
└─────────────────────────────────────────┘
```

The factor of 2 comes from the **power rule** in calculus:

```
d/dx(x²) = 2x
```

### Does it Matter in Practice?

**No!** The factor of 2 doesn't significantly affect training because:

1. It's absorbed into the learning rate
2. If we use `(2/n)`, we might use learning rate `α = 0.01`
3. If we use `(1/n)`, we might use learning rate `α = 0.02`
4. The effect is the same!

Many implementations omit the factor of 2 for simplicity, but we include it here for **mathematical correctness**.

---

## 🎓 Key Insights

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  1. Chain Rule is Fundamental                             │
│     → Backpropagation = repeated chain rule application   │
│                                                            │
│  2. Error Signal: (ŷᵢ - yᵢ)                               │
│     → Represents how wrong our prediction is              │
│                                                            │
│  3. Input Scaling for Weight                              │
│     → Multiply by xᵢ (input contribution)                 │
│                                                            │
│  4. Bias Simplicity                                       │
│     → Doesn't depend on input (affects all equally)       │
│                                                            │
│  5. Averaging: (1/n) factor                               │
│     → Ensures gradients don't grow with dataset size      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 📚 Summary

### Weight Gradient Formula:

```
╔═════════════════════════════════════════════╗
║                                             ║
║   ∂L/∂W = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) · xᵢ      ║
║                                             ║
╚═════════════════════════════════════════════╝
```

**Intuition:** "How much does changing the weight affect the loss, weighted by the input values?"

---

### Bias Gradient Formula:

```
╔═════════════════════════════════════╗
║                                     ║
║   ∂L/∂B = 2/n × Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)    ║
║                                     ║
╚═════════════════════════════════════╝
```

**Intuition:** "How much does changing the bias affect the loss, averaged across all samples?"

---

### Update Rules:

```
╔═════════════════════════════════════╗
║                                     ║
║   W ← W - α × (∂L/∂W)              ║
║                                     ║
║   B ← B - α × (∂L/∂B)              ║
║                                     ║
╚═════════════════════════════════════╝
```

**Intuition:** "Move parameters in the direction that reduces the loss, scaled by the learning rate."

---

## 🔗 Connection to the Code

In our `SinglePerceptron` class, the `backward()` method implements these formulas:

```python
def backward(self, X, y_true, y_pred):
    n = tf.cast(tf.shape(X)[0], tf.float32)
    error = y_pred - y_true
    
    # Weight gradient: (2/n) × Σ(error × input)
    weight_grad = (2.0 / n) * tf.reduce_sum(error * X)
    
    # Bias gradient: (2/n) × Σ(error)
    bias_grad = (2.0 / n) * tf.reduce_sum(error)
    
    return weight_grad, bias_grad
```

**This is the mathematical derivation brought to life in code!** 🎉

---

## 🎯 Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    GRADIENT DESCENT FLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Forward Pass:  ŷᵢ = W·xᵢ + B                          │
│                                                             │
│  2. Compute Loss:  L = 1/n × Σ(ŷᵢ - yᵢ)²                  │
│                                                             │
│  3. Compute Gradients:                                      │
│     • ∂L/∂W = 2/n × Σ(ŷᵢ - yᵢ)·xᵢ                         │
│     • ∂L/∂B = 2/n × Σ(ŷᵢ - yᵢ)                            │
│                                                             │
│  4. Update Parameters:                                      │
│     • W ← W - α·(∂L/∂W)                                    │
│     • B ← B - α·(∂L/∂B)                                    │
│                                                             │
│  5. Repeat until convergence                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This is how a single perceptron learns through gradient descent! 🚀
