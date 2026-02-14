# Gradient Derivation for Two-Layer Neural Network: Drug Dosage Response

This document provides a detailed mathematical derivation of the gradients used in backpropagation for the **two-layer neural network** that learns optimal drug dosage relationships (inverted U-shaped curve).

## 📐 Network Architecture

```
Input (dosage) → Hidden Layer 1 (1 neuron + tanh) → Hidden Layer 2 (1 neuron + tanh) → Output (effectiveness)
```

### Mathematical Notation

- **Input**: d (dosage in mg)
- **Layer 1**: 
  - Weight: W₁, Bias: b₁
  - Linear: z₁ = W₁·d + b₁
  - Activation: h₁ = tanh(z₁)
- **Layer 2**: 
  - Weight: W₂, Bias: b₂
  - Linear: z₂ = W₂·h₁ + b₂
  - Activation: h₂ = tanh(z₂)
- **Output Layer**: 
  - Weight: W₃, Bias: b₃
  - Linear: z₃ = W₃·h₂ + b₃
  - Output: ê = z₃ (effectiveness prediction, no activation)

### True Dose-Response Relationship

The true relationship follows an inverted parabola:
```
Effectiveness = MAX - (dosage - optimal)² / scale_factor
```

For our example:
- Optimal dosage: 50 mg
- Maximum effectiveness: 100%
- Scale factor: 25 (chosen so effectiveness reaches ~0 at boundaries)

---

## 🎯 Forward Propagation

### Step-by-Step Computation

1. **Layer 1 (Hidden Layer 1)**:
   ```
   z₁ = W₁·d + b₁
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
   ê = z₃  (predicted effectiveness)
   ```

### Example with Numbers

Let's say:
- d = 30 mg (underdose region)
- W₁ = 0.6, b₁ = -0.3
- W₂ = 0.9, b₂ = 0.1
- W₃ = 1.5, b₃ = 0.5

**Forward Pass**:
```
z₁ = 0.6 × 30 + (-0.3) = 17.7
h₁ = tanh(17.7) ≈ 1.0 (saturated)

z₂ = 0.9 × 1.0 + 0.1 = 1.0
h₂ = tanh(1.0) ≈ 0.7616

z₃ = 1.5 × 0.7616 + 0.5 = 1.6424
ê = 1.6424
```

**True effectiveness** at 30mg:
```
e = 100 - (30 - 50)² / 25 = 100 - 400/25 = 100 - 16 = 84%
```

After normalization (assuming mean=50, std=20):
- Normalized: (84 - 50) / 20 = 1.7

Error = 1.6424 - 1.7 = -0.0576

---

## 📉 Loss Function

We use **Mean Squared Error (MSE)**:

```
L = (1/n) × Σᵢ (êᵢ - eᵢ)²
```

For a single sample:
```
L = (ê - e)²
```

Where:
- ê = predicted effectiveness
- e = true effectiveness

### Derivative of Loss

```
∂L/∂ê = 2(ê - e)
```

For batch training with n samples:
```
∂L/∂ê = (2/n) × Σᵢ (êᵢ - eᵢ)
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
∂L/∂ê = 2(ê - e)
```

**Step 2**: Since ê = z₃ (no activation):
```
∂ê/∂z₃ = 1
```

**Step 3**: Chain rule gives us:
```
∂L/∂z₃ = ∂L/∂ê × ∂ê/∂z₃ = 2(ê - e) × 1 = 2(ê - e)
```

**Step 4**: Now compute weight gradient. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂W₃ = h₂
```

**Step 5**: Apply chain rule:
```
∂L/∂W₃ = ∂L/∂z₃ × ∂z₃/∂W₃ = 2(ê - e) × h₂
```

**Step 6**: Compute bias gradient. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂b₃ = 1
```

**Step 7**: Apply chain rule:
```
∂L/∂b₃ = ∂L/∂z₃ × ∂z₃/∂b₃ = 2(ê - e) × 1 = 2(ê - e)
```

### Summary for Layer 3:
```
∂L/∂W₃ = 2(ê - e) × h₂
∂L/∂b₃ = 2(ê - e)
```

### Numerical Example:
Using our example where ê = 1.6424, e = 1.7, h₂ = 0.7616:
```
∂L/∂W₃ = 2(1.6424 - 1.7) × 0.7616 = 2(-0.0576) × 0.7616 ≈ -0.0877
∂L/∂b₃ = 2(1.6424 - 1.7) = -0.1152
```

---

## 🎓 Layer 2 Gradients (Hidden Layer 2)

### Goal: Compute ∂L/∂W₂ and ∂L/∂b₂

**Step 1**: We already have ∂L/∂z₃ = 2(ê - e)

**Step 2**: Compute how z₃ depends on h₂. Since z₃ = W₃·h₂ + b₃:
```
∂z₃/∂h₂ = W₃
```

**Step 3**: Chain rule to get gradient at h₂:
```
∂L/∂h₂ = ∂L/∂z₃ × ∂z₃/∂h₂ = 2(ê - e) × W₃
```

**Step 4**: Now we need to go through the activation. Since h₂ = tanh(z₂):
```
∂h₂/∂z₂ = tanh'(z₂) = 1 - tanh²(z₂) = 1 - h₂²
```

**Step 5**: Chain rule to get gradient at z₂:
```
∂L/∂z₂ = ∂L/∂h₂ × ∂h₂/∂z₂ = 2(ê - e) × W₃ × (1 - h₂²)
```

**Step 6**: Compute weight gradient. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂W₂ = h₁
```

**Step 7**: Apply chain rule:
```
∂L/∂W₂ = ∂L/∂z₂ × ∂z₂/∂W₂ = 2(ê - e) × W₃ × (1 - h₂²) × h₁
```

**Step 8**: Compute bias gradient. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂b₂ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₂ = ∂L/∂z₂ × ∂z₂/∂b₂ = 2(ê - e) × W₃ × (1 - h₂²)
```

### Summary for Layer 2:
```
∂L/∂W₂ = 2(ê - e) × W₃ × (1 - h₂²) × h₁
∂L/∂b₂ = 2(ê - e) × W₃ × (1 - h₂²)
```

### Numerical Example:
Using ê = 1.6424, e = 1.7, W₃ = 1.5, h₂ = 0.7616, h₁ = 1.0:
```
1 - h₂² = 1 - 0.7616² ≈ 0.4199

∂L/∂W₂ = 2(-0.0576) × 1.5 × 0.4199 × 1.0 ≈ -0.0726
∂L/∂b₂ = 2(-0.0576) × 1.5 × 0.4199 ≈ -0.0726
```

---

## 🎓 Layer 1 Gradients (Hidden Layer 1)

### Goal: Compute ∂L/∂W₁ and ∂L/∂b₁

**Step 1**: We already have ∂L/∂z₂ = 2(ê - e) × W₃ × (1 - h₂²)

**Step 2**: Compute how z₂ depends on h₁. Since z₂ = W₂·h₁ + b₂:
```
∂z₂/∂h₁ = W₂
```

**Step 3**: Chain rule to get gradient at h₁:
```
∂L/∂h₁ = ∂L/∂z₂ × ∂z₂/∂h₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂
```

**Step 4**: Go through the activation. Since h₁ = tanh(z₁):
```
∂h₁/∂z₁ = tanh'(z₁) = 1 - tanh²(z₁) = 1 - h₁²
```

**Step 5**: Chain rule to get gradient at z₁:
```
∂L/∂z₁ = ∂L/∂h₁ × ∂h₁/∂z₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

**Step 6**: Compute weight gradient. Since z₁ = W₁·d + b₁:
```
∂z₁/∂W₁ = d
```

**Step 7**: Apply chain rule:
```
∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × d
```

**Step 8**: Compute bias gradient. Since z₁ = W₁·d + b₁:
```
∂z₁/∂b₁ = 1
```

**Step 9**: Apply chain rule:
```
∂L/∂b₁ = ∂L/∂z₁ × ∂z₁/∂b₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

### Summary for Layer 1:
```
∂L/∂W₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × d
∂L/∂b₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
```

### Numerical Example:
Using previous values plus W₂ = 0.9, h₁ = 1.0, d = 30:
```
1 - h₁² = 1 - 1.0² = 0.0 (saturated!)

∂L/∂W₁ = 2(-0.0576) × 1.5 × 0.4199 × 0.9 × 0.0 × 30 ≈ 0.0
∂L/∂b₁ = 2(-0.0576) × 1.5 × 0.4199 × 0.9 × 0.0 ≈ 0.0
```

**Note**: When h₁ is saturated (≈1.0), the gradient vanishes! This is the **vanishing gradient problem**.

---

## 📊 Complete Gradient Summary

For a two-layer network learning drug dosage response:

### Output Layer (Layer 3):
```
∂L/∂W₃ = 2(ê - e) × h₂
∂L/∂b₃ = 2(ê - e)
```

### Hidden Layer 2:
```
∂L/∂W₂ = 2(ê - e) × W₃ × (1 - h₂²) × h₁
∂L/∂b₂ = 2(ê - e) × W₃ × (1 - h₂²)
```

### Hidden Layer 1:
```
∂L/∂W₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²) × d
∂L/∂b₁ = 2(ê - e) × W₃ × (1 - h₂²) × W₂ × (1 - h₁²)
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
W₃ ← 1.5 - 0.01 × (-0.0877) = 1.5 + 0.000877 = 1.501
b₃ ← 0.5 - 0.01 × (-0.1152) = 0.5 + 0.001152 = 0.501

W₂ ← 0.9 - 0.01 × (-0.0726) = 0.9 + 0.000726 = 0.901
b₂ ← 0.1 - 0.01 × (-0.0726) = 0.1 + 0.000726 = 0.101

W₁ ← 0.6 - 0.01 × 0.0 = 0.6 (no change due to saturation)
b₁ ← -0.3 - 0.01 × 0.0 = -0.3 (no change due to saturation)
```

---

## 🧮 Batch Training

For multiple samples (batch size n), we average the gradients:

```
∂L/∂W₃ = (2/n) × Σᵢ (êᵢ - eᵢ) × h₂ᵢ
∂L/∂b₃ = (2/n) × Σᵢ (êᵢ - eᵢ)

∂L/∂W₂ = (2/n) × Σᵢ (êᵢ - eᵢ) × W₃ × (1 - h₂ᵢ²) × h₁ᵢ
∂L/∂b₂ = (2/n) × Σᵢ (êᵢ - eᵢ) × W₃ × (1 - h₂ᵢ²)

∂L/∂W₁ = (2/n) × Σᵢ (êᵢ - eᵢ) × W₃ × (1 - h₂ᵢ²) × W₂ × (1 - h₁ᵢ²) × dᵢ
∂L/∂b₁ = (2/n) × Σᵢ (êᵢ - eᵢ) × W₃ × (1 - h₂ᵢ²) × W₂ × (1 - h₁ᵢ²)
```

---

## 🎯 Key Insights for Drug Dosage Application

### 1. **Learning the Inverted U-Shape**
The network must learn:
- Low dosage → low effectiveness (underdose)
- Optimal dosage → maximum effectiveness
- High dosage → low effectiveness (overdose)

### 2. **Gradient Flow Across Dosage Ranges**

**Underdose Region (0-25mg)**:
- Large errors drive strong gradients
- Network learns to increase effectiveness prediction

**Therapeutic Window (25-75mg)**:
- Smaller errors, moderate gradients
- Network fine-tunes the peak

**Overdose Region (75-100mg)**:
- Large errors again
- Network learns to decrease effectiveness prediction

### 3. **Activation Saturation Issues**

When tanh saturates (output ≈ ±1):
```
tanh'(x) ≈ 0  →  vanishing gradients
```

This can happen at extreme dosages, slowing learning.

**Solution**: Proper weight initialization and input normalization.

### 4. **Why Two Layers Work for Inverted Parabola**

- **Layer 1**: Captures initial non-linear transformation of dosage
- **Layer 2**: Refines the curve to create the peak
- **Together**: Form the inverted U-shape

### 5. **Medical Interpretation**

The gradients tell us:
- How to adjust the model to better predict effectiveness
- Which dosages need more learning (larger gradients)
- When the model has converged (small gradients)

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
- Saturates at extremes: tanh(±∞) = ±1
```

### Why Tanh for Drug Dosage?
1. **Non-linearity**: Essential for learning bell curves
2. **Zero-centered**: Helps with gradient flow
3. **Smooth derivative**: Enables stable learning
4. **Bounded output**: Represents bounded effectiveness (0-100%)

---

## 💊 Dose-Response Specific Considerations

### 1. **Optimal Dosage Detection**

The network learns to maximize output at optimal dosage:
```
∂ê/∂d = 0  at  d = d_optimal
```

Gradients guide the network to create this peak.

### 2. **Safety Margins**

Gradients in overdose region should be:
- **Negative**: Decreasing effectiveness with increasing dose
- **Large magnitude**: Strong signal to avoid this region

### 3. **Therapeutic Window**

In the therapeutic window (25-75mg):
- Gradients are smaller (model is more confident)
- Fine-tuning occurs to perfect the peak shape

### 4. **Underdose vs Overdose Symmetry**

For symmetric dose-response curves:
```
Effectiveness(50 - x) ≈ Effectiveness(50 + x)
```

The network learns this symmetry through balanced gradients.

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
   - Loss should decrease over iterations
   - Predicted optimal dosage should approach true optimal (50mg)

4. **Dose-Response Curve Check**:
   - Predicted curve should be inverted U-shaped
   - Peak should be near 50mg
   - Effectiveness should decrease on both sides

---

## 💡 Practical Tips for Drug Dosage Models

1. **Normalize Dosages**: Scale to [0, 1] or standardize (mean=0, std=1)
2. **Normalize Effectiveness**: Scale to [0, 1] or standardize
3. **Initialize Carefully**: Use Xavier/He initialization
4. **Monitor Gradients**: Watch for vanishing/exploding gradients
5. **Learning Rate**: Start with 0.01, adjust based on convergence
6. **Batch Size**: Use full batch or large batches for stable gradients
7. **Validation**: Check predicted optimal dosage against known value
8. **Safety**: Ensure model doesn't predict high effectiveness in overdose region

---

## 🏥 Clinical Implications

### Understanding the Gradients

**Large Gradients** indicate:
- Model is uncertain about effectiveness at this dosage
- More training data needed in this region
- Potential safety concerns if in overdose region

**Small Gradients** indicate:
- Model is confident about predictions
- Well-learned region
- Stable therapeutic window

### Model Confidence

Gradient magnitude can inform clinical decisions:
- **High confidence** (small gradients): Safe to use predictions
- **Low confidence** (large gradients): Need more data or caution

---

## 🎓 Advanced Topics

### 1. **Asymmetric Dose-Response**

Real drugs may have asymmetric curves:
```
Underdose slope ≠ Overdose slope
```

The network can learn this through different gradient patterns.

### 2. **Multiple Peaks**

Some drugs have multiple therapeutic windows. This would require:
- More hidden layers
- More neurons per layer
- More complex gradient patterns

### 3. **Patient-Specific Dosing**

Adding patient features (age, weight, metabolism):
- Input becomes multi-dimensional
- Gradients computed for each feature
- Personalized dosing recommendations

---

**This derivation shows how backpropagation enables neural networks to learn complex medical relationships like optimal drug dosing, with direct applications to patient safety and treatment optimization!**
