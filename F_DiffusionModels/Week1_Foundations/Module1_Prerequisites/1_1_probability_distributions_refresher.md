# Probability Distributions Refresher

## Overview

This module refreshes essential probability concepts needed for understanding diffusion models. We'll focus on Gaussian distributions, multivariate normal distributions, and KL divergence.

---

## 1. Gaussian (Normal) Distribution

### Definition

A Gaussian distribution is defined by two parameters: mean (μ) and variance (σ²).

**Probability Density Function (PDF):**

```
         1                 -(x - μ)²
p(x) = ───────── × exp( ──────────── )
       √(2πσ²)              2σ²
```

### Visual Representation

```
     p(x)
      │
      │     ╱‾‾‾╲
      │    ╱     ╲
      │   ╱       ╲
      │  ╱         ╲
      │ ╱           ╲___
      │╱                 ╲___
      └────────────────────────── x
           μ-σ  μ  μ+σ
```

### Key Properties

1. **Mean (μ)**: Center of the distribution
2. **Variance (σ²)**: Spread of the distribution
3. **Standard Deviation (σ)**: Square root of variance
4. **68-95-99.7 Rule**: 
   - 68% of data within μ ± σ
   - 95% of data within μ ± 2σ
   - 99.7% of data within μ ± 3σ

### Why Gaussians Matter in Diffusion Models

Diffusion models use Gaussian distributions because:
- They are **mathematically tractable** (easy to work with)
- They have **closed-form solutions** for many operations
- **Central Limit Theorem**: Sum of many random variables → Gaussian
- **Reparameterization trick** works naturally with Gaussians

---

## 2. Multivariate Gaussian Distribution

### Definition

For a d-dimensional vector **x**, the multivariate Gaussian is:

```
                    1                      -1
p(x) = ──────────────────── × exp( -½(x-μ)ᵀΣ  (x-μ) )
       (2π)^(d/2) |Σ|^(1/2)
```

Where:
- **μ** ∈ ℝᵈ is the mean vector
- **Σ** ∈ ℝᵈˣᵈ is the covariance matrix
- |Σ| is the determinant of Σ

### Covariance Matrix

The covariance matrix captures relationships between dimensions:

```
     ┌                    ┐
     │ σ₁²   σ₁₂   σ₁₃  │
Σ =  │ σ₂₁   σ₂²   σ₂₃  │
     │ σ₃₁   σ₃₂   σ₃²  │
     └                    ┘
```

- **Diagonal elements (σᵢ²)**: Variance of each dimension
- **Off-diagonal elements (σᵢⱼ)**: Covariance between dimensions i and j

### Special Case: Isotropic Gaussian

When all dimensions are independent with equal variance:

```
Σ = σ²I

where I is the identity matrix
```

This simplifies to:

```
                1              -‖x-μ‖²
p(x) = ──────────────── × exp( ──────── )
       (2πσ²)^(d/2)             2σ²
```

**This is the form used in diffusion models!**

---

## 3. Sampling from Gaussians

### Univariate Case

To sample x ~ N(μ, σ²):

```python
# Method 1: Direct sampling
x = np.random.normal(μ, σ)

# Method 2: Reparameterization trick
ε = np.random.normal(0, 1)  # Standard normal
x = μ + σ * ε
```

### Multivariate Case

To sample **x** ~ N(**μ**, **Σ**):

```python
# Method 1: Direct sampling
x = np.random.multivariate_normal(μ, Σ)

# Method 2: Reparameterization trick
ε = np.random.normal(0, 1, size=d)  # Standard normal vector
L = np.linalg.cholesky(Σ)  # Cholesky decomposition: Σ = LLᵀ
x = μ + L @ ε
```

### The Reparameterization Trick

**Key Insight**: Instead of sampling directly from N(μ, σ²), we:
1. Sample ε ~ N(0, 1) (standard normal)
2. Transform: x = μ + σε

**Why this matters:**
- Makes sampling **differentiable** with respect to μ and σ
- Essential for training neural networks with stochastic nodes
- Used extensively in VAEs and diffusion models

---

## 4. KL Divergence

### Definition

The Kullback-Leibler (KL) divergence measures how one probability distribution differs from another:

```
         ⌠      p(x)
D_KL(p‖q) = │ p(x) log ──── dx
         ⌡      q(x)
```

### Properties

1. **Non-negative**: D_KL(p‖q) ≥ 0
2. **Zero iff identical**: D_KL(p‖q) = 0 ⟺ p = q
3. **Asymmetric**: D_KL(p‖q) ≠ D_KL(q‖p)
4. **Not a distance metric** (doesn't satisfy triangle inequality)

### KL Divergence Between Two Gaussians

For two univariate Gaussians p = N(μ₁, σ₁²) and q = N(μ₂, σ₂²):

```
              σ₁²     (μ₁ - μ₂)²      σ₂²
D_KL(p‖q) = log ─── + ──────────── + ─── - ½
              σ₂²        2σ₂²         σ₁²
```

### Special Case: KL to Standard Normal

For p = N(μ, σ²) and q = N(0, 1):

```
D_KL(p‖q) = ½(μ² + σ² - log(σ²) - 1)
```

**This appears in VAE loss functions!**

### Multivariate Case

For p = N(μ₁, Σ₁) and q = N(μ₂, Σ₂):

```
                                                    -1
D_KL(p‖q) = ½[tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂  (μ₂-μ₁) - d + log(|Σ₂|/|Σ₁|)]
```

---

## 5. Conditional Distributions

### Bayes' Theorem

```
              p(x|y) × p(y)
p(y|x) = ─────────────────
                p(x)
```

### Conditional Gaussian

If (x, y) ~ N(μ, Σ) where:

```
    ┌   ┐       ┌        ┐
    │ x │       │ Σ_xx  Σ_xy │
    │ y │ ~ N(μ,│ Σ_yx  Σ_yy │)
    └   ┘       └        ┘
```

Then the conditional distribution is:

```
                                    -1
x|y ~ N(μ_x + Σ_xy Σ_yy  (y - μ_y), Σ_xx - Σ_xy Σ_yy⁻¹ Σ_yx)
```

**This is crucial for understanding the reverse diffusion process!**

---

## 6. Practical Examples

### Example 1: Sampling and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
μ = 0
σ = 1
n_samples = 1000

# Sample using reparameterization trick
ε = np.random.randn(n_samples)
x = μ + σ * ε

# Visualize
plt.hist(x, bins=50, density=True, alpha=0.7, label='Samples')
x_range = np.linspace(-4, 4, 100)
pdf = (1/np.sqrt(2*np.pi*σ**2)) * np.exp(-(x_range-μ)**2/(2*σ**2))
plt.plot(x_range, pdf, 'r-', linewidth=2, label='True PDF')
plt.legend()
plt.title('Gaussian Distribution')
plt.show()
```

### Example 2: KL Divergence Calculation

```python
def kl_divergence_gaussians(μ1, σ1, μ2, σ2):
    """
    KL divergence between N(μ1, σ1²) and N(μ2, σ2²)
    """
    return np.log(σ2/σ1) + (σ1**2 + (μ1-μ2)**2)/(2*σ2**2) - 0.5

# Example
μ1, σ1 = 0, 1
μ2, σ2 = 1, 2
kl = kl_divergence_gaussians(μ1, σ1, μ2, σ2)
print(f"KL(N({μ1},{σ1²})‖N({μ2},{σ2²})) = {kl:.4f}")
```

---

## 7. Connection to Diffusion Models

### Forward Process

In diffusion models, we add Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

This is a **conditional Gaussian distribution**.

### Reverse Process

We learn to reverse the noise:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training Objective

The loss involves **KL divergence** between distributions:

```
L = E_q[D_KL(q(x_{t-1}|x_t, x_0) ‖ p_θ(x_{t-1}|x_t))]
```

---

## Summary

Key takeaways:
1. **Gaussian distributions** are the foundation of diffusion models
2. **Reparameterization trick** makes sampling differentiable
3. **KL divergence** measures distribution similarity
4. **Conditional distributions** are used in both forward and reverse processes
5. **Multivariate Gaussians** handle high-dimensional data (images, audio)

---

## Exercises

1. **Sampling**: Implement the reparameterization trick for a 2D Gaussian with correlation
2. **KL Divergence**: Calculate D_KL between N(2, 1) and N(0, 4)
3. **Visualization**: Plot how KL divergence changes as you vary μ and σ
4. **Conditional**: Given a 2D Gaussian, compute p(x|y=1)

---

## Next Steps

Continue to `1_2_probability_visualizations.ipynb` for interactive demonstrations of these concepts.
