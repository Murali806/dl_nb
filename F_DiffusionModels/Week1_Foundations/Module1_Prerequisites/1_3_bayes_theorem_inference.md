# Bayes' Theorem and Bayesian Inference

## Overview

Bayes' theorem is fundamental to understanding diffusion models, particularly the reverse process. This module covers Bayesian inference, posterior distributions, and their connection to generative modeling.

---

## 1. Bayes' Theorem

### The Formula

```
              P(A|B) × P(A)
P(B|A) = ─────────────────
                P(B)
```

Or in more common notation:

```
              p(θ|x) × p(θ)
p(θ|x) = ─────────────────
                p(x)
```

Where:
- **p(θ|x)**: Posterior (what we want to know)
- **p(x|θ)**: Likelihood (how likely is data given parameters)
- **p(θ)**: Prior (what we believe before seeing data)
- **p(x)**: Evidence (normalizing constant)

### Intuitive Understanding

```
                    How well does θ explain x?  ×  What we believed about θ
Posterior belief = ─────────────────────────────────────────────────────────
                              How likely is x overall?
```

---

## 2. Components of Bayes' Theorem

### Prior Distribution p(θ)

The prior represents our belief about parameters **before** seeing data.

**Example**: Coin flip
```
If we believe a coin is fair:
p(θ = 0.5) is high
p(θ = 0.1) is low
```

**Visual Representation**:
```
p(θ)
  │
  │    ╱‾‾‾╲
  │   ╱     ╲
  │  ╱       ╲
  │ ╱         ╲___
  │╱               ╲___
  └────────────────────── θ
      0.3  0.5  0.7
```

### Likelihood p(x|θ)

The likelihood measures how probable the observed data is for different parameter values.

**Example**: Observing 7 heads in 10 flips
```
p(x|θ) = θ⁷(1-θ)³

For θ = 0.5: p(x|0.5) = 0.117
For θ = 0.7: p(x|0.7) = 0.267  ← More likely!
```

### Evidence p(x)

The evidence is the marginal probability of the data:

```
p(x) = ∫ p(x|θ) p(θ) dθ
```

This is often **intractable** to compute, which is why we need approximation methods!

### Posterior Distribution p(θ|x)

The posterior combines prior beliefs with observed data:

```
p(θ|x) ∝ p(x|θ) × p(θ)
```

**Visual Evolution**:
```
Prior          Likelihood      Posterior
p(θ)           p(x|θ)          p(θ|x)
  │              │               │
  │  ╱‾╲         │    ╱╲         │   ╱╲
  │ ╱  ╲        │   ╱  ╲        │  ╱  ╲
  │╱    ╲       │  ╱    ╲       │ ╱    ╲
  └──────       └─────────       └────────
   0.5           0.7             0.6
```

---

## 3. Bayesian Inference

### The Process

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  1. Choose Prior p(θ)                          │
│     ↓                                           │
│  2. Observe Data x                             │
│     ↓                                           │
│  3. Compute Likelihood p(x|θ)                  │
│     ↓                                           │
│  4. Apply Bayes' Theorem                       │
│     ↓                                           │
│  5. Get Posterior p(θ|x)                       │
│     ↓                                           │
│  6. Make Predictions                           │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Example: Coin Flip Inference

**Setup**:
- Prior: p(θ) = Beta(2, 2) (slightly biased toward fair)
- Data: 7 heads in 10 flips
- Likelihood: p(x|θ) = Binomial(10, θ)

**Posterior**:
```
p(θ|x) = Beta(2+7, 2+3) = Beta(9, 5)
```

**Code Implementation**:
```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Prior
α_prior, β_prior = 2, 2

# Data
heads, tails = 7, 3

# Posterior (conjugate prior property)
α_post = α_prior + heads
β_post = β_prior + tails

# Visualize
θ = np.linspace(0, 1, 100)
prior = beta.pdf(θ, α_prior, β_prior)
posterior = beta.pdf(θ, α_post, β_post)

plt.plot(θ, prior, label='Prior')
plt.plot(θ, posterior, label='Posterior')
plt.xlabel('θ (probability of heads)')
plt.ylabel('Density')
plt.legend()
plt.title('Bayesian Inference for Coin Flip')
plt.show()
```

---

## 4. Maximum A Posteriori (MAP) Estimation

### Definition

MAP finds the most probable parameter value:

```
θ_MAP = argmax p(θ|x)
         θ

      = argmax p(x|θ) p(θ)
         θ
```

### Comparison with MLE

**Maximum Likelihood Estimation (MLE)**:
```
θ_MLE = argmax p(x|θ)
         θ
```

**Key Difference**:
- MLE: Ignores prior, only uses data
- MAP: Incorporates prior knowledge

**Visual Comparison**:
```
p(θ|x)
  │
  │      MAP
  │       ↓
  │      ╱╲
  │     ╱  ╲
  │    ╱    ╲
  │   ╱      ╲
  │  ╱        ╲___
  │ ╱             ╲___
  └────────────────────── θ
         ↑
        MLE (if prior is uniform)
```

### When to Use Each

| Method | Use When |
|--------|----------|
| MLE | Large data, weak prior knowledge |
| MAP | Limited data, strong prior knowledge |
| Full Bayesian | Need uncertainty quantification |

---

## 5. Conjugate Priors

### Definition

A prior is **conjugate** if the posterior has the same functional form.

### Common Conjugate Pairs

| Likelihood | Conjugate Prior | Posterior |
|------------|----------------|-----------|
| Bernoulli | Beta | Beta |
| Gaussian (known σ²) | Gaussian | Gaussian |
| Gaussian (known μ) | Inverse-Gamma | Inverse-Gamma |
| Poisson | Gamma | Gamma |

### Why Conjugacy Matters

1. **Analytical posterior**: No need for approximation
2. **Interpretable updates**: Clear how data changes beliefs
3. **Computational efficiency**: Fast inference

### Example: Gaussian-Gaussian Conjugacy

**Setup**:
```
Likelihood: x ~ N(θ, σ²)  (known σ²)
Prior:      θ ~ N(μ₀, σ₀²)
```

**Posterior**:
```
θ|x ~ N(μ_post, σ²_post)

where:
       σ²μ₀ + σ₀²x
μ_post = ─────────────
         σ² + σ₀²

         σ²σ₀²
σ²_post = ─────────
         σ² + σ₀²
```

**Intuition**:
- Posterior mean is weighted average of prior mean and data
- Posterior variance is smaller (more certain after seeing data)

---

## 6. Connection to Diffusion Models

### Reverse Process as Bayesian Inference

In diffusion models, the reverse process is Bayesian inference:

```
p(x_{t-1} | x_t) = ?
```

We want to infer the **less noisy** image given the **noisy** image.

### Using Bayes' Theorem

```
                p(x_t | x_{t-1}) × p(x_{t-1})
p(x_{t-1} | x_t) = ─────────────────────────────
                        p(x_t)
```

Where:
- **p(x_t | x_{t-1})**: Forward process (known)
- **p(x_{t-1})**: Marginal distribution (intractable)
- **p(x_t)**: Evidence (intractable)

### The Challenge

The denominator p(x_t) is intractable because:

```
p(x_t) = ∫ p(x_t | x_{t-1}) p(x_{t-1}) dx_{t-1}
```

This integral is over all possible images!

### The Solution

**Condition on x₀** (the original image):

```
p(x_{t-1} | x_t, x_0) = ?
```

This becomes tractable! (We'll derive this in Module 4)

---

## 7. Variational Inference

### The Problem

For complex models, the posterior p(θ|x) is intractable.

### The Solution

Approximate p(θ|x) with a simpler distribution q(θ):

```
q(θ) ≈ p(θ|x)
```

### Variational Lower Bound (ELBO)

We maximize the Evidence Lower Bound:

```
ELBO = E_q[log p(x|θ)] - D_KL(q(θ) ‖ p(θ))
```

**Intuition**:
- First term: How well does q explain the data?
- Second term: How different is q from the prior?

### Connection to VAEs and Diffusion

Both VAEs and diffusion models use variational inference:

**VAE**:
```
q(z|x) ≈ p(z|x)
```

**Diffusion**:
```
q(x_{1:T}|x_0) ≈ p(x_{1:T}|x_0)
```

---

## 8. Practical Examples

### Example 1: Gaussian Inference

```python
import numpy as np
from scipy.stats import norm

# Prior
μ_prior = 0
σ_prior = 2

# Data
data = np.array([1.5, 2.0, 1.8, 2.2])
σ_likelihood = 0.5

# Posterior (using conjugate prior formula)
n = len(data)
x_bar = np.mean(data)

precision_prior = 1 / σ_prior**2
precision_likelihood = n / σ_likelihood**2

precision_post = precision_prior + precision_likelihood
σ_post = 1 / np.sqrt(precision_post)

μ_post = (precision_prior * μ_prior + precision_likelihood * x_bar) / precision_post

print(f"Prior: N({μ_prior}, {σ_prior²})")
print(f"Posterior: N({μ_post:.2f}, {σ_post²:.2f})")
```

### Example 2: MAP vs MLE

```python
def negative_log_posterior(θ, data, μ_prior, σ_prior):
    """Negative log posterior for optimization"""
    # Likelihood term
    likelihood = -np.sum((data - θ)**2) / (2 * σ_likelihood**2)
    
    # Prior term
    prior = -(θ - μ_prior)**2 / (2 * σ_prior**2)
    
    return -(likelihood + prior)

# MAP estimate
from scipy.optimize import minimize
result = minimize(negative_log_posterior, x0=0, 
                 args=(data, μ_prior, σ_prior))
θ_MAP = result.x[0]

# MLE estimate
θ_MLE = np.mean(data)

print(f"MAP: {θ_MAP:.2f}")
print(f"MLE: {θ_MLE:.2f}")
```

---

## 9. Key Insights for Diffusion Models

### 1. Posterior Inference

The reverse diffusion process is posterior inference:
```
p(x_{t-1} | x_t, x_0) ∝ p(x_t | x_{t-1}) × p(x_{t-1} | x_0)
```

### 2. Intractability

Direct computation is intractable, requiring:
- Variational approximation (DDPM)
- Score matching (NCSN)
- Flow matching (CNF)

### 3. Conditioning

Conditioning on x₀ makes inference tractable:
```
q(x_{t-1} | x_t, x_0) is Gaussian!
```

### 4. ELBO Connection

The diffusion training objective is derived from ELBO:
```
L = E_q[log p(x_0)] ≥ ELBO
```

---

## Summary

Key concepts:
1. **Bayes' theorem** combines prior and likelihood to get posterior
2. **MAP estimation** finds most probable parameters
3. **Conjugate priors** enable analytical inference
4. **Variational inference** approximates intractable posteriors
5. **Diffusion models** use Bayesian inference for the reverse process

---

## Exercises

1. **Coin Flip**: Given 15 heads in 20 flips, compute posterior with Beta(2,2) prior
2. **Gaussian Inference**: Implement full Bayesian inference for Gaussian mean
3. **MAP vs MLE**: Compare estimates with different prior strengths
4. **ELBO**: Derive the ELBO for a simple Gaussian model
5. **Conditioning**: Show that q(x_{t-1}|x_t, x_0) is Gaussian (preview of Module 4)

---

## Next Steps

Continue to `1_4_markov_chains_basics.md` to learn about sequential probabilistic models.
