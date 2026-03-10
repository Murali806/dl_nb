# Forward Process Mathematics

## Overview

This module provides rigorous mathematical derivation of the forward diffusion process. You'll understand how noise is added gradually and derive the closed-form sampling equation.

---

## 1. The Forward Process Definition

### Markov Chain Formulation

The forward process is a **Markov chain** that gradually adds Gaussian noise:

```
q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})
```

### Single Step Transition

At each timestep, we add Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Parameters**:
- `β_t ∈ (0, 1)`: Variance schedule (how much noise to add)
- `√(1-β_t)`: Scaling factor for signal
- `β_t I`: Variance of added noise

### Reparameterization

We can sample from this distribution:

```
x_t = √(1-β_t) x_{t-1} + √β_t ε_{t-1}

where ε_{t-1} ~ N(0, I)
```

---

## 2. Alternative Parameterization

### Using α_t

Define `α_t = 1 - β_t`:

```
q(x_t | x_{t-1}) = N(x_t; √α_t x_{t-1}, (1-α_t) I)

Sampling:
x_t = √α_t x_{t-1} + √(1-α_t) ε_{t-1}
```

### Cumulative Product

Define `ᾱ_t = ∏_{s=1}^t α_s`:

```
ᾱ_t = α_1 · α_2 · ... · α_t
```

This will be crucial for the closed-form solution!

---

## 3. Closed-Form Forward Process

### Theorem: Direct Sampling

**Theorem**: We can sample x_t directly from x_0:

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

Equivalently:
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

where ε ~ N(0, I)
```

### Proof by Induction

**Base case** (t=1):
```
x_1 = √α_1 x_0 + √(1-α_1) ε_0
    = √ᾱ_1 x_0 + √(1-ᾱ_1) ε_0  ✓
```

**Inductive step**: Assume true for t-1, prove for t.

Given:
```
x_{t-1} = √ᾱ_{t-1} x_0 + √(1-ᾱ_{t-1}) ε̄_{t-1}
```

Then:
```
x_t = √α_t x_{t-1} + √(1-α_t) ε_{t-1}
    = √α_t (√ᾱ_{t-1} x_0 + √(1-ᾱ_{t-1}) ε̄_{t-1}) + √(1-α_t) ε_{t-1}
    = √(α_t ᾱ_{t-1}) x_0 + √α_t √(1-ᾱ_{t-1}) ε̄_{t-1} + √(1-α_t) ε_{t-1}
    = √ᾱ_t x_0 + √α_t √(1-ᾱ_{t-1}) ε̄_{t-1} + √(1-α_t) ε_{t-1}
```

**Key insight**: The noise terms combine!

```
√α_t √(1-ᾱ_{t-1}) ε̄_{t-1} + √(1-α_t) ε_{t-1}
```

Since both are Gaussian with mean 0, their sum is Gaussian:

```
Variance = α_t(1-ᾱ_{t-1}) + (1-α_t)
         = α_t - α_t ᾱ_{t-1} + 1 - α_t
         = 1 - α_t ᾱ_{t-1}
         = 1 - ᾱ_t  ✓
```

Therefore:
```
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε  ✓
```

---

## 4. Properties of the Forward Process

### Property 1: Signal Decay

As t increases, the signal decays:

```
Signal strength: √ᾱ_t
Noise strength: √(1-ᾱ_t)

As t → T:
- ᾱ_t → 0
- √ᾱ_t → 0 (signal vanishes)
- √(1-ᾱ_t) → 1 (pure noise)
```

### Property 2: Variance Preservation

The total variance is preserved (approximately):

```
Var[x_t] = ᾱ_t Var[x_0] + (1-ᾱ_t)
```

If Var[x_0] ≈ 1, then Var[x_t] ≈ 1 for all t.

### Property 3: Markov Property

```
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
```

Only the previous state matters!

### Property 4: Gaussian Preservation

If x_0 is Gaussian, then x_t is Gaussian for all t.

---

## 5. Noise Schedule β_t

### Linear Schedule

```
β_t = β_min + (β_max - β_min) × (t-1)/(T-1)

Typical values:
- β_min = 0.0001
- β_max = 0.02
- T = 1000
```

### Cosine Schedule

```
ᾱ_t = f(t) / f(0)

where f(t) = cos((t/T + s)/(1+s) × π/2)²

s = 0.008 (small offset)
```

**Advantage**: More gradual at beginning and end.

### Comparison

```
Linear Schedule:
β_t: ___/‾‾‾‾‾
     (linear increase)

Cosine Schedule:
β_t: ___/‾‾‾‾‾
     (smoother curve)
```

---

## 6. Mathematical Properties

### Lemma 1: Gaussian Convolution

If X ~ N(μ_x, σ_x²) and Y ~ N(μ_y, σ_y²) are independent, then:

```
X + Y ~ N(μ_x + μ_y, σ_x² + σ_y²)
```

**Used in**: Combining noise terms in the proof.

### Lemma 2: Gaussian Scaling

If X ~ N(μ, σ²), then:

```
aX ~ N(aμ, a²σ²)
```

**Used in**: Scaling x_{t-1} by √α_t.

### Lemma 3: Product of Gaussians

```
N(x; μ_1, σ_1²) · N(x; μ_2, σ_2²) ∝ N(x; μ, σ²)

where:
μ = (σ_2² μ_1 + σ_1² μ_2) / (σ_1² + σ_2²)
σ² = (σ_1² σ_2²) / (σ_1² + σ_2²)
```

**Used in**: Deriving the reverse process.

---

## 7. Practical Implementation

### Computing ᾱ_t

```python
import numpy as np

def get_alpha_schedule(T, schedule='linear'):
    """Compute α_t and ᾱ_t"""
    if schedule == 'linear':
        beta = np.linspace(0.0001, 0.02, T)
    elif schedule == 'cosine':
        s = 0.008
        t = np.arange(T + 1)
        f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f[1:] / f[0]
        beta = 1 - alpha_bar / np.concatenate([alpha_bar[:1], alpha_bar[:-1]])
        beta = np.clip(beta, 0, 0.999)
    
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    
    return alpha, alpha_bar, beta
```

### Forward Sampling

```python
def forward_sample(x0, t, alpha_bar):
    """Sample x_t from x_0"""
    # x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    noise = np.random.randn(*x0.shape)
    sqrt_alpha_bar = np.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
    
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise
```

### Visualization

```python
import matplotlib.pyplot as plt

def visualize_forward_process(x0, T=10):
    """Visualize forward diffusion"""
    alpha, alpha_bar, beta = get_alpha_schedule(T)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    timesteps = np.linspace(0, T-1, 10, dtype=int)
    
    for i, t in enumerate(timesteps):
        if t == 0:
            xt = x0
        else:
            xt, _ = forward_sample(x0, t, alpha_bar)
        
        axes[i].imshow(xt, cmap='gray')
        axes[i].set_title(f't={t}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## 8. Connection to Intuition

### From Module 3 to Mathematics

| Intuition | Mathematics |
|-----------|-------------|
| "Add noise gradually" | q(x_t\|x_{t-1}) = N(...) |
| "Markov chain" | ∏_t q(x_t\|x_{t-1}) |
| "Closed form" | q(x_t\|x_0) = N(√ᾱ_t x_0, ...) |
| "Signal decays" | √ᾱ_t → 0 as t → T |
| "Ends in noise" | x_T ~ N(0, I) |

---

## 9. Key Equations Summary

### Forward Transition

```
q(x_t | x_{t-1}) = N(x_t; √α_t x_{t-1}, (1-α_t) I)
```

### Closed-Form Sampling

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
```

### Definitions

```
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^t α_s
```

---

## 10. Common Mistakes

### Mistake 1: Forgetting the Square Root

```
❌ Wrong: x_t = ᾱ_t x_0 + (1-ᾱ_t) ε
✓ Correct: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
```

### Mistake 2: Using β_t Instead of α_t

```
❌ Wrong: x_t = √β_t x_{t-1} + ...
✓ Correct: x_t = √α_t x_{t-1} + ...
```

### Mistake 3: Wrong Variance

```
❌ Wrong: x_t = √α_t x_{t-1} + β_t ε
✓ Correct: x_t = √α_t x_{t-1} + √(1-α_t) ε
```

---

## 11. Exercises

### Exercise 1: Verify Closed Form

Prove that for t=2:
```
x_2 = √ᾱ_2 x_0 + √(1-ᾱ_2) ε
```

Starting from:
```
x_1 = √α_1 x_0 + √(1-α_1) ε_0
x_2 = √α_2 x_1 + √(1-α_2) ε_1
```

### Exercise 2: Implement Schedule

Implement both linear and cosine schedules. Plot β_t, α_t, and ᾱ_t.

### Exercise 3: Forward Sampling

Implement forward sampling and visualize the diffusion process on a simple image.

### Exercise 4: Variance Analysis

Verify that Var[x_t] ≈ 1 for all t when Var[x_0] = 1.

### Exercise 5: Schedule Comparison

Compare linear vs cosine schedules. Which preserves signal longer?

---

## Summary

Key concepts:
1. **Forward process**: Markov chain adding Gaussian noise
2. **Closed form**: Can sample x_t directly from x_0
3. **Signal decay**: √ᾱ_t controls signal strength
4. **Noise schedule**: β_t determines diffusion speed
5. **Gaussian preservation**: Process maintains Gaussian structure

---

## Next Steps

Continue to `4_2_elbo_derivation.md` to derive the variational lower bound and understand the training objective.
