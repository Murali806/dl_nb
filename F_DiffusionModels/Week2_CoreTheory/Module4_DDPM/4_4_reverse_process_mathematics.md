# Reverse Process Mathematics

## Overview

This module derives the mathematics of the reverse diffusion process - how we sample from the trained model to generate new data.

---

## 1. The Reverse Process

### Goal

Given a trained model, generate samples:

```
x_T ~ N(0, I)  (start from noise)
x_{T-1} ~ p_θ(x_{T-1}|x_T)
x_{T-2} ~ p_θ(x_{T-2}|x_{T-1})
...
x_0 ~ p_θ(x_0|x_1)  (final sample)
```

### The Challenge

We need to define p_θ(x_{t-1}|x_t) such that:
1. It's tractable to sample from
2. It approximates the true posterior q(x_{t-1}|x_t,x_0)

---

## 2. Reverse Transition Distribution

### Model Definition

```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

### Mean Parameterization

Using noise prediction:

```
μ_θ(x_t, t) = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t))
```

### Variance Options

**Option 1: Fixed (DDPM)**
```
Σ_θ(x_t, t) = β_t I  or  β̃_t I
```

**Option 2: Learned (Improved DDPM)**
```
Σ_θ(x_t, t) = exp(v_θ(x_t,t)) I
```

---

## 3. Deriving the Mean Formula

### Starting from True Posterior

Recall:
```
q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)

where:
μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
```

### Expressing x_0 in terms of x_t

From forward process:
```
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

Therefore:
x_0 = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t
```

### Substituting into μ̃_t

```
μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) · (x_t - √(1-ᾱ_t) ε)/√ᾱ_t 
     + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
```

After simplification:
```
μ̃_t = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε)
```

### Model Prediction

Replace true ε with predicted ε_θ:
```
μ_θ(x_t,t) = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t))
```

---

## 4. Sampling Algorithm

### Algorithm: DDPM Sampling

```
Input: Trained model ε_θ, timesteps T

1. Sample x_T ~ N(0, I)

2. For t = T, T-1, ..., 1:
   a. If t > 1:
      z ~ N(0, I)
   Else:
      z = 0
   
   b. Compute:
      ε_pred = ε_θ(x_t, t)
      μ_t = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_pred)
      σ_t = √β_t  (or √β̃_t)
   
   c. Sample:
      x_{t-1} = μ_t + σ_t · z

3. Return x_0
```

### Why z = 0 at t = 1?

At the final step, we want deterministic output:
```
x_0 = μ_1  (no noise)
```

---

## 5. Variance Schedule for Sampling

### Option 1: β_t (Original DDPM)

```
σ_t² = β_t
```

**Properties**:
- Simple
- Works well

### Option 2: β̃_t (Improved)

```
σ_t² = β̃_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) β_t
```

**Properties**:
- Theoretically motivated
- Slightly better quality

### Option 3: Interpolated

```
σ_t² = η · β_t + (1-η) · β̃_t

where η ∈ [0, 1]
```

**Properties**:
- Flexible
- η = 0: deterministic (DDIM)
- η = 1: stochastic (DDPM)

---

## 6. Connection to Score Function

### Score-Based View

The score function is:
```
∇_x log p(x_t) ≈ -ε_θ(x_t,t) / √(1-ᾱ_t)
```

### Reverse SDE

The reverse process can be written as:
```
dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dW̄

where W̄ is reverse-time Brownian motion
```

### Discretization

DDPM sampling is a discretization of this SDE!

---

## 7. Practical Implementation

### Sampling Function

```python
import torch

@torch.no_grad()
def sample_ddpm(model, shape, T, alpha, alpha_bar, beta):
    """
    Sample from DDPM
    
    Args:
        model: Trained noise prediction network
        shape: Shape of samples (batch_size, channels, height, width)
        T: Number of timesteps
        alpha, alpha_bar, beta: Noise schedule parameters
    
    Returns:
        Generated samples
    """
    device = next(model.parameters()).device
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    # Iteratively denoise
    for t in reversed(range(T)):
        # Create time tensor
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        eps_pred = model(x, t_tensor)
        
        # Compute mean
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        )
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * noise
        else:
            x = mean
    
    return x
```

### Batch Sampling

```python
def sample_batch(model, num_samples, img_shape, T, alpha, alpha_bar, beta):
    """Sample a batch of images"""
    shape = (num_samples,) + img_shape
    samples = sample_ddpm(model, shape, T, alpha, alpha_bar, beta)
    return samples
```

---

## 8. Accelerated Sampling (DDIM)

### The Idea

Skip timesteps to sample faster!

### DDIM Formula

```
x_{t-Δt} = √ᾱ_{t-Δt} x̂_0 + √(1-ᾱ_{t-Δt}-σ_t²) ε_θ + σ_t ε

where:
x̂_0 = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
```

### Implementation

```python
@torch.no_grad()
def sample_ddim(model, shape, timesteps, alpha_bar):
    """
    DDIM sampling (faster)
    
    Args:
        timesteps: List of timesteps to use (e.g., [999, 899, 799, ...])
    """
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        eps_pred = model(x, t_tensor)
        
        # Predict x_0
        alpha_bar_t = alpha_bar[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        if i > 0:
            t_prev = timesteps[i-1]
            alpha_bar_t_prev = alpha_bar[t_prev]
            
            # DDIM update (deterministic, σ=0)
            x = torch.sqrt(alpha_bar_t_prev) * x0_pred + \
                torch.sqrt(1 - alpha_bar_t_prev) * eps_pred
        else:
            x = x0_pred
    
    return x
```

---

## 9. Conditional Sampling

### Classifier Guidance

```
ε̃_θ(x_t,t,y) = ε_θ(x_t,t) - √(1-ᾱ_t) ∇_x log p(y|x_t)
```

### Classifier-Free Guidance

```
ε̃_θ(x_t,t,y) = ε_θ(x_t,t,∅) + w(ε_θ(x_t,t,y) - ε_θ(x_t,t,∅))

where w is guidance scale
```

### Implementation

```python
@torch.no_grad()
def sample_conditional(model, shape, T, alpha, alpha_bar, beta, 
                      condition, guidance_scale=7.5):
    """Sample with classifier-free guidance"""
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict with and without condition
        eps_cond = model(x, t_tensor, condition)
        eps_uncond = model(x, t_tensor, None)
        
        # Apply guidance
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # Denoise step
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    
    return x
```

---

## 10. Sampling Quality Analysis

### Metrics

1. **FID (Fréchet Inception Distance)**
   - Measures distribution similarity
   - Lower is better

2. **IS (Inception Score)**
   - Measures quality and diversity
   - Higher is better

3. **Precision/Recall**
   - Precision: quality
   - Recall: diversity

### Computing FID

```python
from scipy import linalg
import numpy as np

def calculate_fid(real_features, fake_features):
    """Calculate FID score"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid
```

---

## 11. Visualization

### Progressive Denoising

```python
import matplotlib.pyplot as plt

def visualize_sampling(model, T, alpha, alpha_bar, beta, num_steps=10):
    """Visualize the sampling process"""
    shape = (1, 3, 32, 32)
    device = next(model.parameters()).device
    
    x = torch.randn(shape, device=device)
    images = [x.cpu()]
    
    step_size = T // num_steps
    
    for t in reversed(range(T)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor)
        
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
        
        if t % step_size == 0:
            images.append(x.cpu())
    
    # Plot
    fig, axes = plt.subplots(1, len(images), figsize=(20, 2))
    for i, img in enumerate(images):
        axes[i].imshow(img[0].permute(1, 2, 0))
        axes[i].axis('off')
    plt.show()
```

---

## Summary

Key concepts:
1. **Reverse process**: p_θ(x_{t-1}|x_t) = N(μ_θ, Σ_θ)
2. **Mean formula**: μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ)
3. **Sampling**: Iterative denoising from x_T to x_0
4. **DDIM**: Faster sampling by skipping steps
5. **Guidance**: Conditional generation

---

## Exercises

1. **Derivation**: Derive the mean formula from scratch
2. **Implementation**: Implement DDPM sampling
3. **DDIM**: Implement and compare with DDPM
4. **Guidance**: Implement classifier-free guidance
5. **Visualization**: Create sampling visualizations

---

## Next Steps

Continue to `4_5_ddpm_implementation.ipynb` for a complete end-to-end implementation of DDPM with training and sampling on real data.
