# ELBO Derivation for DDPM

## Overview

This module derives the Evidence Lower Bound (ELBO) for Denoising Diffusion Probabilistic Models. Understanding this derivation is crucial for understanding the training objective.

---

## 1. The Goal

### What We Want

Maximize the log-likelihood of the data:

```
max E_data[log p_θ(x_0)]
 θ
```

### The Problem

Direct optimization is intractable because:

```
p_θ(x_0) = ∫ p_θ(x_{0:T}) dx_{1:T}
```

This integral over all latent variables is intractable!

### The Solution

Use **variational inference** to derive a lower bound (ELBO) that we can optimize instead.

---

## 2. Variational Lower Bound

### Jensen's Inequality

For any distribution q:

```
log p_θ(x_0) = log ∫ p_θ(x_{0:T}) dx_{1:T}
             = log ∫ p_θ(x_{0:T}) q(x_{1:T}|x_0)/q(x_{1:T}|x_0) dx_{1:T}
             = log E_q[p_θ(x_{0:T})/q(x_{1:T}|x_0)]
             ≥ E_q[log p_θ(x_{0:T})/q(x_{1:T}|x_0)]  (Jensen's)
```

### ELBO Definition

```
ELBO = E_q[log p_θ(x_{0:T})/q(x_{1:T}|x_0)]
     = E_q[log p_θ(x_{0:T})] - E_q[log q(x_{1:T}|x_0)]
```

**Key property**: log p_θ(x_0) ≥ ELBO

---

## 3. Expanding the ELBO

### Forward Process

Recall:
```
q(x_{1:T}|x_0) = ∏_{t=1}^T q(x_t|x_{t-1})
```

### Reverse Process

Define:
```
p_θ(x_{0:T}) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1}|x_t)
```

### ELBO Expansion

```
ELBO = E_q[log p(x_T) + ∑_{t=1}^T log p_θ(x_{t-1}|x_t) 
           - ∑_{t=1}^T log q(x_t|x_{t-1})]
```

---

## 4. Rearranging Terms

### Telescoping Sum

Rewrite using Bayes' theorem:

```
q(x_t|x_{t-1}) = q(x_{t-1}|x_t,x_0) q(x_t|x_0) / q(x_{t-1}|x_0)
```

### After Rearrangement

```
ELBO = E_q[log p(x_T)/q(x_T|x_0)]
     + E_q[∑_{t=2}^T log p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0)]
     + E_q[log p_θ(x_0|x_1)]
```

---

## 5. KL Divergence Form

### Definition of KL Divergence

```
D_KL(p ‖ q) = E_p[log p/q]
```

### ELBO in KL Form

```
ELBO = E_q[log p_θ(x_0|x_1)]
     - D_KL(q(x_T|x_0) ‖ p(x_T))
     - ∑_{t=2}^T E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

### Interpretation

```
Term 1: Reconstruction term
Term 2: Prior matching (usually ignored)
Term 3: Denoising matching (main term)
```

---

## 6. Detailed Derivation

### Step 1: Start with ELBO

```
ELBO = E_q[log p_θ(x_{0:T})/q(x_{1:T}|x_0)]
```

### Step 2: Expand Joint Distributions

```
= E_q[log p(x_T) ∏_{t=1}^T p_θ(x_{t-1}|x_t) / ∏_{t=1}^T q(x_t|x_{t-1})]
```

### Step 3: Separate Logs

```
= E_q[log p(x_T)] 
  + E_q[∑_{t=1}^T log p_θ(x_{t-1}|x_t)]
  - E_q[∑_{t=1}^T log q(x_t|x_{t-1})]
```

### Step 4: Use Bayes' Theorem

For t ≥ 2:
```
q(x_t|x_{t-1}) = q(x_{t-1}|x_t,x_0) q(x_t|x_0) / q(x_{t-1}|x_0)
```

### Step 5: Substitute and Simplify

After telescoping:
```
ELBO = E_q[log p(x_T)/q(x_T|x_0)]
     + E_q[log p_θ(x_0|x_1)]
     + ∑_{t=2}^T E_q[log p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0)]
```

### Step 6: Convert to KL Divergences

```
ELBO = E_q[log p_θ(x_0|x_1)]
     - D_KL(q(x_T|x_0) ‖ p(x_T))
     - ∑_{t=2}^T E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

---

## 7. Understanding Each Term

### Term 1: Reconstruction

```
L_0 = E_q[log p_θ(x_0|x_1)]
```

**Interpretation**: How well can we reconstruct x_0 from x_1?

**In practice**: Often approximated or ignored.

### Term 2: Prior Matching

```
L_T = D_KL(q(x_T|x_0) ‖ p(x_T))
```

**Interpretation**: How close is q(x_T|x_0) to p(x_T) = N(0,I)?

**In practice**: Usually very small, can be ignored.

### Term 3: Denoising Matching

```
L_{t-1} = E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

**Interpretation**: How well does p_θ match the true posterior?

**In practice**: This is the main term we optimize!

---

## 8. Computing the KL Divergence

### Both Distributions are Gaussian

```
q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

### KL Between Gaussians

```
D_KL(N(μ_1,Σ_1) ‖ N(μ_2,Σ_2)) = 
    ½[tr(Σ_2⁻¹Σ_1) + (μ_2-μ_1)ᵀΣ_2⁻¹(μ_2-μ_1) - k + log(det(Σ_2)/det(Σ_1))]
```

### Simplified (Fixed Variance)

If Σ_θ = β̃_t I (fixed):

```
D_KL = 1/(2β̃_t) ‖μ̃_t(x_t,x_0) - μ_θ(x_t,t)‖²
```

---

## 9. Parameterization Choices

### Option 1: Predict Mean Directly

```
μ_θ(x_t, t) = neural network output
```

### Option 2: Predict x_0

```
μ_θ(x_t, t) = function of x̂_0 = f_θ(x_t, t)
```

### Option 3: Predict Noise (DDPM)

```
μ_θ(x_t, t) = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t))
```

**DDPM uses Option 3!**

---

## 10. Simplified Objective

### From ELBO to L_simple

Starting from:
```
L_t = E_q[1/(2β̃_t) ‖μ̃_t - μ_θ‖²]
```

With noise prediction:
```
μ̃_t = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε)
μ_θ = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ)
```

Substituting:
```
L_t ∝ E_q[‖ε - ε_θ(x_t,t)‖²]
```

### Final Simplified Loss

```
L_simple = E_t,x_0,ε[‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)‖²]
```

**This is what DDPM actually optimizes!**

---

## 11. Why the Simplification Works

### Empirical Observation

Ho et al. (2020) found that:
```
L_simple works better than full ELBO in practice!
```

### Reasons

1. **Simpler**: No need to compute variance terms
2. **Stable**: Uniform weighting across timesteps
3. **Effective**: Focuses on noise prediction

### Weighting Comparison

```
Full ELBO: L_t has weight 1/(2β̃_t)
Simplified: L_t has weight 1

Result: More emphasis on high noise levels
```

---

## 12. Connection to VAE

### VAE ELBO

```
ELBO = E_q[log p_θ(x|z)] - D_KL(q(z|x) ‖ p(z))
```

### DDPM ELBO

```
ELBO = E_q[log p_θ(x_0|x_1)]
     - D_KL(q(x_T|x_0) ‖ p(x_T))
     - ∑_t D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))
```

### Key Differences

1. **Hierarchy**: DDPM has T latent variables, VAE has 1
2. **Markov**: DDPM uses Markov chain, VAE doesn't
3. **Fixed encoder**: DDPM's q is fixed, VAE's q is learned

---

## 13. Practical Implementation

### Computing the Loss

```python
def compute_loss(model, x0, t, alpha_bar):
    """Compute DDPM training loss"""
    # Sample noise
    noise = torch.randn_like(x0)
    
    # Forward process: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t])
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    # Predict noise
    predicted_noise = model(xt, t)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss
```

### Training Loop

```python
def train_ddpm(model, dataloader, optimizer, T, alpha_bar):
    """Train DDPM"""
    model.train()
    
    for epoch in range(num_epochs):
        for x0 in dataloader:
            # Sample random timestep
            t = torch.randint(0, T, (x0.shape[0],))
            
            # Compute loss
            loss = compute_loss(model, x0, t, alpha_bar)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 14. Visualization

### ELBO Components

```
Total ELBO:
│
├── L_0: Reconstruction
│   └── Usually small
│
├── L_T: Prior matching
│   └── Usually negligible
│
└── L_{1:T-1}: Denoising
    └── Main contribution
    
Simplified:
Just optimize L_simple = E[‖ε - ε_θ‖²]
```

---

## Summary

Key concepts:
1. **ELBO**: Variational lower bound on log p(x_0)
2. **KL decomposition**: Three terms (reconstruction, prior, denoising)
3. **Main term**: Denoising matching L_{t-1}
4. **Simplification**: L_simple = E[‖ε - ε_θ‖²]
5. **Why it works**: Empirically better than full ELBO

---

## Exercises

1. **Derivation**: Derive the ELBO from scratch
2. **KL computation**: Compute D_KL for Gaussian distributions
3. **Simplification**: Show how ELBO simplifies to L_simple
4. **Implementation**: Implement ELBO computation
5. **Comparison**: Compare full ELBO vs L_simple empirically

---

## Next Steps

Continue to `4_3_training_objective_derivation.md` to understand the complete training procedure and why the simplified objective works so well.
