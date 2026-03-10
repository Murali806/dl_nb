# Module 4: DDPM Mathematics

## Overview

This module provides rigorous mathematical derivations of Denoising Diffusion Probabilistic Models (DDPM). You'll understand the theory behind the training objective and sampling procedure.

## Learning Objectives

By the end of this module, you will:
- Derive the forward process mathematically
- Understand the ELBO derivation
- Derive the simplified training objective
- Prove key properties of the diffusion process
- Understand the reverse process formulation
- Implement DDPM from scratch

## Time Estimate

**2-3 days** (6-8 hours total)

## Files in This Module

### Day 8: Forward Process Theory
1. **4_1_forward_process_mathematics.md** (90 min)
   - Markov chain formulation
   - Gaussian transitions
   - Closed-form sampling
   - Noise schedule properties

2. **4_2_elbo_derivation.md** (90 min)
   - Variational lower bound
   - KL divergence decomposition
   - Simplification steps
   - Connection to VAE

### Day 9: Training Objective
3. **4_3_training_objective_derivation.md** (90 min)
   - From ELBO to simple loss
   - Noise prediction formulation
   - Why it works
   - Practical considerations

4. **4_4_reverse_process_mathematics.md** (60 min)
   - Reverse transition distribution
   - Mean and variance formulas
   - Sampling algorithm
   - Connection to forward process

### Day 10: Implementation
5. **4_5_ddpm_implementation.ipynb** (120 min)
   - Complete DDPM from scratch
   - Training on MNIST
   - Sampling and visualization
   - Hyperparameter analysis

## Prerequisites

From Week 1:
- ✅ Gaussian distributions
- ✅ KL divergence
- ✅ Bayes' theorem
- ✅ Markov chains
- ✅ ELBO derivation
- ✅ Diffusion intuition

## Key Concepts

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Forward Process                                │
│  q(x_{1:T}|x_0) = ∏ q(x_t|x_{t-1})            │
│                                                 │
│  Reverse Process                                │
│  p_θ(x_{0:T}) = p(x_T) ∏ p_θ(x_{t-1}|x_t)     │
│                                                 │
│  Training Objective                             │
│  L_simple = E[‖ε - ε_θ(x_t, t)‖²]             │
│                                                 │
│  Sampling                                       │
│  x_T ~ N(0,I), then iteratively sample         │
│  x_{t-1} ~ p_θ(x_{t-1}|x_t)                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Study Tips

1. **Work through derivations**: Don't just read, derive yourself
2. **Check dimensions**: Verify matrix/vector dimensions
3. **Connect to intuition**: Link math to Module 3 concepts
4. **Implement**: Code helps solidify understanding

## The Big Picture

### Forward Process (Mathematical)

```
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

Closed form:
q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

where ᾱ_t = ∏_{s=1}^t (1-β_s)
```

### Reverse Process (Mathematical)

```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

where μ_θ and Σ_θ are learned
```

### Training Objective

```
Maximize: E_q[log p_θ(x_0)]

Equivalent to minimizing:
L = E_t,x_0,ε[‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)‖²]
```

---

## Mathematical Notation

### Variables
- `x_0`: Clean data
- `x_t`: Noisy data at timestep t
- `x_T`: Pure noise
- `ε`: Noise (Gaussian)
- `β_t`: Noise schedule
- `α_t = 1 - β_t`
- `ᾱ_t = ∏_{s=1}^t α_s`

### Distributions
- `q(x_t|x_{t-1})`: Forward transition
- `q(x_t|x_0)`: Forward marginal
- `p_θ(x_{t-1}|x_t)`: Reverse transition
- `p_θ(x_0)`: Model distribution

### Functions
- `ε_θ(x_t, t)`: Noise prediction network
- `μ_θ(x_t, t)`: Mean prediction
- `Σ_θ(x_t, t)`: Variance prediction

---

## Key Theorems

### Theorem 1: Closed-Form Forward

```
If q(x_t|x_{t-1}) = N(√α_t x_{t-1}, β_t I)

Then q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
```

**Proof**: By induction on Gaussian convolution

### Theorem 2: Reverse Process

```
If q(x_t|x_0) is Gaussian, then:

q(x_{t-1}|x_t, x_0) = N(μ̃_t(x_t, x_0), β̃_t I)

where:
μ̃_t = (√ᾱ_{t-1}β_t)/(1-ᾱ_t) x_0 + (√α_t(1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t
```

**Proof**: By Bayes' theorem and Gaussian identities

### Theorem 3: ELBO Decomposition

```
log p_θ(x_0) ≥ E_q[log p_θ(x_{0:T})/q(x_{1:T}|x_0)]

= E_q[log p(x_T)] 
  - E_q[D_KL(q(x_T|x_0) ‖ p(x_T))]
  - ∑_t E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
  + E_q[log p_θ(x_0|x_1)]
```

---

## Common Questions

**Q: Why use noise prediction instead of direct x_0 prediction?**
A: Empirically works better. Theoretically equivalent but different inductive bias.

**Q: Why is the variance fixed in DDPM?**
A: Simplifies training. Can be learned but doesn't improve much.

**Q: How to choose the noise schedule β_t?**
A: Linear or cosine. Start small (0.0001), end larger (0.02).

**Q: Why T=1000 timesteps?**
A: Trade-off between quality and speed. Can use fewer with DDIM.

---

## Connection to Intuition

This module formalizes concepts from Module 3:

| Intuition (Module 3) | Mathematics (Module 4) |
|---------------------|------------------------|
| "Add noise gradually" | q(x_t\|x_{t-1}) = N(...) |
| "Markov chain" | ∏_t q(x_t\|x_{t-1}) |
| "Closed form" | q(x_t\|x_0) = N(√ᾱ_t x_0, ...) |
| "Predict noise" | ε_θ(x_t, t) |
| "Training loss" | ‖ε - ε_θ(x_t, t)‖² |
| "Sampling" | x_{t-1} ~ p_θ(x_{t-1}\|x_t) |

---

## Exercises

- [ ] Derive the closed-form forward process
- [ ] Prove the reverse process formula
- [ ] Derive the ELBO decomposition
- [ ] Simplify ELBO to L_simple
- [ ] Implement forward process
- [ ] Implement reverse process
- [ ] Train DDPM on toy data

---

## Module Structure

```
4_1: Forward Process
  ↓ (mathematical formulation)
  
4_2: ELBO Derivation
  ↓ (variational bound)
  
4_3: Training Objective
  ↓ (simplified loss)
  
4_4: Reverse Process
  ↓ (sampling algorithm)
  
4_5: Implementation
  ↓ (complete DDPM)
```

---

## Success Criteria

After this module, you should be able to:
- [ ] Derive all key equations
- [ ] Explain each term in ELBO
- [ ] Implement DDPM from scratch
- [ ] Train on simple datasets
- [ ] Debug training issues
- [ ] Understand hyperparameters

---

## Next Steps

After completing this module:
1. Understand DDPM mathematics completely
2. Can derive key results
3. Ready for **Module 5: Score Matching**
4. Move to score-based perspective

---

## Resources

- **Paper**: Ho et al. (2020) - DDPM
- **Paper**: Sohl-Dickstein et al. (2015) - Original diffusion
- **Blog**: Lilian Weng - Mathematical details
- **Tutorial**: Hugging Face DDPM tutorial

---

**Ready to start?** Open `4_1_forward_process_mathematics.md`

Let's formalize the intuition! 📐
