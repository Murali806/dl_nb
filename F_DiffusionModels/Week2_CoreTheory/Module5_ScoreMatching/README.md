# Module 5: Score Matching

## Overview

This module introduces score-based generative models and denoising score matching. You'll learn how DDPM connects to score-based models and understand the score function perspective.

## Learning Objectives

By the end of this module, you will:
- Understand the score function ∇_x log p(x)
- Learn denoising score matching
- Connect DDPM to score-based models
- Understand Langevin dynamics sampling
- Implement score-based models

## Time Estimate

**2-3 days** (6-8 hours total)

## Files in This Module

### Day 11: Score Function Basics
1. **5_1_score_function_introduction.md** (90 min)
   - What is the score function?
   - Properties and intuition
   - Why scores are useful
   - Connection to energy-based models

2. **5_2_score_matching_theory.md** (90 min)
   - Score matching objective
   - Denoising score matching
   - Sliced score matching
   - Practical considerations

### Day 12: Connection to Diffusion
3. **5_3_score_based_diffusion.md** (90 min)
   - DDPM as score-based model
   - Noise conditional score networks (NCSN)
   - Equivalence proofs
   - Unified perspective

4. **5_4_langevin_dynamics.md** (60 min)
   - Langevin MCMC
   - Annealed Langevin dynamics
   - Connection to sampling
   - Convergence properties

### Day 13: Implementation
5. **5_5_score_matching_implementation.ipynb** (120 min)
   - Implement score network
   - Train with denoising score matching
   - Sample with Langevin dynamics
   - Compare with DDPM

## Prerequisites

From previous modules:
- ✅ Probability distributions
- ✅ Stochastic processes
- ✅ DDPM mathematics
- ✅ Gradient-based optimization

## Key Concepts

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Score Function                                 │
│  s(x) = ∇_x log p(x)                           │
│                                                 │
│  Score Matching                                 │
│  min E_data[‖s_θ(x) - ∇_x log p(x)‖²]         │
│   θ                                             │
│                                                 │
│  Denoising Score Matching                       │
│  min E_x,σ,ε[‖s_θ(x+σε,σ) + ε/σ‖²]            │
│   θ                                             │
│                                                 │
│  Langevin Dynamics                              │
│  x_{t+1} = x_t + ε/2 s_θ(x_t) + √ε z          │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Study Tips

1. **Visualize scores**: Draw score fields for simple distributions
2. **Connect to DDPM**: See how ε_θ relates to s_θ
3. **Understand Langevin**: It's gradient ascent + noise
4. **Practice derivations**: Score matching objectives

## The Big Picture

### Score Function

```
For p(x) = exp(-E(x))/Z:

Score: s(x) = ∇_x log p(x) = -∇_x E(x)

Properties:
- Points toward high probability
- Independent of Z (normalization)
- Defines probability up to constant
```

### Score Matching

```
Goal: Learn s_θ(x) ≈ ∇_x log p_data(x)

Problem: Can't compute ∇_x log p_data(x) directly

Solution: Denoising score matching
- Add noise: x̃ = x + σε
- Learn: s_θ(x̃, σ) ≈ ∇_x̃ log p(x̃|x)
- Equivalent to: s_θ(x̃, σ) ≈ -ε/σ
```

### Connection to DDPM

```
DDPM noise prediction:
ε_θ(x_t, t) ≈ ε

Score function:
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)

Therefore:
DDPM ≡ Score-based model!
```

---

## Mathematical Notation

### Score Function
- `s(x) = ∇_x log p(x)`: Score function
- `E(x)`: Energy function
- `Z`: Partition function

### Score Matching
- `s_θ(x)`: Learned score function
- `σ`: Noise level
- `ε ~ N(0, I)`: Noise

### Langevin Dynamics
- `ε`: Step size
- `T`: Number of steps
- `x_t`: Sample at step t

---

## Key Theorems

### Theorem 1: Score Matching Equivalence

```
The following are equivalent:

1. min E_data[‖s_θ(x) - ∇_x log p(x)‖²]
    θ

2. min E_data[tr(∇_x s_θ(x)) + ½‖s_θ(x)‖²]
    θ
```

**Proof**: Integration by parts

### Theorem 2: Denoising Score Matching

```
For x̃ = x + σε where ε ~ N(0,I):

∇_x̃ log p(x̃|x) = -(x̃ - x)/σ² = -ε/σ

Therefore:
min E_x,ε[‖s_θ(x+σε, σ) + ε/σ‖²]
 θ
```

### Theorem 3: Langevin Convergence

```
Under certain conditions, Langevin dynamics:

x_{t+1} = x_t + ε/2 ∇_x log p(x_t) + √ε z_t

converges to p(x) as t → ∞ and ε → 0
```

---

## Common Questions

**Q: What is the score function intuitively?**
A: It points in the direction of increasing probability density.

**Q: Why is the score useful?**
A: It defines the distribution without needing the normalization constant Z.

**Q: How does score matching avoid computing Z?**
A: The score is the gradient of log p, so Z cancels out.

**Q: What's the connection to DDPM?**
A: DDPM's noise prediction is proportional to the negative score.

**Q: Why use multiple noise levels?**
A: Different noise levels capture different scales of the data distribution.

---

## Connection to Previous Modules

This module builds on:

| Previous Concept | Score-Based View |
|-----------------|------------------|
| DDPM noise ε_θ | Score s_θ = -ε_θ/√(1-ᾱ_t) |
| Forward process | Adding noise for score matching |
| Reverse process | Langevin dynamics |
| Training loss | Denoising score matching |

---

## Exercises

- [ ] Compute score for simple distributions
- [ ] Derive score matching objective
- [ ] Prove denoising score matching
- [ ] Implement Langevin dynamics
- [ ] Connect DDPM to score-based models

---

## Module Structure

```
5_1: Score Function
  ↓ (definition and properties)
  
5_2: Score Matching
  ↓ (training objective)
  
5_3: Score-Based Diffusion
  ↓ (connection to DDPM)
  
5_4: Langevin Dynamics
  ↓ (sampling algorithm)
  
5_5: Implementation
  ↓ (complete score-based model)
```

---

## Success Criteria

After this module, you should be able to:
- [ ] Explain the score function
- [ ] Derive score matching objectives
- [ ] Understand NCSN architecture
- [ ] Implement Langevin sampling
- [ ] Connect DDPM and score-based models

---

## Next Steps

After completing this module:
1. Understand score-based perspective
2. See DDPM as special case
3. Ready for **Module 6: SDE Framework**
4. Unify discrete and continuous views

---

## Resources

- **Paper**: Song & Ermon (2019) - NCSN
- **Paper**: Song et al. (2021) - Score-Based Generative Modeling
- **Blog**: Yang Song's blog on score-based models
- **Tutorial**: Score-based generative modeling tutorial

---

**Ready to start?** Open `5_1_score_function_introduction.md`

Let's explore the score-based perspective! 📊
