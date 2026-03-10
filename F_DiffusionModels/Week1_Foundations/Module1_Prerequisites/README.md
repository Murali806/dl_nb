# Module 1: Prerequisites Refresher

## Overview

This module refreshes the mathematical foundations needed for diffusion models. Even if you're familiar with these concepts, reviewing them in the context of diffusion models will be valuable.

## Learning Objectives

By the end of this module, you will:
- Understand Gaussian distributions and their properties
- Master the reparameterization trick
- Calculate KL divergence between distributions
- Work with Markov chains and stochastic processes
- Understand basic differential equations (ODEs and SDEs)

## Time Estimate

**2 days** (4-6 hours total)

## Files in This Module

### Day 1: Probability Theory
1. **1_1_probability_distributions_refresher.md** (60 min)
   - Gaussian distributions
   - Multivariate normal distributions
   - KL divergence
   - Conditional distributions

2. **1_2_probability_visualizations.ipynb** (45 min)
   - Interactive visualizations
   - Sampling demonstrations
   - KL divergence plots

3. **1_3_bayes_theorem_inference.md** (45 min)
   - Bayes' theorem review
   - Posterior distributions
   - Maximum likelihood vs MAP

### Day 2: Stochastic Processes
4. **1_4_markov_chains_basics.md** (60 min)
   - Markov property
   - Transition matrices
   - Stationary distributions

5. **1_5_stochastic_processes_intro.md** (45 min)
   - Random walks
   - Brownian motion
   - Wiener processes

6. **1_6_differential_equations_refresher.md** (45 min)
   - Ordinary Differential Equations (ODEs)
   - Stochastic Differential Equations (SDEs)
   - Numerical solvers

## Prerequisites

- Basic calculus (derivatives, integrals)
- Linear algebra (matrices, vectors)
- Python programming
- NumPy basics

## Key Concepts

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Probability Theory                             │
│  ├── Gaussian Distributions                     │
│  ├── KL Divergence                              │
│  └── Conditional Distributions                  │
│                                                 │
│  Stochastic Processes                           │
│  ├── Markov Chains                              │
│  ├── Brownian Motion                            │
│  └── SDEs                                       │
│                                                 │
│  Mathematical Tools                             │
│  ├── Reparameterization Trick                   │
│  ├── Bayes' Theorem                             │
│  └── Differential Equations                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Study Tips

1. **Don't skip the basics**: These concepts are fundamental to understanding diffusion models
2. **Work through examples**: Run the code in the notebooks
3. **Visualize**: Use the interactive plots to build intuition
4. **Connect concepts**: Think about how each concept relates to diffusion

## Exercises

Each file includes exercises. Complete them before moving on:

- [ ] Implement reparameterization trick for 2D Gaussian
- [ ] Calculate KL divergence between different distributions
- [ ] Simulate a simple Markov chain
- [ ] Implement Brownian motion
- [ ] Solve a simple ODE numerically

## Common Questions

**Q: Do I need to master all the math before proceeding?**
A: No, but you should understand the key concepts. You can always return to this module for reference.

**Q: How much time should I spend on this module?**
A: 2 days is recommended, but adjust based on your background. If you're comfortable with probability theory, you might move faster.

**Q: Can I skip the differential equations part?**
A: Not recommended. SDEs are crucial for understanding the continuous-time view of diffusion models.

## Connection to Diffusion Models

These concepts directly map to diffusion models:

| Concept | Application in Diffusion |
|---------|-------------------------|
| Gaussian distributions | Forward and reverse processes |
| Reparameterization trick | Training with stochastic nodes |
| KL divergence | Loss function component |
| Markov chains | Discrete-time diffusion |
| SDEs | Continuous-time diffusion |
| Conditional distributions | Reverse process modeling |

## Next Steps

After completing this module:
1. Review your exercise solutions
2. Ensure you understand the reparameterization trick
3. Move to **Module 2: Generative Models Landscape**

## Resources

- **Book**: "Pattern Recognition and Machine Learning" by Bishop (Chapter 2)
- **Video**: 3Blue1Brown's series on probability
- **Paper**: "A Tutorial on Stochastic Differential Equations" by Särkkä

---

**Ready to start?** Open `1_1_probability_distributions_refresher.md`
