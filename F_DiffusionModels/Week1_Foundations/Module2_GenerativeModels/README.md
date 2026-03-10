# Module 2: Generative Models Landscape

## Overview

This module provides a comprehensive overview of deep generative modeling, placing diffusion models in context with other approaches. You'll understand the taxonomy of generative models and why diffusion models have become so powerful.

## Learning Objectives

By the end of this module, you will:
- Understand the goal of generative modeling
- Know the major families of generative models
- Compare strengths and weaknesses of different approaches
- Understand why diffusion models are powerful
- Recognize the three theoretical perspectives on diffusion

## Time Estimate

**2 days** (4-6 hours total)

## Files in This Module

### Day 3: Generative Modeling Overview
1. **2_1_deep_generative_modeling_overview.md** (60 min)
   - What is generative modeling?
   - Likelihood-based vs implicit models
   - Key challenges
   - Evaluation metrics

2. **2_2_taxonomy_of_generative_models.md** (60 min)
   - Autoregressive models
   - VAEs (Variational Autoencoders)
   - GANs (Generative Adversarial Networks)
   - Normalizing flows
   - Energy-based models
   - Diffusion models

### Day 4: Comparative Analysis
3. **2_3_comparative_analysis.ipynb** (45 min)
   - Side-by-side comparisons
   - Visualizations
   - Trade-offs

4. **2_4_why_diffusion_models.md** (45 min)
   - Advantages of diffusion
   - Success stories
   - Current limitations

5. **2_5_three_perspectives_overview.md** (45 min)
   - Variational perspective (DDPM)
   - Score-based perspective (NCSN)
   - Flow-based perspective (CNF)
   - Unified view

## Prerequisites

From Module 1:
- ✅ Probability distributions
- ✅ Bayes' theorem
- ✅ Markov chains
- ✅ Stochastic processes

## Key Concepts

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Generative Models                              │
│  ├── Likelihood-Based                           │
│  │   ├── Autoregressive (PixelCNN, GPT)        │
│  │   ├── VAEs                                   │
│  │   ├── Normalizing Flows                      │
│  │   └── Diffusion Models                       │
│  │                                               │
│  ├── Implicit                                    │
│  │   └── GANs                                    │
│  │                                               │
│  └── Energy-Based                                │
│      └── Score-Based Models                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Study Tips

1. **Compare and contrast**: Think about trade-offs between models
2. **Visualize**: Use the provided diagrams to build intuition
3. **Connect**: Link each model to concepts from Module 1
4. **Question**: Why would you choose one model over another?

## Exercises

- [ ] Classify a new generative model into the taxonomy
- [ ] Compare training objectives of VAE, GAN, and diffusion
- [ ] Explain when to use each type of model
- [ ] Identify which perspective (variational/score/flow) a paper uses

## Common Questions

**Q: Do I need to master all generative models?**
A: No, but understanding the landscape helps you appreciate diffusion models' unique advantages.

**Q: Which perspective should I focus on?**
A: All three! They're different views of the same process. Start with variational (DDPM), then score-based, then flows.

**Q: Are diffusion models always better?**
A: No. Each model type has trade-offs. Diffusion models excel at sample quality but can be slow.

## Connection to Diffusion Models

This module answers:
- **What problem** do diffusion models solve?
- **How** do they compare to alternatives?
- **Why** have they become so successful?
- **Which perspective** should you learn first?

## Comparison Table

| Model | Training | Sampling | Quality | Speed | Likelihood |
|-------|----------|----------|---------|-------|------------|
| Autoregressive | Stable | Slow | High | Slow | Exact |
| VAE | Stable | Fast | Medium | Fast | Lower bound |
| GAN | Unstable | Fast | High | Fast | No |
| Flow | Stable | Fast | Medium | Fast | Exact |
| Diffusion | Stable | Slow* | Very High | Slow* | Lower bound |

*Recent advances have significantly improved sampling speed

## Timeline of Generative Models

```
2013: VAE (Kingma & Welling)
2014: GAN (Goodfellow et al.)
2015: PixelCNN (van den Oord et al.)
2017: Normalizing Flows (Rezende & Mohamed)
2019: NCSN (Song & Ermon)
2020: DDPM (Ho et al.)
2021: Score-Based SDEs (Song et al.)
2022: Stable Diffusion (Rombach et al.)
2023: Flow Matching (Lipman et al.)
```

## Next Steps

After completing this module:
1. Understand the generative modeling landscape
2. Know why diffusion models are powerful
3. Recognize the three theoretical perspectives
4. Move to **Module 3: Diffusion Intuition**

## Resources

- **Survey**: "Generative Modeling by Estimating Gradients" (Yang Song)
- **Blog**: Lilian Weng's "What are Diffusion Models?"
- **Video**: Pieter Abbeel's Deep Unsupervised Learning course
- **Paper**: "Understanding Diffusion Models: A Unified Perspective" (Luo, 2022)

---

**Ready to start?** Open `2_1_deep_generative_modeling_overview.md`
