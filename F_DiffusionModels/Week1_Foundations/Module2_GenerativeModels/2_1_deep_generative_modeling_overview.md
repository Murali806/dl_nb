# Deep Generative Modeling Overview

## Overview

This module introduces the fundamental concepts of generative modeling, setting the stage for understanding diffusion models in the broader context of deep learning.

---

## 1. What is Generative Modeling?

### The Goal

**Generative modeling** aims to learn the probability distribution of data:

```
Given dataset: {x₁, x₂, ..., xₙ} ~ p_data(x)
Goal: Learn p_model(x) ≈ p_data(x)
```

### Two Key Capabilities

1. **Density Estimation**: Evaluate p(x) for any x
2. **Sampling**: Generate new samples x ~ p(x)

### Visual Intuition

```
Data Distribution          Learned Distribution
    p_data(x)                  p_model(x)
    
    ●  ●●                       ●  ●●
   ●●  ●●●                     ●●  ●●●
  ●●●●●●●●●                   ●●●●●●●●●
   ●●●●●●●                     ●●●●●●●
    ●●●●●                       ●●●●●
    
    Real data              Generated samples
```

---

## 2. Discriminative vs Generative Models

### Discriminative Models

Learn p(y|x): "Given input x, what is output y?"

```
Examples:
- Image classification: p(class | image)
- Object detection: p(bbox | image)
- Sentiment analysis: p(sentiment | text)
```

### Generative Models

Learn p(x) or p(x, y): "What does the data look like?"

```
Examples:
- Image generation: p(image)
- Text generation: p(text)
- Conditional generation: p(image | text)
```

### Comparison

```
┌─────────────────────────────────────────────┐
│                                             │
│  Discriminative: x → [Model] → y           │
│                                             │
│  Generative:     noise → [Model] → x       │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 3. Why Generative Modeling?

### Applications

1. **Content Creation**
   - Images: DALL-E, Midjourney, Stable Diffusion
   - Text: GPT, Claude
   - Audio: WaveNet, MusicGen
   - Video: Sora, Gen-2

2. **Data Augmentation**
   - Generate synthetic training data
   - Balance imbalanced datasets
   - Privacy-preserving data sharing

3. **Representation Learning**
   - Learn meaningful features
   - Unsupervised pretraining
   - Anomaly detection

4. **Scientific Discovery**
   - Drug design
   - Protein folding
   - Materials science

### Example Use Cases

```
Medical Imaging:
Real MRI → [Generative Model] → Synthetic MRI
                                 (for training)

Drug Discovery:
Properties → [Generative Model] → New molecules

Creative Tools:
Text prompt → [Generative Model] → Image
```

---

## 4. The Fundamental Challenge

### High-Dimensional Data

Images are high-dimensional:
```
28×28 grayscale image: 784 dimensions
256×256 RGB image: 196,608 dimensions
```

**Problem**: Curse of dimensionality!

### Data Manifold Hypothesis

Real data lies on a **low-dimensional manifold** in high-dimensional space.

```
High-Dimensional Space (ℝᵈ)
┌─────────────────────────────┐
│                             │
│    ╱‾‾‾╲                   │
│   ╱     ╲  ← Data manifold │
│  │   ●●● │    (low-dim)    │
│   ╲  ●●●╱                  │
│    ╲___╱                    │
│                             │
└─────────────────────────────┘
```

**Implication**: We don't need to model the entire space, just the manifold!

---

## 5. Likelihood-Based vs Implicit Models

### Likelihood-Based Models

Explicitly model p(x) and maximize likelihood:

```
max E_data[log p_θ(x)]
 θ
```

**Examples**: VAEs, Autoregressive models, Normalizing Flows, Diffusion Models

**Pros**:
- Principled training objective
- Can evaluate likelihood
- Stable training

**Cons**:
- May require approximations
- Can be computationally expensive

### Implicit Models

Don't explicitly model p(x), learn to sample:

```
z ~ p(z) → G_θ(z) → x
```

**Examples**: GANs

**Pros**:
- Fast sampling
- High-quality samples

**Cons**:
- No likelihood evaluation
- Training instability
- Mode collapse

### Comparison

```
Likelihood-Based:           Implicit:
    
p(x) = ...                 z → [G] → x
  ↓                           ↓
max log p(x)               Adversarial training
  ↓                           ↓
Can evaluate p(x)          Cannot evaluate p(x)
```

---

## 6. Key Challenges in Generative Modeling

### 1. Mode Coverage

**Problem**: Model may miss modes of the data distribution.

```
True Distribution:         Learned Distribution:
    
  ●●●        ●●●            ●●●        
  ●●●        ●●●            ●●●        ✗
  ●●●        ●●●            ●●●        
  
  Mode 1     Mode 2         Mode 1     Missing!
```

### 2. Sample Quality vs Diversity

**Trade-off**: High quality samples vs covering all modes

```
High Quality, Low Diversity:
  ●●●●●●●
  ●●●●●●●  (all similar)
  ●●●●●●●

Low Quality, High Diversity:
  ●  ●  ●
  ●  ●  ●  (varied but noisy)
  ●  ●  ●
```

### 3. Computational Cost

**Training**: Can take days/weeks on GPUs
**Sampling**: May require many iterations

### 4. Evaluation

**Problem**: How do we measure quality?

---

## 7. Evaluation Metrics

### Inception Score (IS)

Measures quality and diversity:

```
IS = exp(E_x[D_KL(p(y|x) ‖ p(y))])
```

**Higher is better**

### Fréchet Inception Distance (FID)

Measures similarity to real data:

```
FID = ‖μ_real - μ_gen‖² + Tr(Σ_real + Σ_gen - 2√(Σ_real Σ_gen))
```

**Lower is better**

### Precision and Recall

- **Precision**: Quality (are generated samples realistic?)
- **Recall**: Coverage (do we cover all modes?)

### Likelihood

For likelihood-based models:
```
E_test[log p_θ(x)]
```

**Higher is better** (but doesn't always correlate with sample quality!)

---

## 8. The Generative Modeling Pipeline

### Training Phase

```
1. Collect Data
   ↓
2. Define Model Architecture
   ↓
3. Define Training Objective
   ↓
4. Optimize Parameters
   ↓
5. Evaluate
```

### Sampling Phase

```
1. Sample Noise: z ~ p(z)
   ↓
2. Transform: x = G(z)
   ↓
3. (Optional) Refine
   ↓
4. Output Sample
```

---

## 9. Types of Generative Models

### By Training Objective

```
┌─────────────────────────────────────────┐
│                                         │
│  Maximum Likelihood                     │
│  ├── Autoregressive                     │
│  ├── VAE                                │
│  ├── Flow                               │
│  └── Diffusion                          │
│                                         │
│  Adversarial                            │
│  └── GAN                                │
│                                         │
│  Score-Based                            │
│  └── Energy-Based Models                │
│                                         │
└─────────────────────────────────────────┘
```

### By Architecture

```
┌─────────────────────────────────────────┐
│                                         │
│  Sequential                             │
│  └── Autoregressive (PixelCNN, GPT)    │
│                                         │
│  Latent Variable                        │
│  ├── VAE                                │
│  └── Diffusion                          │
│                                         │
│  Invertible                             │
│  └── Normalizing Flow                   │
│                                         │
│  Adversarial                            │
│  └── GAN                                │
│                                         │
└─────────────────────────────────────────┘
```

---

## 10. Connection to Diffusion Models

### Where Diffusion Fits

Diffusion models are **likelihood-based** models that:
1. Use a **latent variable** formulation
2. Employ a **Markov chain** structure
3. Can be viewed from **three perspectives**

### Key Advantages

1. **Stable Training**: No adversarial training
2. **High Quality**: State-of-the-art sample quality
3. **Mode Coverage**: Better than GANs
4. **Flexible**: Can be conditional or unconditional

### The Diffusion Approach

```
Forward Process:
x₀ → x₁ → x₂ → ... → x_T
│    │    │         │
Clean  Noisy  More   Pure
Data   Data   Noisy  Noise

Reverse Process:
x_T → x_{T-1} → ... → x₁ → x₀
│     │              │     │
Noise  Denoise      Denoise Clean
                            Sample
```

---

## 11. Historical Context

### Evolution of Generative Models

```
2013: VAE
  ↓ (stable but blurry)
  
2014: GAN
  ↓ (sharp but unstable)
  
2015: PixelCNN
  ↓ (slow sampling)
  
2017: Normalizing Flows
  ↓ (exact likelihood)
  
2020: DDPM
  ↓ (high quality + stable)
  
2022: Stable Diffusion
  ↓ (practical applications)
  
2023: Flow Matching
  ↓ (faster training)
```

---

## 12. Practical Considerations

### Choosing a Model

| Use Case | Recommended Model |
|----------|------------------|
| High-quality images | Diffusion, GAN |
| Fast sampling | GAN, Flow |
| Exact likelihood | Autoregressive, Flow |
| Stable training | VAE, Diffusion |
| Text generation | Autoregressive |
| Conditional generation | Diffusion, GAN |

### Computational Requirements

```
Training:
- VAE: Moderate (hours)
- GAN: Moderate (hours-days)
- Diffusion: High (days)
- Autoregressive: High (days-weeks)

Sampling:
- VAE: Fast (milliseconds)
- GAN: Fast (milliseconds)
- Flow: Fast (milliseconds)
- Diffusion: Slow* (seconds)
- Autoregressive: Slow (seconds)

*Recent advances have improved this significantly
```

---

## 13. Current State of the Art

### Image Generation

- **Stable Diffusion**: Text-to-image
- **Midjourney**: Artistic images
- **DALL-E 3**: High-quality, controllable

### Text Generation

- **GPT-4**: General text
- **Claude**: Conversational AI
- **Llama**: Open-source

### Video Generation

- **Sora**: Text-to-video
- **Gen-2**: Video editing
- **Pika**: Animation

### Audio Generation

- **MusicGen**: Music generation
- **AudioLDM**: Text-to-audio
- **Bark**: Speech synthesis

---

## Summary

Key concepts:
1. **Generative modeling** learns p(x) to generate new samples
2. **Likelihood-based** vs **implicit** models
3. **Key challenges**: mode coverage, quality vs diversity, evaluation
4. **Diffusion models** are likelihood-based with stable training
5. **Applications** span images, text, audio, video, and science

---

## Exercises

1. **Conceptual**: Explain the difference between discriminative and generative models
2. **Analysis**: Why is the manifold hypothesis important?
3. **Comparison**: Compare likelihood-based and implicit models
4. **Evaluation**: What are the pros/cons of FID vs Inception Score?
5. **Application**: Choose a generative model for a specific use case and justify

---

## Next Steps

Continue to `2_2_taxonomy_of_generative_models.md` to learn about specific model families in detail.
