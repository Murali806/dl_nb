# Three Perspectives on Diffusion Models

## Overview

Diffusion models can be understood from three different theoretical perspectives. Each provides unique insights and leads to different algorithmic choices. Understanding all three perspectives gives you a complete picture of how diffusion models work.

---

## 1. The Three Perspectives

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Perspective 1: Variational (DDPM)             │
│  └── Maximize ELBO with hierarchical VAE       │
│                                                 │
│  Perspective 2: Score-Based (NCSN)             │
│  └── Learn score function ∇_x log p(x)         │
│                                                 │
│  Perspective 3: Flow-Based (CNF)               │
│  └── Learn continuous normalizing flow         │
│                                                 │
│  All three are equivalent!                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 2. Perspective 1: Variational (DDPM)

### Core Idea

View diffusion as a **hierarchical VAE** with:
- Many latent variables: x₁, x₂, ..., x_T
- Markov structure
- Fixed encoder (forward process)
- Learned decoder (reverse process)

### Mathematical Framework

**Forward Process (Encoder)**:
```
q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})

where q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Reverse Process (Decoder)**:
```
p_θ(x_{0:T}) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1} | x_t)

where p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training Objective (ELBO)

```
L = E_q[log p_θ(x_0)] ≥ E_q[log p_θ(x_{0:T})/q(x_{1:T}|x_0)]
```

Simplifies to:
```
L_simple = E_t,x_0,ε[‖ε - ε_θ(x_t, t)‖²]
```

**Intuition**: Predict the noise that was added!

### Visual Representation

```
Forward (Fixed):
x₀ → x₁ → x₂ → ... → x_T
│    │    │         │
Clean  +noise  +noise  Pure noise

Reverse (Learned):
x_T → x_{T-1} → ... → x₁ → x₀
│     │              │     │
Noise  -noise       -noise Clean
```

### Key Papers

- **DDPM** (Ho et al., 2020)
- **Improved DDPM** (Nichol & Dhariwal, 2021)

---

## 3. Perspective 2: Score-Based (NCSN)

### Core Idea

Learn the **score function** (gradient of log density):
```
s_θ(x, t) ≈ ∇_x log p_t(x)
```

Then use **Langevin dynamics** to sample.

### Mathematical Framework

**Score Function**:
```
∇_x log p(x) points toward higher density
```

**Langevin Dynamics**:
```
x_{i+1} = x_i + ε∇_x log p(x_i) + √(2ε) z_i

where z_i ~ N(0, I)
```

**Multiple Noise Levels**:
```
p_σ(x) = ∫ p(y) N(x; y, σ²I) dy

Learn: s_θ(x, σ) ≈ ∇_x log p_σ(x)
```

### Training Objective (Score Matching)

**Denoising Score Matching**:
```
L = E_t,x_0,x_t[‖s_θ(x_t, t) - ∇_{x_t} log q(x_t|x_0)‖²]
```

Which equals:
```
L = E_t,x_0,ε[‖s_θ(x_t, t) - (-ε/σ_t)‖²]
```

**Connection to DDPM**: Same objective, different interpretation!

### Visual Representation

```
Score Function:
    
High density ●●●●●●●
            ↗ ↑ ↖
           ↗  ↑  ↖
          ↗   ↑   ↖
         ●    ↑    ●
    
Arrows point toward data
```

### Key Papers

- **NCSN** (Song & Ermon, 2019)
- **Score-Based SDEs** (Song et al., 2021)

---

## 4. Perspective 3: Flow-Based (CNF)

### Core Idea

Model data generation as a **continuous normalizing flow**:
```
dx/dt = f(x, t)

where f is learned
```

### Mathematical Framework

**Probability Flow ODE**:
```
dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt
```

**Flow Matching**:
```
Learn velocity field v_θ(x, t) to match:
v_θ(x_t, t) ≈ dx_t/dt
```

### Training Objective

**Conditional Flow Matching**:
```
L = E_t,x_0,x_1[‖v_θ(x_t, t) - (x_1 - x_0)‖²]
```

**Optimal Transport**:
```
Minimize transport cost between p_0 and p_1
```

### Visual Representation

```
Flow Field:
    
x_T (noise) ──→ ──→ ──→ x_0 (data)
    │         ↘   ↓   ↙     │
    │           Flow         │
    │         ↗   ↑   ↖     │
    └──────────────────────┘
    
Deterministic paths
```

### Key Papers

- **Neural ODEs** (Chen et al., 2018)
- **Flow Matching** (Lipman et al., 2023)
- **Rectified Flows** (Liu et al., 2022)

---

## 5. Unified View

### All Three Are Equivalent!

```
Variational:  Maximize ELBO
    ↓
Score-Based:  Learn ∇_x log p(x)
    ↓
Flow-Based:   Learn velocity field
    ↓
Same model, different interpretations!
```

### The Connection

**DDPM noise prediction**:
```
ε_θ(x_t, t)
```

**Score function**:
```
s_θ(x_t, t) = -ε_θ(x_t, t) / σ_t
```

**Velocity field**:
```
v_θ(x_t, t) = -½σ_t ε_θ(x_t, t)
```

### Sampling Methods

```
SDE Sampling (Stochastic):
dx = [f(x,t) - g²(t)s_θ(x,t)]dt + g(t)dW

ODE Sampling (Deterministic):
dx = [f(x,t) - ½g²(t)s_θ(x,t)]dt

Flow Sampling (Deterministic):
dx = v_θ(x,t)dt
```

---

## 6. Comparison of Perspectives

| Aspect | Variational | Score-Based | Flow-Based |
|--------|-------------|-------------|------------|
| **View** | Hierarchical VAE | Score matching | Continuous flow |
| **Learn** | Noise predictor | Score function | Velocity field |
| **Training** | ELBO | Score matching | Flow matching |
| **Sampling** | SDE or ODE | Langevin | ODE |
| **Intuition** | Denoise step-by-step | Follow gradient | Follow flow |
| **Papers** | DDPM | NCSN | Flow Matching |

---

## 7. When to Use Each Perspective

### Variational (DDPM)

**Use when**:
- Want discrete-time formulation
- Need clear probabilistic interpretation
- Building on VAE knowledge

**Advantages**:
- Intuitive (just predict noise)
- Well-established theory
- Easy to implement

**Example Code**:
```python
def ddpm_loss(model, x0, t):
    """DDPM training loss"""
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise
    predicted_noise = model(xt, t)
    return F.mse_loss(predicted_noise, noise)
```

---

### Score-Based (NCSN)

**Use when**:
- Want continuous-time formulation
- Need theoretical guarantees
- Working with SDEs

**Advantages**:
- Elegant theory
- Flexible noise schedules
- Connection to physics

**Example Code**:
```python
def score_matching_loss(model, x0, t):
    """Score matching loss"""
    noise = torch.randn_like(x0)
    xt = x0 + sigma[t] * noise
    predicted_score = model(xt, t)
    target_score = -noise / sigma[t]
    return F.mse_loss(predicted_score, target_score)
```

---

### Flow-Based (CNF)

**Use when**:
- Want fastest training
- Need deterministic sampling
- Working with optimal transport

**Advantages**:
- Simpler training objective
- Faster convergence
- Straight paths

**Example Code**:
```python
def flow_matching_loss(model, x0, x1, t):
    """Flow matching loss"""
    xt = (1 - t) * x0 + t * x1
    predicted_velocity = model(xt, t)
    target_velocity = x1 - x0
    return F.mse_loss(predicted_velocity, target_velocity)
```

---

## 8. Practical Implications

### Training

```
Variational:
- Sample t uniformly
- Add noise to x0
- Predict noise

Score-Based:
- Sample t and noise level
- Perturb data
- Predict score

Flow-Based:
- Sample t and endpoints
- Interpolate
- Predict velocity
```

### Sampling

```
Variational (DDPM):
for t in reversed(range(T)):
    xt = denoise_step(xt, t)

Score-Based (Langevin):
for t in reversed(range(T)):
    xt = xt + ε*score(xt, t) + √(2ε)*noise

Flow-Based (ODE):
xt = odeint(velocity, x0, [0, 1])
```

---

## 9. Historical Development

### Timeline

```
2015: Variational perspective (Sohl-Dickstein)
  ↓
2019: Score-based perspective (Song & Ermon)
  ↓
2020: DDPM makes it practical (Ho et al.)
  ↓
2021: Unified SDE framework (Song et al.)
  ↓
2023: Flow matching perspective (Lipman et al.)
  ↓
2024+: Continued unification
```

### Evolution of Understanding

```
Initially: Three separate approaches
    ↓
Realization: They're equivalent!
    ↓
Current: Unified framework
    ↓
Future: Even deeper connections?
```

---

## 10. Choosing Your Perspective

### For Learning

**Start with**: Variational (DDPM)
- Most intuitive
- Easiest to implement
- Best tutorials available

**Then learn**: Score-Based (NCSN)
- Deeper understanding
- Continuous-time view
- Theoretical foundations

**Finally**: Flow-Based (CNF)
- Latest developments
- Fastest methods
- Optimal transport view

### For Research

**Variational**: If working on discrete-time methods
**Score-Based**: If working on theory or SDEs
**Flow-Based**: If working on efficiency or optimal transport

### For Applications

**All perspectives** lead to similar practical implementations!

Choose based on:
- Your background (VAE → Variational, Physics → Score-Based)
- Your goals (Theory → Score-Based, Speed → Flow-Based)
- Your constraints (Discrete → Variational, Continuous → Flow-Based)

---

## 11. The Big Picture

### Unified Framework

```
┌─────────────────────────────────────────────┐
│                                             │
│         Diffusion Models                    │
│                                             │
│  Variational ←→ Score-Based ←→ Flow-Based  │
│      ↓              ↓              ↓        │
│   Predict       Predict        Predict      │
│    Noise         Score        Velocity      │
│      ↓              ↓              ↓        │
│         Same underlying process             │
│                                             │
└─────────────────────────────────────────────┘
```

### Key Insight

**All three perspectives describe the same process**:
- Gradually transforming noise into data
- Using learned neural networks
- With stable, scalable training

The perspective you choose affects:
- How you think about the problem
- How you derive the algorithm
- How you implement the code

But the **end result is equivalent**!

---

## Summary

Key concepts:
1. **Variational**: Hierarchical VAE, predict noise
2. **Score-Based**: Learn score function, Langevin dynamics
3. **Flow-Based**: Continuous flow, velocity field
4. **All equivalent**: Different views of same process
5. **Choose based on**: Background, goals, constraints

---

## Exercises

1. **Derivation**: Show that DDPM and score matching objectives are equivalent
2. **Implementation**: Implement all three perspectives for 2D toy data
3. **Comparison**: Compare sampling quality of SDE vs ODE
4. **Analysis**: When would you prefer flow matching over DDPM?
5. **Unification**: Explain how all three perspectives are connected

---

## Next Steps

You've completed Module 2! You now understand:
- The generative modeling landscape
- Why diffusion models are powerful
- Three theoretical perspectives

Continue to **Module 3: Diffusion Intuition** to build deeper intuition through examples and visualizations.
