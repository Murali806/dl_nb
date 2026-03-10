# Markov Chains Basics

## Overview

Markov chains are fundamental to understanding diffusion models. The forward diffusion process is a Markov chain, and understanding this structure is key to deriving the training objective.

---

## 1. What is a Markov Chain?

### Definition

A **Markov chain** is a sequence of random variables where the future depends only on the present, not the past.

**Markov Property**:
```
P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)
```

**In words**: "The future is independent of the past given the present."

### Visual Representation

```
X_0 ──→ X_1 ──→ X_2 ──→ X_3 ──→ ... ──→ X_t
 │       │       │       │              │
 └───────┴───────┴───────┴──────────────┘
         All history is summarized in X_t
```

### Intuitive Example: Weather

```
Today's weather depends only on yesterday's weather,
not on the weather from last week.

Sunny ──0.7──→ Sunny
  │            ↑
  └──0.3──→ Rainy
            ↓
          Rainy ──0.6──→ Rainy
```

---

## 2. Transition Probabilities

### Transition Matrix

For a discrete Markov chain with states {1, 2, ..., n}:

```
     ┌                    ┐
     │ p₁₁  p₁₂  ...  p₁ₙ │
P =  │ p₂₁  p₂₂  ...  p₂ₙ │
     │  ⋮    ⋮    ⋱    ⋮  │
     │ pₙ₁  pₙ₂  ...  pₙₙ │
     └                    ┘
```

Where p_{ij} = P(X_{t+1} = j | X_t = i)

### Properties

1. **Non-negative**: p_{ij} ≥ 0
2. **Row sums to 1**: Σⱼ p_{ij} = 1 (probability distribution)

### Example: Weather Model

```
         Sunny  Rainy
Sunny  [  0.7    0.3  ]
Rainy  [  0.4    0.6  ]
```

**Interpretation**:
- If sunny today, 70% chance sunny tomorrow
- If rainy today, 60% chance rainy tomorrow

---

## 3. Multi-Step Transitions

### n-Step Transition Probability

```
P(X_{t+n} = j | X_t = i) = (P^n)_{ij}
```

Where P^n is the matrix P multiplied by itself n times.

### Example Calculation

```python
import numpy as np

# Transition matrix
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# 2-step transition
P2 = P @ P
print("2-step transitions:")
print(P2)

# 10-step transition
P10 = np.linalg.matrix_power(P, 10)
print("\n10-step transitions:")
print(P10)
```

### Visualization

```
Step 0:  Sunny (100%)
         │
Step 1:  Sunny (70%)  Rainy (30%)
         │              │
Step 2:  Sunny (61%)  Rainy (39%)
         │              │
Step 3:  Sunny (58%)  Rainy (42%)
         ⋮              ⋮
Step ∞:  Sunny (57%)  Rainy (43%)  ← Stationary!
```

---

## 4. Stationary Distribution

### Definition

A distribution π is **stationary** if:

```
π = πP
```

In other words, applying the transition doesn't change the distribution.

### Finding Stationary Distribution

For the weather example:
```
[π_s, π_r] = [π_s, π_r] × [[0.7, 0.3],
                            [0.4, 0.6]]
```

This gives:
```
π_s = 0.7π_s + 0.4π_r
π_r = 0.3π_s + 0.6π_r
π_s + π_r = 1
```

**Solution**: π_s = 4/7 ≈ 0.57, π_r = 3/7 ≈ 0.43

### Code Implementation

```python
def find_stationary(P, tol=1e-10):
    """Find stationary distribution using power method"""
    n = P.shape[0]
    π = np.ones(n) / n  # Start with uniform
    
    for _ in range(1000):
        π_new = π @ P
        if np.allclose(π, π_new, atol=tol):
            break
        π = π_new
    
    return π

π = find_stationary(P)
print(f"Stationary distribution: {π}")
```

---

## 5. Continuous-State Markov Chains

### Transition Kernel

For continuous states (like images), we use a **transition kernel**:

```
p(x_{t+1} | x_t)
```

This is a conditional probability density function.

### Example: Gaussian Random Walk

```
X_{t+1} = X_t + ε_t

where ε_t ~ N(0, σ²)
```

**Transition kernel**:
```
p(x_{t+1} | x_t) = N(x_{t+1}; x_t, σ²)
```

### Visual Representation

```
     x_t
      │
      ├──→ x_{t+1} (most likely)
      │
      ├──→ x_{t+1} (less likely)
      │
      └──→ x_{t+1} (least likely)
```

---

## 6. Time-Homogeneous vs Time-Inhomogeneous

### Time-Homogeneous

Transition probabilities don't change over time:
```
P(X_{t+1} | X_t) = P(X_1 | X_0)  for all t
```

### Time-Inhomogeneous

Transition probabilities depend on time:
```
P(X_{t+1} | X_t) = P_t(X_{t+1} | X_t)
```

### In Diffusion Models

The forward process is **time-inhomogeneous**:
```
q(x_t | x_{t-1}) depends on t
```

Different noise levels at different timesteps!

---

## 7. Joint and Marginal Distributions

### Joint Distribution

For a Markov chain:
```
p(x_0, x_1, ..., x_T) = p(x_0) ∏_{t=1}^T p(x_t | x_{t-1})
```

**Key insight**: The joint factorizes due to Markov property!

### Marginal Distribution

```
p(x_t) = ∫ p(x_t | x_{t-1}) p(x_{t-1}) dx_{t-1}
```

Recursively:
```
p(x_t) = ∫...∫ p(x_0) ∏_{s=1}^t p(x_s | x_{s-1}) dx_0...dx_{t-1}
```

---

## 8. Connection to Diffusion Models

### Forward Process

The forward diffusion is a Markov chain:

```
q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})
```

**Visual**:
```
x_0 ──→ x_1 ──→ x_2 ──→ ... ──→ x_T
│       │       │              │
Clean   Slightly  More        Pure
Image   Noisy     Noisy       Noise
```

### Each Transition

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

This is a **Gaussian transition kernel**.

### Why Markov Property Matters

1. **Simplifies joint distribution**:
   ```
   q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})
   ```

2. **Enables closed-form q(x_t | x_0)**:
   ```
   q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)
   ```

3. **Makes training tractable**:
   - Can sample any timestep directly
   - Don't need to simulate entire chain

---

## 9. Reversibility and Detailed Balance

### Reversible Markov Chain

A chain is **reversible** if:
```
π(i) p_{ij} = π(j) p_{ji}
```

**Detailed balance**: Flow from i to j equals flow from j to i.

### Visual Representation

```
State i ⇄ State j
   π(i)p_{ij} = π(j)p_{ji}
```

### In Diffusion Models

The reverse process aims to reverse the forward Markov chain:
```
Forward:  x_0 → x_1 → ... → x_T
Reverse:  x_T → x_{T-1} → ... → x_0
```

But the reverse is **not** simply running forward backward!

---

## 10. Practical Examples

### Example 1: Simple Random Walk

```python
import numpy as np
import matplotlib.pyplot as plt

def random_walk(n_steps, σ=1.0):
    """Simulate 1D random walk"""
    x = np.zeros(n_steps)
    x[0] = 0
    
    for t in range(1, n_steps):
        x[t] = x[t-1] + np.random.randn() * σ
    
    return x

# Simulate
walk = random_walk(1000, σ=0.5)

# Plot
plt.plot(walk)
plt.xlabel('Time step')
plt.ylabel('Position')
plt.title('Random Walk (Markov Chain)')
plt.show()
```

### Example 2: Discrete Markov Chain Simulation

```python
def simulate_markov_chain(P, initial_state, n_steps):
    """Simulate discrete Markov chain"""
    n_states = P.shape[0]
    states = np.zeros(n_steps, dtype=int)
    states[0] = initial_state
    
    for t in range(1, n_steps):
        # Sample next state based on current state
        states[t] = np.random.choice(n_states, p=P[states[t-1]])
    
    return states

# Weather model
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Simulate starting from sunny
weather = simulate_markov_chain(P, initial_state=0, n_steps=100)

# Visualize
plt.plot(weather)
plt.yticks([0, 1], ['Sunny', 'Rainy'])
plt.xlabel('Day')
plt.title('Weather Simulation')
plt.show()
```

### Example 3: Gaussian Markov Chain (Diffusion-like)

```python
def gaussian_markov_chain(x0, betas, n_steps):
    """Simulate Gaussian Markov chain (like forward diffusion)"""
    x = np.zeros((n_steps, *x0.shape))
    x[0] = x0
    
    for t in range(1, n_steps):
        # Transition: x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε
        noise = np.random.randn(*x0.shape)
        x[t] = np.sqrt(1 - betas[t]) * x[t-1] + np.sqrt(betas[t]) * noise
    
    return x

# Simulate
x0 = np.array([1.0, 2.0])  # 2D starting point
betas = np.linspace(0.01, 0.1, 100)
trajectory = gaussian_markov_chain(x0, betas, 100)

# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', alpha=0.5)
plt.plot(x0[0], x0[1], 'ro', markersize=10, label='Start')
plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=10, label='End')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.legend()
plt.title('Gaussian Markov Chain Trajectory')
plt.show()
```

---

## 11. Key Properties for Diffusion

### 1. Markov Property

```
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
```

**Benefit**: Simplifies joint distribution.

### 2. Closed-Form Marginals

For Gaussian transitions:
```
q(x_t | x_0) can be computed directly!
```

**Benefit**: No need to simulate entire chain during training.

### 3. Factorization

```
q(x_{1:T} | x_0) = ∏_{t=1}^T q(x_t | x_{t-1})
```

**Benefit**: Can optimize each transition independently.

### 4. Time-Inhomogeneity

```
β_t increases with t
```

**Benefit**: Gradual noise addition, better training.

---

## 12. Chapman-Kolmogorov Equation

### Statement

For any s < t:
```
p(x_t | x_s) = ∫ p(x_t | x_r) p(x_r | x_s) dx_r
```

for any s < r < t.

### Intuition

To go from s to t, we can go through any intermediate point r.

### In Diffusion

```
q(x_t | x_0) = ∫ q(x_t | x_s) q(x_s | x_0) dx_s
```

This is used to derive the closed-form q(x_t | x_0)!

---

## Summary

Key concepts:
1. **Markov property**: Future independent of past given present
2. **Transition probabilities**: p(x_{t+1} | x_t)
3. **Stationary distribution**: Long-run behavior
4. **Joint factorization**: p(x_{0:T}) = p(x_0) ∏ p(x_t | x_{t-1})
5. **Forward diffusion is a Markov chain**
6. **Closed-form marginals** enable efficient training

---

## Exercises

1. **Weather Model**: Compute 5-step transition probabilities
2. **Stationary Distribution**: Find π for a 3-state Markov chain
3. **Random Walk**: Simulate and analyze variance growth
4. **Gaussian Chain**: Implement forward diffusion for 1D signal
5. **Chapman-Kolmogorov**: Verify the equation numerically

---

## Next Steps

Continue to `1_5_stochastic_processes_intro.md` to learn about continuous-time processes and Brownian motion.
