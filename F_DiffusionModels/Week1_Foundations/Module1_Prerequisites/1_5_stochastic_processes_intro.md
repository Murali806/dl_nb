# Stochastic Processes Introduction

## Overview

Stochastic processes are the foundation for understanding continuous-time diffusion models. This module covers random walks, Brownian motion, and Wiener processes - essential for the SDE framework.

---

## 1. What is a Stochastic Process?

### Definition

A **stochastic process** is a collection of random variables indexed by time:

```
{X(t) : t ∈ T}
```

Where:
- **X(t)**: Random variable at time t
- **T**: Time index set (discrete or continuous)

### Types

**Discrete Time**:
```
X₀, X₁, X₂, ..., Xₙ
```
Example: Daily stock prices

**Continuous Time**:
```
X(t) for t ∈ [0, ∞)
```
Example: Temperature over time

---

## 2. Random Walks

### Simple Random Walk

At each step, move up (+1) or down (-1) with equal probability:

```
X₀ = 0
Xₙ = Xₙ₋₁ + εₙ

where εₙ ∈ {-1, +1} with P(εₙ = 1) = 0.5
```

### Visualization

```
Position
   │
 3 │         ●
 2 │       ●   ●
 1 │     ●       ●
 0 │ ●─●           ●
-1 │                 ●
   └─────────────────── Time
     0 1 2 3 4 5 6 7
```

### Properties

1. **Mean**: E[Xₙ] = 0
2. **Variance**: Var(Xₙ) = n
3. **Standard Deviation**: σ(Xₙ) = √n

**Key Insight**: Uncertainty grows with √time!

### Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def random_walk(n_steps, p=0.5):
    """Simulate simple random walk"""
    steps = np.random.choice([-1, 1], size=n_steps, p=[1-p, p])
    return np.cumsum(steps)

# Simulate multiple walks
n_walks = 5
n_steps = 100

plt.figure(figsize=(10, 6))
for _ in range(n_walks):
    walk = random_walk(n_steps)
    plt.plot(walk, alpha=0.7)

plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Random Walks')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Continuous Random Walk

### Scaling Limit

As we take smaller steps more frequently:

```
Step size: Δx → 0
Time step: Δt → 0
But: (Δx)² / Δt → constant
```

This limit gives us **Brownian motion**!

### Intuition

```
Discrete (Random Walk)     Continuous (Brownian Motion)
        
    ●─●─●─●                    ~~~
   ╱       ╲                  ╱   ╲
  ●         ●                ╱     ╲
                            ╱       ~~~
```

---

## 4. Brownian Motion (Wiener Process)

### Definition

A stochastic process W(t) is **Brownian motion** if:

1. **W(0) = 0** (starts at origin)
2. **Independent increments**: W(t) - W(s) independent of W(s) - W(r) for t > s > r
3. **Gaussian increments**: W(t) - W(s) ~ N(0, t-s)
4. **Continuous paths**: W(t) is continuous in t

### Key Properties

```
Mean:     E[W(t)] = 0
Variance: Var(W(t)) = t
Covariance: Cov(W(s), W(t)) = min(s, t)
```

### Visual Representation

```
W(t)
  │
  │    ╱╲  ╱╲
  │   ╱  ╲╱  ╲╱╲
  │  ╱          ╲  ╱╲
  │ ╱            ╲╱  ╲
  │╱                  ╲
  └────────────────────── t
```

### Simulation

```python
def brownian_motion(T, n_steps):
    """Simulate Brownian motion"""
    dt = T / n_steps
    dW = np.random.randn(n_steps) * np.sqrt(dt)
    W = np.cumsum(dW)
    return np.concatenate([[0], W])

# Simulate
T = 1.0  # Total time
n_steps = 1000
t = np.linspace(0, T, n_steps + 1)
W = brownian_motion(T, n_steps)

# Plot
plt.plot(t, W)
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.title('Brownian Motion')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Properties of Brownian Motion

### Non-Differentiability

Brownian motion is **nowhere differentiable**!

```
dW/dt does not exist

But: dW ~ N(0, dt) makes sense
```

**Intuition**: The path is so jagged that you can't define a tangent.

### Quadratic Variation

```
∫₀ᵗ (dW)² = t
```

This is **not zero** like for smooth functions!

### Scaling Property

```
If W(t) is Brownian motion, then:
W(ct) / √c is also Brownian motion
```

### Time Reversal

```
W̃(t) = W(T) - W(T-t) is also Brownian motion
```

This is important for reverse diffusion!

---

## 6. Geometric Brownian Motion

### Definition

```
dS(t) = μS(t)dt + σS(t)dW(t)
```

Where:
- **μ**: Drift (average growth rate)
- **σ**: Volatility (randomness)

### Solution

```
S(t) = S(0) exp((μ - σ²/2)t + σW(t))
```

### Applications

- **Stock prices**: S&P 500, etc.
- **Option pricing**: Black-Scholes model
- **Population dynamics**: With random fluctuations

### Visualization

```
S(t)
  │
  │         ╱‾‾‾‾╲
  │        ╱      ╲╱╲
  │       ╱           ╲
  │      ╱             ╲╱╲
  │     ╱                 ╲
  │____╱___________________╲__ t
  S(0)
```

---

## 7. Ornstein-Uhlenbeck Process

### Definition

```
dX(t) = -θX(t)dt + σdW(t)
```

Where:
- **θ > 0**: Mean reversion rate
- **σ**: Volatility

### Properties

1. **Mean reversion**: Pulled toward zero
2. **Stationary distribution**: N(0, σ²/(2θ))
3. **Gaussian**: Always Gaussian distributed

### Intuition

```
X(t)
  │
  │  ╱╲    ╱╲
  │ ╱  ╲  ╱  ╲
  │╱    ╲╱    ╲╱╲
  ├─────────────────── 0 (mean)
  │      ╲╱    ╱
  │            ╱
  └──────────────────── t

Pulled back to zero!
```

### Connection to Diffusion

The **Variance Preserving (VP) SDE** in diffusion models is related to OU process:

```
dx = -½β(t)x dt + √β(t) dW
```

---

## 8. Itô's Lemma

### Statement

For a function f(t, X(t)) where dX = μdt + σdW:

```
df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ∂f/∂x dW
```

### Why It Matters

- Generalizes chain rule to stochastic calculus
- Essential for deriving SDEs
- Used in diffusion model theory

### Example: f(X) = X²

```
Given: dX = μdt + σdW
Find: d(X²)

Using Itô's lemma:
d(X²) = 2X dX + ½ · 2 · σ² dt
      = 2X(μdt + σdW) + σ²dt
      = (2μX + σ²)dt + 2σX dW
```

**Note the extra σ² term!** This comes from (dW)² = dt.

---

## 9. Connection to Diffusion Models

### Forward Process as Stochastic Process

The forward diffusion can be written as:

```
dx = f(x, t)dt + g(t)dW
```

This is a **Stochastic Differential Equation (SDE)**!

### Example: DDPM as SDE

```
dx = -½β(t)x dt + √β(t) dW
```

Where:
- **f(x,t) = -½β(t)x**: Drift (pulls toward zero)
- **g(t) = √β(t)**: Diffusion coefficient

### Reverse Process

The reverse SDE is:

```
dx = [f(x,t) - g(t)²∇ₓ log p(x,t)]dt + g(t)dW̄
```

Where ∇ₓ log p(x,t) is the **score function**!

---

## 10. Practical Examples

### Example 1: Simulating Brownian Motion

```python
def simulate_brownian_paths(T, n_steps, n_paths):
    """Simulate multiple Brownian motion paths"""
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        dW = np.random.randn(n_steps) * np.sqrt(dt)
        paths[i, 1:] = np.cumsum(dW)
    
    return t, paths

# Simulate
T = 1.0
n_steps = 1000
n_paths = 10

t, paths = simulate_brownian_paths(T, n_steps, n_paths)

# Plot
plt.figure(figsize=(12, 6))
for path in paths:
    plt.plot(t, path, alpha=0.5)
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.title('Multiple Brownian Motion Paths')
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Ornstein-Uhlenbeck Process

```python
def ornstein_uhlenbeck(T, n_steps, theta, sigma, x0=0):
    """Simulate OU process"""
    dt = T / n_steps
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    for i in range(n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        x[i+1] = x[i] - theta * x[i] * dt + sigma * dW
    
    return x

# Simulate
T = 10.0
n_steps = 1000
theta = 0.5  # Mean reversion rate
sigma = 1.0  # Volatility

t = np.linspace(0, T, n_steps + 1)
x = ornstein_uhlenbeck(T, n_steps, theta, sigma)

# Plot
plt.plot(t, x)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.title('Ornstein-Uhlenbeck Process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 3: Verifying Brownian Motion Properties

```python
def verify_brownian_properties(n_simulations=10000):
    """Verify Brownian motion properties"""
    T = 1.0
    n_steps = 100
    dt = T / n_steps
    
    # Simulate many paths
    final_values = []
    for _ in range(n_simulations):
        dW = np.random.randn(n_steps) * np.sqrt(dt)
        W_T = np.sum(dW)
        final_values.append(W_T)
    
    final_values = np.array(final_values)
    
    # Check properties
    print(f"Theoretical mean: 0")
    print(f"Empirical mean: {np.mean(final_values):.4f}")
    print(f"\nTheoretical variance: {T}")
    print(f"Empirical variance: {np.var(final_values):.4f}")
    
    # Plot histogram
    plt.hist(final_values, bins=50, density=True, alpha=0.7)
    
    # Overlay theoretical distribution
    x = np.linspace(-4, 4, 100)
    plt.plot(x, (1/np.sqrt(2*np.pi*T)) * np.exp(-x**2/(2*T)), 
             'r-', linewidth=2, label='N(0,1)')
    plt.xlabel('W(T)')
    plt.ylabel('Density')
    plt.title('Distribution of W(T)')
    plt.legend()
    plt.show()

verify_brownian_properties()
```

---

## 11. Key Concepts for Diffusion

### 1. Continuous-Time Processes

Diffusion models can be formulated in continuous time using SDEs:
```
dx = f(x,t)dt + g(t)dW
```

### 2. Brownian Motion as Noise

The noise in diffusion is Brownian motion:
```
x_t = x_0 + ∫₀ᵗ f(x_s, s)ds + ∫₀ᵗ g(s)dW_s
```

### 3. Itô's Lemma for Derivations

Used to derive the reverse SDE and score matching objectives.

### 4. Time Reversal

Brownian motion can be reversed, enabling the reverse diffusion process.

---

## Summary

Key concepts:
1. **Random walks** → Brownian motion in the limit
2. **Brownian motion**: Continuous, nowhere differentiable, Gaussian increments
3. **Ornstein-Uhlenbeck**: Mean-reverting process (like VP-SDE)
4. **Itô's lemma**: Chain rule for stochastic calculus
5. **SDEs**: Continuous-time formulation of diffusion models

---

## Exercises

1. **Simulation**: Implement and visualize geometric Brownian motion
2. **Properties**: Verify that Cov(W(s), W(t)) = min(s,t) empirically
3. **OU Process**: Show that OU process has stationary distribution N(0, σ²/(2θ))
4. **Itô's Lemma**: Apply to f(X) = eˣ where dX = μdt + σdW
5. **Scaling**: Verify the scaling property of Brownian motion

---

## Next Steps

Continue to `1_6_differential_equations_refresher.md` to learn about ODEs and SDEs in detail.
