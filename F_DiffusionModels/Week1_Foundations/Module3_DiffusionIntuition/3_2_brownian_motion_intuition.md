# Brownian Motion Intuition

## Overview

Brownian motion is the mathematical foundation of diffusion models. This module builds intuitive understanding of Brownian motion and its connection to the diffusion process.

---

## 1. From Random Walk to Brownian Motion

### Simple Random Walk

Start with a discrete random walk:

```
Step 1: ●
Step 2: ●─●
Step 3: ●─●─●
Step 4: ●─●─●
        │
Step 5: ●─●─●─●
```

Each step: move left or right randomly.

### Taking the Limit

```
Discrete Random Walk:
- Steps: Δx = ±1
- Time: Δt = 1

Continuous Limit:
- Δx → 0
- Δt → 0
- But: (Δx)²/Δt → constant

Result: Brownian Motion!
```

### Visual Comparison

```
Random Walk (Discrete):     Brownian Motion (Continuous):
    
    ●─●─●                       ~~~
   ╱     ╲                     ╱   ╲
  ●       ●                   ╱     ╲
                             ╱       ~~~
```

---

## 2. Properties of Brownian Motion

### Definition

A process W(t) is Brownian motion if:

1. **W(0) = 0**: Starts at origin
2. **Independent increments**: Future independent of past
3. **Gaussian increments**: W(t) - W(s) ~ N(0, t-s)
4. **Continuous paths**: No jumps

### Key Properties

```
Mean:     E[W(t)] = 0
Variance: Var(W(t)) = t
Std Dev:  σ(W(t)) = √t

Uncertainty grows with √time!
```

### Visual Representation

```
Multiple Brownian Paths:
    
W(t)
  │    ╱╲  ╱╲
  │   ╱  ╲╱  ╲╱╲
  │  ╱          ╲  ╱╲
  │ ╱            ╲╱  ╲
  │╱                  ╲
  └────────────────────── t
  
Each path is different but follows same statistics
```

---

## 3. The Scaling Property

### Key Insight

```
If W(t) is Brownian motion, then:

W(ct) / √c is also Brownian motion

where c > 0
```

### Intuition

**Time scaling**: If you speed up time by factor c, you must scale position by √c to maintain Brownian motion properties.

### Example

```
Original:
W(1) ~ N(0, 1)

Speed up 4x:
W(4) ~ N(0, 4) = N(0, 2²)
W(4)/2 ~ N(0, 1)  ← Still Brownian!
```

---

## 4. Non-Differentiability

### The Surprising Fact

Brownian motion is **nowhere differentiable**!

```
dW/dt does not exist

But: dW ~ N(0, dt) makes sense
```

### Why?

The path is so jagged that you can't define a tangent anywhere.

### Visual Intuition

```
Zoom in on any point:
    
Original:     Zoomed 10x:    Zoomed 100x:
   ╱╲            ╱╲╱╲           ╱╲╱╲╱╲
  ╱  ╲          ╱    ╲         ╱      ╲
 ╱    ╲        ╱      ╲       ╱        ╲

Still jagged at every scale!
```

---

## 5. Quadratic Variation

### Definition

The quadratic variation of Brownian motion is:

```
∫₀ᵗ (dW)² = t
```

### Intuition

For smooth functions: ∫(df)² = 0
For Brownian motion: ∫(dW)² = t ≠ 0!

This is because:
```
(dW)² ≈ dt  (not 0!)
```

### Why It Matters

This property is crucial for:
- Itô's lemma
- Stochastic calculus
- Diffusion model theory

---

## 6. Connection to Gaussian Distribution

### Central Limit Theorem

```
Sum of many small random steps → Gaussian

Random Walk:
X_n = ε₁ + ε₂ + ... + εₙ

As n → ∞:
X_n / √n → N(0, 1)
```

### In Continuous Time

```
W(t) = lim_{n→∞} (ε₁ + ε₂ + ... + εₙ) / √n

Result: W(t) ~ N(0, t)
```

### Visual Proof

```
Distribution at different times:
    
t=0.1:  ╱╲
       ╱  ╲
      ╱    ╲

t=1:     ╱‾╲
        ╱   ╲
       ╱     ╲

t=4:       ╱‾‾╲
          ╱    ╲
         ╱      ╲

Wider as t increases!
```

---

## 7. Brownian Motion in Diffusion Models

### Forward Process

Adding noise is like adding Brownian motion:

```
x_t = x_0 + noise

where noise ~ Brownian motion
```

### Mathematical Form

```
dx = f(x,t)dt + g(t)dW
     ↑          ↑
   Drift    Diffusion
```

### Example: DDPM

```
x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

where ε ~ N(0, I) is like W(1)
```

---

## 8. Geometric Brownian Motion

### Definition

```
dS = μS dt + σS dW

where:
- μ: drift rate
- σ: volatility
- S: value (e.g., stock price)
```

### Solution

```
S(t) = S(0) exp((μ - σ²/2)t + σW(t))
```

### Visual Representation

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
  
Always positive, multiplicative noise
```

### Connection to Diffusion

Some diffusion models use geometric Brownian motion for:
- Positive data (e.g., images with pixel values > 0)
- Multiplicative noise

---

## 9. Ornstein-Uhlenbeck Process

### Definition

```
dX = -θX dt + σ dW

where:
- θ > 0: mean reversion rate
- σ: volatility
```

### Properties

1. **Mean reversion**: Pulled toward zero
2. **Stationary**: Long-run distribution is N(0, σ²/(2θ))
3. **Gaussian**: Always Gaussian

### Visual Representation

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
  
Oscillates around zero
```

### Connection to Diffusion

The Variance Preserving (VP) SDE is related:

```
dx = -½β(t)x dt + √β(t) dW
```

This is like OU process with time-varying parameters!

---

## 10. Time Reversal

### Forward Brownian Motion

```
W(t) for t ∈ [0, T]
```

### Reverse Brownian Motion

```
W̃(t) = W(T) - W(T-t)

This is also Brownian motion!
```

### Why It Matters

**Key Insight**: We can reverse diffusion!

```
Forward:  x₀ → x_T  (add noise)
Reverse:  x_T → x₀  (remove noise)

Both involve Brownian motion
```

### The Catch

Reverse requires knowing the **score function**:

```
dx = [f(x,t) - g²(t)∇log p(x,t)]dt + g(t)dW̃
                      ↑
              Need to learn this!
```

---

## 11. Practical Simulation

### Euler-Maruyama Method

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(T, n_steps):
    """Simulate Brownian motion"""
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # Generate increments
    dW = np.random.randn(n_steps) * np.sqrt(dt)
    
    # Cumulative sum
    W = np.concatenate([[0], np.cumsum(dW)])
    
    return t, W

# Simulate multiple paths
T = 1.0
n_steps = 1000
n_paths = 5

plt.figure(figsize=(12, 6))
for _ in range(n_paths):
    t, W = simulate_brownian_motion(T, n_steps)
    plt.plot(t, W, alpha=0.7)

plt.xlabel('Time')
plt.ylabel('W(t)')
plt.title('Brownian Motion Paths')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.show()
```

### Verification

```python
def verify_properties(n_simulations=10000):
    """Verify Brownian motion properties"""
    T = 1.0
    n_steps = 100
    
    # Simulate many final values
    final_values = []
    for _ in range(n_simulations):
        _, W = simulate_brownian_motion(T, n_steps)
        final_values.append(W[-1])
    
    final_values = np.array(final_values)
    
    # Check properties
    print(f"Mean: {np.mean(final_values):.4f} (should be 0)")
    print(f"Variance: {np.var(final_values):.4f} (should be {T})")
    print(f"Std Dev: {np.std(final_values):.4f} (should be {np.sqrt(T)})")
    
    # Plot histogram
    plt.hist(final_values, bins=50, density=True, alpha=0.7)
    
    # Overlay theoretical distribution
    x = np.linspace(-4, 4, 100)
    plt.plot(x, (1/np.sqrt(2*np.pi*T)) * np.exp(-x**2/(2*T)), 
             'r-', linewidth=2, label='N(0,1)')
    plt.xlabel('W(T)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Distribution of W(T)')
    plt.show()

verify_properties()
```

---

## 12. Intuitive Understanding

### Key Insights

1. **Random but structured**: Not just any random process
2. **Gaussian increments**: Each step is Gaussian
3. **Independent increments**: Past doesn't affect future
4. **Continuous paths**: No jumps
5. **Nowhere differentiable**: Infinitely jagged

### Physical Intuition

```
Pollen grain in water:
- Bombarded by water molecules
- Each collision is random
- Net effect: Brownian motion

Image in diffusion:
- Bombarded by noise
- Each noise addition is random
- Net effect: Brownian-like process
```

### Mathematical Beauty

```
Simple definition:
- Start at 0
- Independent Gaussian increments
- Continuous paths

Rich properties:
- Scaling
- Time reversal
- Quadratic variation
- Connection to PDEs
```

---

## 13. Common Misconceptions

### Misconception 1: "Brownian motion is just random noise"

**Reality**: It has specific mathematical properties
- Gaussian increments
- Independent increments
- Specific variance growth (√t)

### Misconception 2: "You can differentiate Brownian motion"

**Reality**: Nowhere differentiable
- dW/dt doesn't exist
- But dW ~ N(0, dt) makes sense

### Misconception 3: "All random walks are Brownian motion"

**Reality**: Brownian motion is the continuous limit
- Need proper scaling
- Need Gaussian increments
- Need independence

---

## 14. Connection to Diffusion Process

### Forward Diffusion

```
Add Brownian motion gradually:

x_t = x_0 + ∫₀ᵗ √β(s) dW(s)
```

### Reverse Diffusion

```
Remove Brownian motion using score:

dx = [f(x,t) - g²(t)∇log p(x,t)]dt + g(t)dW̃
```

### Why Brownian Motion?

1. **Natural**: Physical diffusion uses Brownian motion
2. **Tractable**: Gaussian properties are well-understood
3. **Reversible**: Can be reversed with score function
4. **Universal**: Central limit theorem guarantees it

---

## Summary

Key concepts:
1. **Brownian motion**: Continuous limit of random walk
2. **Properties**: Gaussian increments, independent, continuous
3. **Scaling**: W(ct)/√c is also Brownian
4. **Non-differentiable**: Infinitely jagged
5. **Quadratic variation**: ∫(dW)² = t
6. **Time reversal**: Can be reversed with score
7. **Connection**: Foundation of diffusion models

---

## Exercises

1. **Simulation**: Implement and visualize Brownian motion
2. **Properties**: Verify E[W(t)] = 0 and Var(W(t)) = t empirically
3. **Scaling**: Verify the scaling property numerically
4. **OU Process**: Simulate Ornstein-Uhlenbeck process
5. **Comparison**: Compare random walk to Brownian motion

---

## Next Steps

Continue to `3_3_forward_process_visualization.ipynb` to see how Brownian motion is used in the forward diffusion process with interactive visualizations.
