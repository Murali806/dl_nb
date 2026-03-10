# Differential Equations Refresher

## Overview

This module covers Ordinary Differential Equations (ODEs) and Stochastic Differential Equations (SDEs) - essential for understanding the continuous-time formulation of diffusion models and flow matching.

---

## 1. Ordinary Differential Equations (ODEs)

### Definition

An ODE relates a function to its derivatives:

```
dy/dt = f(t, y)
```

Where:
- **y(t)**: Unknown function
- **f(t, y)**: Known function
- **dy/dt**: Derivative (rate of change)

### Example: Exponential Growth

```
dy/dt = ky

Solution: y(t) = y(0)e^(kt)
```

### Visual Representation

```
y(t)
  │
  │         ╱
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  │____╱____________ t
  y(0)
```

---

## 2. First-Order ODEs

### General Form

```
dy/dt = f(t, y)
y(0) = y₀  (initial condition)
```

### Separable ODEs

If f(t, y) = g(t)h(y):

```
dy/h(y) = g(t)dt

∫ dy/h(y) = ∫ g(t)dt
```

### Example: Logistic Growth

```
dy/dt = ry(1 - y/K)

where:
- r: growth rate
- K: carrying capacity
```

**Solution**:
```
y(t) = K / (1 + ((K-y₀)/y₀)e^(-rt))
```

---

## 3. Systems of ODEs

### Definition

Multiple coupled equations:

```
dx/dt = f(x, y, t)
dy/dt = g(x, y, t)
```

### Example: Predator-Prey (Lotka-Volterra)

```
dx/dt = αx - βxy  (prey)
dy/dt = δxy - γy  (predator)
```

### Vector Form

```
d/dt [x] = [f(x,y,t)]
     [y]   [g(x,y,t)]

Or: dy/dt = F(y, t)
```

---

## 4. Numerical Methods for ODEs

### Euler's Method

Simplest numerical solver:

```
y_{n+1} = y_n + h·f(t_n, y_n)
```

Where h is the step size.

### Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, t_span, n_steps):
    """
    Solve dy/dt = f(t, y) using Euler's method
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = (t_span[1] - t_span[0]) / n_steps
    
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        y[i+1] = y[i] + h * f(t[i], y[i])
    
    return t, y

# Example: dy/dt = -y
f = lambda t, y: -y
t, y = euler_method(f, y0=1.0, t_span=(0, 5), n_steps=100)

plt.plot(t, y, label='Numerical')
plt.plot(t, np.exp(-t), '--', label='Analytical')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Euler Method vs Analytical Solution')
plt.show()
```

### Runge-Kutta Methods

More accurate than Euler:

**RK4 (4th order)**:
```
k₁ = f(t_n, y_n)
k₂ = f(t_n + h/2, y_n + h·k₁/2)
k₃ = f(t_n + h/2, y_n + h·k₂/2)
k₄ = f(t_n + h, y_n + h·k₃)

y_{n+1} = y_n + h/6(k₁ + 2k₂ + 2k₃ + k₄)
```

---

## 5. Stochastic Differential Equations (SDEs)

### Definition

An SDE adds random noise to an ODE:

```
dX(t) = f(X, t)dt + g(X, t)dW(t)
```

Where:
- **f(X, t)dt**: Drift term (deterministic)
- **g(X, t)dW(t)**: Diffusion term (stochastic)
- **dW(t)**: Brownian motion increment

### Intuition

```
ODE:  dy/dt = f(y, t)           Deterministic
      ↓
SDE:  dX = f(X, t)dt + g(X, t)dW   Add randomness
```

### Visual Comparison

```
ODE Solution:              SDE Solution:
    
    ╱‾‾‾‾╲                   ╱╲╱╲╱‾╲
   ╱      ╲                 ╱      ╲╱╲
  ╱        ╲               ╱          ╲
 ╱          ╲             ╱            ╲╱
╱            ╲           ╱                ╲
Smooth, deterministic    Noisy, random
```

---

## 6. Itô vs Stratonovich SDEs

### Itô Interpretation

```
dX = f(X, t)dt + g(X, t)dW
```

**Properties**:
- Martingale property
- Easier to work with mathematically
- Used in finance and diffusion models

### Stratonovich Interpretation

```
dX = f(X, t)dt + g(X, t)∘dW
```

**Properties**:
- Chain rule works as in calculus
- More intuitive physically
- Used in physics

### Conversion

```
Stratonovich → Itô:
dX = [f - ½g(∂g/∂x)]dt + g dW
```

---

## 7. Solving SDEs: Euler-Maruyama Method

### Algorithm

```
X_{n+1} = X_n + f(X_n, t_n)Δt + g(X_n, t_n)ΔW_n
```

Where ΔW_n ~ N(0, Δt)

### Code Implementation

```python
def euler_maruyama(f, g, X0, T, n_steps):
    """
    Solve dX = f(X,t)dt + g(X,t)dW using Euler-Maruyama
    
    Parameters:
    - f: drift function
    - g: diffusion function
    - X0: initial condition
    - T: final time
    - n_steps: number of steps
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros(n_steps + 1)
    X[0] = X0
    
    for i in range(n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        X[i+1] = X[i] + f(X[i], t[i]) * dt + g(X[i], t[i]) * dW
    
    return t, X

# Example: Ornstein-Uhlenbeck process
# dX = -θX dt + σ dW
theta = 0.5
sigma = 1.0

f = lambda X, t: -theta * X
g = lambda X, t: sigma

t, X = euler_maruyama(f, g, X0=1.0, T=10.0, n_steps=1000)

plt.plot(t, X)
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.title('Ornstein-Uhlenbeck Process (Euler-Maruyama)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. Fokker-Planck Equation

### Definition

The Fokker-Planck equation describes how the probability density evolves:

```
∂p/∂t = -∂/∂x[f(x,t)p] + ½∂²/∂x²[g²(x,t)p]
```

For SDE: dX = f(X,t)dt + g(X,t)dW

### Intuition

```
∂p/∂t = -∇·(drift) + ½∇²(diffusion)
```

- **Drift term**: Moves probability mass
- **Diffusion term**: Spreads probability mass

### Connection to Diffusion Models

The forward diffusion process satisfies a Fokker-Planck equation:

```
∂p_t/∂t = -∇·[f(x,t)p_t] + ½∇²[g²(t)p_t]
```

---

## 9. Reverse-Time SDEs

### Theorem (Anderson, 1982)

If the forward SDE is:
```
dx = f(x,t)dt + g(t)dW
```

Then the reverse SDE is:
```
dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dW̄
```

Where:
- **∇_x log p_t(x)**: Score function
- **dW̄**: Reverse-time Brownian motion

### Key Insight

To reverse diffusion, we need the **score function**!

```
Forward:  x₀ → x_T  (add noise)
          ↓
Reverse:  x_T → x₀  (remove noise using score)
```

---

## 10. Probability Flow ODE

### Definition

For an SDE:
```
dx = f(x,t)dt + g(t)dW
```

The probability flow ODE is:
```
dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt
```

### Properties

1. **Deterministic**: No randomness
2. **Same marginals**: p_t(x) is the same as SDE
3. **Faster sampling**: Can use ODE solvers

### Visual Comparison

```
SDE Sampling:              ODE Sampling:
    
  ╱╲╱╲╱‾╲                   ╱‾‾‾╲
 ╱      ╲╱╲                ╱     ╲
╱          ╲              ╱       ╲
Random path              Smooth path
```

---

## 11. Connection to Diffusion Models

### Forward Process (VP-SDE)

```
dx = -½β(t)x dt + √β(t) dW
```

Where:
- **f(x,t) = -½β(t)x**: Drift toward zero
- **g(t) = √β(t)**: Time-dependent diffusion

### Reverse Process

```
dx = [-½β(t)x - β(t)∇_x log p_t(x)]dt + √β(t) dW̄
```

The neural network learns ∇_x log p_t(x)!

### Probability Flow ODE

```
dx = [-½β(t)x - ½β(t)∇_x log p_t(x)]dt
```

Used in DDIM and fast samplers.

---

## 12. Practical Examples

### Example 1: Comparing ODE and SDE

```python
def compare_ode_sde():
    """Compare ODE and SDE solutions"""
    # Parameters
    T = 5.0
    n_steps = 1000
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # ODE: dy/dt = -y
    y_ode = np.zeros(n_steps + 1)
    y_ode[0] = 1.0
    for i in range(n_steps):
        y_ode[i+1] = y_ode[i] - y_ode[i] * dt
    
    # SDE: dX = -X dt + 0.5 dW
    X_sde = np.zeros(n_steps + 1)
    X_sde[0] = 1.0
    for i in range(n_steps):
        dW = np.random.randn() * np.sqrt(dt)
        X_sde[i+1] = X_sde[i] - X_sde[i] * dt + 0.5 * dW
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, y_ode, label='ODE')
    plt.plot(t, np.exp(-t), '--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.title('ODE Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(t, X_sde, label='SDE')
    plt.plot(t, np.exp(-t), '--', label='ODE mean')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.title('SDE Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_ode_sde()
```

### Example 2: Simulating Forward Diffusion

```python
def forward_diffusion_sde(x0, T, n_steps, beta_schedule='linear'):
    """
    Simulate forward diffusion as SDE
    dx = -0.5*β(t)*x dt + sqrt(β(t)) dW
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # Beta schedule
    if beta_schedule == 'linear':
        beta = np.linspace(0.0001, 0.02, n_steps + 1)
    else:
        beta = np.ones(n_steps + 1) * 0.01
    
    # Simulate
    x = np.zeros((n_steps + 1, *x0.shape))
    x[0] = x0
    
    for i in range(n_steps):
        dW = np.random.randn(*x0.shape) * np.sqrt(dt)
        drift = -0.5 * beta[i] * x[i] * dt
        diffusion = np.sqrt(beta[i]) * dW
        x[i+1] = x[i] + drift + diffusion
    
    return t, x, beta

# Simulate 2D point
x0 = np.array([2.0, 2.0])
t, x, beta = forward_diffusion_sde(x0, T=1.0, n_steps=100)

# Plot trajectory
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x[:, 0], x[:, 1], 'o-', alpha=0.5, markersize=2)
plt.plot(x0[0], x0[1], 'ro', markersize=10, label='Start')
plt.plot(x[-1, 0], x[-1, 1], 'go', markersize=10, label='End')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Forward Diffusion Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(t, np.linalg.norm(x, axis=1))
plt.xlabel('Time')
plt.ylabel('||x(t)||')
plt.title('Distance from Origin')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 3: ODE vs SDE Sampling

```python
def compare_sampling_methods():
    """Compare ODE and SDE sampling"""
    # Simulate score function (simplified)
    def score_fn(x, t):
        # Simplified: score points toward origin
        return -x / (1 - t + 0.1)
    
    T = 1.0
    n_steps = 100
    dt = T / n_steps
    
    # Start from noise
    x0 = np.random.randn(2) * 2
    
    # SDE sampling
    x_sde = np.zeros((n_steps + 1, 2))
    x_sde[0] = x0
    for i in range(n_steps):
        t = i * dt
        score = score_fn(x_sde[i], t)
        dW = np.random.randn(2) * np.sqrt(dt)
        x_sde[i+1] = x_sde[i] + score * dt + 0.5 * dW
    
    # ODE sampling (deterministic)
    x_ode = np.zeros((n_steps + 1, 2))
    x_ode[0] = x0
    for i in range(n_steps):
        t = i * dt
        score = score_fn(x_ode[i], t)
        x_ode[i+1] = x_ode[i] + score * dt
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_sde[:, 0], x_sde[:, 1], 'o-', alpha=0.5, label='SDE')
    plt.plot(x_sde[0, 0], x_sde[0, 1], 'ro', markersize=10)
    plt.plot(x_sde[-1, 0], x_sde[-1, 1], 'go', markersize=10)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('SDE Sampling (Stochastic)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_ode[:, 0], x_ode[:, 1], 'o-', alpha=0.5, label='ODE')
    plt.plot(x_ode[0, 0], x_ode[0, 1], 'ro', markersize=10)
    plt.plot(x_ode[-1, 0], x_ode[-1, 1], 'go', markersize=10)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('ODE Sampling (Deterministic)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

compare_sampling_methods()
```

---

## 13. Key Concepts for Diffusion

### 1. Forward Process is an SDE

```
dx = f(x,t)dt + g(t)dW
```

### 2. Reverse Process Uses Score

```
dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dW̄
```

### 3. Probability Flow ODE

```
dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt
```

Enables deterministic sampling!

### 4. Numerical Methods

- **Euler-Maruyama**: For SDEs
- **RK4, DPM-Solver**: For ODEs
- **Trade-off**: Speed vs quality

---

## Summary

Key concepts:
1. **ODEs**: Deterministic evolution equations
2. **SDEs**: ODEs + random noise
3. **Euler-Maruyama**: Numerical method for SDEs
4. **Fokker-Planck**: Evolution of probability density
5. **Reverse-time SDE**: Requires score function
6. **Probability flow ODE**: Deterministic alternative

---

## Exercises

1. **ODE Solver**: Implement RK4 method and compare with Euler
2. **SDE Simulation**: Simulate geometric Brownian motion
3. **Convergence**: Study how step size affects accuracy
4. **Forward Diffusion**: Implement VP-SDE from scratch
5. **ODE vs SDE**: Compare sampling quality and speed

---

## Next Steps

You've completed Module 1! Continue to `Week1_Foundations/Module2_GenerativeModels/` to learn about the landscape of generative models.
