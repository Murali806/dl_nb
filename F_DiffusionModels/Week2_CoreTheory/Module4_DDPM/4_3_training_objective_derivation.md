# Training Objective Derivation

## Overview

This module derives the practical training objective for DDPM, showing how the ELBO simplifies to the elegant L_simple loss that's actually used in practice.

---

## 1. From ELBO to Practice

### Recall: ELBO in KL Form

```
ELBO = E_q[log p_θ(x_0|x_1)]
     - D_KL(q(x_T|x_0) ‖ p(x_T))
     - ∑_{t=2}^T E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

### The Main Term

Focus on the denoising term:

```
L_t = E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

This is what we need to minimize!

---

## 2. The True Posterior

### Bayes' Theorem

```
q(x_{t-1}|x_t,x_0) = q(x_t|x_{t-1},x_0) q(x_{t-1}|x_0) / q(x_t|x_0)
```

### Using Markov Property

```
q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1})
```

### Gaussian Form

Both q(x_t|x_{t-1}) and q(x_{t-1}|x_0) are Gaussian, so their product is Gaussian:

```
q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)
```

---

## 3. Computing μ̃_t and β̃_t

### Using Gaussian Identities

For Gaussians:
```
N(x; μ_1, σ_1²) · N(x; μ_2, σ_2²) ∝ N(x; μ, σ²)

where:
1/σ² = 1/σ_1² + 1/σ_2²
μ/σ² = μ_1/σ_1² + μ_2/σ_2²
```

### Applying to Our Case

```
q(x_t|x_{t-1}) = N(x_t; √α_t x_{t-1}, (1-α_t)I)
q(x_{t-1}|x_0) = N(x_{t-1}; √ᾱ_{t-1} x_0, (1-ᾱ_{t-1})I)
```

### Result

```
μ̃_t(x_t, x_0) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

β̃_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) β_t
```

---

## 4. Parameterizing the Model

### Option 1: Predict Mean Directly

```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
```

**Problem**: Network must learn complex function μ_θ

### Option 2: Predict x_0

```
μ_θ(x_t,t) = μ̃_t(x_t, x̂_0)

where x̂_0 = f_θ(x_t, t)
```

**Better**: Leverage known formula for μ̃_t

### Option 3: Predict Noise (DDPM Choice)

```
x̂_0 = (x_t - √(1-ᾱ_t) ε_θ(x_t,t)) / √ᾱ_t

μ_θ(x_t,t) = μ̃_t(x_t, x̂_0)
```

**Best**: Empirically works best!

---

## 5. Deriving L_simple

### Starting Point

```
L_t = E_q[D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))]
```

### KL for Gaussians (Fixed Variance)

```
D_KL = 1/(2σ²) ‖μ̃_t(x_t,x_0) - μ_θ(x_t,t)‖²
```

### Substituting Noise Prediction

```
μ̃_t = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε)
μ_θ = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ)
```

### Simplification

```
‖μ̃_t - μ_θ‖² = ‖1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε) 
                 - 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ)‖²

              = ((1-α_t)/(α_t(1-ᾱ_t))) ‖ε - ε_θ‖²
```

### Weighted Loss

```
L_t = (1/(2σ²)) · ((1-α_t)/(α_t(1-ᾱ_t))) · E[‖ε - ε_θ(x_t,t)‖²]
```

---

## 6. The Simplified Objective

### Removing Weights

Ho et al. (2020) found that **removing the weighting** works better:

```
L_simple = E_t,x_0,ε[‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)‖²]
```

### Why This Works

1. **Simpler**: No complex weighting terms
2. **Uniform**: Equal weight to all timesteps
3. **Empirical**: Better results in practice

### Comparison

```
Full ELBO:
L_t ∝ ((1-α_t)/(α_t(1-ᾱ_t))) ‖ε - ε_θ‖²
     ↑
Complex weighting

Simplified:
L_simple ∝ ‖ε - ε_θ‖²
           ↑
No weighting
```

---

## 7. Complete Training Algorithm

### Algorithm: DDPM Training

```
Input: Dataset D, timesteps T, noise schedule β_1,...,β_T

1. Initialize network ε_θ
2. Compute α_t = 1 - β_t and ᾱ_t = ∏_{s=1}^t α_s

3. Repeat until converged:
   a. Sample x_0 ~ D
   b. Sample t ~ Uniform(1, T)
   c. Sample ε ~ N(0, I)
   d. Compute x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
   e. Compute loss: L = ‖ε - ε_θ(x_t, t)‖²
   f. Update θ: θ ← θ - η ∇_θ L

Output: Trained network ε_θ
```

---

## 8. Practical Implementation

### Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x0, t, alpha_bar):
    """Single training step"""
    batch_size = x0.shape[0]
    
    # Sample noise
    noise = torch.randn_like(x0)
    
    # Forward diffusion
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    # Predict noise
    predicted_noise = model(xt, t)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss

def train_ddpm(model, dataloader, optimizer, num_epochs, T, alpha_bar):
    """Complete training loop"""
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (x0, _) in enumerate(dataloader):
            # Sample random timesteps
            t = torch.randint(0, T, (x0.shape[0],))
            
            # Compute loss
            loss = train_step(model, x0, t, alpha_bar)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader):.4f}")
```

---

## 9. Variance Parameterization

### Fixed Variance (DDPM)

```
Σ_θ(x_t, t) = β̃_t I  (fixed)
```

**Advantage**: Simpler, one less thing to learn

### Learned Variance (Improved DDPM)

```
Σ_θ(x_t, t) = exp(v_θ(x_t, t)) I  (learned)
```

**Advantage**: Slightly better likelihood

### Interpolated Variance

```
Σ_θ = exp(v · log β_t + (1-v) · log β̃_t)

where v = sigmoid(v_θ(x_t, t))
```

**Advantage**: Stable learning, better results

---

## 10. Time Embedding

### Why Time Embedding?

The network needs to know the noise level (timestep t).

### Sinusoidal Embedding

```python
def get_time_embedding(t, dim):
    """Sinusoidal time embedding"""
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

### Usage in Network

```python
class NoisePredictor(nn.Module):
    def __init__(self, img_channels, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        # ... rest of network
    
    def forward(self, x, t):
        # Embed time
        t_emb = get_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Use t_emb in network
        # ...
```

---

## 11. Hyperparameters

### Noise Schedule

```python
# Linear schedule
beta = torch.linspace(0.0001, 0.02, T)

# Cosine schedule (better)
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

### Learning Rate

```python
# Typical values
lr = 2e-4  # Adam optimizer
```

### Batch Size

```python
# Depends on GPU memory
batch_size = 128  # for MNIST
batch_size = 64   # for CIFAR-10
batch_size = 32   # for ImageNet
```

### Number of Timesteps

```python
T = 1000  # Standard DDPM
T = 50    # With DDIM (faster)
```

---

## 12. Loss Weighting Variants

### Standard (L_simple)

```
L = E[‖ε - ε_θ‖²]
```

### SNR Weighting

```
L = E[SNR(t) · ‖ε - ε_θ‖²]

where SNR(t) = ᾱ_t / (1 - ᾱ_t)
```

### Min-SNR Weighting

```
L = E[min(SNR(t), k) · ‖ε - ε_θ‖²]

where k = 5 (typical)
```

---

## 13. Training Tips

### 1. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 2. EMA (Exponential Moving Average)

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] = self.decay * self.shadow[name] + \
                                       (1 - self.decay) * param.data
```

### 3. Learning Rate Schedule

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)
```

---

## 14. Monitoring Training

### Metrics to Track

```python
def evaluate(model, val_loader, T, alpha_bar):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x0, _ in val_loader:
            t = torch.randint(0, T, (x0.shape[0],))
            loss = train_step(model, x0, t, alpha_bar)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
```

### Visualization

```python
def visualize_diffusion(model, x0, T, alpha_bar):
    """Visualize forward and reverse process"""
    model.eval()
    
    # Forward process
    forward_images = [x0]
    for t in range(0, T, T//10):
        xt, _ = forward_sample(x0, t, alpha_bar)
        forward_images.append(xt)
    
    # Reverse process (sampling)
    samples = sample_ddpm(model, x0.shape, T, alpha_bar)
    
    # Plot
    # ...
```

---

## Summary

Key concepts:
1. **True posterior**: q(x_{t-1}|x_t,x_0) is Gaussian
2. **Noise prediction**: Parameterize with ε_θ
3. **L_simple**: Simplified objective works best
4. **Training**: Simple MSE loss on noise
5. **Implementation**: Straightforward PyTorch code

---

## Exercises

1. **Derivation**: Derive L_simple from ELBO step-by-step
2. **Implementation**: Implement complete training loop
3. **Comparison**: Compare different variance parameterizations
4. **Weighting**: Experiment with different loss weightings
5. **Visualization**: Plot loss curves and generated samples

---

## Next Steps

Continue to `4_4_reverse_process_mathematics.md` to understand the sampling procedure and how to generate new samples from the trained model.
