# Taxonomy of Generative Models

## Overview

This module provides a comprehensive taxonomy of generative models, explaining each family's approach, strengths, and weaknesses. Understanding this landscape is crucial for appreciating where diffusion models fit.

---

## 1. Complete Taxonomy

```
Generative Models
│
├── Likelihood-Based (Explicit Density)
│   ├── Tractable
│   │   ├── Autoregressive Models
│   │   │   ├── PixelCNN
│   │   │   ├── WaveNet
│   │   │   └── GPT (Transformers)
│   │   │
│   │   └── Normalizing Flows
│   │       ├── RealNVP
│   │       ├── Glow
│   │       └── Neural ODEs
│   │
│   └── Approximate
│       ├── Variational Autoencoders (VAEs)
│       │   ├── Standard VAE
│       │   ├── β-VAE
│       │   └── VQ-VAE
│       │
│       └── Diffusion Models
│           ├── DDPM (Variational)
│           ├── NCSN (Score-Based)
│           └── Flow Matching
│
├── Implicit (No Explicit Density)
│   └── Generative Adversarial Networks (GANs)
│       ├── DCGAN
│       ├── StyleGAN
│       └── Conditional GAN
│
└── Energy-Based Models
    ├── Boltzmann Machines
    └── Score-Based Models
```

---

## 2. Autoregressive Models

### Core Idea

Model the joint distribution as a product of conditionals:

```
p(x) = p(x₁) × p(x₂|x₁) × p(x₃|x₁,x₂) × ... × p(xₙ|x₁,...,xₙ₋₁)
```

### Visual Representation

```
Generation Process:
    
x₁ → x₂ → x₃ → x₄ → ... → xₙ
│    │    │    │         │
Sample each pixel/token sequentially
```

### Examples

**PixelCNN** (Images):
```
p(image) = ∏ p(pixel_i | previous pixels)
```

**GPT** (Text):
```
p(text) = ∏ p(token_i | previous tokens)
```

### Pros and Cons

**Pros**:
- ✅ Exact likelihood
- ✅ Stable training
- ✅ High quality

**Cons**:
- ❌ Slow sampling (sequential)
- ❌ No latent representation
- ❌ Hard to parallelize

### Code Example

```python
class SimpleAutoregressive(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: [seq_len, batch]
        embedded = self.embedding(x)
        hidden, _ = self.rnn(embedded)
        logits = self.output(hidden)
        return logits
    
    def sample(self, start_token, max_len):
        """Generate sequence autoregressively"""
        x = start_token
        sequence = [x]
        
        for _ in range(max_len):
            logits = self.forward(x)
            probs = F.softmax(logits[-1], dim=-1)
            x_next = torch.multinomial(probs, 1)
            sequence.append(x_next)
            x = torch.cat([x, x_next])
        
        return sequence
```

---

## 3. Variational Autoencoders (VAEs)

### Core Idea

Learn a latent variable model with variational inference:

```
Encoder:  q_φ(z|x) ≈ p(z|x)
Decoder:  p_θ(x|z)
```

### Architecture

```
    Encoder          Latent          Decoder
    
x → [Neural Net] → z ~ N(μ,σ²) → [Neural Net] → x̂
                      ↑
                Reparameterization
                z = μ + σ·ε
```

### Training Objective (ELBO)

```
L = E_q[log p_θ(x|z)] - D_KL(q_φ(z|x) ‖ p(z))
    ↑                    ↑
Reconstruction loss   Regularization
```

### Pros and Cons

**Pros**:
- ✅ Fast sampling
- ✅ Latent representation
- ✅ Stable training
- ✅ Interpretable latent space

**Cons**:
- ❌ Blurry samples
- ❌ Posterior collapse
- ❌ Lower bound (not exact likelihood)

### Code Example

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
```

---

## 4. Generative Adversarial Networks (GANs)

### Core Idea

Two networks compete in a minimax game:

```
Generator:     G(z) → x
Discriminator: D(x) → [0,1] (real or fake?)
```

### Training Objective

```
min max E_data[log D(x)] + E_z[log(1 - D(G(z)))]
 G   D
```

### Visual Representation

```
    Generator              Discriminator
    
z → [G] → fake image → [D] → Real/Fake?
                       ↑
Real image ────────────┘
```

### Pros and Cons

**Pros**:
- ✅ Sharp, high-quality samples
- ✅ Fast sampling
- ✅ No explicit likelihood needed

**Cons**:
- ❌ Training instability
- ❌ Mode collapse
- ❌ No likelihood evaluation
- ❌ Difficult to train

### Code Example

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, np.prod(img_shape)),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

# Training loop
def train_gan(generator, discriminator, dataloader):
    for real_imgs in dataloader:
        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        
        d_loss = -torch.mean(torch.log(discriminator(real_imgs)) + 
                            torch.log(1 - discriminator(fake_imgs)))
        
        # Train Generator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        g_loss = -torch.mean(torch.log(discriminator(fake_imgs)))
```

---

## 5. Normalizing Flows

### Core Idea

Use invertible transformations to map simple → complex distributions:

```
z ~ p(z) → f₁ → f₂ → ... → fₙ → x
         ↑                      ↑
      Simple              Complex
    (Gaussian)          (Data dist)
```

### Change of Variables

```
p(x) = p(z) |det(∂f/∂z)|⁻¹
```

### Pros and Cons

**Pros**:
- ✅ Exact likelihood
- ✅ Fast sampling
- ✅ Invertible (can encode/decode)

**Cons**:
- ❌ Architecture constraints (must be invertible)
- ❌ Limited expressiveness
- ❌ Computational cost of Jacobian

### Example: Coupling Layer

```python
class CouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, dim // 2)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, dim // 2)
        )
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1), s
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)
```

---

## 6. Diffusion Models

### Core Idea

Gradually add noise (forward), then learn to denoise (reverse):

```
Forward:  x₀ → x₁ → x₂ → ... → x_T
Reverse:  x_T → x_{T-1} → ... → x₀
```

### Three Perspectives

1. **Variational (DDPM)**:
   ```
   Maximize ELBO with Markov chain
   ```

2. **Score-Based (NCSN)**:
   ```
   Learn ∇_x log p(x) at multiple noise levels
   ```

3. **Flow-Based (CNF)**:
   ```
   Learn continuous normalizing flow
   ```

### Pros and Cons

**Pros**:
- ✅ State-of-the-art quality
- ✅ Stable training
- ✅ Mode coverage
- ✅ Flexible conditioning

**Cons**:
- ❌ Slow sampling (many steps)
- ❌ Computationally expensive
- ❌ Lower bound (not exact likelihood)

### Simple Example

```python
class SimpleDiffusion(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model  # Noise prediction network
        self.T = T
        
        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffusion(self, x0, t):
        """Add noise to x0 at timestep t"""
        noise = torch.randn_like(x0)
        alpha_t = self.alphas_cumprod[t]
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise
    
    def reverse_diffusion(self, xt, t):
        """Denoise xt at timestep t"""
        predicted_noise = self.model(xt, t)
        alpha_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # Sample xt-1
        if t > 0:
            noise = torch.randn_like(xt)
            xt_prev = torch.sqrt(self.alphas[t]) * x0_pred + torch.sqrt(beta_t) * noise
        else:
            xt_prev = x0_pred
        
        return xt_prev
```

---

## 7. Comparison Table

| Model | Likelihood | Sampling | Quality | Training | Latent |
|-------|-----------|----------|---------|----------|--------|
| **Autoregressive** | Exact | Slow | High | Stable | No |
| **VAE** | Lower bound | Fast | Medium | Stable | Yes |
| **GAN** | No | Fast | High | Unstable | Yes |
| **Flow** | Exact | Fast | Medium | Stable | Yes |
| **Diffusion** | Lower bound | Slow* | Very High | Stable | Yes |

*Recent advances (DDIM, DPM-Solver) have significantly improved sampling speed

---

## 8. When to Use Each Model

### Autoregressive
- **Use when**: Need exact likelihood, sequential data (text)
- **Examples**: Language modeling, music generation

### VAE
- **Use when**: Need fast sampling, interpretable latents
- **Examples**: Data compression, representation learning

### GAN
- **Use when**: Need fast sampling, highest quality
- **Examples**: Real-time generation, style transfer

### Flow
- **Use when**: Need exact likelihood and fast sampling
- **Examples**: Density estimation, anomaly detection

### Diffusion
- **Use when**: Need highest quality, stable training
- **Examples**: Image generation, text-to-image, inpainting

---

## 9. Hybrid Approaches

### Latent Diffusion Models

Combine VAE + Diffusion:
```
x → [VAE Encoder] → z → [Diffusion] → z' → [VAE Decoder] → x'
```

**Advantages**:
- Faster than pixel-space diffusion
- Better quality than VAE alone
- Used in Stable Diffusion!

### VQ-VAE + Autoregressive

```
x → [VQ-VAE] → discrete codes → [Autoregressive] → new codes → [VQ-VAE] → x'
```

**Advantages**:
- Discrete latents
- Combines benefits of both

---

## Summary

Key concepts:
1. **Autoregressive**: Sequential generation, exact likelihood
2. **VAE**: Latent variables, fast sampling, blurry
3. **GAN**: Adversarial training, sharp but unstable
4. **Flow**: Invertible, exact likelihood
5. **Diffusion**: Iterative denoising, state-of-the-art quality

---

## Exercises

1. **Classification**: Given a paper, identify which model family it belongs to
2. **Trade-offs**: Explain when you'd choose VAE vs GAN vs Diffusion
3. **Implementation**: Implement a simple VAE from scratch
4. **Analysis**: Why do GANs suffer from mode collapse?
5. **Comparison**: Compare training objectives of VAE, GAN, and Diffusion

---

## Next Steps

Continue to `2_4_why_diffusion_models.md` to understand why diffusion models have become so successful.
