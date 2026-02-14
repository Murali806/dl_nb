---

## 📊 Complete Gradient Summary

For a three-layer network with **n neurons per layer**:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  OUTPUT LAYER (Layer 3):                                     ║
║  ∂L/∂W₃ = h₂ᵀ × (ŷ - y)          Shape: (n₂, k)            ║
║  ∂L/∂b₃ = ŷ - y                   Shape: (k,)               ║
║                                                               ║
║  HIDDEN LAYER 2 (with tanh):                                 ║
║  ∂L/∂h₂ = (ŷ - y) × W₃ᵀ          Shape: (n₂,)              ║
║  ∂L/∂z₂ = (∂L/∂h₂) ⊙ (1 - h₂²)   Shape: (n₂,)              ║
║  ∂L/∂W₂ = h₁ᵀ × (∂L/∂z₂)         Shape: (n₁, n₂)           ║
║  ∂L/∂b₂ = ∂L/∂z₂                  Shape: (n₂,)              ║
║                                                               ║
║  HIDDEN LAYER 1 (with tanh):                                 ║
║  ∂L/∂h₁ = (∂L/∂z₂) × W₂ᵀ         Shape: (n₁,)              ║
║  ∂L/∂z₁ = (∂L/∂h₁) ⊙ (1 - h₁²)   Shape: (n₁,)              ║
║  ∂L/∂W₁ = xᵀ × (∂L/∂z₁)          Shape: (d, n₁)            ║
║  ∂L/∂b₁ = ∂L/∂z₁                  Shape: (n₁,)              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Key Pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  For each layer:                                            │
│                                                             │
│  1. Propagate gradient backward: ∂L/∂h = (∂L/∂z_next) × Wᵀ │
│  2. Apply activation derivative: ∂L/∂z = (∂L/∂h) ⊙ (1-h²) │
│  3. Compute weight gradient:     ∂L/∂W = h_prevᵀ × (∂L/∂z) │
│  4. Compute bias gradient:       ∂L/∂b = ∂L/∂z            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Batch Training (Multiple Samples)

For a batch of m samples: X ∈ ℝᵐˣᵈ

### Forward Pass:
```
Z₁ = XW₁ + b₁        (m, n₁)
H₁ = tanh(Z₁)        (m, n₁)

Z₂ = H₁W₂ + b₂       (m, n₂)
H₂ = tanh(Z₂)        (m, n₂)

Z₃ = H₂W₃ + b₃       (m, k)
Ŷ = Z₃               (m, k)
```

### Backward Pass:
```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  ∂L/∂Z₃ = Ŷ - Y                    Shape: (m, k)              ║
║  ∂L/∂W₃ = (1/m) × H₂ᵀ × (∂L/∂Z₃)   Shape: (n₂, k)             ║
║  ∂L/∂b₃ = (1/m) × Σ(∂L/∂Z₃)        Shape: (k,)                ║
║                                                               ║
║  ∂L/∂H₂ = (∂L/∂Z₃) × W₃ᵀ           Shape: (m, n₂)             ║
║  ∂L/∂Z₂ = (∂L/∂H₂) ⊙ (1 - H₂²)    Shape: (m, n₂)             ║
║  ∂L/∂W₂ = (1/m) × H₁ᵀ × (∂L/∂Z₂)   Shape: (n₁, n₂)            ║
║  ∂L/∂b₂ = (1/m) × Σ(∂L/∂Z₂)        Shape: (n₂,)               ║
║                                                               ║
║  ∂L/∂H₁ = (∂L/∂Z₂) × W₂ᵀ           Shape: (m, n₁)             ║
║  ∂L/∂Z₁ = (∂L/∂H₁) ⊙ (1 - H₁²)    Shape: (m, n₁)             ║
║  ∂L/∂W₁ = (1/m) × Xᵀ × (∂L/∂Z₁)    Shape: (d, n₁)             ║
║  ∂L/∂b₁ = (1/m) × Σ(∂L/∂Z₁)        Shape: (n₁,)               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

**Note:** Σ means sum over the batch dimension (axis=0)

---

## 💻 Python/NumPy Implementation

```python
import numpy as np

class ThreeLayerNetwork:
    """
    Three-layer neural network with n neurons per layer.
    """
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, lr=0.01):
        self.lr = lr
        
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1_dim)
        
        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / hidden1_dim)
        self.b2 = np.zeros(hidden2_dim)
        
        self.W3 = np.random.randn(hidden2_dim, output_dim) * np.sqrt(2.0 / hidden2_dim)
        self.b3 = np.zeros(output_dim)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, h):
        """Derivative: 1 - tanh²(x) = 1 - h²"""
        return 1 - h**2
    
    def forward(self, X):
        """
        Forward pass.
        X: (batch_size, input_dim)
        """
        # Layer 1
        self.Z1 = X @ self.W1 + self.b1  # (batch, hidden1)
        self.H1 = self.tanh(self.Z1)
        
        # Layer 2
        self.Z2 = self.H1 @ self.W2 + self.b2  # (batch, hidden2)
        self.H2 = self.tanh(self.Z2)
        
        # Layer 3
        self.Z3 = self.H2 @ self.W3 + self.b3  # (batch, output)
        self.Y_pred = self.Z3
        
        return self.Y_pred
    
    def backward(self, X, Y_true):
        """
        Backward pass and parameter update.
        X: (batch_size, input_dim)
        Y_true: (batch_size, output_dim)
        """
        m = X.shape[0]  # batch size
        
        # Output layer gradients
        dL_dZ3 = self.Y_pred - Y_true  # (batch, output)
        dL_dW3 = (1/m) * (self.H2.T @ dL_dZ3)  # (hidden2, output)
        dL_db3 = (1/m) * np.sum(dL_dZ3, axis=0)  # (output,)
        
        # Layer 2 gradients
        dL_dH2 = dL_dZ3 @ self.W3.T  # (batch, hidden2)
        dL_dZ2 = dL_dH2 * self.tanh_derivative(self.H2)  # (batch, hidden2)
        dL_dW2 = (1/m) * (self.H1.T @ dL_dZ2)  # (hidden1, hidden2)
        dL_db2 = (1/m) * np.sum(dL_dZ2, axis=0)  # (hidden2,)
        
        # Layer 1 gradients
        dL_dH1 = dL_dZ2 @ self.W2.T  # (batch, hidden1)
        dL_dZ1 = dL_dH1 * self.tanh_derivative(self.H1)  # (batch, hidden1)
        dL_dW1 = (1/m) * (X.T @ dL_dZ1)  # (input, hidden1)
        dL_db1 = (1/m) * np.sum(dL_dZ1, axis=0)  # (hidden1,)
        
        # Update parameters
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
    
    def compute_loss(self, Y_true, Y_pred):
        """MSE loss"""
        return 0.5 * np.mean((Y_pred - Y_true)**2)
    
    def train(self, X, Y, epochs=1000):
        """Training loop"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y, Y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, Y)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    Y = np.sum(X**2, axis=1, keepdims=True)  # Target: sum of squares
    
    # Create and train network
    model = ThreeLayerNetwork(
        input_dim=2,
        hidden1_dim=10,
        hidden2_dim=5,
        output_dim=1,
        lr=0.01
    )
    
    losses = model.train(X, Y, epochs=1000)
    
    # Test
    Y_pred = model.forward(X)
    final_loss = model.compute_loss(Y, Y_pred)
    print(f"\nFinal Loss: {final_loss:.4f}")
```

---
