# Mathematical Derivation of Gradients for N-Layer Neural Network with General Activation Functions

This document provides a **generalized mathematical framework** for backpropagation in an **n-layer neural network** with **arbitrary activation functions**.

---

## ✅ Problem Setup

### Network Architecture (N Layers, General Activation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Input (x) → Layer 1 (σ₁) → Layer 2 (σ₂) → ... → Layer L (σₗ) → ŷ    │
│                                                                         │
│  Where σᵢ represents the activation function for layer i               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### General Notation for L-Layer Network

**Layer l (for l = 1, 2, ..., L):**
- **Input**: h⁽ˡ⁻¹⁾ ∈ ℝⁿˡ⁻¹ (output from previous layer)
  - Special case: h⁽⁰⁾ = x (network input)
- **Weights**: W⁽ˡ⁾ ∈ ℝⁿˡ⁻¹ˣⁿˡ
- **Bias**: b⁽ˡ⁾ ∈ ℝⁿˡ
- **Linear output**: z⁽ˡ⁾ = h⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾ ∈ ℝⁿˡ
- **Activation**: h⁽ˡ⁾ = σₗ(z⁽ˡ⁾) ∈ ℝⁿˡ
  - Special case: h⁽ᴸ⁾ = ŷ (network output)

### Dimensions Summary

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Layer l:                                                   │
│  h⁽ˡ⁻¹⁾: (nₗ₋₁,)  →  W⁽ˡ⁾: (nₗ₋₁, nₗ)  →  z⁽ˡ⁾: (nₗ,)    │
│                      b⁽ˡ⁾: (nₗ,)        →  h⁽ˡ⁾: (nₗ,)    │
│                                                             │
│  For batch of m samples:                                    │
│  H⁽ˡ⁻¹⁾: (m, nₗ₋₁) → W⁽ˡ⁾: (nₗ₋₁, nₗ) → Z⁽ˡ⁾: (m, nₗ)    │
│                      b⁽ˡ⁾: (nₗ,)       → H⁽ˡ⁾: (m, nₗ)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Forward Propagation (General Form)

### Single Sample

```
For l = 1 to L:
    z⁽ˡ⁾ = h⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾
    h⁽ˡ⁾ = σₗ(z⁽ˡ⁾)

Output: ŷ = h⁽ᴸ⁾
```

### Batch of m Samples

```
For l = 1 to L:
    Z⁽ˡ⁾ = H⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾    (m, nₗ)
    H⁽ˡ⁾ = σₗ(Z⁽ˡ⁾)              (m, nₗ)

Output: Ŷ = H⁽ᴸ⁾
```

---

## 📉 Loss Function

For regression (MSE):
```
L = (1/2m) × ||Ŷ - Y||²
```

For classification (Cross-Entropy):
```
L = -(1/m) × Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
```

**General derivative with respect to output:**
```
∂L/∂Ŷ = f(Ŷ, Y)
```

Examples:
- MSE: ∂L/∂Ŷ = Ŷ - Y
- Cross-Entropy with Softmax: ∂L/∂Ŷ = Ŷ - Y (simplified)

---

## 🔄 Backpropagation: General Framework

### Key Principle: Chain Rule

For any layer l, we compute gradients by propagating the error backward:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ∂L/∂z⁽ˡ⁾ = ∂L/∂h⁽ˡ⁾ ⊙ σₗ'(z⁽ˡ⁾)                          │
│                                                             │
│  where σₗ'(z⁽ˡ⁾) is the derivative of activation function │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 General Gradient Formulas

### For Layer l (l = 1, 2, ..., L)

**Step 1: Gradient with respect to activation output**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  If l = L (output layer):                                  │
│      ∂L/∂h⁽ᴸ⁾ = ∂L/∂Ŷ                                     │
│                                                             │
│  If l < L (hidden layer):                                  │
│      ∂L/∂h⁽ˡ⁾ = (∂L/∂z⁽ˡ⁺¹⁾) × (W⁽ˡ⁺¹⁾)ᵀ                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Step 2: Gradient with respect to linear output (pre-activation)**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ∂L/∂z⁽ˡ⁾ = (∂L/∂h⁽ˡ⁾) ⊙ σₗ'(z⁽ˡ⁾)                        │
│                                                             │
│  where ⊙ is element-wise multiplication                    │
│  and σₗ'(z⁽ˡ⁾) is the activation derivative               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Step 3: Gradient with respect to weights**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Single sample:                                             │
│  ∂L/∂W⁽ˡ⁾ = (h⁽ˡ⁻¹⁾)ᵀ × (∂L/∂z⁽ˡ⁾)                       │
│                                                             │
│  Batch (m samples):                                         │
│  ∂L/∂W⁽ˡ⁾ = (1/m) × (H⁽ˡ⁻¹⁾)ᵀ × (∂L/∂Z⁽ˡ⁾)               │
│                                                             │
│  Shape: (nₗ₋₁, nₗ)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Step 4: Gradient with respect to bias**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Single sample:                                             │
│  ∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾                                       │
│                                                             │
│  Batch (m samples):                                         │
│  ∂L/∂b⁽ˡ⁾ = (1/m) × Σ(∂L/∂Z⁽ˡ⁾)                           │
│                                                             │
│  Shape: (nₗ,)                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Complete Backpropagation Algorithm

### Batch Training (m samples)

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  FORWARD PASS:                                                ║
║  ─────────────                                                ║
║  H⁽⁰⁾ = X                                                     ║
║  For l = 1 to L:                                              ║
║      Z⁽ˡ⁾ = H⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾                                ║
║      H⁽ˡ⁾ = σₗ(Z⁽ˡ⁾)                                         ║
║  Ŷ = H⁽ᴸ⁾                                                     ║
║                                                               ║
║  COMPUTE LOSS:                                                ║
║  ──────────────                                               ║
║  L = loss_function(Ŷ, Y)                                     ║
║                                                               ║
║  BACKWARD PASS:                                               ║
║  ───────────────                                              ║
║  ∂L/∂H⁽ᴸ⁾ = ∂L/∂Ŷ                                           ║
║                                                               ║
║  For l = L down to 1:                                         ║
║      # Gradient through activation                           ║
║      ∂L/∂Z⁽ˡ⁾ = (∂L/∂H⁽ˡ⁾) ⊙ σₗ'(Z⁽ˡ⁾)                      ║
║                                                               ║
║      # Weight and bias gradients                             ║
║      ∂L/∂W⁽ˡ⁾ = (1/m) × (H⁽ˡ⁻¹⁾)ᵀ × (∂L/∂Z⁽ˡ⁾)             ║
║      ∂L/∂b⁽ˡ⁾ = (1/m) × Σ(∂L/∂Z⁽ˡ⁾)                         ║
║                                                               ║
║      # Propagate to previous layer (if not input layer)      ║
║      If l > 1:                                                ║
║          ∂L/∂H⁽ˡ⁻¹⁾ = (∂L/∂Z⁽ˡ⁾) × (W⁽ˡ⁾)ᵀ                  ║
║                                                               ║
║  UPDATE PARAMETERS:                                           ║
║  ───────────────────                                          ║
║  For l = 1 to L:                                              ║
║      W⁽ˡ⁾ ← W⁽ˡ⁾ - α × (∂L/∂W⁽ˡ⁾)                           ║
║      b⁽ˡ⁾ ← b⁽ˡ⁾ - α × (∂L/∂b⁽ˡ⁾)                           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 🔧 Activation Function Derivatives

### Common Activation Functions and Their Derivatives

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. SIGMOID: σ(z) = 1 / (1 + e⁻ᶻ)                         │
│     σ'(z) = σ(z) × (1 - σ(z))                             │
│     σ'(z) = h ⊙ (1 - h)    [where h = σ(z)]              │
│                                                             │
│  2. TANH: σ(z) = tanh(z)                                   │
│     σ'(z) = 1 - tanh²(z)                                   │
│     σ'(z) = 1 - h²         [where h = tanh(z)]            │
│                                                             │
│  3. RELU: σ(z) = max(0, z)                                 │
│     σ'(z) = 1 if z > 0, else 0                            │
│     σ'(z) = (z > 0)        [indicator function]            │
│                                                             │
│  4. LEAKY RELU: σ(z) = max(αz, z)  [α = 0.01]            │
│     σ'(z) = 1 if z > 0, else α                            │
│                                                             │
│  5. ELU: σ(z) = z if z > 0, else α(eᶻ - 1)               │
│     σ'(z) = 1 if z > 0, else σ(z) + α                    │
│                                                             │
│  6. SOFTMAX (for output layer):                            │
│     σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ                                  │
│     With cross-entropy: ∂L/∂z = ŷ - y (simplified)        │
│                                                             │
│  7. LINEAR (no activation):                                │
│     σ(z) = z                                               │
│     σ'(z) = 1                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Python/NumPy Implementation (General Framework)

```python
import numpy as np

class ActivationFunction:
    """Base class for activation functions"""
    def forward(self, z):
        raise NotImplementedError
    
    def derivative(self, z, h=None):
        """
        Compute derivative.
        z: pre-activation values
        h: post-activation values (optional, for efficiency)
        """
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z, h=None):
        if h is None:
            h = self.forward(z)
        return h * (1 - h)

class Tanh(ActivationFunction):
    def forward(self, z):
        return np.tanh(z)
    
    def derivative(self, z, h=None):
        if h is None:
            h = self.forward(z)
        return 1 - h**2

class ReLU(ActivationFunction):
    def forward(self, z):
        return np.maximum(0, z)
    
    def derivative(self, z, h=None):
        return (z > 0).astype(float)

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)
    
    def derivative(self, z, h=None):
        return np.where(z > 0, 1.0, self.alpha)

class Linear(ActivationFunction):
    def forward(self, z):
        return z
    
    def derivative(self, z, h=None):
        return np.ones_like(z)


class NLayerNetwork:
    """
    General N-layer neural network with arbitrary activation functions.
    """
    def __init__(self, layer_sizes, activations, lr=0.01):
        """
        Args:
            layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activations: List of activation functions for each layer
            lr: Learning rate
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # Number of weight layers
        self.activations = activations
        self.lr = lr
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for l in range(self.num_layers):
            # Xavier/He initialization
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            
            # Use He initialization for ReLU-like, Xavier for others
            if isinstance(activations[l], (ReLU, LeakyReLU)):
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Storage for forward pass (needed for backward pass)
        self.Z = []  # Pre-activation values
        self.H = []  # Post-activation values
    
    def forward(self, X):
        """
        Forward pass through all layers.
        X: (batch_size, input_dim)
        """
        self.Z = []
        self.H = [X]  # H[0] = input
        
        for l in range(self.num_layers):
            # Linear transformation
            Z = self.H[l] @ self.weights[l] + self.biases[l]
            self.Z.append(Z)
            
            # Activation
            H = self.activations[l].forward(Z)
            self.H.append(H)
        
        return self.H[-1]  # Return output
    
    def backward(self, X, Y_true, Y_pred):
        """
        Backward pass through all layers.
        X: (batch_size, input_dim)
        Y_true: (batch_size, output_dim)
        Y_pred: (batch_size, output_dim)
        """
        m = X.shape[0]  # Batch size
        
        # Initialize gradient storage
        dL_dW = [None] * self.num_layers
        dL_db = [None] * self.num_layers
        
        # Output layer gradient (assuming MSE loss)
        dL_dH = Y_pred - Y_true  # (batch, output_dim)
        
        # Backward pass through layers
        for l in range(self.num_layers - 1, -1, -1):
            # Gradient through activation
            activation_derivative = self.activations[l].derivative(
                self.Z[l], 
                self.H[l + 1]
            )
            dL_dZ = dL_dH * activation_derivative  # Element-wise
            
            # Weight gradient
            dL_dW[l] = (1/m) * (self.H[l].T @ dL_dZ)
            
            # Bias gradient
            dL_db[l] = (1/m) * np.sum(dL_dZ, axis=0)
            
            # Propagate to previous layer (if not input layer)
            if l > 0:
                dL_dH = dL_dZ @ self.weights[l].T
        
        # Update parameters
        for l in range(self.num_layers):
            self.weights[l] -= self.lr * dL_dW[l]
            self.biases[l] -= self.lr * dL_db[l]
    
    def compute_loss(self, Y_true, Y_pred):
        """MSE loss"""
        return 0.5 * np.mean((Y_pred - Y_true)**2)
    
    def train(self, X, Y, epochs=1000, verbose=True):
        """Training loop"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y, Y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, Y, Y_pred)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    Y = np.sum(X**2, axis=1, keepdims=True)  # Target: sum of squares
    
    # Define network architecture
    layer_sizes = [2, 10, 8, 5, 1]  # 4-layer network
    activations = [
        ReLU(),      # Layer 1: ReLU
        Tanh(),      # Layer 2: Tanh
        ReLU(),      # Layer 3: ReLU
        Linear()     # Layer 4 (output): Linear
    ]
    
    # Create and train network
    model = NLayerNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        lr=0.01
    )
    
    print("Training 4-layer network with mixed activations...")
    losses = model.train(X, Y, epochs=1000)
    
    # Test
    Y_pred = model.forward(X)
    final_loss = model.compute_loss(Y, Y_pred)
    print(f"\nFinal Loss: {final_loss:.4f}")
    
    # Example with different architecture
    print("\n" + "="*60)
    print("Training 2-layer network with Sigmoid...")
    
    model2 = NLayerNetwork(
        layer_sizes=[2, 15, 1],
        activations=[Sigmoid(), Linear()],
        lr=0.1
    )
    
    losses2 = model2.train(X, Y, epochs=1000)
    Y_pred2 = model2.forward(X)
    final_loss2 = model2.compute_loss(Y, Y_pred2)
    print(f"\nFinal Loss: {final_loss2:.4f}")
```

---

## 🎯 Key Insights

### 1. **Modularity**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  The general framework separates:                          │
│  • Network structure (layer sizes)                         │
│  • Activation functions (pluggable)                        │
│  • Loss function (can be changed)                          │
│  • Optimization (gradient descent)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Activation Function Abstraction**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Each activation function provides:                        │
│  • forward(z): Compute activation                          │
│  • derivative(z, h): Compute derivative                    │
│                                                             │
│  This allows easy addition of new activations!             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. **Gradient Flow Pattern**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  For each layer l (backward):                              │
│  1. ∂L/∂Z⁽ˡ⁾ = (∂L/∂H⁽ˡ⁾) ⊙ σₗ'(Z⁽ˡ⁾)                    │
│  2. ∂L/∂W⁽ˡ⁾ = (H⁽ˡ⁻¹⁾)ᵀ × (∂L/∂Z⁽ˡ⁾)                    │
│  3. ∂L/∂b⁽ˡ⁾ = Σ(∂L/∂Z⁽ˡ⁾)                                │
│  4. ∂L/∂H⁽ˡ⁻¹⁾ = (∂L/∂Z⁽ˡ⁾) × (W⁽ˡ⁾)ᵀ                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. **Computational Efficiency**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  • Store activations during forward pass                   │
│  • Reuse for derivative computation                        │
│  • Example: σ'(z) = h(1-h) uses h, not z                  │
│  • Saves computation in backward pass                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. **Scalability**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  This framework scales to:                                 │
│  • Any number of layers (2, 10, 100, ...)                 │
│  • Any layer sizes (10, 1000, 10000, ...)                 │
│  • Any activation functions                                │
│  • Any loss function (with appropriate ∂L/∂Ŷ)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Dimension Tracking Template

For an L-layer network with batch size m:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  FORWARD PASS DIMENSIONS:                                     ║
║  ─────────────────────────                                    ║
║  H⁽⁰⁾ = X:           (m, n₀)                                 ║
║  Z⁽¹⁾ = H⁽⁰⁾W⁽¹⁾:    (m, n₁)    W⁽¹⁾: (n₀, n₁)             ║
║  H⁽¹⁾ = σ₁(Z⁽¹⁾):    (m, n₁)                                 ║
║  Z⁽²⁾ = H⁽¹⁾W⁽²⁾:    (m, n₂)    W⁽²⁾: (n₁, n₂)             ║
║  H⁽²⁾ = σ₂(Z⁽²⁾):    (m, n₂)                                 ║
║  ...                                                          ║
║  Z⁽ᴸ⁾ = H⁽ᴸ⁻¹⁾W⁽ᴸ⁾:  (m, nₗ)    W⁽ᴸ⁾: (nₗ₋₁, nₗ)           ║
║  Ŷ = H⁽ᴸ⁾:           (m, nₗ)                                 ║
║                                                               ║
║  BACKWARD PASS DIMENSIONS:                                    ║
║  ──────────────────────────                                   ║
║  ∂L/∂H⁽ᴸ⁾:           (m, nₗ)                                 ║
║  ∂L/∂Z⁽ᴸ⁾:           (m, nₗ)                                 ║
║  ∂L/∂W⁽ᴸ⁾:           (nₗ₋₁, nₗ)                              ║
║  ∂L/∂b⁽ᴸ⁾:           (nₗ,)                                   ║
║  ∂L/∂H⁽ᴸ⁻¹⁾:         (m, nₗ₋₁)                               ║
║  ...                                                          ║
║  ∂L/∂Z⁽¹⁾:           (m, n₁)                                 ║
║  ∂L/∂W⁽¹⁾:           (n₀, n₁)                                ║
║  ∂L/∂b⁽¹⁾:           (n₁,)                                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 🌟 Summary

### Universal Backpropagation Formula:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  For any layer l in an L-layer network:                      ║
║                                                               ║
║  1. FORWARD:                                                  ║
║     Z⁽ˡ⁾ = H⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾                                 ║
║     H⁽ˡ⁾ = σₗ(Z⁽ˡ⁾)                                          ║
║                                                               ║
║  2. BACKWARD:                                                 ║
║     ∂L/∂Z⁽ˡ⁾ = (∂L/∂H⁽ˡ⁾) ⊙ σₗ'(Z⁽ˡ⁾)                       ║
║     ∂L/∂W⁽ˡ⁾ = (1/m) × (H⁽ˡ⁻¹⁾)ᵀ × (∂L/∂Z⁽ˡ⁾)              ║
║     ∂L/∂b⁽ˡ⁾ = (1/m) × Σ(∂L/∂Z⁽ˡ⁾)                          ║
║     ∂L/∂H⁽ˡ⁻¹⁾ = (∂L/∂Z⁽ˡ⁾) × (W⁽ˡ⁾)ᵀ                       ║
║                                                               ║
║  3. UPDATE:                                                   ║
║     W⁽ˡ⁾ ← W⁽ˡ⁾ - α × (∂L/∂W⁽ˡ⁾)                            ║
║     b⁽ˡ⁾ ← b⁽ˡ⁾ - α × (∂L/∂b⁽ˡ⁾)                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Key Advantages:

1. **Works for any number of layers**
2. **Works with any activation function** (just provide σ and σ')
3. **Works with any loss function** (just provide ∂L/∂Ŷ)
4. **Efficient** (same complexity as forward pass)
5. **Modular** (easy to extend and modify)

---

**This is the universal framework for deep learning!** 🎉

All modern deep learning frameworks (PyTorch, TensorFlow, JAX) implement this general pattern with automatic differentiation. Understanding this framework is key to understanding how neural networks learn! 🚀
