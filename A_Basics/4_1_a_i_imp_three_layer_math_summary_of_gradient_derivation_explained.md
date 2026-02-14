### The Complete Backpropagation Algorithm:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. FORWARD PASS:                                      │
│     • Compute h₁ = W₁·x + b₁                          │
│     • Compute h₂ = W₂·h₁ + b₂                         │
│     • Compute ŷ = W₃·h₂ + b₃                          │
│                                                         │
│  2. COMPUTE LOSS:                                      │
│     • L = (ŷ - y)²                                    │
│                                                         │
│  3. BACKWARD PASS (Compute Gradients):                  |
|     Generalized = ∂L/∂Wn = 2 * error * Wn+1 * hn-1      |
|                   ∂L/∂Wn = 2 * error * Wn+1             |
│     • ∂L/∂W₃ = 2(ŷ - y) × h₂                        │
│     • ∂L/∂b₃ = 2(ŷ - y)                              │
│     • ∂L/∂W₂ = 2(ŷ - y) × W₃ × h₁                   │
│     • ∂L/∂b₂ = 2(ŷ - y) × W₃                         │
│     • ∂L/∂W₁ = 2(ŷ - y) × W₃ × W₂ × x               │
│     • ∂L/∂b₁ = 2(ŷ - y) × W₃ × W₂                   │
│                                                        │
│  4. UPDATE PARAMETERS:                                 │
│     • W₃ ← W₃ - α × (∂L/∂W₃)                         │
│     • b₃ ← b₃ - α × (∂L/∂b₃)                         │
│     • W₂ ← W₂ - α × (∂L/∂W₂)                         │
│     • b₂ ← b₂ - α × (∂L/∂b₂)                         │
│     • W₁ ← W₁ - α × (∂L/∂W₁)                         │
│     • b₁ ← b₁ - α × (∂L/∂b₁)                         │
│                                                         │
│  5. REPEAT until convergence                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔗 Connection to Code

In Python/NumPy, the backward pass would look like:

```python
def backward(x, y, y_pred, h1, h2, W3, W2, learning_rate=0.01):
    """
    Compute gradients and update parameters for 3-layer linear network.
    """
    n = len(x)  # batch size
    
    # Error signal
    error = y_pred - y
    
    # Layer 3 gradients
    dW3 = (2.0 / n) * np.sum(error * h2)
    db3 = (2.0 / n) * np.sum(error)
    
    # Layer 2 gradients
    dW2 = (2.0 / n) * np.sum(error * W3 * h1)
    db2 = (2.0 / n) * np.sum(error * W3)
    
    # Layer 1 gradients
    dW1 = (2.0 / n) * np.sum(error * W3 * W2 * x)
    db1 = (2.0 / n) * np.sum(error * W3 * W2)
    
    # Update parameters
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    return W3, b3, W2, b2, W1, b1
```
