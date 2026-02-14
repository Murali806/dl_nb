# Bug Fix: Non-Linear Network Not Learning

## Problem
The demo notebook showed that both linear and non-linear networks had the same loss (~7.5303), with the non-linear network failing to learn the parabola despite having activation functions.

## Root Cause
The `NonLinearNetwork` class was using a **1→1→1 architecture** (single neuron per hidden layer):
```python
self.W1 = np.random.randn(1, 1) * 0.1  # 1 input → 1 neuron
self.W2 = np.random.randn(1, 1) * 0.1  # 1 neuron → 1 neuron  
self.W3 = np.random.randn(1, 1) * 0.1  # 1 neuron → 1 output
```

**Why this failed:**
- With only 1 neuron per layer, the network has insufficient capacity to learn non-linear patterns
- Even with activation functions, a single neuron can only create simple transformations
- The Universal Approximation Theorem requires "sufficient neurons" - 1 neuron is not sufficient!

## Solution
Changed the architecture to **1→10→10→1** (10 neurons in each hidden layer):
```python
def __init__(self, learning_rate=0.01, hidden_size=10):
    self.W1 = np.random.randn(1, hidden_size) * 0.1  # 1 input → 10 neurons
    self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1  # 10 → 10 neurons
    self.W3 = np.random.randn(hidden_size, 1) * 0.1  # 10 neurons → 1 output
```

## Expected Results
After the fix, you should see:
- **Linear Network Loss**: ~7.5 (unchanged - still can only fit a straight line)
- **Non-Linear Network Loss**: ~0.01-0.1 (dramatic improvement!)
- **Improvement Factor**: 75-750x better!

The non-linear network will now successfully learn the parabola shape.

## Key Lesson
**Activation functions alone are not enough** - you also need:
1. ✅ Non-linear activation functions (tanh, ReLU, etc.)
2. ✅ **Sufficient neurons** in hidden layers
3. ✅ Proper initialization and learning rate

The combination of these three factors enables neural networks to approximate complex non-linear functions!
