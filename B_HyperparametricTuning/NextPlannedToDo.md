# Next Planned Topics - Deep Learning Notebook Series

This document outlines the advanced topics planned for future notebooks in this series. These topics build upon the fundamentals covered in notebooks 1-6.

---

## 📚 Current Progress

### ✅ Completed Topics
1. **Single Perceptron** - House price prediction, gradient derivation
2. **Multi-layer Networks** - Polynomial approximation, activation functions
3. **Activation Functions** - ReLU, sigmoid, tanh, softmax
4. **Loss Functions** - MSE, cross-entropy, entropy
5. **Logistic Regression** - Binary classification, with/without hidden layers
6. **Vectorization** - Batch processing, forward/backward propagation

---

## 🎯 Planned Topics

### 7. Optimizers (High Priority)

Advanced optimization algorithms that improve upon basic gradient descent.

#### 7.1 Stochastic Gradient Descent (SGD)
- **Concepts**:
  - Difference between batch, mini-batch, and stochastic GD
  - Trade-offs: speed vs convergence stability
  - When to use each variant
- **Implementation**:
  - Mini-batch creation and shuffling
  - Update rules for SGD
  - Convergence analysis

#### 7.2 Learning Rate
- **Concepts**:
  - Impact of learning rate on convergence
  - Learning rate too high vs too low
  - Visualizing loss landscapes
- **Implementation**:
  - Experiments with different learning rates
  - Finding optimal learning rate
  - Learning rate range test

#### 7.3 Learning Rate Decay
- **Concepts**:
  - Why decay learning rate over time
  - Different decay schedules (step, exponential, cosine)
  - Warm-up strategies
- **Implementation**:
  - Step decay
  - Exponential decay
  - Time-based decay
  - Performance comparison

#### 7.4 Stochastic Gradient Descent with Momentum
- **Concepts**:
  - Physics intuition: ball rolling down hill
  - Exponentially weighted averages
  - Bias correction
- **Mathematical Formulation**:
  ```
  v_dW = β * v_dW + (1-β) * dW
  v_db = β * v_db + (1-β) * db
  W = W - α * v_dW
  b = b - α * v_db
  ```
- **Implementation**:
  - Momentum parameter β (typically 0.9)
  - Comparison with vanilla SGD
  - Convergence speed analysis

#### 7.5 AdaGrad (Adaptive Gradient)
- **Concepts**:
  - Adaptive learning rates per parameter
  - Accumulation of squared gradients
  - Good for sparse data
- **Mathematical Formulation**:
  ```
  cache_dW = cache_dW + dW²
  cache_db = cache_db + db²
  W = W - α * dW / (√cache_dW + ε)
  b = b - α * db / (√cache_db + ε)
  ```
- **Implementation**:
  - Per-parameter learning rates
  - Epsilon for numerical stability
  - Limitations (diminishing learning rates)

#### 7.6 RMSProp (Root Mean Square Propagation)
- **Concepts**:
  - Fixes AdaGrad's diminishing learning rate
  - Exponentially weighted moving average of squared gradients
  - Better for non-convex optimization
- **Mathematical Formulation**:
  ```
  cache_dW = β * cache_dW + (1-β) * dW²
  cache_db = β * cache_db + (1-β) * db²
  W = W - α * dW / (√cache_dW + ε)
  b = b - α * db / (√cache_db + ε)
  ```
- **Implementation**:
  - Decay rate β (typically 0.9 or 0.999)
  - Comparison with AdaGrad
  - Performance on different problems

#### 7.7 Adam (Adaptive Moment Estimation)
- **Concepts**:
  - Combines momentum and RMSProp
  - Most popular optimizer in practice
  - Bias correction for first and second moments
- **Mathematical Formulation**:
  ```
  # First moment (momentum)
  v_dW = β₁ * v_dW + (1-β₁) * dW
  v_db = β₁ * v_db + (1-β₁) * db
  
  # Second moment (RMSProp)
  s_dW = β₂ * s_dW + (1-β₂) * dW²
  s_db = β₂ * s_db + (1-β₂) * db²
  
  # Bias correction
  v_dW_corrected = v_dW / (1 - β₁ᵗ)
  v_db_corrected = v_db / (1 - β₁ᵗ)
  s_dW_corrected = s_dW / (1 - β₂ᵗ)
  s_db_corrected = s_db / (1 - β₂ᵗ)
  
  # Update
  W = W - α * v_dW_corrected / (√s_dW_corrected + ε)
  b = b - α * v_db_corrected / (√s_db_corrected + ε)
  ```
- **Implementation**:
  - Hyperparameters: α=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸
  - Bias correction importance
  - Variants: AdaMax, Nadam

#### 7.8 Full Code Implementation
- **Complete Optimizer Class**:
  - Unified interface for all optimizers
  - Easy switching between optimizers
  - Performance benchmarking
- **Comparative Analysis**:
  - Training curves for each optimizer
  - Convergence speed comparison
  - Memory requirements
  - When to use which optimizer

---

### 8. L1 and L2 Regularization (High Priority)

Techniques to prevent overfitting by adding penalty terms to the loss function.

#### 8.1 Concepts
- **Overfitting Problem**:
  - High training accuracy, low test accuracy
  - Model memorizes training data
  - Poor generalization
- **Regularization Intuition**:
  - Penalize large weights
  - Encourage simpler models
  - Bias-variance tradeoff

#### 8.2 L2 Regularization (Weight Decay)
- **Mathematical Formulation**:
  ```
  Loss = Original_Loss + (λ/2m) * Σ(W²)
  ```
- **Forward Pass**:
  - Add regularization term to loss
  - λ (lambda) is regularization strength
  - Only regularize weights, not biases
- **Backward Pass**:
  ```
  dW = dW_original + (λ/m) * W
  ```
- **Implementation**:
  - Computing regularized loss
  - Modified gradient computation
  - Effect on weight magnitudes
- **Visualization**:
  - Weight distributions with/without L2
  - Decision boundaries comparison
  - Validation curves

#### 8.3 L1 Regularization (Lasso)
- **Mathematical Formulation**:
  ```
  Loss = Original_Loss + (λ/m) * Σ|W|
  ```
- **Forward Pass**:
  - Add L1 penalty to loss
  - Encourages sparsity (many weights → 0)
- **Backward Pass**:
  ```
  dW = dW_original + (λ/m) * sign(W)
  ```
- **Implementation**:
  - Sign function for gradients
  - Sparse weight matrices
  - Feature selection property
- **Comparison with L2**:
  - L1 creates sparse models
  - L2 creates small but non-zero weights
  - When to use each

#### 8.4 Elastic Net (L1 + L2)
- **Mathematical Formulation**:
  ```
  Loss = Original_Loss + λ₁/m * Σ|W| + λ₂/2m * Σ(W²)
  ```
- **Implementation**:
  - Combining both penalties
  - Balancing λ₁ and λ₂
  - Best of both worlds

#### 8.5 Practical Considerations
- **Choosing λ**:
  - Cross-validation
  - Regularization path
  - Grid search
- **When to Regularize**:
  - Small datasets
  - High model complexity
  - Signs of overfitting
- **Alternatives**:
  - Early stopping
  - Data augmentation
  - Dropout (next section)

---

### 9. Dropout (High Priority)

A powerful regularization technique that randomly drops neurons during training.

#### 9.1 Concepts
- **Core Idea**:
  - Randomly "drop" neurons during training
  - Forces network to learn redundant representations
  - Prevents co-adaptation of neurons
- **Ensemble Interpretation**:
  - Training many sub-networks
  - Averaging predictions at test time
  - Reduces overfitting

#### 9.2 Forward Pass (Training)
- **Mathematical Formulation**:
  ```
  # Training mode
  mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
  A_dropout = A * mask
  ```
- **Implementation**:
  - Generate random mask
  - Apply mask to activations
  - Inverted dropout (scale by 1/keep_prob)
- **Keep Probability**:
  - Typical values: 0.5 to 0.9
  - Higher for input layers (0.8-0.9)
  - Lower for hidden layers (0.5-0.7)

#### 9.3 Forward Pass (Testing)
- **No Dropout at Test Time**:
  ```
  # Test mode
  A_test = A  # No dropout, no scaling needed (due to inverted dropout)
  ```
- **Why No Dropout**:
  - Want deterministic predictions
  - Already scaled during training
  - Ensemble effect achieved

#### 9.4 Backward Pass
- **Gradient Flow**:
  ```
  dA_prev = dA * mask  # Apply same mask used in forward pass
  ```
- **Implementation**:
  - Store mask from forward pass
  - Apply mask to gradients
  - No gradient flows through dropped neurons
- **Cache Management**:
  - Save masks for backpropagation
  - Different masks for each layer
  - Different masks for each training iteration

#### 9.5 The Code
- **Complete Implementation**:
  - Dropout layer class
  - Training vs testing modes
  - Integration with existing network
- **Practical Tips**:
  - Start with keep_prob = 0.5
  - Use higher keep_prob for input layer
  - Monitor training vs validation loss
  - Can use different keep_prob per layer

#### 9.6 Visualization and Analysis
- **Effects of Dropout**:
  - Training curves with/without dropout
  - Weight distributions
  - Activation patterns
- **Hyperparameter Tuning**:
  - Grid search for keep_prob
  - Layer-wise dropout rates
  - Interaction with other regularization

#### 9.7 Variants
- **DropConnect**:
  - Drop connections instead of neurons
  - More fine-grained
- **Spatial Dropout**:
  - For convolutional layers
  - Drop entire feature maps
- **Variational Dropout**:
  - Bayesian interpretation
  - Learnable dropout rates

---

## 🔄 Integration with Existing Notebooks

### Connections to Previous Work
- **Notebook 6 (Vectorization)** → Optimizers use vectorized operations
- **Notebook 5 (Logistic Regression)** → Add regularization and dropout
- **Notebook 4 (Activation Functions)** → Dropout affects activation patterns
- **Notebook 3 (Gradient Derivation)** → Modified gradients with regularization

### Suggested Notebook Structure
```
7_optimizers_complete_guide.ipynb
  - All optimizers with side-by-side comparison
  - Performance benchmarks
  - When to use which optimizer

8_regularization_l1_l2.ipynb
  - L1 and L2 regularization
  - Mathematical derivations
  - Overfitting prevention

9_dropout_regularization.ipynb
  - Dropout implementation
  - Training vs testing modes
  - Comparison with L1/L2
```

---

## 📊 Future Topics (After 7-9)

### 10. Batch Normalization
- Internal covariate shift
- Normalization during training
- Inference mode

### 11. Convolutional Neural Networks (CNNs)
- Convolution operation
- Pooling layers
- CNN architectures

### 12. Recurrent Neural Networks (RNNs)
- Sequence modeling
- LSTM and GRU
- Backpropagation through time

### 13. Advanced Architectures
- ResNet (skip connections)
- Attention mechanisms
- Transformers

---

## 📝 Notes

### Implementation Philosophy
- **From Scratch**: Pure NumPy implementations for understanding
- **Mathematical Rigor**: Complete derivations with step-by-step explanations
- **Practical Code**: Working examples with real datasets
- **Visualizations**: Plots showing concepts and results
- **Comparisons**: Side-by-side analysis of different approaches

### Learning Objectives
By completing topics 7-9, you will:
1. ✅ Understand modern optimization algorithms
2. ✅ Know when to use which optimizer
3. ✅ Implement regularization techniques
4. ✅ Prevent overfitting effectively
5. ✅ Build production-ready neural networks

---

## 🎯 Priority Order

1. **Optimizers** (Topic 7) - Essential for efficient training
2. **L1/L2 Regularization** (Topic 8) - Prevent overfitting
3. **Dropout** (Topic 9) - Modern regularization technique

These three topics form the foundation for training robust neural networks and should be completed before moving to advanced architectures.

---

**Last Updated**: February 2, 2026
**Status**: Planning Phase
**Next Action**: Begin implementation of Topic 7 (Optimizers)
