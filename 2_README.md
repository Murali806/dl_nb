# Deep Learning Neural Network Implementations

This repository contains educational implementations of neural networks from scratch, demonstrating fundamental concepts in deep learning with manual backpropagation and comprehensive visualizations.

## 📋 Project Overview

This project includes three progressively complex neural network implementations:

1. **Single Perceptron** - Linear relationships (house price prediction)
2. **Two-Layer Network (Polynomial)** - Quadratic functions (y = x²)
3. **Two-Layer Network (Drug Dosage)** - Inverted U-shaped curves (optimal dosage)

**Key Features**:
- ✅ Manual forward/backward propagation implementation
- ✅ Gradient descent optimization from scratch
- ✅ Synthetic data generation for various relationships
- ✅ Real-time training visualization with TensorBoard
- ✅ Experiment tracking with MLflow
- ✅ Comprehensive performance metrics and visualizations
- ✅ Educational focus with detailed explanations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Choose a notebook to explore**:
   - `single_perceptron_house_price.ipynb` - Start here for basics
   - `two_layer_polynomial_approximation.ipynb` - Non-linear functions
   - `two_layer_drug_dosage_response.ipynb` - Real-world medical application

---

## 📚 Notebooks

### 0. Demo: How Activation Functions Enable Non-Linearity

**File**: `demo_how_activation_functions_enable_nonlinearity.ipynb`

**Purpose**: Educational demonstration answering the fundamental question: "How can neural networks learn non-linear functions like x² when forward propagation seems to collapse to linear form?"

**What's Inside**:
- Mathematical proof of why linear networks collapse
- Side-by-side comparison: Linear vs Non-linear networks
- Visual demonstrations of layer-by-layer transformations
- Taylor series analysis of tanh activation
- Function composition visualizations
- Universal Approximation Theorem demonstrations
- Interactive experiments with different target functions

**Key Insights**:
- Without activation functions: Multiple layers = waste (collapses to linear)
- With activation functions: Networks can approximate ANY continuous function
- The x³ term in tanh's Taylor series enables non-linear learning
- Composition of non-linearities creates complex patterns

**Perfect for**: Understanding the fundamental role of activation functions in neural networks

---

### 1. Single Perceptron: House Price Prediction

**File**: `single_perceptron_house_price.ipynb`

**Use Case**: Predict house prices based on square footage (linear relationship)

**Architecture**: Input (1) → Output (1)

**What's Inside**:

- Synthetic house price data (linear relationship: `Price = 150 × Size + 50,000`)
- Single perceptron implementation with manual backpropagation
- Training: 500 epochs, learning rate 0.01
- Visualizations: loss curves, parameter evolution, predictions vs actual
- Performance metrics: MSE, R², MAE, MAPE

**Learning Objectives**:
- Understand single perceptron fundamentals
- Learn gradient descent optimization
- Master manual backpropagation
- Visualize training dynamics

---

### 2. Two-Layer Network: Polynomial Function Approximation

**File**: `two_layer_polynomial_approximation.ipynb`

**Use Case**: Approximate quadratic function y = x² (parabola)

**Architecture**: Input (1) → Hidden1 (1 + tanh) → Hidden2 (1 + tanh) → Output (1)

**What's Inside**:
- Synthetic polynomial data (y = x² with noise)
- Two hidden layers with tanh activation
- Manual backpropagation through 2 layers
- Training: 2000 epochs, learning rate 0.01
- Visualizations: parabola approximation, parameter evolution, gradient flow
- Performance metrics: MSE, R², MAE

**Learning Objectives**:
- Understand why hidden layers are needed for non-linear functions
- Learn multi-layer backpropagation with chain rule
- Visualize how networks learn curved relationships
- Compare network predictions with true mathematical functions

**Key Insight**: Single perceptrons can only learn linear relationships. Two hidden layers enable learning quadratic (parabolic) patterns.

---

### 3. Two-Layer Network: Drug Dosage Response

**File**: `two_layer_drug_dosage_response.ipynb`

**Use Case**: Learn optimal drug dosage (inverted U-shaped curve)

**Architecture**: Input (1) → Hidden1 (1 + tanh) → Hidden2 (1 + tanh) → Output (1)

**What's Inside**:
- Synthetic drug dosage-response data (bell curve)
- Optimal dosage at 50mg with effectiveness peaking at 100%
- Two hidden layers to capture inverted parabola
- Training: 2000 epochs, learning rate 0.01
- Visualizations: dose-response curve, therapeutic window, safety zones
- Performance metrics: MSE, R², MAE, optimal dosage prediction

**Learning Objectives**:
- Apply neural networks to real-world medical problems
- Understand dose-response relationships
- Visualize therapeutic windows and safety margins
- Identify optimal dosage from learned patterns

**Key Insight**: Neural networks can learn complex medical relationships like optimal drug dosing, helping identify therapeutic windows and avoid underdose/overdose risks.

**Real-World Applications**:
- Personalized medicine
- Drug development
- Clinical decision support
- Safety analysis

---

## 📊 Comparison of Implementations

| Feature | Single Perceptron | Polynomial (2-Layer) | Drug Dosage (2-Layer) |
|---------|------------------|---------------------|---------------------|
| **Architecture** | 1 → 1 | 1 → 1 → 1 → 1 | 1 → 1 → 1 → 1 |
| **Activation** | None | tanh | tanh |
| **Relationship** | Linear | Quadratic (U-shape) | Inverted Quadratic |
| **Use Case** | House prices | Math function | Medical dosing |
| **Complexity** | Simple | Moderate | Moderate |
| **Epochs** | 500 | 2000 | 2000 |
| **Key Learning** | Gradient descent | Non-linear patterns | Real-world application |

## 🔍 Viewing Training Metrics

### TensorBoard

After running the notebook, view real-time training metrics:

```bash
tensorboard --logdir=logs/tensorboard
```

Then open your browser and navigate to: `http://localhost:6006`

**TensorBoard shows**:
- Loss curves
- Weight/bias values over time
- Gradient magnitudes
- Parameter distributions

### MLflow

View experiment tracking and compare runs:

```bash
mlflow ui
```

Then open your browser and navigate to: `http://localhost:5000`

**MLflow shows**:
- Hyperparameters
- Metrics comparison
- Model artifacts
- Experiment history

## 📈 Expected Results

After training, you should see:

- **Final Training Loss**: ~0.001-0.01 (normalized)
- **Test R² Score**: ~0.95-0.99
- **Mean Absolute Error**: ~$5,000-$15,000
- **Learned Relationship**: Close to `Price = 150 × Size + 50,000`

## 🧠 Key Concepts Demonstrated

### 1. Forward Propagation
```
y_pred = weight × input + bias
```

### 2. Loss Function (MSE)
```
Loss = (1/n) × Σ(y_pred - y_true)²
```

### 3. Backpropagation
```
∂Loss/∂weight = (2/n) × Σ(y_pred - y_true) × input
∂Loss/∂bias = (2/n) × Σ(y_pred - y_true)
```

### 4. Gradient Descent Update
```
weight = weight - learning_rate × ∂Loss/∂weight
bias = bias - learning_rate × ∂Loss/∂bias
```

## 📁 Project Structure

```
.
├── single_perceptron_house_price.ipynb          # Single perceptron (linear)
├── two_layer_polynomial_approximation.ipynb     # 2-layer network (quadratic)
├── two_layer_drug_dosage_response.ipynb         # 2-layer network (medical)
├── requirements.txt                                      # Python dependencies
├── README.md                                             # This file
├── single_perceptron_gradient_derivation_explained.md    # Single perceptron math
├── two_layer_polynomial_gradient_derivation_explained.md # Polynomial math
├── two_layer_drug_dosage_gradient_derivation_explained.md # Drug dosage math
├── why_averaging_in_loss_function.md                     # Loss function details
├── logs/                                        # TensorBoard logs (generated)
│   └── tensorboard/
├── mlruns/                                      # MLflow tracking (generated)
└── *.png                                        # Generated visualizations
```

## 🎯 Learning Path

**Recommended Order**:

1. **Start with Single Perceptron** (`single_perceptron_house_price.ipynb`)
   - Learn the basics: forward propagation, backpropagation, gradient descent
   - Understand linear relationships
   - Master visualization tools (TensorBoard, MLflow)

2. **Progress to Polynomial Approximation** (`two_layer_polynomial_approximation.ipynb`)
   - Understand why hidden layers are necessary
   - Learn multi-layer backpropagation
   - See how networks learn non-linear patterns

3. **Apply to Real-World Problem** (`two_layer_drug_dosage_response.ipynb`)
   - Apply concepts to medical domain
   - Understand practical implications
   - Learn about safety and optimization

## 🎯 Key Learning Objectives

This project helps you understand:

1. **Neural Network Fundamentals**: From single perceptrons to multi-layer networks
2. **Gradient Descent**: How models learn through iterative optimization
3. **Backpropagation**: Manual gradient computation through multiple layers
4. **Activation Functions**: Role of non-linear activations (tanh)
5. **Experiment Tracking**: Using TensorBoard and MLflow for ML workflows
6. **Problem Complexity**: Linear vs non-linear relationships
7. **Real-World Applications**: Medical dosing, function approximation

## 🔧 Customization

Each notebook allows you to modify hyperparameters:

**Common Parameters**:
- `LEARNING_RATE`: Default 0.01 (try 0.001, 0.1)
- `EPOCHS`: 500-2000 depending on complexity
- `NUM_SAMPLES`: Number of training samples
- `NOISE_STD`: Amount of noise in synthetic data

**Network-Specific**:
- Single Perceptron: `true_weight`, `true_bias`
- Polynomial: `INPUT_RANGE` (-3 to 3)
- Drug Dosage: `OPTIMAL_DOSAGE`, `MAX_EFFECTIVENESS`

## 📝 Notes

- The model uses **normalized data** for stable training
- All gradients are computed **manually** (not using automatic differentiation)
- The implementation is **educational** - production code would use higher-level APIs
- TensorBoard and MLflow logs are saved locally

## 🤝 Contributing

Feel free to:
- Experiment with different hyperparameters
- Add more features (e.g., multiple inputs)
- Try different optimizers
- Extend to multi-layer networks

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Neural Networks Basics](https://www.tensorflow.org/guide/keras/sequential_model)

## ✨ Key Takeaways

1. **Single perceptrons** can only learn linear relationships
2. **Hidden layers with activation functions** enable learning non-linear patterns
3. **Two hidden layers** (even with 1 neuron each) can approximate quadratic functions
4. **Gradient descent** iteratively updates parameters to minimize loss
5. **Backpropagation** uses chain rule to compute gradients through multiple layers
6. **Visualization tools** (TensorBoard, MLflow) are essential for understanding training
7. **Neural networks** can solve real-world problems like optimal drug dosing
8. Understanding **fundamentals** is crucial before moving to complex architectures

## 🌟 Why This Matters

- **Educational**: Learn by implementing from scratch, not using black-box libraries
- **Foundational**: Master concepts that apply to all neural networks
- **Practical**: See real-world applications (medical dosing)
- **Visual**: Comprehensive visualizations aid understanding
- **Progressive**: Build complexity gradually from simple to advanced

---

**Happy Learning! 🎓**

For questions or issues, please refer to the notebook's detailed markdown explanations.
