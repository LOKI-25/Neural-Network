# Assignment 1: Neural Network Fundamentals
**CS5720 - Deep Learning**

## Learning Objectives

By completing this assignment, you will:
1. **Understand Forward Propagation**: Implement the forward pass through a neural network from scratch
2. **Master Backward Propagation**: Derive and implement backpropagation algorithm step by step
3. **Implement Core Components**: Build essential neural network components (layers, activations, loss functions)
4. **Apply Gradient Descent**: Implement various optimization algorithms (SGD, Momentum)
5. **Debug Neural Networks**: Use gradient checking to verify your implementation
6. **Solve Real Problems**: Apply your network to MNIST digit classification

## Assignment Overview

In this assignment, you will build a fully-functional neural network library from scratch using only NumPy. You'll implement all core components of a modern neural network and train it on the MNIST dataset to classify handwritten digits.

### Key Components to Implement:
- **Layers**: Dense (fully connected) layers with forward and backward passes
- **Activation Functions**: ReLU, Sigmoid, and Softmax
- **Loss Functions**: Mean Squared Error and Cross-Entropy
- **Optimizers**: Stochastic Gradient Descent with optional momentum
- **Neural Network Class**: Orchestrates all components for training and inference

## Detailed Specifications

### 1. Layer Implementation (15 points)
Implement the following layer types in `starter_code.py`:

#### Dense Layer
```python
class Dense:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and biases
        # W shape: (input_dim, output_dim)
        # b shape: (output_dim,)
        
    def forward(self, X):
        # Compute: output = XW + b
        # Store inputs for backward pass
        
    def backward(self, dL_dY):
        # Compute gradients w.r.t weights, biases, and inputs
        # dL_dW = X.T @ dL_dY
        # dL_db = sum(dL_dY, axis=0)
        # dL_dX = dL_dY @ W.T
```

### 2. Activation Functions (10 points)
Implement these activation functions with their derivatives:

#### ReLU
- Forward: `f(x) = max(0, x)`
- Backward: `f'(x) = 1 if x > 0 else 0`

#### Sigmoid
- Forward: `f(x) = 1 / (1 + exp(-x))`
- Backward: `f'(x) = f(x) * (1 - f(x))`

#### Softmax
- Forward: `f(x_i) = exp(x_i) / sum(exp(x))`
- Backward: Jacobian matrix computation

### 3. Loss Functions (10 points)
Implement loss functions with their gradients:

#### Mean Squared Error (MSE)
- Loss: `L = 0.5 * mean((y_pred - y_true)^2)`
- Gradient: `dL/dy_pred = (y_pred - y_true) / batch_size`

#### Cross-Entropy Loss
- Loss: `L = -mean(sum(y_true * log(y_pred)))`
- Gradient: Simplified for softmax output

### 4. Optimizers (5 points)
Implement optimization algorithms:

#### SGD
```python
class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        
    def update(self, params, grads):
        # params = params - lr * grads
```

#### SGD with Momentum
```python
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, params, grads):
        # v = momentum * v - lr * grads
        # params = params + v
```

### 5. Neural Network Class (5 points)

### 6. Gradient Checking (10 points)
Implement numerical gradient verification to validate your backpropagation:
- Compare analytical gradients with numerical approximations
- Use relative error threshold (< 1e-5) for validation
- Test gradients for all layer parameters

### 7. MNIST Performance Testing (20 points)
Train and evaluate your network on MNIST digit classification:
- Achieve reasonable accuracy on test data
- Demonstrate proper training convergence
- Handle real dataset preprocessing and batching

### 8. Code Quality and Documentation (10 points)
Maintain professional code standards:
- Clean, readable implementation
- Proper error handling and edge cases
- No use of prohibited deep learning frameworks

### 9. Numerical Stability (5 points)
Ensure robust numerical implementations:
- Handle large values in softmax and sigmoid
- Prevent overflow/underflow in loss computations
- Maintain numerical precision throughout training

### 10. Efficiency and Performance (10 points)
Optimize for computational efficiency:
- Use vectorized operations where possible
- Reasonable training speed on provided datasets
- Memory-efficient implementation
Complete the `NeuralNetwork` class that:
- Manages layers in sequence
- Implements forward propagation through all layers
- Implements backward propagation in reverse order
- Provides training loop with batch processing
- Includes prediction and evaluation methods

## Implementation Requirements

### Forward Pass
1. Input data flows through each layer sequentially
2. Each layer transforms its input and passes to the next
3. Final layer produces predictions
4. Loss is computed from predictions and targets

### Backward Pass
1. Compute loss gradient w.r.t. predictions
2. Propagate gradients backward through each layer
3. Each layer computes gradients w.r.t. its parameters and inputs
4. Store parameter gradients for optimizer updates

### Gradient Checking
Implement numerical gradient checking to verify your backward pass:
```python
def gradient_check(network, X, y, epsilon=1e-7):
    # For each parameter:
    # 1. Compute numerical gradient using finite differences
    # 2. Compare with analytical gradient from backprop
    # 3. Ensure relative error < 1e-5
```

## MNIST Classification Task

### Dataset Details
- **Training Set**: 60,000 grayscale images (28x28 pixels)
- **Test Set**: 10,000 grayscale images
- **Classes**: 10 (digits 0-9)
- **Preprocessing**: Flatten to 784 features, normalize to [0, 1]

### Network Architecture
Implement and train this architecture:
```
Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax
```

### Training Requirements
- **Batch Size**: 32
- **Epochs**: 10
- **Learning Rate**: Start with 0.01, tune as needed
- **Target Accuracy**: Achieve at least 95% test accuracy

### Performance Benchmarks
Your implementation should achieve:
- Training time: < 5 minutes for 10 epochs
- Memory usage: < 500 MB
- Test accuracy: > 95%

## Submission Guidelines

### File Naming Convention
**IMPORTANT**: Your submission file must be named using your 9-digit student ID followed by `_pa1.py`

**Format**: `[9-digit-student-id]_pa1.py`  
**Example**: `700762718_pa1.py`

### Submission Requirements
1. **Single Python File**: Submit only your completed `700762718_pa1.py` file (replace with your actual student ID)
2. **File Contents**: Your complete implementation with all required classes and methods
3. **Student Information**: Include your name and 9-digit student ID in comments at the top of your file:
   ```python
   """
   CS5720 - Assignment 1: Neural Network Fundamentals
   Student Name: [Your Name]
   Student ID: 700762718
   """
   ```

### Code Requirements
- Use only NumPy (no TensorFlow, PyTorch, scikit-learn, etc.)
- Follow the provided class structure in `starter_code.py`
- Include docstrings for all methods
- Add comments explaining key algorithms (when necessary)
- Pass all unit tests in `test_solution.py`
- Handle all edge cases properly

### Brightspace Submission Instructions

1. **File Name**: Your Python file must be named `700762718_pa1.py` (replace with your actual 9-digit student ID)
2. **File Format**: Submit only the Python file (.py) - no zip files, reports, or additional materials
3. **Student ID**: Include your 9-digit student ID as a comment at the top of your file
4. **Submit to**: CS5720 Assignment 1 submission area in Brightspace
5. **Deadline**: Check the course schedule in Brightspace for the exact due date

### Before Submission Checklist:
□ Run `python test_solution.py` to verify your implementation passes all tests  
□ Run `python validate_submission.py 700762718_pa1.py` to check submission format  
□ Ensure your file follows the naming convention: `[9-digit-ID]_pa1.py`  
□ Test your network on MNIST and verify it achieves target accuracy  
□ Include your name and student ID in the file header comments  
□ Verify no prohibited libraries are imported

## Grading Criteria

### Correctness (50%)
- Layer implementations pass unit tests (15%)
- Activation functions are correct (10%)
- Loss functions compute proper gradients (10%)
- Gradient checking passes (10%)
- Network trains successfully (5%)

### Performance (30%)
- Achieves required test accuracy (20%)
- Training completes within time limit (5%)
- Memory usage is efficient (5%)

### Code Quality (20%)
- Clear code structure and organization (10%)
- Comprehensive documentation (5%)
- Follows Python best practices (5%)

## Tips for Success

1. **Start with Gradient Checking**: Implement and verify each component individually
2. **Use Small Networks First**: Test with 2-3 neurons before scaling up
3. **Monitor Training**: Plot loss curves to diagnose issues
4. **Debug Systematically**: Check shapes, verify mathematically, use print statements
5. **Vectorize Operations**: Use NumPy broadcasting for efficiency

## Common Pitfalls to Avoid

1. **Shape Mismatches**: Always verify tensor dimensions
2. **Numerical Instability**: Add small epsilon to denominators
3. **Gradient Explosions**: Check for NaN/Inf values
4. **Memory Leaks**: Don't store unnecessary intermediate values
5. **Wrong Gradient Signs**: Remember to negate loss gradients

## Resources

- [Backpropagation Explained](http://cs231n.github.io/optimization-2/)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)

## Self-Directed Learning Approach

This course follows a self-directed learning model. Use the provided materials, test your implementation thoroughly with the testing script, and refer to the neural network resources for guidance. Focus on understanding backpropagation and gradient descent fundamentals.

## Academic Integrity

This is an individual assignment. You may discuss concepts with classmates, but all code must be your own. Do not share implementations or look at others' code.

---

Good luck! Building a neural network from scratch is challenging but incredibly rewarding. You'll gain deep insights into how modern deep learning frameworks operate under the hood.