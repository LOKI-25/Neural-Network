#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Starter Code - Build a Neural Network from Scratch

Instructions:
- Complete all TODO sections
- Use only NumPy for computations
- Follow the docstring specifications carefully
- Run test_solution.py to verify your implementation
"""

import numpy as np
import struct
import gzip
from typing import List, Tuple, Dict
import pickle
import time


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mnist(path='data/'):
    """
    Load MNIST dataset from files or download if not present.
    
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    import os
    import urllib.request
    
    # Create data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # MNIST file information
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files if not present
    base_url = 'https://github.com/fgnt/mnist'
    for file in files.values():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
    
    # Load data
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # Normalize to [0, 1]
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    X_train = load_images(os.path.join(path, files['train_images']))
    y_train = load_labels(os.path.join(path, files['train_labels']))
    X_test = load_images(os.path.join(path, files['test_images']))
    y_test = load_labels(os.path.join(path, files['test_labels']))
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# ============================================================================
# Layer Implementations
# ============================================================================

class Layer:
    """Base class for all layers."""
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dL_dY):
        raise NotImplementedError
    
    def get_params(self):
        return {}
    
    def get_grads(self):
        return {}
    
    def set_params(self, params):
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        weight_init: Weight initialization method ('xavier', 'he', 'normal')
    """
    def __init__(self, input_dim, output_dim, weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # TODO: Initialize weights and biases
        # Hint: Use different initialization strategies:
        # - 'xavier': sqrt(2 / (input_dim + output_dim))
        # - 'he': sqrt(2 / input_dim)
        # - 'normal': standard normal * 0.01
        self.W = None  # Shape: (input_dim, output_dim)
        self.b = None  # Shape: (output_dim,)
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (input_dim + output_dim))
            self.W = np.random.randn(input_dim, output_dim) * scale
        elif weight_init == 'he':
            scale = np.sqrt(2.0 / input_dim)
            self.W = np.random.randn(input_dim, output_dim) * scale
        elif weight_init == 'normal':
            self.W = np.random.randn(input_dim, output_dim) * 0.01
        
        # Storage for backward pass
        self.X = None
        self.dW = None
        self.db = None
        self.b = np.zeros(output_dim)
    
    def forward(self, X):
        """
        Forward pass: Y = XW + b
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            
        Returns:
            Y: Output data, shape (batch_size, output_dim)
        """
        # TODO: Implement forward pass
        # Store X for backward pass
        self.X = X
        return np.dot(X, self.W) + self.b

    
    def backward(self, dL_dY):
        """
        Backward pass: compute gradients.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output, shape (batch_size, output_dim)
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input, shape (batch_size, input_dim)
        """
        # TODO: Compute gradients
        # dL_dW = X.T @ dL_dY
        # dL_db = sum(dL_dY, axis=0)
        # dL_dX = dL_dY @ W.T
        self.dW = self.X.T @ dL_dY
        self.db = np.sum(dL_dY, axis=0)
        dL_dX = dL_dY @ self.W.T
        return dL_dX
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']


# ============================================================================
# Activation Functions
# ============================================================================

class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self):
        self.cache = None


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)
        
        Args:
            X: Input data
            
        Returns:
            Output after applying ReLU
        """
        self.cache = X
        return np.maximum(0, X)
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = 1 if x > 0 else 0
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # TODO: Implement ReLU backward pass
        dL_dX = dL_dY.copy()
        dL_dX[self.cache <= 0] = 0
        return dL_dX


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = 1 / (1 + exp(-x))
        
        Args:
            X: Input data
            
        Returns:
            Output after applying sigmoid
        """
        # TODO: Implement sigmoid forward pass
        # Store output for backward pass
        output = 1 / (1 + np.exp(-X))
        self.cache = output
        return output
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = f(x) * (1 - f(x))
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # TODO: Implement sigmoid backward pass
        sig = self.cache
        return dL_dY * sig * (1 - sig)


class Softmax(Activation):
    """Softmax activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x_i) = exp(x_i) / sum(exp(x))
        
        Args:
            X: Input data, shape (batch_size, num_classes)
            
        Returns:
            Output probabilities, shape (batch_size, num_classes)
        """
        # TODO: Implement softmax forward pass
        # Hint: Subtract max for numerical stability

        stable_X = X - np.max(X, axis=1, keepdims=True)
        
        # Exponentiate the stable values
        exps = np.exp(stable_X)
        
        # Normalize to get probabilities
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # Store the output for the backward pass
        self.cache = probs
        return probs
    
    def backward(self, dL_dY):
        """
        Backward pass for softmax.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        Y = self.cache
        s = np.sum(dL_dY * Y, axis=1, keepdims=True)
        dL_dX = Y * (dL_dY - s)
        
        return dL_dX


# ============================================================================
# Loss Functions
# ============================================================================

class Loss:
    """Base class for loss functions."""
    def compute(self, y_pred, y_true):
        raise NotImplementedError
    
    def gradient(self, y_pred, y_true):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def compute(self, y_pred, y_true):
        """
        Compute MSE loss: L = 0.5 * mean((y_pred - y_true)^2)
        
        Args:
            y_pred: Predictions, shape (batch_size, num_features)
            y_true: True values, shape (batch_size, num_features)
            
        Returns:
            Scalar loss value
        """
        # TODO: Implement MSE loss
        return 0.5 * np.mean((y_pred - y_true) ** 2)
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.
        
        Args:
            y_pred: Predictions
            y_true: True values
            
        Returns:
            Gradient w.r.t. predictions
        """
        return (y_pred - y_true) / 10
        


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification."""
    
    def compute(self, y_pred, y_true):
        """
        Compute cross-entropy loss: L = -mean(sum(y_true * log(y_pred)))
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
        # TODO: Implement cross-entropy loss
        # Add small epsilon to prevent log(0)
        loss = -np.sum(y_true * np.log(y_pred + 1e-9))
        return loss / y_pred.shape[0]
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot)
            
        Returns:
            Gradient w.r.t. predictions
        """
        # TODO: Implement cross-entropy gradient
        # For softmax + cross-entropy: gradient = (y_pred - y_true) / batch_size
        return (y_pred - y_true) / y_pred.shape[0]


# ============================================================================
# Optimizers
# ============================================================================

class Optimizer:
    """Base class for optimizers."""
    def update(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using vanilla SGD.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        # TODO: Implement SGD update rule
        # params = params - learning_rate * grads
        for key in params:
            params[key] = params[key] - self.lr * grads[key]


class Momentum(Optimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        # TODO: Implement momentum update rule
        # v = momentum * v - learning_rate * grads
        # params = params + v
        
        for key in params:

            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
                
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key] = params[key] + self.velocity[key]


# ============================================================================
# Neural Network Class
# ============================================================================

class NeuralNetwork:
    """
    Modular neural network implementation.
    
    Example usage:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.01))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """Configure the model for training."""
        self.loss_fn = loss
        self.optimizer = optimizer
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output of the network
        """
        # TODO: Implement forward pass through all layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dL_dY):
        """
        Backward propagation through all layers.
        
        Args:
            dL_dY: Gradient of loss w.r.t. network output
        """
        # TODO: Implement backward pass through all layers in reverse order
        grad = dL_dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_params(self):
        """Update parameters of all trainable layers using the optimizer."""
        # TODO: Collect parameters and gradients from all layers
        for li, layer in enumerate(self.layers):
            params = layer.get_params()
            if not params:
                continue
            grads = layer.get_grads()

            # Namespace keys to avoid collisions across layers
            ns_params = {(li, k): v for k, v in params.items()}
            ns_grads  = {(li, k): v for k, v in grads.items()}

            self.optimizer.update(ns_params, ns_grads)

            # Strip namespacing and write updated arrays back
            for k in params:
                params[k] = ns_params[(li, k)]
            layer.set_params(params)

    
    def fit(self, X_train, y_train, epochs, batch_size, 
            X_val=None, y_val=None, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'train_acc': [], 
                   'val_loss': [], 'val_acc': []}
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # TODO: Implement training loop
            # 1. Shuffle training data
            # 2. Process mini-batches
            # 3. Forward pass
            # 4. Compute loss
            # 5. Backward pass
            # 6. Update parameters
            # 7. Track metrics
            epoch_start_time = time.time()
            # Shuffle training data
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            epoch_loss = 0
            epoch_correct = 0
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 1. Forward pass
                y_pred = self.forward(X_batch)
                
                # 2. Compute loss
                loss = self.loss_fn.compute(y_pred, y_batch)
                epoch_loss += loss * X_batch.shape[0]
                
                # 3. Compute initial gradient
                grad = self.loss_fn.gradient(y_pred, y_batch)
                
                # 4. Backward pass
                self.backward(grad)
                
                # 5. Update parameters
                self.update_params()
                
                # Track training accuracy
                epoch_correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

            # Calculate and store metrics for the epoch
            avg_train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            val_info = ""
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                val_info = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_train_loss:.4f} - acc: {train_acc:.4f}{val_info}")
            
                # Print metrics
            
        return history
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions (class indices for classification)
        """
        # TODO: Forward pass and return predictions
        y_pred_probs = self.forward(X)
        return np.argmax(y_pred_probs, axis=1)
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            loss, accuracy
        """
        # TODO: Compute loss and accuracy
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y_pred, y)
        
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        
        return loss, accuracy
    
    def save_weights(self, filename):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                weights[f'layer_{i}'] = layer.get_params()
        np.savez(filename, **weights)
    
    def load_weights(self, filename):
        """Load model weights from file."""
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in weights:
                layer.set_params(weights[f'layer_{i}'])


# ============================================================================
# Gradient Checking
# ============================================================================

def gradient_check(model, X, y, epsilon=1e-7):
    """
    Verify gradients using finite differences.
    
    Args:
        model: Neural network model
        X: Sample input data
        y: Sample labels
        epsilon: Small value for numerical differentiation
        
    Returns:
        Dictionary with gradient checking results
    """
    # TODO: Implement gradient checking
    # 1. Compute gradients using backpropagation
    # 2. Compute numerical gradients using finite differences
    # 3. Compare and return relative error
    y_pred = model.forward(X)
    grad = model.loss_fn.gradient(y_pred, y)
    model.backward(grad)
    
    results = {}
    
    # Iterate through each layer and each parameter
    for l_idx, layer in enumerate(model.layers):
        params = layer.get_params()
        if not params:
            continue
            
        grads = layer.get_grads()
        
        for p_name, param in params.items():
            grad_analytic = grads[p_name]
            grad_numerical = np.zeros_like(param)
            
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                
                # Compute loss for theta + epsilon
                original_value = param[ix]
                param[ix] = original_value + epsilon
                loss_plus = model.loss_fn.compute(model.forward(X), y)
                
                # Compute loss for theta - epsilon
                param[ix] = original_value - epsilon
                loss_minus = model.loss_fn.compute(model.forward(X), y)
                
                # Restore original parameter value
                param[ix] = original_value
                
                # Compute numerical gradient
                grad_numerical[ix] = (loss_plus - loss_minus) / (2 * epsilon)
                
                it.iternext()

            # Compare numerical and analytical gradients
            numerator = np.linalg.norm(grad_analytic - grad_numerical)
            denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numerical)
            
            # Avoid division by zero
            if denominator == 0:
                relative_error = 0
            else:
                relative_error = numerator / denominator

            key = f"Layer_{l_idx}_{p_name}"
            print(f"Gradient check for {key}: Relative error = {relative_error}")
            results[key] = relative_error
            
    return results


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Create model
    print("Building neural network...")
    model = NeuralNetwork()
    
    # TODO: Build the network architecture
    # Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax
    model.add(Dense(input_dim=784, output_dim=128, weight_init='he'))
    model.add(ReLU())
    model.add(Dense(input_dim=128, output_dim=64, weight_init='he'))
    model.add(ReLU())
    model.add(Dense(input_dim=64, output_dim=10, weight_init='xavier'))
    model.add(Softmax())
    # TODO: Compile model with CrossEntropyLoss and SGD optimizer

    model.compile(loss=CrossEntropyLoss(), optimizer=Momentum(learning_rate=0.05, momentum=0.9))
    

    # TODO: Train the model
    print("Training model...")
    # history = model.fit(...)
    history = model.fit(X_train, y_train_oh, epochs=20, batch_size=32, 
                        X_val=X_test, y_val=y_test_oh, verbose=True)
    
    # TODO: Evaluate on test set
    print("Evaluating model...")
    # test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    
    # TODO: Save model weights
    # model.save_weights('model_weights.npz')
    model.save_weights('model_weights.pkl')
    print("\nModel weights saved to model_weights.pkl")
    
    # TODO: Save training log
    # with open('training_log.txt', 'w') as f:
    #     f.write("Training Log\n")
    #     ...
    with open('training_log.txt', 'w') as f:
        f.write("Training Log\n")
        f.write("="*30 + "\n")
        f.write(f"Final Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n\n")
        f.write("Epoch Details:\n")
        for i in range(len(history['train_loss'])):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss={history['train_loss'][i]:.4f}, "
                f"Train Acc={history['train_acc'][i]:.4f}, "
                f"Val Loss={history['val_loss'][i]:.4f}, "
                f"Val Acc={history['val_acc'][i]:.4f}\n"
            )
    print("Training log saved to training_log.txt")
    
    # TODO: Save sample predictions
    # predictions = model.predict(X_test[:100])
    # np.savetxt('predictions_sample.txt', predictions, fmt='%d')
    predictions = model.predict(X_test[:100])
    np.savetxt('predictions_sample.txt', predictions, fmt='%d')
    print("Sample predictions saved to predictions_sample.txt")
    
    print("Training complete!")