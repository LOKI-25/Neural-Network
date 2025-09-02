#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Unit Tests for Student Implementation

Run this file to test your implementation:
    python test_solution.py

All tests should pass before submission.
"""

import numpy as np
import unittest
from starter_code import (
    Dense, ReLU, Sigmoid, Softmax,
    MSELoss, CrossEntropyLoss,
    SGD, Momentum,
    NeuralNetwork,
    gradient_check,
    one_hot_encode
)
import time

class TestLayers(unittest.TestCase):
    """Test layer implementations."""
    
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 32
        self.input_dim = 10
        self.output_dim = 5
    
    def test_dense_forward_shape(self):
        """Test Dense layer forward pass output shape."""
        layer = Dense(self.input_dim, self.output_dim)
        X = np.random.randn(self.batch_size, self.input_dim)
        Y = layer.forward(X)
        
        self.assertEqual(Y.shape, (self.batch_size, self.output_dim),
                        "Dense forward output shape incorrect")
    
    def test_dense_backward_shape(self):
        """Test Dense layer backward pass gradient shapes."""
        layer = Dense(self.input_dim, self.output_dim)
        X = np.random.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        Y = layer.forward(X)
        
        # Backward pass
        dL_dY = np.random.randn(*Y.shape)
        dL_dX = layer.backward(dL_dY)
        
        # Check shapes
        self.assertEqual(dL_dX.shape, X.shape,
                        "Dense backward input gradient shape incorrect")
        self.assertEqual(layer.dW.shape, layer.W.shape,
                        "Dense weight gradient shape incorrect")
        self.assertEqual(layer.db.shape, layer.b.shape,
                        "Dense bias gradient shape incorrect")
    
    def test_dense_gradient_computation(self):
        """Test Dense layer gradient computation correctness."""
        layer = Dense(3, 2)
        
        # Simple test case
        X = np.array([[1, 2, 3], [4, 5, 6]])
        layer.W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        layer.b = np.array([0.1, 0.2])
        
        # Forward pass
        Y = layer.forward(X)
        
        # Backward pass
        dL_dY = np.ones_like(Y)
        dL_dX = layer.backward(dL_dY)
        
        # Check gradient computations
        expected_dW = X.T @ dL_dY
        expected_db = np.sum(dL_dY, axis=0)
        expected_dX = dL_dY @ layer.W.T
        
        np.testing.assert_array_almost_equal(layer.dW, expected_dW,
                                           err_msg="Weight gradient incorrect")
        np.testing.assert_array_almost_equal(layer.db, expected_db,
                                           err_msg="Bias gradient incorrect")
        np.testing.assert_array_almost_equal(dL_dX, expected_dX,
                                           err_msg="Input gradient incorrect")


class TestActivations(unittest.TestCase):
    """Test activation function implementations."""
    
    def setUp(self):
        np.random.seed(42)
        self.X = np.array([[-2, -1, 0, 1, 2],
                          [-1, 0, 1, 2, 3]])
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLU()
        Y = relu.forward(self.X)
        
        expected = np.maximum(0, self.X)
        np.testing.assert_array_equal(Y, expected,
                                    "ReLU forward pass incorrect")
    
    def test_relu_backward(self):
        """Test ReLU backward pass."""
        relu = ReLU()
        Y = relu.forward(self.X)
        
        dL_dY = np.ones_like(Y)
        dL_dX = relu.backward(dL_dY)
        
        expected = (self.X > 0).astype(float)
        np.testing.assert_array_equal(dL_dX, expected,
                                    "ReLU backward pass incorrect")
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = Sigmoid()
        Y = sigmoid.forward(self.X)
        
        # Check output range
        self.assertTrue(np.all(Y >= 0) and np.all(Y <= 1),
                       "Sigmoid output not in range [0, 1]")
        
        # Check specific values
        self.assertAlmostEqual(Y[0, 2], 0.5, places=5,
                             msg="Sigmoid(0) should be 0.5")
    
    def test_sigmoid_backward(self):
        """Test Sigmoid backward pass."""
        sigmoid = Sigmoid()
        Y = sigmoid.forward(self.X)
        
        dL_dY = np.ones_like(Y)
        dL_dX = sigmoid.backward(dL_dY)
        
        # Gradient should be Y * (1 - Y)
        expected = Y * (1 - Y)
        np.testing.assert_array_almost_equal(dL_dX, expected,
                                           err_msg="Sigmoid backward pass incorrect")
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        softmax = Softmax()
        X = np.array([[1, 2, 3], [2, 3, 4]])
        Y = softmax.forward(X)
        
        # Check that each row sums to 1
        row_sums = np.sum(Y, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2),
                                           err_msg="Softmax rows don't sum to 1")
        
        # Check that all values are positive
        self.assertTrue(np.all(Y > 0), "Softmax output contains non-positive values")
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        softmax = Softmax()
        X = np.array([[1000, 1001, 1002], [500, 501, 502]])
        Y = softmax.forward(X)
        
        # Should not have NaN or Inf
        self.assertFalse(np.any(np.isnan(Y)), "Softmax produced NaN values")
        self.assertFalse(np.any(np.isinf(Y)), "Softmax produced Inf values")


class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""
    
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 10
        self.num_features = 5
    
    def test_mse_loss(self):
        """Test MSE loss computation."""
        mse = MSELoss()
        
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1.5, 2.5], [2.5, 3.5]])
        
        loss = mse.compute(y_pred, y_true)
        expected_loss = 0.5 * np.mean((y_pred - y_true) ** 2)
        
        self.assertAlmostEqual(loss, expected_loss, places=5,
                             msg="MSE loss computation incorrect")
    
    def test_mse_gradient(self):
        """Test MSE gradient computation."""
        mse = MSELoss()
        
        y_pred = np.random.randn(self.batch_size, self.num_features)
        y_true = np.random.randn(self.batch_size, self.num_features)
        
        grad = mse.gradient(y_pred, y_true)
        expected_grad = (y_pred - y_true) / self.batch_size
        
        np.testing.assert_array_almost_equal(grad, expected_grad,
                                           err_msg="MSE gradient incorrect")
    
    def test_cross_entropy_loss(self):
        """Test Cross-Entropy loss computation."""
        ce = CrossEntropyLoss()
        
        # Perfect prediction
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        y_true = np.array([[1, 0], [0, 1]])
        
        loss = ce.compute(y_pred, y_true)
        self.assertLess(loss, 0.2, "Cross-entropy loss too high for good predictions")
        
        # Bad prediction
        y_pred_bad = np.array([[0.1, 0.9], [0.9, 0.1]])
        loss_bad = ce.compute(y_pred_bad, y_true)
        self.assertGreater(loss_bad, loss, 
                          "Bad predictions should have higher loss")
    
    def test_cross_entropy_gradient(self):
        """Test Cross-Entropy gradient with softmax."""
        ce = CrossEntropyLoss()
        
        # For softmax + cross-entropy, gradient is simply y_pred - y_true
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        
        grad = ce.gradient(y_pred, y_true)
        expected_grad = (y_pred - y_true) / 2  # batch_size = 2
        
        np.testing.assert_array_almost_equal(grad, expected_grad,
                                           err_msg="Cross-entropy gradient incorrect")


class TestOptimizers(unittest.TestCase):
    """Test optimizer implementations."""
    
    def test_sgd_update(self):
        """Test SGD parameter update."""
        sgd = SGD(learning_rate=0.1)
        
        params = {'W': np.array([[1, 2], [3, 4]]), 
                  'b': np.array([1, 2])}
        grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]]),
                 'b': np.array([0.1, 0.2])}
        
        # Make copies for comparison
        W_old = params['W'].copy()
        b_old = params['b'].copy()
        
        sgd.update(params, grads)
        
        # Check updates
        np.testing.assert_array_almost_equal(
            params['W'], W_old - 0.1 * grads['W'],
            err_msg="SGD weight update incorrect"
        )
        np.testing.assert_array_almost_equal(
            params['b'], b_old - 0.1 * grads['b'],
            err_msg="SGD bias update incorrect"
        )
    
    def test_momentum_update(self):
        """Test Momentum optimizer update."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        
        params = {'W': np.array([[1, 2], [3, 4]]), 
                  'b': np.array([1, 2])}
        grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]]),
                 'b': np.array([0.1, 0.2])}
        
        # First update
        params_copy = {k: v.copy() for k, v in params.items()}
        momentum.update(params, grads)
        
        # Velocity should be initialized
        self.assertIn('W', momentum.velocity)
        self.assertIn('b', momentum.velocity)
        
        # Second update to test momentum
        momentum.update(params, grads)
        
        # Parameters should have changed
        self.assertFalse(np.allclose(params['W'], params_copy['W']))
        self.assertFalse(np.allclose(params['b'], params_copy['b']))


class TestNeuralNetwork(unittest.TestCase):
    """Test neural network integration."""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_network_construction(self):
        """Test building a neural network."""
        model = NeuralNetwork()
        model.add(Dense(10, 5))
        model.add(ReLU())
        model.add(Dense(5, 2))
        model.add(Softmax())
        
        self.assertEqual(len(model.layers), 4,
                        "Network should have 4 layers")
    
    def test_forward_pass(self):
        """Test forward propagation through network."""
        model = NeuralNetwork()
        model.add(Dense(10, 5))
        model.add(ReLU())
        model.add(Dense(5, 2))
        model.add(Softmax())
        
        X = np.random.randn(32, 10)
        Y = model.forward(X)
        
        self.assertEqual(Y.shape, (32, 2),
                        "Network output shape incorrect")
        
        # Check softmax properties
        row_sums = np.sum(Y, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(32),
                                           err_msg="Softmax output doesn't sum to 1")
    
    def test_backward_pass(self):
        """Test backward propagation through network."""
        model = NeuralNetwork()
        model.add(Dense(10, 5))
        model.add(ReLU())
        model.add(Dense(5, 2))
        
        model.compile(loss=MSELoss(), optimizer=SGD(0.01))
        
        # Forward pass
        X = np.random.randn(32, 10)
        Y = model.forward(X)
        
        # Compute loss gradient
        y_true = np.random.randn(32, 2)
        dL_dY = model.loss_fn.gradient(Y, y_true)
        
        # Backward pass should not raise errors
        try:
            model.backward(dL_dY)
        except Exception as e:
            self.fail(f"Backward pass failed: {str(e)}")
    
    def test_parameter_update(self):
        """Test parameter updates during training."""
        model = NeuralNetwork()
        model.add(Dense(5, 3))
        model.add(ReLU())
        model.add(Dense(3, 2))
        
        model.compile(loss=MSELoss(), optimizer=SGD(0.1))
        
        # Get initial parameters
        initial_params = []
        for layer in model.layers:
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                initial_params.append({k: v.copy() for k, v in params.items()})
        
        # Perform one training step
        X = np.random.randn(10, 5)
        y_true = np.random.randn(10, 2)
        
        Y = model.forward(X)
        loss = model.loss_fn.compute(Y, y_true)
        dL_dY = model.loss_fn.gradient(Y, y_true)
        model.backward(dL_dY)
        model.update_params()
        
        # Check that parameters changed
        param_changed = False
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                for key in params:
                    if not np.allclose(params[key], initial_params[i][key]):
                        param_changed = True
                        break
        
        self.assertTrue(param_changed, "Parameters should change after update")


class TestGradientChecking(unittest.TestCase):
    """Test gradient checking functionality."""
    
    def test_gradient_check_dense(self):
        """Test gradient checking for dense layer."""
        # Create simple network
        model = NeuralNetwork()
        model.add(Dense(3, 2))
        model.compile(loss=MSELoss(), optimizer=SGD(0.01))
        
        # Small batch for gradient checking
        X = np.random.randn(5, 3)
        y = np.random.randn(5, 2)
        
        # Run gradient check
        results = gradient_check(model, X, y, epsilon=1e-7)
        
        # Check that relative errors are small
        if results is not None:
            for layer_name, error in results.items():
                self.assertLess(error, 1e-5,
                              f"Gradient check failed for {layer_name}")


class TestMNISTPerformance(unittest.TestCase):
    """Test model performance on MNIST subset."""
    
    @unittest.skipIf(not hasattr(unittest, 'skip'), "Skipping performance test")
    def test_mnist_training(self):
        """Test that model can train on MNIST subset."""
        # Create small dataset for testing
        np.random.seed(42)
        X_train = np.random.rand(1000, 784)
        y_train = np.random.randint(0, 10, 1000)
        y_train_oh = one_hot_encode(y_train)
        
        # Build model
        model = NeuralNetwork()
        model.add(Dense(784, 64))
        model.add(ReLU())
        model.add(Dense(64, 10))
        model.add(Softmax())
        
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.1))
        
        # Train for a few epochs
        history = model.fit(X_train, y_train_oh, epochs=5, batch_size=32, verbose=False)
        
        # Check that loss decreases
        if 'train_loss' in history and len(history['train_loss']) > 1:
            self.assertLess(history['train_loss'][-1], history['train_loss'][0],
                          "Training loss should decrease")


def run_basic_tests():
    """Run basic tests that must pass."""
    print("Running basic shape tests...")
    
    # Test Dense layer shapes
    layer = Dense(10, 5)
    X = np.random.randn(32, 10)
    Y = layer.forward(X)
    print(f"✓ Dense forward shape: {Y.shape}")
    
    # Test ReLU
    relu = ReLU()
    X = np.array([[-1, 0, 1], [2, -2, 3]])
    Y = relu.forward(X)
    print(f"✓ ReLU forward: {Y}")
    
    # Test Softmax
    softmax = Softmax()
    X = np.array([[1, 2, 3], [2, 3, 4]])
    Y = softmax.forward(X)
    print(f"✓ Softmax sums: {np.sum(Y, axis=1)}")
    
    print("\nAll basic tests passed!")


if __name__ == "__main__":
    # Run basic tests first
    run_basic_tests()
    
    print("\nRunning full test suite...")
    unittest.main(verbosity=2)