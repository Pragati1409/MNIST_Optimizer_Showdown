import numpy as np
from models.neural_net import NeuralNetwork
from optimizers.sgd import SGDOptimizer
from optimizers.backtrack import BacktrackLineSearchOptimizer
from utils.trainer import train_model

def run_experiments(X_train, y_train, X_val, y_val):
    """Run experiments with different hyperparameters"""
    results = {}

    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 128, 512]
    architectures = [
        [784, 128, 10],
        [784, 256, 128, 10],
        [784, 512, 256, 128, 10]
    ]
    regularization_strengths = [0.0, 0.01, 0.1]

    # Base configuration for comparison
    base_config = {
        'lr': 0.01,
        'batch_size': 128,
        'architecture': [784, 256, 128, 10],
        'regularization': 0.01
    }

    # Test SGD optimizer
    print("Testing SGD Optimizer...")
    sgd_optimizer = SGDOptimizer(learning_rate=base_config['lr'], momentum=0.9)
    sgd_model = NeuralNetwork(base_config['architecture'],
                             regularization=base_config['regularization'])
    sgd_history = train_model(sgd_model, sgd_optimizer, X_train, y_train, X_val, y_val,
                             epochs=30, batch_size=base_config['batch_size'])
    results['SGD_base'] = sgd_history

    # Test Backtrack Line Search
    print("\nTesting Backtrack Line Search Optimizer...")
    bls_optimizer = BacktrackLineSearchOptimizer()
    bls_model = NeuralNetwork(base_config['architecture'],
                             regularization=base_config['regularization'])
    bls_history = train_model(bls_model, bls_optimizer, X_train, y_train, X_val, y_val,
                             epochs=30, batch_size=base_config['batch_size'])
    results['BLS_base'] = bls_history

    # Learning rate experiments with SGD
    print("\nTesting different learning rates with SGD...")
    for lr in learning_rates:
        print(f"Learning rate: {lr}")
        optimizer = SGDOptimizer(learning_rate=lr, momentum=0.9)
        model = NeuralNetwork(base_config['architecture'],
                             regularization=base_config['regularization'])
        history = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             epochs=20, batch_size=base_config['batch_size'],
                             verbose=False)
        results[f'SGD_lr_{lr}'] = history

    # Batch size experiments with SGD
    print("\nTesting different batch sizes with SGD...")
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        optimizer = SGDOptimizer(learning_rate=base_config['lr'], momentum=0.9)
        model = NeuralNetwork(base_config['architecture'],
                             regularization=base_config['regularization'])
        history = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             epochs=20, batch_size=batch_size, verbose=False)
        results[f'SGD_batch_{batch_size}'] = history

    # Architecture experiments
    print("\nTesting different architectures with SGD...")
    for i, architecture in enumerate(architectures):
        print(f"Architecture: {architecture}")
        optimizer = SGDOptimizer(learning_rate=base_config['lr'], momentum=0.9)
        model = NeuralNetwork(architecture, regularization=base_config['regularization'])
        history = train_model(model, optimizer, X_train, y_train, X_val, y_val,
                             epochs=20, batch_size=base_config['batch_size'],
                             verbose=False)
        results[f'SGD_arch_{i}'] = history

    return results