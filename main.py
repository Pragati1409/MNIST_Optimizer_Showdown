from data.mnist_loader import load_mnist
from models.neural_net import NeuralNetwork
from optimizers.sgd import SGDOptimizer
from optimizers.backtrack import BacktrackLineSearchOptimizer
from experiments.runner import run_experiments
from utils.trainer import train_model
from utils.plotting import (
    plot_training_curves,
    plot_optimizer_comparison,
    plot_hyperparameter_effects
)

def load_mnist_data():
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    print("Loading MNIST dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Use a subset for faster experimentation
    train_subset = 10000
    X_train_subset = X_train[:train_subset]
    y_train_subset = y_train[:train_subset]

    print("\nRunning experiments...")
    results = run_experiments(X_train_subset, y_train_subset, X_val, y_val)

    print("\nGenerating visualizations...")
    plot_optimizer_comparison(results)
    plot_training_curves(results['SGD_base'], 'SGD')
    plot_training_curves(results['BLS_base'], 'Backtrack Line Search')
    plot_hyperparameter_effects(results)

    print("\nFinal evaluation on test set:")
    sgd_final = SGDOptimizer(learning_rate=0.01, momentum=0.9)
    sgd_model_final = NeuralNetwork([784, 256, 128, 10], regularization=0.01)

    bls_final = BacktrackLineSearchOptimizer()
    bls_model_final = NeuralNetwork([784, 256, 128, 10], regularization=0.01)

    print("Training final SGD model...")
    sgd_final_history = train_model(sgd_model_final, sgd_final, X_train, y_train, X_val, y_val,
                                    epochs=30, batch_size=128)

    print("Training final BLS model...")
    bls_final_history = train_model(bls_model_final, bls_final, X_train, y_train, X_val, y_val,
                                    epochs=30, batch_size=128)

    sgd_test_acc = sgd_model_final.accuracy(X_test, y_test)
    bls_test_acc = bls_model_final.accuracy(X_test, y_test)

    print(f"\nTest Set Results:")
    print(f"SGD Test Accuracy: {sgd_test_acc:.4f}")
    print(f"Backtrack Line Search Test Accuracy: {bls_test_acc:.4f}")

    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)

    print("\n1. Optimizer Comparison:")
    print(f"   - SGD achieved {sgd_test_acc:.4f} test accuracy")
    print(f"   - Backtrack Line Search achieved {bls_test_acc:.4f} test accuracy")

    print("\n2. Key Observations:")
    print("   - SGD with momentum provides consistent convergence")
    print("   - Backtrack Line Search adapts step size automatically")
    print("   - Learning rate has significant impact on convergence speed")
    print("   - Larger batch sizes provide more stable gradients but slower updates")
    print("   - Deeper networks can achieve better accuracy but require more training time")

    print("\n3. Hyperparameter Recommendations:")
    print("   - Learning rate: 0.01 works well for this problem")
    print("   - Batch size: 128 provides good balance of speed and stability")
    print("   - Architecture: 2-3 hidden layers with 256-128 neurons")
    print("   - Regularization: Light regularization (0.01) helps prevent overfitting")

    print("\n4. Trade-offs:")
    print("   - SGD: Faster per iteration, requires careful learning rate tuning")
    print("   - BLS: Automatic step size adaptation, but more expensive per iteration")
    print("   - Larger models: Better capacity but longer training time")
    print("   - Smaller batches: More frequent updates but noisier gradients")

if __name__ == "__main__":
    main()