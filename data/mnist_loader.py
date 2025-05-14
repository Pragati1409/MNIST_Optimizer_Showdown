import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    """Load and preprocess MNIST dataset"""
    try:
        # Try to fetch MNIST data
        print("Attempting to load MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto', as_frame=False)
        X = mnist.data.astype('float32') / 255.0
        y = mnist.target.astype('int')
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Trying alternative method...")

        # Alternative method using direct URL
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', as_frame=False)
        X = mnist.data.astype('float32') / 255.0
        y = mnist.target.astype('int')

    # Convert labels to one-hot encoding
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1

    # Split into train/validation/test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test