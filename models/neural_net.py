import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', regularization=0.0):
        """
        Initialize neural network with given architecture
        layer_sizes: list of layer sizes (e.g., [784, 256, 128, 10])
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.regularization = regularization
        self.num_layers = len(layer_sizes)

        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def softmax(self, z):
        """Softmax activation for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == self.num_layers - 2:  # Output layer
                a = self.softmax(z)
            else:  # Hidden layers
                if self.activation == 'relu':
                    a = self.relu(z)
                else:
                    a = self.sigmoid(z)

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]

        # Initialize gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient
        delta = output - y

        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            dw[i] = np.dot(self.activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Add L2 regularization to weight gradients
            if self.regularization > 0:
                dw[i] += (self.regularization / m) * self.weights[i]

            if i > 0:
                if self.activation == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
                else:
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])

        return dw, db

    def compute_loss(self, X, y):
        """Compute cross-entropy loss with L2 regularization"""
        output = self.forward(X)
        m = X.shape[0]

        # Cross-entropy loss
        ce_loss = -np.sum(y * np.log(output + 1e-8)) / m

        # L2 regularization term
        reg_loss = 0
        if self.regularization > 0:
            for w in self.weights:
                reg_loss += 0.5 * self.regularization * np.sum(w**2) / m

        return ce_loss + reg_loss

    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def accuracy(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)