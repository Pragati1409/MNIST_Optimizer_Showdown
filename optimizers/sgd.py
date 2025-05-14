import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, model, dw, db):
        """Update weights using SGD with optional momentum"""
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in model.weights]
            self.velocity_b = [np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            # Update velocities
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dw[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db[i]

            # Update weights
            model.weights[i] += self.velocity_w[i]
            model.biases[i] += self.velocity_b[i]