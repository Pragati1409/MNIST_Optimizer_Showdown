import numpy as np

class BacktrackLineSearchOptimizer:
    def __init__(self, alpha=1.0, beta=0.5, c=0.5):
        """
        Backtrack line search optimizer
        alpha: initial step size
        beta: step size reduction factor
        c: Armijo condition parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def update(self, model, dw, db, X_batch, y_batch):
        """Update weights using backtrack line search"""
        # Compute initial loss
        initial_loss = model.compute_loss(X_batch, y_batch)

        # Compute gradient norm for Armijo condition
        grad_norm_sq = 0
        for i in range(len(dw)):
            grad_norm_sq += np.sum(dw[i]**2) + np.sum(db[i]**2)

        # Backtrack line search
        step_size = self.alpha
        while step_size > 1e-10:
            # Try update with current step size
            for i in range(len(model.weights)):
                model.weights[i] -= step_size * dw[i]
                model.biases[i] -= step_size * db[i]

            # Compute new loss
            new_loss = model.compute_loss(X_batch, y_batch)

            # Check Armijo condition
            if new_loss <= initial_loss - self.c * step_size * grad_norm_sq:
                break

            # Revert update
            for i in range(len(model.weights)):
                model.weights[i] += step_size * dw[i]
                model.biases[i] += step_size * db[i]

            # Reduce step size
            step_size *= self.beta