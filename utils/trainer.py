import numpy as np
import time
from optimizers.backtrack import BacktrackLineSearchOptimizer

def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=128, verbose=True):
    """Train the neural network model"""
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }

    for epoch in range(epochs):
        start_time = time.time()

        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Train on batches
        train_losses = []
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # Forward pass
            output = model.forward(X_batch)

            # Backward pass
            dw, db = model.backward(X_batch, y_batch, output)

            # Update weights
            if isinstance(optimizer, BacktrackLineSearchOptimizer):
                optimizer.update(model, dw, db, X_batch, y_batch)
            else:
                optimizer.update(model, dw, db)

            # Track batch loss
            batch_loss = model.compute_loss(X_batch, y_batch)
            train_losses.append(batch_loss)

        # Compute epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = model.compute_loss(X_val, y_val)
        train_acc = model.accuracy(X_train[:5000], y_train[:5000])  # Sample for speed
        val_acc = model.accuracy(X_val, y_val)

        epoch_time = time.time() - start_time

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                  f"train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f} - "
                  f"time: {epoch_time:.2f}s")

    return history