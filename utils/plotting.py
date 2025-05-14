import matplotlib.pyplot as plt

def plot_training_curves(history, title):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_optimizer_comparison(results):
    """Compare SGD and Backtrack Line Search"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Validation loss comparison
    ax1.plot(results['SGD_base']['val_loss'], label='SGD', linewidth=2)
    ax1.plot(results['BLS_base']['val_loss'], label='Backtrack Line Search', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Optimizer Comparison - Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Validation accuracy comparison
    ax2.plot(results['SGD_base']['val_acc'], label='SGD', linewidth=2)
    ax2.plot(results['BLS_base']['val_acc'], label='Backtrack Line Search', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Optimizer Comparison - Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_hyperparameter_effects(results):
    """Plot effects of different hyperparameters"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Learning rate effects
    learning_rates = [0.001, 0.01, 0.1]
    for lr in learning_rates:
        key = f'SGD_lr_{lr}'
        if key in results:
            ax1.plot(results[key]['val_acc'], label=f'LR={lr}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Learning Rate Effects')
    ax1.legend()
    ax1.grid(True)

    # Batch size effects
    batch_sizes = [32, 128, 512]
    for bs in batch_sizes:
        key = f'SGD_batch_{bs}'
        if key in results:
            ax2.plot(results[key]['val_acc'], label=f'Batch={bs}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Batch Size Effects')
    ax2.legend()
    ax2.grid(True)

    # Architecture effects
    arch_names = ['Small (1 hidden)', 'Medium (2 hidden)', 'Large (3 hidden)']
    for i, name in enumerate(arch_names):
        key = f'SGD_arch_{i}'
        if key in results:
            ax3.plot(results[key]['val_acc'], label=name)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Architecture Effects')
    ax3.legend()
    ax3.grid(True)

    # Training time comparison
    optimizers = ['SGD_base', 'BLS_base']
    total_times = []
    for opt in optimizers:
        if opt in results:
            total_time = sum(results[opt]['epoch_time'])
            total_times.append(total_time)

    ax4.bar(range(len(optimizers)), total_times)
    ax4.set_xticks(range(len(optimizers)))
    ax4.set_xticklabels(['SGD', 'Backtrack LS'])
    ax4.set_ylabel('Total Training Time (s)')
    ax4.set_title('Training Time Comparison')
    ax4.grid(True, axis='y')

    plt.tight_layout()
    plt.show()