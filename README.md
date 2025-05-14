# MNIST Optimizer Showdown 🚀

This project explores how different optimization strategies affect the performance of a custom-built neural network trained on the MNIST dataset. We compare **Stochastic Gradient Descent (SGD)** and **Backtracking Line Search (BLS)** under various hyperparameter configurations.

---

## 📌 Project Overview

- **Dataset**: MNIST Handwritten Digits (70,000 images, 10 classes)
- **Architecture**: Feedforward Neural Network (1–3 hidden layers)
- **Optimizers Compared**:
  - Stochastic Gradient Descent (SGD) with momentum
  - Backtracking Line Search using Armijo condition
- **Evaluated Metrics**:
  - Validation Loss & Accuracy
  - Training Time
  - Hyperparameter effects (learning rate, batch size, architecture)

---

## 📦 Folder Structure

MNIST_Optimizer_Showdown/
│
├── data/
│ └── mnist_loader.py # Loads and preprocesses MNIST dataset
│
├── models/
│ └── neural_net.py # NumPy-based neural network
│
├── optimizers/
│ ├── sgd.py # Stochastic Gradient Descent with momentum
│ └── backtrack.py # Backtracking Line Search optimizer
│
├── utils/
│ ├── trainer.py # Training loop for both optimizers
│ └── plotting.py # Functions for visualization
│
├── experiments/
│ ├── runner.py # Hyperparameter experiments
│ └── plots/ # Generated plots (.png)
│ ├── Optimizer_Comparison.png
│ ├── Backtrack_Line_Search_Loss.png
│ └── Learning_Rate_Batch_Size_Effects.png
│
├── results/
│ └── reflection.md # Summary of observations and conclusions
│
├── notebooks/
│ └── DATA606_Bonus_Assignment.ipynb # Development notebook (for reference)
│
├── main.py # Main script to run full experiment pipeline
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation (this file)

---

## 🚀 How to Run the Project

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/your-username/MNIST_Optimizer_Showdown.git
cd MNIST_Optimizer_Showdown
pip install -r requirements.txt
```

### 2. Run the main experiment

```bash
python main.py
```
- This script will:

    - Load and preprocess the MNIST dataset
    - Train models using both optimizers (SGD & BLS)
    - Run hyperparameter experiments (learning rate, batch size, architecture)
    - Generate training curves and comparison plots
    - Print a final evaluation report with test set accuracy

---

## 📌 Output & Observations

### ✅ Final Test Set Results

| Optimizer                  | Test Accuracy |
|---------------------------|---------------|
| **SGD with Momentum**     | 97.67%        |
| **Backtrack Line Search** | 97.71%        |

Both optimizers achieved strong final performance on the test set, with **Backtrack Line Search (BLS)** slightly outperforming SGD in accuracy. However, BLS took significantly longer to train.

---

### 📉 Training Behavior

- **SGD with Momentum**:
  - Showed fast convergence and consistent improvement across epochs.
  - Training accuracy reached nearly **100%**, and validation accuracy stabilized around **96%**.
  - Required tuning of learning rate and batch size to avoid instability or underfitting.

- **Backtrack Line Search**:
  - Delivered comparable accuracy with less tuning.
  - Automatically adjusted step sizes during training using Armijo condition.
  - Took more time per epoch due to line search overhead.
  - Final model reached **98.4% validation accuracy** in just 30 epochs.

---

### 🔍 Hyperparameter Insights

- **Learning Rate**:
  - Lower rates (0.001) led to slower convergence.
  - Higher rate (0.1) sped up training but risked overshooting.
  - Best performance was observed at **0.01**.

- **Batch Size**:
  - Smaller batches (32) showed noisier convergence but sometimes better generalization.
  - **Batch size of 128** struck the best balance between accuracy and speed.
  - Very large batches (512) were slower to converge and less generalizable.

- **Network Architecture**:
  - Deeper networks (3 hidden layers) improved accuracy but increased training time.
  - The best performing architecture was `[784, 512, 256, 128, 10]`.

---

### ⏱️ Training Time Trade-Off

| Metric                    | SGD           | Backtrack LS     |
|---------------------------|---------------|------------------|
| Total Training Time       | ~54 seconds   | ~91 seconds      |
| Ease of Hyperparameter Tuning | Moderate  | Very Low         |
| Adaptability              | Fixed steps   | Adaptive steps   |

- **SGD** is faster and flexible, but requires careful learning rate tuning.
- **Backtrack Line Search** is adaptive and stable but computationally more expensive.

---

### 📌 Summary

- Both optimizers are effective for training shallow-to-medium depth neural networks.
- **SGD is preferred when speed is critical**, and the user is comfortable tuning learning rates.
- **BLS is preferred when stability and automation are important**, even at the cost of compute time.
- Model performance benefits from deeper architectures, but with diminishing returns beyond 3 layers.
