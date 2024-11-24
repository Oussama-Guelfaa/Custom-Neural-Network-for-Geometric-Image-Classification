# train_radial_dataset.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_neural_network import CustomNeuralNetwork

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of a model."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    Z = model.feedforward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Contour plot for decision boundary
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.Spectral, alpha=0.8)
    plt.contour(xx, yy, Z, colors='k', levels=[0.5])  # Clear line at decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Generate a synthetic radial dataset
np.random.seed(0)
X_radial = np.random.randn(500, 2)
Y_radial = np.array([1 if np.linalg.norm(x) < 1.5 else 0 for x in X_radial])
Y_radial = Y_radial.reshape(-1, 1)

# Split the dataset
X_train_radial, X_test_radial, Y_train_radial, Y_test_radial = train_test_split(
    X_radial, Y_radial, test_size=0.2, random_state=42
)

# Create and train the neural network
neural_network_radial = CustomNeuralNetwork([2, 10, 10, 1])
loss_history_radial = neural_network_radial.train(
    X_train_radial, Y_train_radial, epochs=1000, learning_rate=0.01
)

# Plot loss over epochs
neural_network_radial.plot_loss(loss_history_radial)

# Plot decision boundary
plt.figure()
plot_decision_boundary(neural_network_radial, X_radial, Y_radial)
plt.title("Decision Boundary of Neural Network on Radial Dataset")
plt.show()
