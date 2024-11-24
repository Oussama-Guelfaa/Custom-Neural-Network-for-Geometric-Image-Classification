# custom_neural_network.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    return x * (1 - x)

class CustomNeuralNetwork:
    """A simple feedforward neural network with backpropagation."""

    def __init__(self, neurons_per_layer):
        """
        Initialize the neural network.
        :param neurons_per_layer: List containing the number of neurons in each layer.
        """
        self.layers = neurons_per_layer
        # Initialize weights with random values
        self.weights = [
            np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            for i in range(len(self.layers) - 1)
        ]

    def feedforward(self, inputs):
        """Compute the network output for given inputs."""
        activations = inputs
        for weight in self.weights:
            activations = sigmoid(np.dot(activations, weight))
        return activations

    def backpropagation(self, inputs, expected_output, learning_rate):
        """Adjust the weights based on the error rate."""
        activations = [inputs]
        # Forward pass
        for weight in self.weights:
            activations.append(sigmoid(np.dot(activations[-1], weight)))

        # Compute error
        error = expected_output - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]

        # Backward pass
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * sigmoid_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate

    def train(self, inputs, outputs, epochs, learning_rate):
        """Train the neural network."""
        loss_history = []
        for epoch in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)
            # Compute loss
            loss = np.mean(np.square(outputs - self.feedforward(inputs)))
            loss_history.append(loss)
        return loss_history

    def plot_loss(self, loss_history):
        """Plot the loss over epochs."""
        plt.plot(loss_history)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
