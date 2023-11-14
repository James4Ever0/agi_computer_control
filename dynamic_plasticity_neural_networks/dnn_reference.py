import numpy as np  # Import NumPy for mathematical operations

class Connection:
    def __init__(self, connected_index, weights, bias):
        self.connected_index = connected_index  # Index of the connected neuron
        self.weights = weights  # Weights of the connection
        self.bias = bias  # Bias of the connection

class Neuron:
    def __init__(self, index, current_potential, input_connections, output_connections):
        self.index = index  # Index of the neuron
        self.current_potential = current_potential  # Current membrane potential
        self.input_connections = input_connections  # List of input connections
        self.output_connections = output_connections  # List of output connections

    def forward_propagate(self, inputs):
        # Perform weighted sum of inputs and apply activation function
        weighted_sum = np.dot(inputs, [conn.weights for conn in self.input_connections]) + self.input_connections[0].bias
        self.current_potential = self.activation_function(weighted_sum)

    def activation_function(self, x):
        # For example, using a simple sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def back_propagate(self, error):
        # Update weights and biases based on backpropagated error
        # This is a simplified example; a complete backpropagation algorithm would involve more steps
        for conn in self.input_connections:
            # Update weights using the error and the derivative of the activation function
            conn.weights -= learning_rate * error * self.activation_function_derivative(self.current_potential) * conn.connected_index.current_potential
            # Update bias using the error
            conn.bias -= learning_rate * error * self.activation_function_derivative(self.current_potential)

    def activation_function_derivative(self, x):
        # Derivative of the sigmoid activation function
        return self.activation_function(x) * (1 - self.activation_function(x))
