import random
import math

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Derivative of sigmoid
def dsigmoid(y):
    return y*(1-y)

# XOR Dataset
training_data = [
([0,0],[0]),
([0,1],[1]),
([1,0],[1]),
([1,1],[0])
]

# Network structure
input_nodes = 2
hidden_nodes = 2
output_nodes = 1
learning_rate = 0.5

# Initialize weights randomly
weights_input_hidden = [[random.random() for j in range(hidden_nodes)] for i in range(input_nodes)]
weights_hidden_output = [random.random() for i in range(hidden_nodes)]

# Training
for epoch in range(5000):

    for inputs, target in training_data:

        # Forward propagation
        hidden = []
        for j in range(hidden_nodes):
            total = 0
            for i in range(input_nodes):
                total += inputs[i]*weights_input_hidden[i][j]
            hidden.append(sigmoid(total))

        output = 0
        for j in range(hidden_nodes):
            output += hidden[j]*weights_hidden_output[j]

        output = sigmoid(output)

        # Error
        error = target[0] - output

        # Backpropagation
        d_output = error * dsigmoid(output)

        for j in range(hidden_nodes):
            weights_hidden_output[j] += learning_rate * d_output * hidden[j]

        for i in range(input_nodes):
            for j in range(hidden_nodes):
                weights_input_hidden[i][j] += learning_rate * d_output * weights_hidden_output[j] * dsigmoid(hidden[j]) * inputs[i]

# Testing
print("Testing Neural Network\n")

for inputs, target in training_data:

    hidden = []
    for j in range(hidden_nodes):
        total = 0
        for i in range(input_nodes):
            total += inputs[i]*weights_input_hidden[i][j]
        hidden.append(sigmoid(total))

    output = 0
    for j in range(hidden_nodes):
        output += hidden[j]*weights_hidden_output[j]

    output = sigmoid(output)

    print("Input:",inputs," Predicted Output:",round(output,3))
