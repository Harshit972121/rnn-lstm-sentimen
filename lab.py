import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

input_nodes = 2
hidden_nodes = 3
output_nodes = 1

np.random.seed(1)

W1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.zeros((1, hidden_nodes))

W2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.zeros((1, output_nodes))

learning_rate = 0.1
epochs = 3000
loss_list = []

for i in range(epochs):
    hidden_input = X @ W1 + b1
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ W2 + b2
    prediction = sigmoid(final_input)

    loss = np.mean((y - prediction) ** 2)
    loss_list.append(loss)

    d_output = (prediction - y) * sigmoid_derivative(prediction)

    d_W2 = hidden_output.T @ d_output
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    d_hidden = d_output @ W2.T * sigmoid_derivative(hidden_output)
    d_W1 = X.T @ d_hidden
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

plt.plot(loss_list)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

hidden_output = sigmoid(X @ W1 + b1)
final_output = sigmoid(hidden_output @ W2 + b2)
print(np.round(final_output, 3))
