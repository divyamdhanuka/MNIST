# Simple Neural Network for MNIST Digit Prediction

This project implements a basic neural network to predict handwritten digits using the MNIST dataset.

## 🚀 Key Features

1. **Neural Network Implementation**:
    - Basic weighted sum of inputs to produce predictions.
    - Element-wise multiplication utility for vectors and matrices.
    - Error calculation using squared differences.

2. **Dataset Handling**:
    - Read and convert MNIST dataset from CSV to Numpy arrays.
    - Preprocessing to segregate labels and features.

3. **Training Process**:
    - Loop through the dataset.
    - Calculate predictions, error, and weight updates.
    - Update weights based on calculated gradients.

4. **Evaluation**:
    - Predictions on a test set.
    - Count of wrongly predicted digits.

## 📌 Requirements

- pandas
- numpy

## 🔧 Implementation

```python
import pandas as pd
import numpy as np

# Neural Network Functions
def neural_network(input, weights):
    intermediate = np.zeros((784, 10))
    for i in range(len(weights)):
        intermediate[i] = input[i] * weights[i]
    out = np.sum(intermediate, axis=0)
    return out

def ele_mul(vector, matrix):
    matrix = matrix.reshape((784, 1))
    vector = vector.reshape((1, 10))
    out = np.dot(matrix, vector)
    return out

# Read Dataset
ds = pd.read_csv('/path/to/mnist_train.csv')
dataset_train = ds.to_numpy()

# Data Preprocessing
true_vector = np.zeros(len(dataset_train), dtype=int)
for i in range(len(dataset_train)):
    true_vector[i] = dataset_train[i][0]

test_set = np.zeros((len(dataset_train), len(dataset_train[0]) - 1))
for i in range(len(dataset_train)):
    test_set[i] = dataset_train[i][1:]

# Training Configuration
alpha = 0.00000001
weights = np.zeros((len(test_set[i]), 10))
weights[:] = 0.0000001
true = np.zeros(10)

# Training Process
for iter in range(2):
    print("Iter:" + str(iter))
    for i in range(len(test_set)):
        input = test_set[i]
        true[true_vector[i]] = 1
        pred = neural_network(input, weights)

        error = np.square(pred - true)

        delta = pred - true

        weights_deltas = ele_mul(delta, input)

        weights -= alpha * weights_deltas

        true = np.zeros(10)

# Evaluation
ds = pd.read_csv('/path/to/mnist_test.csv')
dataset_test = ds.to_numpy()

labels = np.zeros(len(dataset_test))

for i in range(len(dataset_test)):
    labels[i] = dataset_test[i][0]

test_vectors = np.zeros((len(dataset_test), len(dataset_test[0]) - 1))

for i in range(len(dataset_test)):
    test_vectors[i] = dataset_test[i][1:]

results = np.zeros(len(dataset_test))

for i in range(len(test_vectors)):
    pred = neural_network(test_vectors[i], weights)
    
    results[i] = np.argmax(pred)
    
wrong_pred = 0
for i in range(len(labels)):
    if (results[i] != labels[i]):
        wrong_pred += 1
print(wrong_pred)

# 📊 Results
Using an alpha (learning rate) of 0.00000005, the neural network achieved 2172 wrong predictions on the test set. 