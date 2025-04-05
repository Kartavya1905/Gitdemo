# Project 1 Solution
# Name: Kartavya Mandora 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset from your hard drive 
dataset = "heart.csv"
df = pd.read_csv(dataset)

# Print the first few rows of the dataset 
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Standardize features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into two parts for training and testing 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 

# Define the sigmoid function 
def sigmoid(a):
    return 1 / (1 + np.exp(-a)) 

# Compute the loss function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost
    
# Define a function to perform stochastic gradient descent  
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=10000):
    m, n = X.shape
    weights = np.zeros(n)
    cost_history = []

    for i in range(num_iterations):
        j = np.random.randint(m)
        xi = X[j:j+1]
        yi = y[j:j+1]
        h = sigmoid(np.dot(xi, weights))
        gradient = np.dot(xi.T, (h - yi))
        weights -= learning_rate * gradient

        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        if (i + 1) % 1000 == 0:
            print(f"Iterations {i + 1}: Cost {cost:.4f}")
 
    return weights, cost_history

# Train logistic regression using SGD
weights, cost_history = stochastic_gradient_descent(X_train, y_train)

# Plot cost function convergence history 
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()


# Define a function to predict patient outcome with learned weights 
def predict(X, weights):
    return  sigmoid(np.dot(X, weights)) >= 0.5

# Prediction on the test set
y_pred = predict(X_test, weights)
# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

for feature, weight in zip(X.columns, weights):
    print(f"{feature}: {weight:.4f}")


 
