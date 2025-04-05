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
y = df['target'].to_numpy() ##Check this later

# Standardize features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled] ##Check this later

# Split dataset into two parts for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the loss function
def compute_cost(X, y, weights, lambda_=0.01):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    h = np.clip(h, 1e-9, 1 - 1e-9) ##Check this later
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(weights**2) ##Check this later
    return cost

# Define a function to perform stochastic gradient descent
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=5000):
    N, M = X.shape
    w = np.zeros(M)
    cost_history = []

    for i in range(num_iterations):
        i = np.random.randint(N)
        X_single = X[i:i + 1]
        y_single = y[i:i + 1]
        z = np.dot(X_single, w) ##Check this later
        h = sigmoid(z) ##Check this later
        errors = h - y_single
        gradient = np.dot(X_single.T, errors)
        w -= learning_rate * gradient

        cost = compute_cost(X, y, w)
        cost_history.append(cost)

        ##Check this later

    return w, cost_history

# Train logistic regression using SGD
w = np.zeros(X_train.shape[1]) ##Check this later
w, cost_history = stochastic_gradient_descent(X_train, y_train)

# Plot cost function convergence history
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Define a function to predict patient outcome with learned weights
def predict(X, w):
    z = np.dot(X, w) ##Check this later
    probabilities = sigmoid(z) ##Check this later
    predictions = (probabilities >= 0.5).astype(int) ##Check this later
    return probabilities, predictions ##Check this later

# Prediction on the test set
y_prob, y_pred = predict(X_test, w)

for i in range(10):
    print(f"Probability: {y_prob[i]:.4f} → Predicted Class: {y_pred[i]}")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nLearned Weights:")
print(w)

for feature, weight in zip(X.columns, w):
    print(f"{feature}: {weight:.4f}")
