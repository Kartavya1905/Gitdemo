# For AI 681 students only 
# Logistic Regression Problem 
# by using the gradient descent algorithm 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create a Dataset with Random Numbers 
def generate_data(n_samples=100):
    np.random.seed(42)  # For reproducibility
    # Generate class 0
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    y0 = np.zeros(n_samples // 2)
    # Generate class 1
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    y1 = np.ones(n_samples // 2)

    # Combine the data
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    return X, y

# Define the Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize Parameters
def initialize_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Compute the Loss function 
def compute_loss(y, y_pred, weights, lamb ):
    n = len(y)
    # To avoid log(0), we add a small epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Binary Cross-Entropy Loss
    loss = - (1/n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    # L2 Term  
    l2_part = (lamb / (2 * n)) * np.sum(np.square(weights))
    total_loss = loss + l2_part
    return total_loss

# Find Coefficients by using the Gradient Descent Algorithm 
def gradient_descent(X, y, weights, bias, learning_rate, num_iterations, lamb ):
    n = len(y)
    loss_history = []

    for i in range(num_iterations):
        # Compute linear combination
        linear_model = np.dot(X, weights) + bias
        # Apply sigmoid to get predictions
        y_pred = sigmoid(linear_model)

        # Compute loss  
        loss = compute_loss(y, y_pred, weights, lamb )
        loss_history.append(loss)

        # Compute gradients
        dw = (1/n) * np.dot(X.T, (y_pred - y)) + (lamb/n) * weights   
        db = (1/n) * np.sum(y_pred - y)  # Bias gradient 

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print loss every 100 iterations
        if (i+1) % 100 == 0:
            print(f'Iteration {i+1}: Loss = {loss:.4f}')

    return weights, bias, loss_history

# Make Predictions 
def predict(X, weights, bias, threshold=0.5):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= threshold).astype(int)

# Main Function for Logistic Regression   
def main():
    # Generate data
    X, y = generate_data(n_samples=200)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    # Initialize parameters
    n_features = X_train.shape[1]
    weights, bias = initialize_parameters(n_features)

    # Set hyperparameters
    learning_rate = 0.1
    num_iterations = 1000
    reg_strength = 0.01  # Regularization strength

    # Perform gradient descent iterations
    weights, bias, loss_history = gradient_descent(X_train, y_train, weights, bias, learning_rate, num_iterations, reg_strength)

    # Plot the loss over iterations
    plt.figure(figsize=(8,6))
    plt.plot(range(num_iterations), loss_history, color='blue')
    plt.title('Loss over Iterations with L2 Regularization')
    plt.xlabel('Iterations')
    plt.ylabel('Binary Cross-Entropy Loss + L2 Penalty')
    plt.grid(True)
    plt.show()

    # Make predictions on the test set
    y_pred = predict(X_test, weights, bias)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

    # Plot decision boundary
    plt.figure(figsize=(8,6))
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color='red', label='Class 0')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color='blue', label='Class 1')

    # Calculate decision boundary
    x_values = [np.min(X_train[:, 0] - 1), np.max(X_train[:, 0] + 1)]
    # Avoid division by zero in case weights[1] is zero
    if weights[1] != 0:
        y_values = -(weights[0] * np.array(x_values) + bias) / weights[1]
        plt.plot(x_values, y_values, label='Decision Boundary')
    else:
        # Vertical line if weights[1] is zero
        x_boundary = -bias / weights[0]
        plt.axvline(x=x_boundary, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary with Regularization')
    plt.legend()
    plt.grid(True)
    #plt.savefig("myplot_hd.pdf", dpi=300)
    plt.show()
	
    

if __name__ == "__main__":
    main()
