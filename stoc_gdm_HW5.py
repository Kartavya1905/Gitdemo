# For AI 681 students only
# Linear model for predicting the median house value
# by using the stochastic gradient descent algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import ssl  # Import the SSL module


def stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Performs stochastic gradient descent to learn w.
    """
    N, M = X.shape
    w = np.zeros(M)  # Initialize w to zeros
    cost_history = []

    for j in range(num_iterations):
    
        # Select a random instance
        random_index = np.random.randint(N)
        xi = X[random_index:random_index + 1]  # Make sure xi is a row vector (2D array)
        yi = y[random_index:random_index + 1]  # Make sure yi is a scalar (or 1D array of length 1)

        # Compute the prediction
        y_pred = np.dot(xi, w)

        # Compute the error
        error = y_pred - yi

        # Compute the gradient for this instance
        gradient = 2 * xi.T.dot(error)

        # Update the parameters
        w -= learning_rate * gradient.flatten()  # Ensure gradient is a 1D array

        # Compute the cost for the entire dataset at the end of each j
        y_pred_all = np.dot(X, w)
        cost = (1 / N) * np.sum((y_pred_all - y)**2) #Using numpy sum
        cost_history.append(cost)

        if (j + 1) % 100 == 0:
            print(f"Iterations {j + 1}: Cost {cost:.4f}")

    return w, cost_history


def main():
    # 0. Bypass SSL certification (added as solution)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    # 1. Load the California Housing Dataset
    california = fetch_california_housing(as_frame=True)
    X = california.data
    y = california.target

    # 2. Select and Explore the Relevant Features
    selected_features = ['MedInc', 'AveBedrms', 'HouseAge']
    X_selected = X[selected_features]
    print("First five rows of selected features:")
    print(X_selected.head())
    print("\nFirst five target values:")
    print(y.head())

    # 3. Preprocessing the Data
    scaler = StandardScaler()
    # removing the mean and scaling to unit variance
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    print("\nFirst five rows of scaled features:")
    print(X_scaled.head())

    # 4. Split the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # 5. Prepare Data for Stochastic Gradient Descent
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    # Add intercept term
    X_train_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    X_test_np = X_test.to_numpy()
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])

    # 6. Define Hyperparameters for Stochastic Gradient Descent
    learning_rate = 0.01  # Adjust learning rate
    num_iterations = 1000  # Adjust number of iterations

    # 7. Train the Model using Stochastic Gradient Descent
    w, cost_history = stochastic_gradient_descent(X_train_b, y_train_np, learning_rate, num_iterations)

    print("\nLearned parameters (w):")
    print(w)

    # 8. Make Predictions
    y_train_pred_sgd = np.dot(X_train_b, w)
    y_test_pred_sgd = np.dot(X_test_b, w)

    # 9. Evaluate the Model
    mse_train_sgd = mean_squared_error(y_train_np, y_train_pred_sgd)
    mse_test_sgd = mean_squared_error(y_test.to_numpy(), y_test_pred_sgd)
    r2_train_sgd = r2_score(y_train_np, y_train_pred_sgd)
    r2_test_sgd = r2_score(y_test.to_numpy(), y_test_pred_sgd)

    print(f"\nStochastic Gradient Descent Linear Regression Performance:")
    print(f"Training MSE: {mse_train_sgd:.4f}")
    print(f"Testing MSE: {mse_test_sgd:.4f}")
    print(f"Training R²: {r2_train_sgd:.4f}")
    print(f"Testing R²: {r2_test_sgd:.4f}")

    # 10. Visualize the Cost Function Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_iterations + 1), cost_history, color='blue')
    plt.xlabel('j')
    plt.ylabel('Cost (MSE)')
    plt.title('Convergence of Stochastic Gradient Descent')
    plt.grid(True)
    plt.show()

    # 11. Plot Actual vs. Predicted Values for Stochastic Gradient Descent
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred_sgd, alpha=0.5, color='blue', label='Stochastic Gradient Descent')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('Actual vs. Predicted Median House Value - Stochastic Gradient Descent')
    plt.legend()
    plt.show()

    # 13. Residual Analysis for Stochastic Gradient Descent
    residuals = y_test.to_numpy() - y_test_pred_sgd

    # Scatter plot of residuals vs. actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Actual Values - Stochastic Gradient Descent')
    plt.show()

if __name__ == "__main__":
    main()
