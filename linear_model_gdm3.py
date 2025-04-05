# For AI 681 students only 
# Linear model for predicting the median house value
# by using the gradient descent algorithm 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Performs gradient descent to learn w.
    
    Parameters:
        X (numpy.ndarray): Feature matrix with shape (N, M)
        y (numpy.ndarray): Target vector with shape (N,)
        learning_rate (float): Learning rate for gradient descent
        num_iterations (int): Number of iterations to run gradient descent
    
    Returns:
        w (numpy.ndarray): Learned parameters with shape (M,)
        cost_history (list): History of cost function values
    """
    N, M = X.shape
    w = np.zeros(M) # Initialize w to zeros
    cost_history = []

    for i in range(num_iterations):
        # Compute the hypothesis/predictions
        y_pred = np.dot(X, w)
        
        # Compute the residuals/errors
        errors = y_pred - y
        
        # Compute the cost (Mean Squared Error)
        cost = (1 /  N) * np.dot(errors, errors)
        cost_history.append(cost)
        
        # Compute the gradient
        gradient = (2 / N) * np.dot(X.T, errors)
        
        # Update the parameters
        w -= learning_rate * gradient
        
        # Print cost every 100 iterations
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}: Cost {cost:.4f}")
    
    return w, cost_history

def main():
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
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # 5. Prepare Data for Gradient Descent
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    
    # Add intercept term
    X_train_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    
    X_test_np = X_test.to_numpy()
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    
    # 6. Define Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000
    
    # 7. Train the Model using Gradient Descent
    w, cost_history = gradient_descent(X_train_b, y_train_np, learning_rate, num_iterations)
    
    print("\nLearned parameters (w):")
    print(w)
    
    # 8. Make Predictions
    y_train_pred_gd = np.dot(X_train_b, w)
    y_test_pred_gd = np.dot(X_test_b, w)
    
    # 9. Evaluate the Model
    mse_train_gd = mean_squared_error(y_train_np, y_train_pred_gd)
    mse_test_gd = mean_squared_error(y_test.to_numpy(), y_test_pred_gd)
    
    r2_train_gd = r2_score(y_train_np, y_train_pred_gd)
    r2_test_gd = r2_score(y_test.to_numpy(), y_test_pred_gd)
    
    print(f"\nGradient Descent Linear Regression Performance:")
    print(f"Training MSE: {mse_train_gd:.4f}")
    print(f"Testing MSE: {mse_test_gd:.4f}")
    print(f"Training R²: {r2_train_gd:.4f}")
    print(f"Testing R²: {r2_test_gd:.4f}")
    
    # 10. Visualize the Cost Function Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_iterations + 1), cost_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Convergence of Gradient Descent')
    plt.grid(True)
    plt.show()
    
    # 11. Plot Actual vs. Predicted Values for Gradient Descent
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred_gd, alpha=0.5, color='blue', label='Gradient Descent')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('Actual vs. Predicted Median House Value')
    plt.legend()
    plt.show()
    
    # 13. Residual Analysis for Gradient Descent
    residuals = y_test.to_numpy() - y_test_pred_gd
    
   
    # Scatter plot of residuals vs. actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Actual Values - Gradient Descent')
    plt.show()

if __name__ == "__main__":
    main()
