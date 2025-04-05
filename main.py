import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=50):
    N, M = X.shape
    w = np.zeros(M)
    cost_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(N):
            xi = X_shuffled[i].reshape(1, -1)
            yi = y_shuffled[i]

            y_pred = np.dot(xi, w)
            error = y_pred - yi

            gradient = 2 * xi.T * error
            w -= learning_rate * gradient.flatten()

        y_pred_full = np.dot(X, w)
        cost = (1 / N) * np.sum((y_pred_full - y) ** 2)
        cost_history.append(cost)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Cost {cost:.4f}")

    return w, cost_history


def main():
    california = fetch_california_housing(as_frame=True)
    X = california.data
    y = california.target

    selected_features = ["MedInc", "AveBedrms", "HouseAge"]
    X_selected = X[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add bias term
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    learning_rate = 0.01
    num_epochs = 50

    w_sgd, cost_history_sgd = stochastic_gradient_descent(
        X_train_b, y_train.to_numpy(), learning_rate, num_epochs
    )

    print("\nLearned parameters (SGD):")
    print(w_sgd)

    y_train_pred_sgd = np.dot(X_train_b, w_sgd)
    y_test_pred_sgd = np.dot(X_test_b, w_sgd)

    mse_train_sgd = mean_squared_error(y_train, y_train_pred_sgd)
    mse_test_sgd = mean_squared_error(y_test, y_test_pred_sgd)

    r2_train_sgd = r2_score(y_train, y_train_pred_sgd)
    r2_test_sgd = r2_score(y_test, y_test_pred_sgd)

    print(f"Training MSE: {mse_train_sgd:.4f}")
    print(f"Testing MSE: {mse_test_sgd:.4f}")
    print(f"Training R²: {r2_train_sgd:.4f}")
    print(f"Testing R²: {r2_test_sgd:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred_sgd, alpha=0.5, color="blue", label="SGD")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", label="Ideal")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Actual vs. Predicted Median House Value - SGD")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), cost_history_sgd, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE)")
    plt.title("Convergence of Stochastic Gradient Descent")
    plt.grid(True)
    plt.show()

    residuals = y_test.to_numpy() - y_test_pred_sgd

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.5, color="purple")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Actual Values - SGD")
    plt.show()


if __name__ == "__main__":
    main()
