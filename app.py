import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_linear_regression():
    st.title('Linear Regression')

    # Load the Iris dataset
    iris = load_iris()
    X = iris["data"][:, 2:3]  # Use petal length as feature (for univariate regression)
    y = iris["data"][:, 3]    # Target is petal width

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add a bias (intercept) term to the dataset
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Cost function and Gradient descent
    def cost_function(X, y, theta):
        m = len(y)
        cost = (1/(2*m)) * np.sum((X.dot(theta) - y) ** 2)
        return cost

    def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000):
        m = len(y)
        cost_history = []
        for i in range(iterations):
            gradients = (1/m) * X.T.dot(X.dot(theta) - y)
            theta = theta - learning_rate * gradients
            cost = cost_function(X, y, theta)
            cost_history.append(cost)
        return theta, cost_history

    # Initialize parameters
    theta_initial = np.zeros((X_train_b.shape[1],))

    # Train the model
    theta_optimal, cost_history = gradient_descent(X_train_b, y_train, theta_initial, learning_rate=0.1, iterations=1000)

    # Predict values on the test set
    y_pred = X_test_b.dot(theta_optimal)

    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color='blue', label="Training Data")
    ax.plot(X_train, X_train_b.dot(theta_optimal), color='red', label="Linear Regression Line")
    ax.set_title("Linear Regression")
    ax.set_xlabel("Petal Length (Standardized Feature)")
    ax.set_ylabel("Petal Width")
    ax.legend()
    st.pyplot(fig)

    st.write("### Explanation")
    st.write("""
    The graph above shows the result of a linear regression model predicting the petal width based on the petal length. 
    The blue scatter points represent the training data, and the red line is the fitted linear regression model. 
    The cost function used is the Mean Squared Error, and the gradient descent algorithm optimizes the parameters to minimize this cost.
    """)

def plot_multiple_regression():
    st.title('Multiple Regression')

    # Load Iris Dataset
    iris = load_iris()
    X = iris.data  # We will use all four features
    y = iris.target

    # For simplicity, we'll only use samples where the class is not 2 (binary classification between class 0 and 1)
    X = X[y != 2]
    y = y[y != 2]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    def normalize(X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Add a bias term
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Initialize coefficients
    def initialize_weights(n):
        return np.zeros((n, 1))

    # Hypothesis function
    def predict(X, weights):
        return np.dot(X, weights)

    # Cost function and Gradient descent
    def compute_cost(X, y, weights):
        m = len(y)
        predictions = predict(X, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def gradient_descent(X, y, weights, learning_rate, iterations):
        m = len(y)
        cost_history = np.zeros(iterations)
        for i in range(iterations):
            predictions = predict(X, weights)
            errors = predictions - y.reshape(-1, 1)
            gradients = (1 / m) * np.dot(X.T, errors)
            weights -= learning_rate * gradients
            cost_history[i] = compute_cost(X, y, weights)
        return weights, cost_history

    # Prepare the y_train as column vector
    y_train = y_train.reshape(-1, 1)

    # Initialize parameters
    weights = initialize_weights(X_train.shape[1])
    learning_rate = 0.01
    iterations = 1000

    # Perform Gradient Descent
    optimal_weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, iterations)

    # Print the final weights
    st.write("Optimal Weights (Coefficients):", optimal_weights.ravel())

    # Make predictions on the test set
    y_pred = predict(X_test, optimal_weights)

    # Convert predictions to binary (0 or 1 based on a threshold of 0.5)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Evaluate model accuracy
    accuracy = np.mean(y_pred_binary == y_test.reshape(-1, 1))
    st.write("Model Accuracy:", accuracy)

    # Plot the cost function over iterations
    fig, ax = plt.subplots()
    ax.plot(range(iterations), cost_history)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost (MSE)")
    ax.set_title("Cost Function during Gradient Descent")
    st.pyplot(fig)

    st.write("### Explanation")
    st.write("""
    The plot above shows the cost function value over iterations during the gradient descent process. 
    The goal of gradient descent is to minimize this cost function. 
    The accuracy of the model on the test set is also provided, indicating how well the model performs on unseen data.
    """)

def plot_logistic_regression():
    st.title('Logistic Regression')

    # Load the Iris dataset
    iris = load_iris()
    X = iris["data"][:, 3:]  # Use petal width as feature
    y = (iris["target"] == 2).astype(int)  # Binary classification (Iris-Virginica)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add a bias term
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Cost function
    def cost_function(X, y, theta):
        m = len(y)
        h = sigmoid(X.dot(theta))
        epsilon = 1e-5  # Small constant to avoid log(0)
        cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    # Gradient descent for Logistic Regression
    def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000):
        m = len(y)
        cost_history = []
        for i in range(iterations):
            gradients = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
            theta = theta - learning_rate * gradients
            cost = cost_function(X, y, theta)
            cost_history.append(cost)
        return theta, cost_history

    # Initialize parameters
    theta_initial = np.zeros((X_train_b.shape[1],))

    # Train the model
    theta_optimal, cost_history = gradient_descent(X_train_b, y_train, theta_initial, learning_rate=0.1, iterations=1000)

    # Predict probabilities
    X_test_range = np.linspace(X_test_b[:, 1].min(), X_test_b[:, 1].max(), 1000).reshape(-1, 1)
    X_test_range_b = np.c_[np.ones((X_test_range.shape[0], 1)), X_test_range]
    y_pred_prob = sigmoid(X_test_range_b.dot(theta_optimal))

    # Plot the results
    fig, ax = plt.subplots()
    ax.scatter(X_test[:, 0], y_test, color='blue', label="Test Data")
    ax.plot(X_test_range, y_pred_prob, color='red', label="Logistic Regression Model")
    ax.set_title("Logistic Regression")
    ax.set_xlabel("Petal Width")
    ax.set_ylabel("Probability of Iris-Virginica")
    ax.legend()
    st.pyplot(fig)

    st.write("### Explanation")
    st.write("""
    The plot above illustrates the logistic regression model's performance on the test data. 
    The blue points represent the test data, and the red curve shows the logistic regression model's decision boundary. 
    The model predicts the probability of a sample being an Iris-Virginica based on petal width.
    """)

def plot_svm():
    st.title('Support Vector Machine (SVM)')

    # Load the Iris dataset
    iris = load_iris()
    X = iris["data"][:, 2:4]  # Use petal length and petal width as features
    y = iris["target"]

    # Only select Setosa and Versicolor for binary classification
    X = X[y != 2]
    y = y[y != 2]

    # Change class labels from 0 and 1 to -1 and 1 for SVM
    y = np.where(y == 0, -1, 1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add bias term (intercept) to X
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # SVM Hyperparameters
    learning_rate = 0.001
    epochs = 1000
    lambda_param = 0.01  # Regularization parameter

    # Initialize weights
    weights = np.zeros(X_train_b.shape[1])

    # Hinge loss function
    def hinge_loss(X, y, weights, lambda_param):
        distances = 1 - y * (X.dot(weights))
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss_value = lambda_param * np.sum(weights ** 2) / 2 + np.mean(distances)
        return hinge_loss_value

    # Gradient descent for SVM
    def gradient_descent(X, y, weights, learning_rate, lambda_param, epochs):
        n = X.shape[0]  # Number of samples
        loss_history = []
        
        for epoch in range(epochs):
            # Compute gradient
            gradient = np.zeros_like(weights)
            for i in range(n):
                if y[i] * (X[i].dot(weights)) < 1:
                    gradient -= y[i] * X[i]
            
            # Update weights
            weights -= learning_rate * (lambda_param * weights - gradient / n)
            
            # Track loss
            loss = hinge_loss(X, y, weights, lambda_param)
            loss_history.append(loss)
        
        return weights, loss_history

    # Train SVM model using gradient descent
    optimal_weights, loss_history = gradient_descent(X_train_b, y_train, weights, learning_rate, lambda_param, epochs)

    # Predict function
    def predict(X, weights):
        predictions = np.sign(X.dot(weights))
        return predictions

    # Evaluate on test set
    y_pred = predict(X_test_b, optimal_weights)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    st.write(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Plot decision boundary
    def plot_svm_decision_boundary(X, y, weights):
        fig, ax = plt.subplots()
        
        # Plot the samples
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=50, label='Samples')
        
        # Create mesh grid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Predict on mesh grid
        Z = predict(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], weights)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        
        ax.set_title("SVM Decision Boundary")
        ax.set_xlabel("Petal Length")
        ax.set_ylabel("Petal Width")
        ax.legend()
        st.pyplot(fig)

    # Plot decision boundary
    plot_svm_decision_boundary(X_train, y_train, optimal_weights)

    st.write("### Explanation")
    st.write("""
    The plot above illustrates the decision boundary of the Support Vector Machine (SVM) model. 
    The blue and red points represent different classes (Setosa and Versicolor), and the decision boundary is shown as the dividing line.
    The shaded regions represent the areas where the SVM predicts each class. 
    The margins are the areas closest to the decision boundary where the SVM is still confident about its prediction.
    """)

def main():
    st.sidebar.title("Model Selector")
    model_choice = st.sidebar.selectbox("Choose a model", 
                                        ("Select a model", 
                                         "Linear Regression", 
                                         "Multiple Regression", 
                                         "Logistic Regression", 
                                         "SVM"))

    if model_choice == "Select a model":
        st.title("Machine Learning")
        st.subheader("BE Comps A")
        st.subheader("Team Members")
        st.write("1. Rudalph Gonsalves 9608")
        st.write("2. Zane Falcao 9603")
        st.write("3. Alroy Pereira 9630")
    elif model_choice == "Linear Regression":
        plot_linear_regression()
    elif model_choice == "Multiple Regression":
        plot_multiple_regression()
    elif model_choice == "Logistic Regression":
        plot_logistic_regression()
    elif model_choice == "SVM":
        plot_svm()

if __name__ == "__main__":
    main()
