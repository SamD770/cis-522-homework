import numpy as np


def add_bias_feature(X):
    samples, features = X.shape
    dummy_feature = np.ones((samples, 1))

    return np.concatenate((X, dummy_feature), axis=1)


def separate_bias(w_b_concat):
    w = w_b_concat[:-1]
    b = w_b_concat[-1]
    return w, b


class LinearRegression:
    """
    A linear regression model that uses analytic results to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def initialise_params(self, features: int) -> None:
        self.w = np.random.uniform(low=-0.1, high=0.1, size=features)
        self.b = np.random.uniform(low=-1, high=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model parameters using the provided data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input labels
        """
        if self.w is None:
            samples, features = X.shape
            self.initialise_params(features)

        X_concat = add_bias_feature(X)

        w_b_concat = np.linalg.inv(X_concat.T @ X_concat) @ X_concat.T @ y

        self.w, self.b = separate_bias(w_b_concat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        if self.w is None:
            samples, features = X.shape
            self.initialise_params(features)

        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-7, epochs: int = 1000):

        n_samples, features = X.shape

        if self.w is None:
            self.initialise_params(features)

        for epoch in range(epochs):

            y_pred = self.predict(X)

            L = ((y_pred - y) ** 2).mean()

            print(f"epoch: {epoch}, loss: {L}")

            X_concat = add_bias_feature(X)

            dL_dW = 2 * X_concat.T @ (y_pred - y) / n_samples

            dL_dw, dL_db = separate_bias(dL_dW)

            self.w -= lr * dL_dw
            self.b -= lr * dL_db

    # def predict(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Predict the output for the given input.
    #
    #     Arguments:
    #         X (np.ndarray): The input data.
    #
    #     Returns:
    #         np.ndarray: The predicted output.
    #
    #     """
    #     super(self).predict(X)
