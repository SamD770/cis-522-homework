import numpy as np
import torch


def add_bias_feature(X):
    samples, features = X.shape
    dummy_feature = np.ones((samples, 1))

    return np.concatenate((X, dummy_feature), axis=1)


def separate_bias(w_b_concat):
    w = w_b_concat[:-1]
    b = w_b_concat[-1]
    return w, b


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.random.uniform()
        self.b = np.random.uniform()

    def fit(self, X, y):
        X_concat = add_bias_feature(X)

        w_b_concat = np.linalg.inv(X_concat.T @ X_concat) @ X_concat.T @ y

        self.w, self.b = separate_bias(w_b_concat)


    def predict(self, X):
        return X @ self.w + self.b



class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ):
        self.w
        self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
