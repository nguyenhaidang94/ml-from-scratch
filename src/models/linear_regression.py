import numpy as np

from .base import BaseModel


class LinearRegression(BaseModel):

    def __init__(self):
        self._beta = None

    def train(self, X: np.ndarray, y: np.ndarray):
        # account for the bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # use Moore-Penrose pseudo-inverse to make sure that matrix always has inverse
        self._beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X: np.ndarray):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return np.dot(X, self._beta)
