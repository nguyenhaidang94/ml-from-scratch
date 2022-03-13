import numpy as np

from base import BaseModel


class LinearRegresson(BaseModel):

    def __init__(self):
        self._beta = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self._beta = np.linalg.pinv(X.dot(X.T)).dot(X.T).dot(y)

    def predict(self, X: np.ndarray):
        return np.dot(X.T, self._beta)
