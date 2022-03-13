from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train model.\n
        Params:
            X: features
            y: labels
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for data.\n
        Params:
            X: features
        Returns:
            Predicted values
        """
        ...
