from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray):
        ...
