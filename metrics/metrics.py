import numpy as np


class Accuracy:
    @staticmethod
    def __call__(predicted: np.ndarray, actual: np.ndarray) -> float:
        if len(actual.shape) == 2:
            actual = actual.argmax(axis=1)
        acc = np.sum(predicted.argmax(axis=1) == actual) / actual.shape[0]
        return acc

