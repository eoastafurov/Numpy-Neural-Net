import numpy as np


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ):
        """
            Given an input and a target, compute the loss function
            associated to the softmax and return the result.

            For consistency this function should not be overridden,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input_, target)

    def backward(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the softmax and return the result.

            For consistency this function should not be overridden,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input_, target)

    def updateOutput(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ):
        raise NotImplementedError("U Should implement it in a child module")

    def updateGradInput(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ):
        raise NotImplementedError("U Should implement it in a child module")

    def __repr__(self):
        """
        Pretty printing.
        """
        return "Criterion"
