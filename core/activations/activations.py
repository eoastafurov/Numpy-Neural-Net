from core.base.module import Module
import numpy as np
from copy import deepcopy


class SoftMax(Module):
    """
    Input:
        batch_size x n_feats

    Output:
        batch_size x n_feats

    Formula:
        softmax(x)_i = exp(x_i) / sum(exp(x_j))

    P.S. softmax(x) == softmax(x - Const)
    It makes possible to avoid computing exp() from large argument.
    """
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = np.subtract(
            input_,
            input_.max(
                axis=1,
                keepdims=True
            )
        )
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(
            self.output,
            axis=1,
            keepdims=True
        )

        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        local_repr_1 = np.einsum(
            'bi,bj->bij',
            self.output,
            self.output
        )
        local_repr_2 = np.einsum(
            'bi,ij->bij',
            self.output,
            np.eye(
                input_.shape[1],
                input_.shape[1]
            )
        )
        local_repr_3 = local_repr_2 - local_repr_1

        self.gradInput = np.einsum(
            'bij,bi->bj',
            local_repr_3,
            gradOutput
        )
        return self.gradInput

    def __repr__(self) -> str:
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input_: np.ndarray) -> np.float64:
        self.output = np.subtract(
            input_,
            input_.max(
                axis=1,
                keepdims=True
            )
        )
        self.output -= np.log(
            np.sum(
                np.exp(self.output),
                axis=1,
                keepdims=True
            )
        )
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        self.gradInput = np.zeros(input_.shape)
        for i in range(input_.shape[0]):
            self.gradInput[i] = gradOutput[i] @ \
                                np.subtract(
                                    np.eye(input_.shape[1]),
                                    np.exp(self.output)[i]
                                )
        return self.gradInput

    def __repr__(self) -> str:
        return "LogSoftMax"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = np.maximum(input_, 0)
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        self.gradInput = np.multiply(gradOutput, input_ > 0)
        return self.gradInput

    def __repr__(self) -> str:
        return "ReLU"


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def updateOutput(self, input_: np.ndarray):
        self.output = 1 / (1 + np.exp(-input_))
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        sig = 1 / (1 + np.exp(-input_))
        self.gradInput = np.multiply(gradOutput, np.multiply(sig, (1 - sig)))
        return self.gradInput

    def __repr__(self) -> str:
        return "Sigmoid"


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = np.tanh(input_)
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        self.gradInput = np.multiply(gradOutput, 1 - np.power(np.tanh(input_), 2))
        return self.gradInput

    def __repr__(self) -> str:
        return "Tanh"


class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit
    https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs
    """
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = np.maximum(input, 0)
        self.output -= self.slope * np.maximum(-input_, 0)
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        mask_positive, mask_negative = input_ >= 0, input_ < 0
        self.gradInput = gradOutput * (mask_positive + mask_negative * self.slope)
        return self.gradInput

    def __repr__(self) -> str:
        return "LeakyReLU"


class ELU(Module):
    """
    Exponential Linear Units
    https://arxiv.org/abs/1511.07289
    """
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = deepcopy(input_)
        self.output[self.output < 0] = (
                np.exp(self.output[self.output < 0]) - 1
        ) * self.alpha
        return self.output

    def updateGradInput(self, input_: np.ndarray, gradOutput: np.ndarray) -> np.ndarray:
        mask_positive, mask_negative = input_ >= 0, input_ < 0
        self.gradInput = gradOutput * (
                mask_positive + np.exp(input) * mask_negative * self.alpha
        )
        return self.gradInput

    def __repr__(self) -> str:
        return "ELU"
