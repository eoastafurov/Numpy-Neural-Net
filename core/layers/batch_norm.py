import numpy as np

from core.base.module import Module


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(
            self,
            alpha=0.,
            **kwargs
    ):
        super(BatchNormalization, self).__init__(**kwargs)
        self.batch_var = None
        self.batch_mean = None
        self.alpha = alpha
        self.moving_mean = 0
        self.moving_variance = 0

    def updateOutput(
            self,
            input_: np.ndarray
    ) -> np.ndarray:
        self.output = np.zeros_like(input_)
        if self.training:
            self.batch_mean = input_.mean(axis=0)
            self.batch_var = input_.var(axis=0)
            self.moving_mean = self.moving_mean * self.alpha + self.batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + self.batch_var * (1 - self.alpha)
            self.output = np.subtract(input_, self.batch_mean) / np.sqrt(self.batch_var + self.EPS)
        else:
            self.output = np.subtract(input_, self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output

    def updateGradInput(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        self.gradInput = np.zeros_like(input_)
        normalized = np.zeros_like(input_)
        normalized = np.subtract(input_, self.batch_mean) / (np.sqrt(np.add(self.batch_var, self.EPS)))
        np.multiply(np.divide(1, np.multiply(np.sqrt(np.add(self.batch_var, self.EPS)), input_.shape[0])),
                    np.subtract(np.subtract(np.multiply(input_.shape[0], gradOutput), gradOutput.sum(axis=0)),
                                np.multiply(normalized, np.sum(np.multiply(gradOutput, normalized), axis=0))),
                    out=self.gradInput)
        return self.gradInput

    def __repr__(self) -> str:
        return "BatchNormalization" if self.name == 'Nameless' else self.name
