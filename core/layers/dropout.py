import numpy as np

from core.base.module import Module


class Dropout(Module):
    def __init__(self, p=0.5, **kwargs):
        super(Dropout, self).__init__(**kwargs)

        self.p = p
        self.mask = None

    def updateOutput(
            self,
            input_: np.ndarray
    ) -> np.ndarray:
        if not self.training:
            self.output = input_
        else:
            self.mask = np.random.binomial(1, 1. - self.p, input_.shape)
            self.output = np.multiply(input_, self.mask)
            self.output /= 1. - self.p
        return self.output

    def updateGradInput(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        if not self.training:
            self.gradInput = gradOutput
        else:
            self.gradInput = np.multiply(gradOutput, self.mask)
            self.gradInput /= (1. - self.p)
        return self.gradInput

    def __repr__(self) -> str:
        return "Dropout"
