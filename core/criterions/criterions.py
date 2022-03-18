import numpy as np

from core.base.criterion import Criterion


class ClassNLLCriterion(Criterion):
    """
    the softmax input is the output of log-softmax layer.
    This decomposition allows us to avoid problems with
    computation of forward and backward of log().
    """
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ) -> np.float64:
        """
        :param input_: batch_size x n_feats - log probabilities
        :param target: batch_size x n_feats - one-hot representation of ground truth
        """
        self.output = -np.sum(np.multiply(target, input_)) / input_.shape[0]
        return self.output

    def updateGradInput(
            self,
            input_: np.ndarray,
            target: np.ndarray
    ) -> np.ndarray:
        """
        :param input_: batch_size x n_feats - log probabilities
        :param target: batch_size x n_feats - one-hot representation of ground truth
        """
        self.gradInput = -target / input_.shape[0]
        return self.gradInput

    def __repr__(self) -> str:
        return "ClassNLLCriterion"
