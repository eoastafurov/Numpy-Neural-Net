from typing import Optional, Tuple, Union, Type, Any, NoReturn, List, TypeVar

import numpy as np
from core.base.module import Module


class UniformInitializer:
    @staticmethod
    def initialize(
            matrix: Union[np.ndarray, Tuple[int, int], int],
            boundaries: Optional[Tuple[float, float]] = (-1.0, 1.0)
    ) -> np.ndarray:
        np.random.seed(42)
        return np.random.uniform(
            low=boundaries[0],
            high=boundaries[1],
            size=matrix.shape if isinstance(matrix, np.ndarray) else matrix
        )


class ZeroInitializer:
    @staticmethod
    def initialize(
            matrix: Union[np.ndarray, Tuple[int, int], int]
    ) -> np.ndarray:
        return np.zeros_like(
            a=matrix
        )


class Linear(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            Initializer: Optional[Type[UniformInitializer]] = UniformInitializer,
            **kwargs
    ):
        super(Linear, self).__init__(**kwargs)

        std = 1. / np.sqrt(in_features)
        self.weight = Initializer.initialize(
            matrix=(out_features, in_features),
            boundaries=(-std, std)
        )
        self.bias = Initializer.initialize(
            matrix=out_features,
            boundaries=(-std, std)
        )

        self.grad_weight = ZeroInitializer.initialize(self.weight)
        self.grad_bias = ZeroInitializer.initialize(self.bias)

    def updateOutput(self, input_: np.ndarray) -> np.ndarray:
        self.output = input_ @ self.weight.T + self.bias
        return self.output

    def updateGradInput(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        self.gradInput = np.zeros_like(input_)
        np.matmul(gradOutput, self.weight.astype(input_.dtype), out=self.gradInput)
        return self.gradInput

    def accGradParameters(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray,
            scale: Optional[Any] = None
    ) -> NoReturn:
        self.grad_weight = gradOutput.T @ input_
        self.grad_bias = gradOutput.sum(axis=0)

    def zeroGradParameters(self) -> NoReturn:
        self.grad_weight.fill(0)
        self.grad_bias.fill(0)

    def getParameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias]

    def getGradParameters(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias]

    def __repr__(self) -> str:
        repr_ = '{}\t|\tInput shape: (batch_size, {})\t|\tOutput shape: (batch_size, {})\t|\t'.format(
            'Linear' if self.name == 'Nameless' else self.name,
            self.weight.shape[1],
            self.weight.shape[0]
        ) + '#Params: {}\n'.format(self.weight.shape[0] * self.weight.shape[1] + self.bias.shape[0])
        return repr_  # + ''.join(['-'] * int(len(repr_) * 1.3))
