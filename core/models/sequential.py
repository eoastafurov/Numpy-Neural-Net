from abc import ABC
from typing import List, Union, NoReturn, List
import numpy as np
from core.base.module import Module


class Sequential(Module, ABC):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.modules = []

    def add(
            self,
            module: Union[Module, List[Module]]
    ):
        if isinstance(module, Module):
            self.modules.append(module)
        elif isinstance(module, list):
            self.modules.extend(module)

    def updateOutput(
            self,
            input_: np.ndarray
    ) -> np.ndarray:
        self.output = input_

        for module in self.modules:
            self.output = module.forward(self.output)

        return self.output

    def backward(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input_, g_1)
        """
        for i in range(len(self.modules) - 1, 0, -1):
            gradOutput = self.modules[i].backward(self.modules[i - 1].output, gradOutput)

        self.gradInput = self.modules[0].backward(input_, gradOutput)

        return self.gradInput

    def zeroGradParameters(self) -> NoReturn:
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self) -> List[np.ndarray]:
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self) -> List[np.ndarray]:
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self) -> str:
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x: int) -> Module:
        return self.modules.__getitem__(x)

    def train(self) -> NoReturn:
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self) -> NoReturn:
        self.training = False
        for module in self.modules:
            module.evaluate()
