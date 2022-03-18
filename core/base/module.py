from typing import Optional, Any
import numpy as np


class Module(object):
    """
    Module is an abstract class which defines fundamental methods
    necessary for a training a neural network.
    Modules are serializable.

    Modules contain two states variables:
        1.  output
        2.  gradInput
    ----
        1.  output = module.forward(input_)
        2.  gradInput = module.backward(input_, gradOutput)
    """

    def __init__(self, **kwargs):
        self.output = None
        self.gradInput = None
        self.training = True

        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        else:
            self.name = 'Nameless'

    def forward(self, input_: np.ndarray):
        """
            Takes an input object, and computes the
            corresponding output of the module.
        ----
            After a forward(), the output state variable
            should have been updated to the new value.
        """
        return self.updateOutput(input_)

    def backward(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        """
            Performs a backpropagation step through the module,
            with respect to the given input_.
        ----
            In general this method makes the assumption forward(input)
            has been called before, with the same input.
            This is necessary for optimization reasons.
            If you do not respect this rule, backward()
            will compute incorrect gradients.

        This includes
            1.  updateGradInput - computing a gradient w.r.t. `input_`
                (is needed for further backprop),
            2.  accGradParameters - computing a gradient w.r.t. parameters
                (to update parameters while optimizing).
        """
        self.updateGradInput(input_, gradOutput)
        self.accGradParameters(input_, gradOutput)
        return self.gradInput

    def updateOutput(
            self,
            input_: np.ndarray
    ) -> np.float64:
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the output field.

        Make sure to both store the data in `output` field and return it.
        """
        raise NotImplementedError("U Should implement it in a child module")

    def updateGradInput(
            self,
            input_: np.ndarray,
            gradOutput: np.ndarray
    ) -> np.ndarray:
        """
        Computing the gradient of the module with respect to its own input_.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input_`.
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        raise NotImplementedError("U Should implement it in a child module")

    def accGradParameters(
            self,
            input_,
            gradOutput,
            scale: Optional[Any] = None
    ):
        """
        Computing the gradient of the module with respect to its
        own parameters. Many modules do not perform
        this step as they do not have any parameters (e.g. ReLU).
        The state variable name for the parameters is module dependent.
        The module is expected to accumulate the gradients
        with respect to the parameters in some variable.

        Zeroing this accumulation is achieved with zeroGradParameters()
        and updating the parameters according to this accumulation
        is done with updateParameters().
        Params:
            1.  scale is a scale factor that is multiplied
                with the gradParameters before being accumulated.
        """
        pass

    def zeroGradParameters(self):
        """
        If the module has parameters, this will zero the accumulation
        of the gradients with respect to these parameters, accumulated
        through accGradParameters(input, gradOutput,scale) calls.
        Otherwise, it does nothing.
        """
        pass

    def getParameters(self):
        """
        This function returns two tensors. One for the flattened learnable
        parameters flatParameters and another for the gradients of
        the energy w.r.t to the learnable parameters flatGradParameters.

        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        This sets the mode of the Module (or sub-modules) to train=true.
        This is useful for modules like Dropout or BatchNormalization
        that have a different behaviour during training vs evaluation.
        """
        self.training = True

    def evaluate(self):
        """
        This sets the mode of the Module (or sub-modules) to train=false.
        This is useful for modules like Dropout or BatchNormalization that
        have a different behaviour during training vs evaluation.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"
