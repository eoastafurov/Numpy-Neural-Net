import numpy as np


def sgd_momentum(
        variables,
        gradients,
        config,
        state
):
    """
    :param variables: list of lists of variables (one list per layer)
    :param gradients: list of lists of current gradients (same structure as for variables, one array for each var)
    :param config: dict with optimization parameters (learning_rate and momentum)
    :param state: dict with optimizator state (used to save accumulated gradients)
    """
    # 'variables' and 'gradients' have complex structure,
    # accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1


def adam_optimizer(
        variables,
        gradients,
        config,
        state
):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)  # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2'] ** state['t']) / \
           (1 - config['beta1'] ** state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * current_grad * current_grad,
                   out=var_second_moment)
            current_var -= lr_t * var_first_moment / np.sqrt((var_second_moment) + config['epsilon'])

            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1
