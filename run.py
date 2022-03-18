import json
import os

from data.data import DataManager
from core.activations.activations import LogSoftMax, ReLU, Sigmoid, Tanh
from core.criterions.criterions import ClassNLLCriterion
from optimizers.optimizers import adam_optimizer, sgd_momentum
from trainer import Trainer
from metrics.metrics import Accuracy
from models_zoo import Zoo


def run():
    dm = DataManager(
        train_data_path='data/Dataset/train',
        test_data_path='data/Dataset/test',
        val_data_path='data/Dataset/test',
        class_names=[str(i) for i in range(10)],
        batch_size=512,
        flatten=True,
        gray=False
    )

    criterion = ClassNLLCriterion()

    optimizers = {
        'Adam': {
            'optimizer': adam_optimizer,
            'config': {'learning_rate': 3e-4, 'beta1': 0.8, 'beta2': 0.8, 'epsilon': 1e-08}
        },
        'SGD': {
            'optimizer': sgd_momentum,
            'config': {'learning_rate': 0.0001, 'momentum': 0.9}
        }
    }

    metrics = {
        Accuracy(): 'acc'
    }

    pure_linear_relu_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=ReLU,
                                     softmax=LogSoftMax(), bn=False, dp=False, depth=7)
    pure_linear_tanh_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Tanh,
                                     softmax=LogSoftMax(), bn=False, dp=False, depth=7)
    pure_linear_sigmoid_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Sigmoid,
                                        softmax=LogSoftMax(), bn=False, dp=False, depth=7)

    pure_linear_relu_SGD = Zoo.auto(input_dim=3072, out_dim=10, Activation=ReLU,
                                    softmax=LogSoftMax(), bn=False, dp=False, depth=7)

    bn_linear_relu_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=ReLU,
                                   softmax=LogSoftMax(), bn=True, dp=False, depth=7)
    bn_linear_tanh_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Tanh,
                                   softmax=LogSoftMax(), bn=True, dp=False, depth=7)
    bn_linear_sigmoid_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Sigmoid,
                                      softmax=LogSoftMax(), bn=True, dp=False, depth=7)

    bn_dp_linear_relu_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=ReLU,
                                      softmax=LogSoftMax(), bn=True, dp=True, depth=7)
    bn_dp_linear_tanh_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Tanh,
                                      softmax=LogSoftMax(), bn=True, dp=True, depth=7)
    bn_dp_linear_sigmoid_adam = Zoo.auto(input_dim=3072, out_dim=10, Activation=Sigmoid,
                                         softmax=LogSoftMax(), bn=True, dp=True, depth=7)

    pipeline = [
        (pure_linear_relu_adam, 'pure_linear_relu_adam'),
        (pure_linear_tanh_adam, 'pure_linear_tanh_adam'),
        (pure_linear_sigmoid_adam, 'pure_linear_sigmoid_adam'),

        (pure_linear_relu_SGD, 'pure_linear_relu_SGD'),

        (bn_linear_relu_adam, 'bn_linear_relu_adam'),
        (bn_linear_tanh_adam, 'bn_linear_tanh_adam'),
        (bn_linear_sigmoid_adam, 'bn_linear_sigmoid_adam'),

        (bn_dp_linear_relu_adam, 'bn_dp_linear_relu_adam'),
        (bn_dp_linear_tanh_adam, 'bn_dp_linear_tanh_adam'),
        (bn_dp_linear_sigmoid_adam, 'bn_dp_linear_sigmoid_adam')
    ]

    def get_adam_trainer(model_):
        trainer_ = Trainer(
            model=model_,
            train_dataloader=dm.get_train_dataloader(),
            test_dataloader=dm.get_test_dataloader(),
            val_dataloader=dm.get_val_dataloader(),
            criterion=criterion,
            metrics_fn=metrics,
            optimizer=optimizers['Adam']['optimizer'],
            optimizer_config=optimizers['Adam']['config'],
            optimizer_state={}
        )
        return trainer_

    def get_sgd_trainer(model_):
        trainer_ = Trainer(
            model=model_,
            train_dataloader=dm.get_train_dataloader(),
            test_dataloader=dm.get_test_dataloader(),
            val_dataloader=dm.get_val_dataloader(),
            criterion=criterion,
            metrics_fn=metrics,
            optimizer=optimizers['SGD']['optimizer'],
            optimizer_config=optimizers['SGD']['config'],
            optimizer_state={}
        )
        return trainer_

    for model, title in pipeline:
        trainer = get_adam_trainer(model) if title.endswith('adam') else get_sgd_trainer(model)
        _, hist = trainer.fit(n_epochs=100)

        os.mkdir('runs/{}'.format(title))
        with open('runs/{}/hist.json'.format(title), 'w+') as f:
            json.dump(hist, f)


if __name__ == '__main__':
    run()
