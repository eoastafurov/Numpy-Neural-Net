from typing import Any, Optional, Dict, Union, NoReturn, Tuple, List

import numpy as np

from data.data import DataLoader
from core.base.module import Module
from core.base.criterion import Criterion

from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model: Module,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            val_dataloader: DataLoader,
            criterion: Criterion,
            metrics_fn: Dict[Any, str],
            optimizer: Any,
            optimizer_config: Any,
            optimizer_state: Any,
            n_classes: Optional[int] = 10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.metrics_fn = metrics_fn
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config
        self.optimizer_state = optimizer_state
        self.n_classes = n_classes

        self.hist = None
        self.val_hist = None

    def val(
            self
    ) -> Union[NoReturn, Tuple[Module, Dict[str, Dict[str, List[float]]]]]:
        pbar = tqdm(total=len(self.val_dataloader.batches), desc='Validating')
        step = 0
        sum_loss = 0
        try:
            self.model.evaluate()
            for inputs, labels in self.val_dataloader:
                step += 1
                labels = np.eye(self.n_classes)[labels]
                self.model.zeroGradParameters()
                predictions = self.model.forward(inputs)
                loss = self.criterion.forward(predictions, labels)
                self.val_hist['loss'].append(loss)
                sum_loss += loss

                pbar_postfix_str = 'val_avg_loss: {}, val_loss_step: {}'.format(
                    round(sum_loss / step, 3), round(loss, 3)
                )
                for metric in self.metrics_fn.keys():
                    val = metric(predicted=predictions, actual=labels)
                    self.val_hist[self.metrics_fn[metric]].append(val)
                    pbar_postfix_str += ', {}_step: {}'.format(
                        self.metrics_fn[metric], round(val, 3)
                    )

                pbar.update(1)
                pbar.set_postfix_str(pbar_postfix_str)
        except KeyboardInterrupt:
            print('Detected KeyBoardInterrupt. Performing graceful shutdown...')
            pbar.close()
            return self.model, {'train': self.hist, 'val': self.val_hist}
        except RuntimeError:
            pbar.close()
            pass
        pbar.close()

    def _mean_grad(self):
        grad = 0
        for layer_grads in self.model.getGradParameters():
            for obj in layer_grads:
                grad += np.linalg.norm(np.ravel(obj))
        return grad

    def fit(
            self,
            n_epochs,
            verbose: Optional[bool] = True
    ) -> Tuple[Module, Dict[str, Dict[str, List[float]]]]:
        self.hist = {
            'loss': [],
            'grad': []
        }
        self.val_hist = {
            'loss': []
        }
        for metric_name in self.metrics_fn.values():
            self.hist[metric_name] = []
            self.val_hist[metric_name] = []

        self.val()

        sum_loss = 0
        step = 0
        for i in range(n_epochs):
            pbar = tqdm(
                total=len(self.train_dataloader.batches),
                desc='Epoch {}/{}'.format(i + 1, n_epochs)
            )
            try:
                self.model.train()
                for inputs, labels in self.train_dataloader:
                    step += 1
                    labels = np.eye(self.n_classes)[labels]
                    self.model.zeroGradParameters()

                    predictions = self.model.forward(inputs)
                    loss = self.criterion.forward(predictions, labels)
                    self.hist['loss'].append(loss)
                    sum_loss += loss
                    self.model.backward(inputs, self.criterion.backward(predictions, labels))
                    self.hist['grad'].append(self._mean_grad())

                    # Update weights
                    self.optimizer(
                        self.model.getParameters(),
                        self.model.getGradParameters(),
                        self.optimizer_config,
                        self.optimizer_state
                    )

                    pbar_postfix_str = 'avg_loss: {}, loss_step: {}'.format(
                        round(sum_loss / step, 3), round(loss, 3)
                    )
                    for metric in self.metrics_fn.keys():
                        val = metric(predicted=predictions, actual=labels)
                        self.hist[self.metrics_fn[metric]].append(val)
                        pbar_postfix_str += ', {}_step: {}'.format(
                            self.metrics_fn[metric], round(val, 3)
                        )

                    pbar.update(1)
                    pbar.set_postfix_str(pbar_postfix_str)
            except KeyboardInterrupt:
                print('Detected KeyBoardInterrupt. Performing graceful shutdown...')
                pbar.close()
                return self.model, {'train': self.hist, 'val': self.val_hist}
            # except StopIteration:
            except RuntimeError:
                pbar.close()
                self.val()

            pbar.close()

        return self.model, {'train': self.hist, 'val': self.val_hist}
