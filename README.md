# Numpy-Neural-Net


Dear comrade! You can see a simple and extendable library for deep learning which is written entirely in numpy.

## Core
The choosen architecture a little similar to PyTorch (`Module`, `Criterion`, and etc entities), see `core` folder for precise information.

##  Data
In `data/data.py` you can find 3 useful abstractions:
1. `DatasetImageFolder`
2. `DataLoader`
3. `DataManager`

DatasetImageFolder and DataLoader are similar to Torch, DataManager similar to Torch Lightning.

P.S. To run `run.py` to test efficiency of nn, download [CIFAR-10 dataset](https://drive.google.com/drive/folders/1M0M8jFpfWyi2G45kVovvVeoPgzGo6vaD?usp=sharing)

## Trainer
I have implement `Trainer` in  `trainer.py`, you can use it to fit your model.

```python
dm = DataManager(
    train_data_path='data/Dataset/train',
    test_data_path='data/Dataset/test',
    val_data_path='data/Dataset/test',
    class_names=[str(i) for i in range(10)],
    batch_size=512
)

trainer = Trainer(
    model=model,
    train_dataloader=dm.get_train_dataloader(),
    test_dataloader=dm.get_test_dataloader(),
    val_dataloader=dm.get_val_dataloader(),
    criterion=criterion,
    metrics_fn=metrics,
    optimizer=optimizer,
    optimizer_config=optimizer_config,
    optimizer_state={}
)

model, hist = trainer.fit(n_epochs=100)
```


## Examples
see `example.ipynb`
