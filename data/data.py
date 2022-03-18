import os
import numpy
import cv2
from typing import Optional, Tuple, List, NoReturn

import numpy as np


class DatasetImageFolder:
    def __init__(
        self,
        root: str,
        class_names: List[str],
        flatten: Optional[bool] = False,
        supported_image_types: Optional[tuple] = ('jpg', 'png'),
        gray: Optional[bool] = False
    ):
        super(DatasetImageFolder, self).__init__()
        self.root = root
        self.class_names = class_names
        self.images_filepaths = []
        self.flatten = flatten
        self.gray = gray

        for class_name in self.class_names:
            for fname in os.listdir(os.path.join(self.root, class_name)):
                if fname.split('.')[-1] in supported_image_types:
                    if not fname.split('/')[-1].startswith('.'):
                        self.images_filepaths.append(
                            os.path.join(self.root, class_name, fname)
                        )

        self.class_names = {
            class_name: int(i)
            for class_name, i in zip(
                self.class_names,
                range(len(self.class_names))
            )
        }

    def __len__(self) -> int:
        return len(self.images_filepaths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        label = self.class_names[image_filepath.split('/')[-2]]

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.flatten:
            image = np.ravel(image)

        return image.astype(np.float) / 255.0, label


class DataLoader:
    def __init__(
            self,
            dataset: DatasetImageFolder,
            batch_size: int,
            shuffle: bool
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = None
        self.batches = None
        self.counter = 0

        self.setup()

    def setup(self) -> NoReturn:
        self.indices = np.array(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batches = np.array_split(
            self.indices, int(len(self.dataset) / float(self.batch_size))
        )

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.counter == len(self.batches):
            self.counter = 0
            # raise StopIteration
            raise RuntimeError
        indices = self.batches[self.counter]
        images, labels = [], []
        for idx in indices:
            img, label = self.dataset[idx]
            images.append(img)
            labels.append(label)
        self.counter += 1
        return np.array(images), np.array(labels)


class DataManager:
    def __init__(
            self,
            train_data_path: str,
            test_data_path: str,
            val_data_path: str,
            class_names: List[str],
            batch_size: int,
            flatten: Optional[bool] = False,
            gray: Optional[bool] = False
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.class_names = class_names
        self.batch_size = batch_size
        self.flatten = flatten
        self.gray = gray

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.setup()

    def setup(self) -> NoReturn:
        train_dataset = DatasetImageFolder(
            root=self.train_data_path,
            class_names=self.class_names,
            flatten=self.flatten,
            gray=self.gray
        )
        test_dataset = DatasetImageFolder(
            root=self.test_data_path,
            class_names=self.class_names,
            flatten=self.flatten,
            gray=self.gray
        )
        val_dataset = DatasetImageFolder(
            root=self.val_data_path,
            class_names=self.class_names,
            flatten=self.flatten,
            gray=self.gray
        )

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader

    def get_val_dataloader(self) -> DataLoader:
        return self.val_dataloader
