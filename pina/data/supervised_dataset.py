from .base_dataset import BaseDataset


class SupervisedDataset(BaseDataset):
    __slots__ = ['input_points', 'output_points']
