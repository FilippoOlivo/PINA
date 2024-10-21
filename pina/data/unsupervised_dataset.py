from .base_dataset import BaseDataset


class UnsupervisedDataset(BaseDataset):
    """
    This class is used to create a dataset of unsupervised data points and, optionally, conditions.
    """
    __slots__ = ['input_points', 'conditional_variables']
