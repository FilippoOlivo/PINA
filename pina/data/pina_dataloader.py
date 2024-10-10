from random import sample

import torch

from .sample_dataset import SamplePointDataset
from .data_dataset import DataPointDataset
from .unsupervised_dataset import UnsupervisedDataset
from .pina_batch import Batch


class SamplePointLoader:
    """
    This class is used to create a dataloader to use during the training.

    :var condition_names: The names of the conditions. The order is consistent
        with the condition indeces in the batches.
    :vartype condition_names: list[str]
    """

    def __init__(
        self, sample_dataset, data_dataset, unsupervised_dataset, batch_size=None, shuffle=True
    ) -> None:
        """
        Constructor.

        :param SamplePointDataset sample_pts: The sample points dataset.
        :param int batch_size: The batch size. If ``None``, the batch size is
            set to the number of sample points. Default is ``None``.
        :param bool shuffle: If ``True``, the sample points are shuffled.
            Default is ``True``.
        """
        if not isinstance(sample_dataset, SamplePointDataset):
            raise TypeError(
                f"Expected SamplePointDataset, got {type(sample_dataset)}"
            )
        if not isinstance(data_dataset, DataPointDataset):
            raise TypeError(
                f"Expected DataPointDataset, got {type(data_dataset)}"
            )
        if not isinstance(unsupervised_dataset, UnsupervisedDataset):
            raise TypeError(
                f"Expected UnsupervisedDataset, got {type(data_dataset)}"
            )
        #Extract number of conditions
        self.n_data_conditions = len(data_dataset.condition_names)
        self.n_phys_conditions = len(sample_dataset.condition_names)
        self.n_unsupervised_conditions = len(unsupervised_dataset.condition_names)

        #Increment indices in data condition and update names dict
        data_dataset.condition_names = {key + self.n_phys_conditions: value
                                          for key, value in data_dataset.condition_names.items()}
        unsupervised_dataset.condition_names = {key + self.n_phys_conditions + self.n_data_conditions: value
                                          for key, value in data_dataset.condition_names.items()}
        data_dataset.condition_indices += self.n_phys_conditions
        unsupervised_dataset.condition_indices += self.n_phys_conditions + self.n_data_conditions

        # Initialize batches
        self._init_batches(sample_dataset, data_dataset, unsupervised_dataset, batch_size, shuffle)

        #Shuffle batches list
        if shuffle:
            self.random_idx = torch.randperm(len(self.batches))
        else:
            self.random_idx = torch.arange(len(self.batches))

    def _init_batches(self, sample_dataset, data_dataset, unsupervised_dataset, batch_size=None, shuffle=True):
        start_data_conditions = len(sample_dataset)
        start_unsupervised_conditions = start_data_conditions + len(data_dataset)
        if batch_size is None:
            batch_size = len(sample_dataset) + len(data_dataset)
        if shuffle:
            idx = torch.randperm(len(sample_dataset) + len(data_dataset) + len(unsupervised_dataset))
        else:
            idx = torch.arange(len(sample_dataset) + len(data_dataset) + len(unsupervised_dataset))
        batch_num = idx.shape[0] // batch_size
        if idx.shape[0] % batch_size != 0:
            batch_num += 1
        batches = torch.tensor_split(idx, batch_num)
        self.batches = []
        for batch in batches:
            sample_mask = batch < start_data_conditions
            data_mask = (batch >= start_data_conditions) & (batch < start_unsupervised_conditions)
            unsupervised_mask = batch >= start_unsupervised_conditions
            sample_conditions_idx = batch[sample_mask].tolist()
            data_conditions_idx = (batch[data_mask] - start_data_conditions).tolist()
            unsupervised_conditions_idx = (batch[unsupervised_mask] - start_unsupervised_conditions).tolist()
            sample_records = sample_dataset[sample_conditions_idx]
            data_records = data_dataset[data_conditions_idx]
            unsupervised_records = unsupervised_dataset[unsupervised_conditions_idx]
            self.batches.append(Batch(sample_records, data_records, unsupervised_records))


    def __iter__(self):
        """
        Return an iterator over the points. Any element of the iterator is a
        dictionary with the following keys:
            - ``pts``: The input sample points. It is a LabelTensor with the
                shape ``(batch_size, input_dimension)``.
            - ``output``: The output sample points. This key is present only
                if data conditions are present. It is a LabelTensor with the
                shape ``(batch_size, output_dimension)``.
            - ``condition``: The integer condition indeces. It is a tensor
                with the shape ``(batch_size, )`` of type ``torch.int64`` and
                indicates for any ``pts`` the corresponding problem condition.

        :return: An iterator over the points.
        :rtype: iter
        """
        for i in self.random_idx:
            yield self.batches[i]

    def __len__(self):
        """
        Return the number of batches.

        :return: The number of batches.
        :rtype: int
        """
        return len(self.batches)
