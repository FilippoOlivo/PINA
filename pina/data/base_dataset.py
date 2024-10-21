from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class BaseDataset(Dataset):
    """
    BaseDataset class, which ensures that __slots__ is defined and initializes
    the problem and device for the dataset.
    """
    __slots__ = []

    def __new__(cls, problem, device):
        if not cls.__slots__:
            raise TypeError('Something is wrong, __slots__ must be defined in subclasses.')
        return super().__new__(cls)

    def __init__(self, problem, device):
        super().__init__()
        self.condition_names = {}
        collector = problem.collector

        for slot in self.__slots__:
            setattr(self, slot, [])

        idx = 0

        for name, data in collector.data_collections.items():
            keys = []
            for k, v in data.items():
                if isinstance(v, LabelTensor):
                    keys.append(k)
            if self.__slots__ == sorted(keys):
                for slot in self.__slots__:
                    current_list = getattr(self, slot)
                    current_list.append(data[slot])
                self.condition_names[idx] = name
                idx += 1

        if len(getattr(self, self.__slots__[0])) > 0:
            input_list = getattr(self, self.__slots__[0])
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(input_list[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
            for slot in self.__slots__:
                current_attribute = getattr(self, slot)
                setattr(self, slot, LabelTensor.vstack(current_attribute))
        else:
            self.condition_indices = torch.tensor([], dtype=torch.uint8)
            for slot in self.__slots__:
                setattr(self, slot, torch.tensor([]))
        self.device = device

    def __len__(self):
        return len(getattr(self, self.__slots__[0]))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx).to(self.device)
        if isinstance(idx, slice):
            to_return_list = []
            for i in self.__slots__:
                to_return_list.append(getattr(self, i)[[idx]].to(self.device))
            return to_return_list
        if isinstance(idx, (tuple, list)):
            if len(idx) == 2 and isinstance(idx[0], str) and isinstance(idx[1], (list, slice)):
                tensor = getattr(self, idx[0])
                return tensor[[idx[1]]].to(self.device)
            if all(isinstance(x, int) for x in idx):
                to_return_list = []
                for i in self.__slots__:
                    to_return_list.append(getattr(self, i)[[idx]].to(self.device))
                return to_return_list
        raise ValueError(f'Invalid index {idx}')
