"""
Basic data module implementation
"""
from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor
from ..graph import Graph


class BaseDataset(Dataset):
    """
    BaseDataset class, which handle initialization and data retrieval
    :var condition_indices: List of indices
    :var device: torch.device
    :var condition_names: dict of condition index and corresponding name
    """

    def __new__(cls, problem=None, device=torch.device('cpu')):
        """
        Ensure correct definition of __slots__ before initialization
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.device device: The device on which the
        dataset will be loaded.
        """
        if cls is BaseDataset:
            raise TypeError(
                'BaseDataset cannot be instantiated directly. Use a subclass.')
        if not hasattr(cls, '__slots__'):
            raise TypeError(
                'Something is wrong, __slots__ must be defined in subclasses.')
        return object.__new__(cls)

    def __init__(self, problem=None, device=torch.device('cpu')):
        """"
        Initialize the object based on __slots__
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.device device: The device on which the
        dataset will be loaded.
        """
        super().__init__()
        self.empty = True
        self.problem = problem
        self.device = device
        self.condition_indices = None
        for slot in self.__slots__:
            setattr(self, slot, [])
        self.num_el_per_condition = []
        self.conditions_idx = []
        if self.problem is not None:
            self._init_from_problem()

    def _init_from_problem(self):
        collector = self.problem.collector
        for name, data in collector.data_collections.items():
            keys = list(data.keys())
            if sorted(self.__slots__) == sorted(keys):
                self._populate_init_list(data)
                idx = [key for key, val in collector.conditions_name.items() if
                       val == name]
                self.conditions_idx.append(idx)
        self.initialize()

    def add_points(self, data_dict, condition_idx):
        if self.empty:
            if list(data_dict.keys()) != self.__slots__:
                raise ValueError('Wrong keys in input dict')
            self._populate_init_list(data_dict)
            self.conditions_idx.append(condition_idx)
        else:
            self._add_point_not_empty(data_dict, condition_idx)

    def _add_point_not_empty(self, data_dict, condition_idx):
        num_el_condition = None
        for k, v in data_dict.items():
            if isinstance(v, (LabelTensor, torch.Tensor)) and isinstance(
                    getattr(self, k), (LabelTensor, torch.Tensor)):
                setattr(self, k, LabelTensor.vstack([getattr(self, k), v]))
            elif isinstance(v, list) and isinstance(getattr(self, k), list):
                setattr(self, k, getattr(self, k).append(v))

    def initialize(self):
        if self.num_el_per_condition:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * self.num_el_per_condition[i],
                                 dtype=torch.uint8)
                    for i in range(len(self.num_el_per_condition))
                ],
                dim=0,
            )
            for slot in self.__slots__:
                current_attribute = getattr(self, slot)
                if all(isinstance(a, LabelTensor) for a in current_attribute):
                    setattr(self, slot, LabelTensor.vstack(current_attribute))
        self.empty = False

    def _populate_init_list(self, data_dict):
        current_cond_num_el = None
        for slot in self.__slots__:
            slot_data = data_dict[slot]
            if isinstance(slot_data, (LabelTensor, torch.Tensor,
                    Graph)) or isinstance(slot_data,
                    list) and len(slot_data) > 1:
                if current_cond_num_el is None:
                    current_cond_num_el = len(slot_data)
                elif current_cond_num_el != len(slot_data):
                    raise ValueError('Different number of conditions')
            current_list = getattr(self, slot)
            current_list += [data_dict[slot]] if not (
                isinstance(data_dict[slot], list)) else data_dict[slot]
        self.num_el_per_condition.append(current_cond_num_el)

    def __len__(self):
        return len(getattr(self, self.__slots__[0]))

    def __getattribute__(self, item):
        attribute = super().__getattribute__(item)
        if isinstance(attribute,
                      LabelTensor) and attribute.dtype == torch.float32:
            attribute = attribute.to(device=self.device).requires_grad_()
        return attribute

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx).to(self.device)
        if isinstance(idx, slice):
            to_return_list = []
            for i in self.__slots__:
                to_return_list.append(getattr(self, i)[idx].to(self.device))
            return to_return_list

        if isinstance(idx, (tuple, list)):
            if (len(idx) == 2 and isinstance(idx[0], str)
                    and isinstance(idx[1], (list, slice))):
                tensor = getattr(self, idx[0])
                return tensor[[idx[1]]].to(self.device)
            if all(isinstance(x, int) for x in idx):
                to_return_list = []
                for i in self.__slots__:
                    to_return_list.append(
                        getattr(self, i)[[idx]].to(self.device))
                return to_return_list

        raise ValueError(f'Invalid index {idx}')
