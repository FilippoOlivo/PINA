"""
This module provide basic data management functionalities
"""
from torch.utils.data import Dataset

class PinaDataset(Dataset):
    def __init__(self, conditions_dict, max_conditions_lengths):
        self.conditions_dict = conditions_dict
        self.length = self._get_max_len()
        self.max_conditions_lengths = max_conditions_lengths
        self.conditions_length = {k: len(v['input_points']) for k, v in self.conditions_dict.items()}

    def _get_max_len(self):
        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition['input_points']))
        return max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[:self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx%condition_len for idx in cond_idx]
            to_return_dict[condition] = {k: v[[cond_idx]] for k, v in data.items()}
        return to_return_dict





