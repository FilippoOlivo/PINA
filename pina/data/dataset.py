"""
This module provide basic data management functionalities
"""
from torch.utils.data import Dataset

class PinaDataset(Dataset):
    def __init__(self, conditions_dict, max_conditions_lengths, resample):
        self.conditions_dict = conditions_dict
        self.max_conditions_lengths = max_conditions_lengths
        self.conditions_length = {k: len(v['input_points']) for k, v in
                                  self.conditions_dict.items()}
        #self.use_small_batch = False
        self.resample = True if resample or len(self.conditions_dict) == 1 \
            else False
        if resample:
            self.length = max(self.conditions_length.values())
        else:
            self.length = sum(self.conditions_length.values())
            self.offsets = [sum(list(self.conditions_length.values())[:i])
                            for i in range(len(self.conditions_length))]

    def _get_max_len(self):
        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition['input_points']))
        return max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Getitem method for large batch size
        """
        if not self.resample:
            return self._resample_getitem(idx)
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[:self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx%condition_len for idx in cond_idx]
            to_return_dict[condition] = {k: v[[cond_idx]]
                                         for k, v in data.items()}
        return to_return_dict

    def _resample_getitem(self, idx):
        """
        Getitem method for resampling
        """

        to_return_dict = {}
        extractor_dict = {k: [] for k in self.conditions_dict.keys()}
        for i in idx:
            cond_name = None
            for idx, name in enumerate(list(self.conditions_dict.keys())[:-1]):
                if i < self.offsets[idx+1]:
                    cond_name = name
                    break
                if idx == len(self.conditions_dict)-2:
                    idx += 1
                    cond_name = list(self.conditions_dict.keys())[idx]
            extractor_dict[cond_name].append(i - self.offsets[idx])
        for cond_name, idx in extractor_dict.items():
            if len(idx) == 0:
                continue
            to_return_dict[cond_name] = {k: v[[idx]]
                        for k, v in self.conditions_dict[cond_name].items()}
        return to_return_dict

    '''
    def _small_batch_getitem(self, idx):
        return {
            k: {k_data: v[k_data][idx % len(v['input_points'])] for k_data
                in v.keys()} for k, v in self.conditions_dict.items()
        }

    def _large_batch_getitem(self, idx):
        """
        Getitem method for large batch size
        """

        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[:self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx%condition_len for idx in cond_idx]
            to_return_dict[condition] = {k: v[[cond_idx]] 
                                         for k, v in data.items()}
        return to_return_dict
    
    def change_getitem_func(self):
        setattr(self, '__getitem__', self._large_batch_getitem)
    '''