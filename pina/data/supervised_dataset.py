from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class SupervisedDataset(Dataset):
    def __init__(self, problem, device) -> None:
        super().__init__()
        input_list = []
        output_list = []
        self.condition_names = {}
        collector = problem.collector
        idx = 0
        for name, data in collector.data_collections.items():
            if 'output_points' in data.keys():
                input_list.append(data['input_points'])
                output_list.append(data['output_points'])
                self.condition_names[idx] = name
                idx += 1

        self.input_points = LabelTensor.vstack(input_list) if len(input_list) > 0 else None
        self.output_points = LabelTensor.vstack(output_list) if len(input_list) > 0 else None

        if self.input_points is not None:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(input_list[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no data points
            self.condition_indices = torch.tensor([])
            self.input_points = torch.tensor([])
            self.output_points = torch.tensor([])

        self.device = device

    def __len__(self):
        return self.input_points.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx).to(self.device)
        if isinstance(idx, slice):
            return (
                self.input_points[[idx]].to(self.device),
                self.output_points[[idx]].to(self.device),
                self.condition_indices[[idx]].to(self.device),
            )
        elif isinstance(idx, (tuple, list)):
            if len(idx) == 2 and isinstance(idx[0], str) and isinstance(idx[1], (list, slice)):
                tensor = getattr(self, idx[0])
                return tensor[[idx[1]]].to(self.device)
            if all(isinstance(x, int) for x in idx):
                return (
                    self.input_points[[idx]].to(self.device),
                    self.output_points[[idx]].to(self.device),
                    self.condition_indices[[idx]].to(self.device),
                )
