from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor


class SamplePointDataset(Dataset):
    """
    This class is used to create a dataset of sample points.
    """

    def __init__(self, problem, device) -> None:
        """
        TODO
        """
        input_points_list = []
        self.condition_names = {}
        collector = problem.collector
        idx = 0
        for name, data in collector.data_collections.items():
            if 'output_points' not in data.keys() and 'input_points' in data.keys():
                input_points_list.append(data['input_points'])
                self.condition_names[idx] = name
                idx += 1

        self.input_points = LabelTensor.vstack(input_points_list) if len(input_points_list) > 0 else None
        if self.input_points is not None:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * len(input_points_list[i]), dtype=torch.uint8)
                    for i in range(len(self.condition_names))
                ],
                dim=0,
            )
        else:  # if there are no sample points
            self.condition_indices = torch.tensor([])
            self.input_points = torch.tensor([])
        self.input_points = self.input_points.to(device)
        self.condition_indices = self.condition_indices.to(device)
        self.splitting_dimension = 0

    def __len__(self):
        """
        TODO
        """
        return self.input_points.shape[0]

    def __getitem__(self, idx):
        """
        TODO
        """

        if isinstance(idx, str):
            return getattr(self, idx)
        elif isinstance(idx, (tuple, list)):
            if len(idx) == 2 and isinstance(idx[0], str) and isinstance(idx[1], list):
                tensor = getattr(self, idx[0])
                return tensor[[idx[1]]]
            if all(isinstance(x, int) for x in idx):
                return (
                    self.input_input_points[[idx]],
                    self.output_input_points[[idx]],
                    self.condition_indices[[idx]]
                )
