class Batch:

    def __init__(self, idx_dict, dataset_dict) -> None:
        """
        TODO
        """
        self.coordinates_dict = idx_dict
        self.dataset_dict = dataset_dict

    def __len__(self):
        """
        TODO
        """
        length = 0
        for v in self.coordinates_dict.values():
            length += len(v)
        return length

    def __getitem__(self, item):
        """
        TODO
        """
        if isinstance(item, str):
            item = [item]
        if len(item) == 1:
            return self.dataset_dict[item[0]][list(
                self.coordinates_dict[item[0]])]
        return self.dataset_dict[item[0]][item[1]][list(
            self.coordinates_dict[item[0]])]
