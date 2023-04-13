import torch


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, pos_class, neg_class):
        data = []
        targets = []

        for X, y in original_dataset:
            if y == pos_class or y == neg_class:
                data.append(torch.unsqueeze(X, dim=0))
                targets.append(
                    torch.ones((1,), dtype=torch.float32) if y == pos_class else torch.zeros((1,), dtype=torch.float32))

        self.data = torch.vstack(data)
        self.targets = torch.hstack(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
