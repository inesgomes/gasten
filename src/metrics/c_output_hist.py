import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .metric import Metric


class OutputsHistogram(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.y_hat = torch.zeros((dataset_size,), dtype=float)

    def update(self, images, batch):
        start_idx, batch_size = batch

        with torch.no_grad():
            c_output = self.C.get(images, start_idx, batch_size)

        self.y_hat[start_idx:start_idx+batch_size] = c_output

    def plot(self):
        sns.histplot(data=self.y_hat, stat='proportion', bins=20)

    def reset(self):
        self.y_hat = torch.zeros((self.dataset_size,), dtype=float)
