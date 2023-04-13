import torch
from .metric import Metric


class LossSecondTerm(Metric):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.count = 0
        self.acc = 0
        self.result = float('inf')

    def update(self, images, batch):
        start_idx, batch_size = batch

        with torch.no_grad():
            c_output = self.C.get(images, start_idx, batch_size)

        term_2 = (0.5 - c_output).abs().sum().item()

        self.acc += term_2
        self.count += images.size(0)

    def finalize(self):
        self.result = self.acc / self.count

        return self.result

    def reset(self):
        self.count = 0
        self.acc = 0
        self.result = float('inf')
