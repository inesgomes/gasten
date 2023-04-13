import torch.nn as nn


class ClassifierCache():
    def __init__(self, C):
        super(ClassifierCache, self).__init__()
        self.C = C
        self.last_output = None
        self.last_output_feature_maps = None
        self.last_batch_idx = None
        self.last_batch_size = None

    def get(self, x, batch_idx, batch_size, output_feature_maps=False):
        if batch_idx != self.last_batch_idx or batch_size != self.last_batch_size:
            output = self.C(x, output_feature_maps=True)
            self.last_output = output[-1]
            self.last_output_feature_maps = output[-2]
            self.last_batch_idx = batch_idx
            self.last_batch_size = batch_size

        if output_feature_maps:
            return self.last_output, self.last_output_feature_maps
        else:
            return self.last_output
