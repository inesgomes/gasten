import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, img_size, num_classes, nf):
        super(Classifier, self).__init__()
        num_channels = img_size[0]

        self.blocks = nn.ModuleList()
        block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels*28*28, nf),
        )
        self.blocks.append(block_1)

        predictor = nn.Sequential(
            nn.Linear(nf, 1 if num_classes ==
                      2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
        )
        self.blocks.append(predictor)

    def forward(self, x, output_feature_maps=False):
        intermediate_outputs = []

        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
