import torch
from torch import nn
from torch.nn import functional as F


class Model(torch.nn.Module):
    """A simple convnet architecture."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialise the model."""
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.linear = nn.Linear(10 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs the logits for a given input."""
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
