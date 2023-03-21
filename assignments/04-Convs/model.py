import torch
from torch import nn
from torch.nn import functional as F


class Model(torch.nn.Module):
    """A simple convnet architecture."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialise the model."""
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x) -> torch.Tensor:
        """Outputs the logits for a given input."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# def __init__(self, num_channels: int, num_classes: int) -> None:
#     """Initialise the model."""
#     super(Model, self).__init__()
#     self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
#     self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#     self.linear1 = nn.Linear(32 * 8 * 8, 128)
#     self.linear2 = nn.Linear(128, num_classes)
#
# def forward(self, x: torch.Tensor) -> torch.Tensor:
#     """Outputs the logits for a given input."""
#     x = self.conv1(x)
#     x = F.relu(x)
#     x = F.max_pool2d(x, 2)
#     x = self.conv2(x)
#     x = F.relu(x)
#     x = F.max_pool2d(x, 2)
#     x = torch.flatten(x, 1)
#     x = self.linear1(x)
#     x = F.relu(x)
#     x = self.linear2(x)
#
#     return x
