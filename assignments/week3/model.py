import torch
from torch import nn

from typing import Callable


class MLP(torch.nn.Module):
    """A torch implementation of a MultiLayerPerceptron."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
        hidden_sizes: list = None,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            hidden_count: The number of hidden layers.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
            hidden_sizes: List of number of neurons in hidden layers, specified instead o hidden_size and hidden_count.
        """
        super(MLP, self).__init__()

        def get_linear(in_dim, out_dim):
            my_layer = nn.Linear(in_dim, out_dim)
            initializer(my_layer.weight)
            return my_layer

        if hidden_sizes is None:
            hidden_sizes = [hidden_size] * hidden_count

        unit_counts = [input_size] + hidden_sizes

        layers = []

        for count, next_count in zip(unit_counts[:-1], unit_counts[1:]):
            layers.append(get_linear(count, next_count))
            layers.append(activation())
            layers.append(nn.BatchNorm1d(next_count))

        self.hidden_layers = nn.ModuleList(layers)

        self.out = nn.Linear(unit_counts[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.hidden_layers:
            x = layer(x)

        logits = self.out(x)

        return logits
