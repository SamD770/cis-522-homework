import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    my_mlp = MLP(
        input_size=input_dim,
        hidden_size=None,
        num_classes=output_dim,
        activation=torch.nn.ReLU,
        initializer=torch.nn.init.xavier_uniform_,
        hidden_sizes=[1000, 1500, 500, 100],
    )

    return my_mlp
