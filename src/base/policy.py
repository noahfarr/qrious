import torch
import torch.nn as nn


class Network(nn.Module):
    """
    A neural network model for representing a policy in reinforcement learning.

    Args:
        layer_sizes (list[int]): The sizes of each layer in the network.
        activation (nn.Module): The activation function used for intermediate layers (default: nn.Tanh).
        output_activation (nn.Module): The activation function used for the output layer (default: nn.Identity).
    """
    def __init__(self, layer_sizes: list[int], activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        layers = []
        num_of_layers = len(layer_sizes)
        for i in range(num_of_layers - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            activation_function = activation if i < (num_of_layers - 2) else output_activation
            layers += [layer, activation_function()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the policy network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)