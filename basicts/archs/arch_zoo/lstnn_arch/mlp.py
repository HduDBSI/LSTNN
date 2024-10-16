import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, P,N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        # hidden = self.fc2(self.act(self.fc1(input_data)))
        hidden = self.norm((hidden + input_data).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)                            # residual
        # hidden = hidden + input_data
        return hidden
