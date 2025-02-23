from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the classifier

        Args:
            x (torch.Tensor): Batched input tensor of images

        Returns:
            torch.Tensor (batch_size, num_classes): Output logits
        """

    @abstractmethod
    def get_conv_layers(self) -> Dict[str, nn.Module]:
        """Get dict of layers to be used in feature extraction

        Returns:
            dict (str, torch.nn.Sequential)
        """

    @abstractmethod
    def get_name(self) -> str:
        """Get name of the model, e.g. AlexNet, ResNet18

        Returns:
            str: Name
        """
