"""Module for Multi FeedForward model"""

from abc import ABC, abstractmethod
import torch
from .feed_forward import FeedForward


class MultiFeedForward(torch.nn.Module, ABC):
    """
    The PINA implementation of MultiFeedForward network.

    This model allows to create a network with multiple FeedForward combined
    together. The user has to define the `forward` method choosing how to
    combine the different FeedForward networks.

    :param dict ffn_dict: dictionary of FeedForward networks.
    """

    def __init__(self, ffn_dict):
        super().__init__()

        if not isinstance(ffn_dict, dict):
            raise TypeError

        for name, constructor_args in ffn_dict.items():
            setattr(self, name, FeedForward(**constructor_args))

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        TODO: Docstring
        """
