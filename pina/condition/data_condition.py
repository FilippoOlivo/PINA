import torch

from . import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency

class DataConditionInterface(ConditionInterface):
    """
    Condition for data. This condition must be used every
    time a Unsupervised Loss is needed in the Solver. The conditionalvariable
    can be passed as extra-input when the model learns a conditional
    distribution
    """

    __slots__ = ["data", "conditional_variable"]

    def __init__(self, data, conditional_variable=None):
        """
        TODO
        """
        super().__init__()
        self.data = data
        self.conditional_variable = conditional_variable
        self.condition_type = 'unsupervised'

    def __setattr__(self, key, value):
        if (key == 'data') or (key == 'conditional_variable'):
            check_consistency(value, (LabelTensor, Graph, torch.Tensor))
            DataConditionInterface.__dict__[key].__set__(self, value)
        elif key in ('_condition_type', '_problem', 'problem', 'condition_type'):
            super().__setattr__(key, value)