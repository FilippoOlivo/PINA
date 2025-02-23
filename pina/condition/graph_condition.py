from .condition_interface import ConditionInterface
from ..graph import Graph
from ..utils import check_consistency
from torch_geometric.data import Data


class GraphCondition(ConditionInterface):
    """
    TODO
    """

    __slots__ = ["graph"]

    def __new__(cls, graph):
        """
        TODO : add docstring
        """
        if not isinstance(graph, (Graph, Data)):
            raise ValueError(
                "GraphCondition takes only a list of Graph objects."
            )

        if isinstance(graph, Graph):
            if all(hasattr(g, "y") for g in graph.data):
                return super().__new__(GraphInputOutputCondition)
            else:
                return super().__new__(GraphDataCondition)
        elif isinstance(graph, Data):
            if hasattr(graph, "x") and hasattr(graph, "y"):
                return super().__new__(GraphInputOutputCondition)
            else:
                return super().__new__(GraphDataCondition)

        raise RuntimeError("Invalid arguments for GraphCondition")

    def __init__(self, graph):

        super().__init__()
        self.graph = graph

    def __setattr__(self, key, value):
        if key == "graph":
            check_consistency(value, (Graph, Data))
            GraphCondition.__dict__[key].__set__(
                self, [value] if isinstance(value, Data) else value.data
            )
        elif key in ("_problem", "_condition_type"):
            super().__setattr__(key, value)


# The split between GraphInputOutputCondition and GraphDataCondition
# distinguishes different types of graph conditions passed to problems.
# This separation simplifies consistency checks during problem creation.
class GraphInputOutputCondition(GraphCondition):
    def __init__(self, graph):
        super().__init__(graph)


class GraphDataCondition(GraphInputOutputCondition):
    def __init__(self, graph):
        super(GraphInputOutputCondition, self).__init__(graph)
