import torch
import pytest

from pina import LabelTensor, Condition
from pina.condition import GraphInputOutputCondition
from pina.domain import CartesianDomain
from pina.equation.equation_factory import FixedValue
from pina.graph import RadiusGraph
from torch_geometric.data import Data

example_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])
example_output_pts = LabelTensor(torch.tensor([[1, 2]]), ["a", "b"])


def test_init_inputoutput():
    Condition(input_points=example_input_pts, output_points=example_output_pts)
    with pytest.raises(ValueError):
        Condition(example_input_pts, example_output_pts)
    with pytest.raises(ValueError):
        Condition(input_points=3.0, output_points="example")
    with pytest.raises(ValueError):
        Condition(input_points=example_domain, output_points=example_domain)


def test_init_domainfunc():
    Condition(domain=example_domain, equation=FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(domain=3.0, equation="example")
    with pytest.raises(ValueError):
        Condition(domain=example_input_pts, equation=example_output_pts)


def test_init_inputfunc():
    Condition(input_points=example_input_pts, equation=FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(input_points=3.0, equation="example")
    with pytest.raises(ValueError):
        Condition(input_points=example_domain, equation=example_output_pts)


def test_graph_condition():
    x = torch.rand(10, 10, 4)
    pos = torch.rand(10, 10, 2)
    y = torch.rand(10, 10, 2)
    graph = RadiusGraph(x=x, pos=pos, r=0.1, build_edge_attr=True, y=y)
    condition = Condition(graph=graph)
    assert isinstance(condition, GraphInputOutputCondition)
    assert isinstance(condition.graph, list)

    x = x[0]
    pos = pos[0]
    y = y[0]
    edge_index = graph.data[0].edge_index
    graph = Data(x=x, pos=pos, edge_index=edge_index, y=y)
    condition = Condition(graph=graph)
    assert isinstance(condition, GraphInputOutputCondition)
    assert isinstance(condition.graph, list)
    assert len(condition.graph) == 1
    assert isinstance(condition.graph[0], Data)
