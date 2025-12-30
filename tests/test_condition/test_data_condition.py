import pytest
import torch
from pina import Condition, LabelTensor
from pina.condition import (
    TensorDataCondition,
    GraphDataCondition,
)
from pina.graph import RadiusGraph
from torch_geometric.data import Data


def _create_tensor_data(use_lt=False, conditional_variables=False):
    input_tensor = torch.rand((10, 3))
    if use_lt:
        input_tensor = LabelTensor(input_tensor, ["x", "y", "z"])
    if conditional_variables:
        cond_vars = torch.rand((10, 2))
        if use_lt:
            cond_vars = LabelTensor(cond_vars, ["a", "b"])
    else:
        cond_vars = None
    return input_tensor, cond_vars


def _create_graph_data(use_lt=False, conditional_variables=False):
    if use_lt:
        x = LabelTensor(torch.rand(10, 20, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(10, 20, 2), ["x", "y"])
    else:
        x = torch.rand(10, 20, 2)
        pos = torch.rand(10, 20, 2)
    radius = 0.1
    input_graph = [
        RadiusGraph(pos=pos[i], radius=radius, x=x[i]) for i in range(len(x))
    ]
    if conditional_variables:
        if use_lt:
            cond_vars = LabelTensor(torch.rand(10, 20, 1), ["f"])
        else:
            cond_vars = torch.rand(10, 20, 1)
    else:
        cond_vars = None
    return input_graph, cond_vars


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_tensor_data_condition(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = TensorDataCondition(
        input=input_tensor, conditional_variables=cond_vars
    )
    type_ = LabelTensor if use_lt else torch.Tensor
    if conditional_variables:
        assert condition.conditional_variables is not None
        assert isinstance(condition.conditional_variables, type_)
        if use_lt:
            assert condition.conditional_variables.labels == ["a", "b"]
    else:
        assert condition.conditional_variables is None
    assert isinstance(condition.input, type_)
    if use_lt:
        assert condition.input.labels == ["x", "y", "z"]


test_init_tensor_data_condition(False, False)


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_graph_data_condition(use_lt, conditional_variables):
    input_graph, cond_vars = _create_graph_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = GraphDataCondition(
        input=input_graph, conditional_variables=cond_vars
    )
    type_ = LabelTensor if use_lt else torch.Tensor
    if conditional_variables:
        assert condition.conditional_variables is not None
        assert isinstance(condition.conditional_variables, type_)
        if use_lt:
            assert condition.conditional_variables.labels == ["f"]
    else:
        assert condition.conditional_variables is None
        # assert "conditional_variables" not in condition.data.keys()
    assert isinstance(condition.input, list)
    for graph in condition.input:
        assert isinstance(graph, Data)
        assert isinstance(graph.x, type_)
        if use_lt:
            assert graph.x.labels == ["u", "v"]
        assert isinstance(graph.pos, type_)
        if use_lt:
            assert graph.pos.labels == ["x", "y"]


test_init_graph_data_condition(False, False)
