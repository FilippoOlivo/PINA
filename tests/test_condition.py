import torch
import pytest

from pina import LabelTensor, Condition
from pina.domain import CartesianDomain
from pina.equation.equation_factory import FixedValue

example_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
example_output_pts = LabelTensor(torch.tensor([[1, 2]]), ['a', 'b'])


def test_init_inputoutput():
    Condition(input_points=example_input_pts, output_points=example_output_pts)
    with pytest.raises(ValueError):
        Condition(example_input_pts, example_output_pts)
    with pytest.raises(ValueError):
        Condition(input_points=3., output_points='example')
    with pytest.raises(ValueError):
        Condition(input_points=example_domain, output_points=example_domain)


test_init_inputoutput()


def test_init_domainfunc():
    Condition(domain=example_domain, equation=FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(domain=3., equation='example')
    with pytest.raises(ValueError):
        Condition(domain=example_input_pts, equation=example_output_pts)


def test_init_inputfunc():
    Condition(input_points=example_input_pts, equation=FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(input_points=3., equation='example')
    with pytest.raises(ValueError):
        Condition(input_points=example_domain, equation=example_output_pts)
