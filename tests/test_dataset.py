from copy import deepcopy

import torch
import pytest
from pina.data import SamplePointDataset, DataPointDataset, SamplePointLoader
from pina import LabelTensor, Condition
from pina.data.unsupervised_dataset import UnsupervisedDataset
from pina.equation import Equation
from pina.domain import CartesianDomain
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import laplacian
from pina.equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                    torch.sin(input_.extract(['y'])*torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term

my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]]), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])
in2_ = LabelTensor(torch.rand(60, 2), ['x', 'y'])
out2_ = LabelTensor(torch.rand(60, 1), ['u'])

class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1': Condition(
            domain=CartesianDomain({'x': [0, 1], 'y':  1}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            domain=CartesianDomain({'x': [0, 1], 'y': 0}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            domain=CartesianDomain({'x':  1, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            domain=CartesianDomain({'x': 0, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'D': Condition(
            input_points=LabelTensor(torch.rand(size=(100, 2)), ['x', 'y']),
            equation=my_laplace),
        'data': Condition(
            input_points=in_,
            output_points=out_),
        'data2': Condition(
            input_points=in2_,
            output_points=out2_)
    }

boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']


def test_sample():
    poisson = Poisson()
    poisson.discretise_domain(10, 'grid', locations=boundaries)
    sample_dataset = SamplePointDataset(poisson, device='mps')
    assert len(sample_dataset) == 140
    assert sample_dataset.pts.shape == (140, 2)
    assert sample_dataset.pts.labels == ['x', 'y']
    assert sample_dataset.condition_indices.dtype == torch.uint8
    assert sample_dataset.condition_indices.max() == torch.tensor(4)
    assert sample_dataset.condition_indices.min() == torch.tensor(0)
test_sample()

def test_data():
    poisson = Poisson()
    poisson.discretise_domain(10, 'grid', locations=boundaries)
    dataset = DataPointDataset(poisson, device='cpu')
    assert len(dataset) == 61
    assert dataset.input_pts.shape == (61, 2)
    assert dataset.input_pts.labels == ['x', 'y']
    assert dataset.output_pts.shape == (61, 1 )
    assert dataset.output_pts.labels == ['u']
    assert dataset.condition_indices.dtype == torch.uint8
    assert dataset.condition_indices.max() == torch.tensor(1)
    assert dataset.condition_indices.min() == torch.tensor(0)

def test_loader():
    poisson = Poisson()
    poisson.discretise_domain(10, 'grid', locations=boundaries)
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    data_dataset = DataPointDataset(poisson, device='cpu')
    unsupervised_dataset = UnsupervisedDataset(poisson, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, unsupervised_dataset, batch_size=10)

    for batch in loader:
        assert len(batch) <= 10
        assert batch['supervised', 'input_points'].requires_grad == True
        assert batch['supervised', 'input_points'].labels == ['x', 'y']

    loader2 = SamplePointLoader(sample_dataset, data_dataset, unsupervised_dataset,batch_size=None)
    assert len(loader2) == 1

def test_loader2():
    poisson2 = Poisson()
    poisson2.discretise_domain(10, 'grid', locations=boundaries)
    del poisson2.conditions['data2']
    del poisson2.conditions['data']
    poisson2.discretise_domain(10, 'grid', locations=boundaries)
    sample_dataset = SamplePointDataset(poisson2, device='cpu')
    data_dataset = DataPointDataset(poisson2, device='cpu')
    unsupervised_dataset = UnsupervisedDataset(poisson2, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, unsupervised_dataset, batch_size=10)

    for batch in loader:
        assert batch['sample', 'input_points'].shape[0] <= 10
        assert batch['sample', 'input_points'].requires_grad == True
        assert batch['sample', 'input_points'].labels == ['x', 'y']
# ----------------------------------
# ----------------------------------


def test_loader3():
    class Poisson(SpatialProblem):
        output_variables = ['u']
        spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

        conditions = {
            'gamma1': Condition(
                domain=CartesianDomain({'x': [0, 1], 'y': 1}),
                equation=FixedValue(0.0)),
            'gamma2': Condition(
                domain=CartesianDomain({'x': [0, 1], 'y': 0}),
                equation=FixedValue(0.0)),
            'gamma3': Condition(
                domain=CartesianDomain({'x': 1, 'y': [0, 1]}),
                equation=FixedValue(0.0)),
            'gamma4': Condition(
                domain=CartesianDomain({'x': 0, 'y': [0, 1]}),
                equation=FixedValue(0.0)),
            'D': Condition(
                input_points=LabelTensor(torch.rand(size=(100, 2)), ['x', 'y']),
                equation=my_laplace),
            'data': Condition(
                input_points=in_,
                output_points=out_),
            'data2': Condition(
                input_points=in2_,
                output_points=out2_)
        }

    #boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']

    poisson3 = Poisson()
    del poisson3.conditions['gamma1']
    del poisson3.conditions['gamma2']
    del poisson3.conditions['gamma3']
    del poisson3.conditions['gamma4']
    del poisson3.conditions['D']
    sample_dataset = SamplePointDataset(poisson3, device='cpu')
    data_dataset = DataPointDataset(poisson3, device='cpu')
    unsupervised_dataset = UnsupervisedDataset(poisson3, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, unsupervised_dataset, batch_size=10)

    for batch in loader:
        #assert len(batch) == 2 # only phys condtions
        assert batch['supervised', 'input_points'].shape[0] <= 10
        assert batch['supervised', 'input_points'].requires_grad == True
        assert batch['supervised', 'input_points'].labels == ['x', 'y']