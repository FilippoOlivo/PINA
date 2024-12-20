import torch
import pytest

from pina import LabelTensor
from pina.operators import grad, div, laplacian


def func_vec(x):
    return x**2


def func_scalar(x):
    print('X')
    x_ = x.extract(['x'])
    y_ = x.extract(['y'])
    mu_ = x.extract(['mu'])
    return x_**2 + y_**2 + mu_**3


data = torch.rand((20, 3), requires_grad=True)
inp = LabelTensor(data, ['x', 'y', 'mu'])
labels = ['a', 'b', 'c']
tensor_v = LabelTensor(func_vec(inp), labels)
tensor_s = LabelTensor(func_scalar(inp).reshape(-1, 1), labels[0])


def test_grad_scalar_output():
    grad_tensor_s = grad(tensor_s, inp)
    assert grad_tensor_s.shape == inp.shape
    assert grad_tensor_s.labels == [
        f'd{tensor_s.labels[0]}d{i}' for i in inp.labels
    ]
    grad_tensor_s = grad(tensor_s, inp, d=['x', 'y'])
    assert grad_tensor_s.shape == (inp.shape[0], 2)
    assert grad_tensor_s.labels == [
        f'd{tensor_s.labels[0]}d{i}' for i in ['x', 'y']
    ]


def test_grad_vector_output():
    grad_tensor_v = grad(tensor_v, inp)
    assert grad_tensor_v.shape == (20, 9)
    grad_tensor_v = grad(tensor_v, inp, d=['x', 'mu'])
    assert grad_tensor_v.shape == (inp.shape[0], 6)


def test_div_vector_output():
    grad_tensor_v = div(tensor_v, inp)
    assert grad_tensor_v.shape == (20, 1)
    grad_tensor_v = div(tensor_v, inp, components=['a', 'b'], d=['x', 'mu'])
    assert grad_tensor_v.shape == (inp.shape[0], 1)


def test_laplacian_scalar_output():
    laplace_tensor_s = laplacian(tensor_s, inp, components=['a'], d=['x', 'y'])
    assert laplace_tensor_s.shape == tensor_s.shape
    assert laplace_tensor_s.labels == [f"dd{tensor_s.labels[0]}"]
    true_val = 4*torch.ones_like(laplace_tensor_s)
    assert all((laplace_tensor_s - true_val == 0).flatten())


def test_laplacian_vector_output():
    laplace_tensor_v = laplacian(tensor_v, inp)
    assert laplace_tensor_v.shape == tensor_v.shape
    assert laplace_tensor_v.labels == [
        f'dd{i}' for i in tensor_v.labels
    ]
    laplace_tensor_v = laplacian(tensor_v,
                                 inp,
                                 components=['a', 'b'],
                                 d=['x', 'y'])
    assert laplace_tensor_v.shape == tensor_v.extract(['a', 'b']).shape
    assert laplace_tensor_v.labels == [
        f'dd{i}' for i in ['a', 'b']
    ]
    true_val = 2*torch.ones_like(tensor_v.extract(['a', 'b']))
    assert all((laplace_tensor_v - true_val == 0).flatten())
