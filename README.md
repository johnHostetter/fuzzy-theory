# fuzzy-theory: Fuzzy Set Theory and Fuzzy Logic Operations in PyTorch :fire:
<a href="https://github.com/johnHostetter/fuzzy-theory/actions"><img alt="Actions Status" src="https://github.com/johnHostetter/fuzzy-theory/workflows/Test/badge.svg"></a>
<a href="https://github.com/johnHostetter/fuzzy-theory/actions"><img alt="Actions Status" src="https://github.com/johnHostetter/fuzzy-theory/workflows/Pylint/badge.svg"></a>
<a href="https://codecov.io/github/johnHostetter/fuzzy-theory"><img src="https://codecov.io/github/johnHostetter/fuzzy-theory/graph/badge.svg?token=WeWKlnVHqj"/></a>
<a href="https://github.com/psf/fuzzy-theory"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

The `fuzzy-theory` library provides a PyTorch interface to fuzzy set theory and fuzzy logic 
operations. It uses minimal dependencies to implement these features and is designed to be
easy to use and understand. The library is designed to be used in conjunction with PyTorch
and is built on top of PyTorch's tensor operations.

A benefit of using `fuzzy-theory` is that it allows for the creation of fuzzy sets and fuzzy
logic operations in a way that is compatible with PyTorch's autograd system. This means that
you can use the library to create fuzzy sets and perform fuzzy logic operations in a way that
is differentiable and can be used in neural networks and other machine learning models.

## Special features :high_brightness:
1. *Compatible with TorchScript*: Some classes may use `torch.jit.script` or `torch.jit.trace` for production environments.
2. *Differentiable*: The library is designed to be used in conjunction with PyTorch and is built on top of PyTorch's tensor operations.
3. *Minimal dependencies*: The library uses minimal dependencies to implement these features.
4. *Easy to use*: The library is designed to be easy to use and understand, with a simple API that is similar to PyTorch's tensor operations.
5. *Visualization*: Formulas are written with `sympy` for LaTeX rendering and plots are stylized with `scienceplots` for publication-ready figures.  
