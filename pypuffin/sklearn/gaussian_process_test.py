import numpy

from unittest import TestCase

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF

from pypuffin.sklearn.gaussian_process import _gradient_constant, _gradient_kernel, _gradient_product, _gradient_rbf


def _reshapex(x):
    ''' Reshape an x value to include a second dimension '''
    assert len(x.shape) == 1
    return x.reshape((x.shape[0], 1))


class TestGaussianProcess(TestCase):
    ''' Tests for the gaussian_process module '''

    def setUp(self):
        super().setUp()
        x = numpy.linspace(1, 8, 6)
        y = x * numpy.sin(x)

        kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.regressor = GaussianProcessRegressor(kernel, n_restarts_optimizer=20)
        self.X = _reshapex(x)
        self.regressor.fit(self.X, y)

        x_eval = numpy.linspace(0, 10, 20)
        self.X_eval = _reshapex(x_eval)

    def _test_kernel_gradient_impl(self, kernel, gradient_func):
        ''' Common implementation for testing the gradient computation of a kernel '''
        gradient = gradient_func(kernel, self.X_eval, self.X)
        # _gradient_kernel should always work
        gradient_2 = _gradient_kernel(kernel, self.X_eval, self.X)

        # Numerically estimate the gradient. Note reshape we have to do, since here we're in the special
        # case of one input dimension
        dx = 1e-6
        expected_gradient = (kernel(self.X_eval + dx, self.X) - kernel(self.X_eval, self.X)) / dx
        expected_gradient = expected_gradient.reshape(expected_gradient.shape + (1,))

        self.assertTrue(numpy.allclose(gradient, expected_gradient))
        self.assertTrue(numpy.allclose(gradient_2, expected_gradient))

    def test_gradient_constant(self):
        kernel = self.regressor.kernel_.k1
        self.assertIsInstance(kernel, ConstantKernel)
        self._test_kernel_gradient_impl(kernel, _gradient_constant)
        

    def test_gradient_rbf(self):
        kernel = self.regressor.kernel_.k2
        self.assertIsInstance(kernel, RBF)
        self._test_kernel_gradient_impl(kernel, _gradient_rbf)

    def test_gradient_product(self):
        kernel = self.regressor.kernel_
        self.assertIsInstance(kernel, Product)
        self._test_kernel_gradient_impl(kernel, _gradient_product)



        
