''' Tests for pypuffin.sklearn.gaussian_process '''

from unittest import TestCase

import numpy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF

from pypuffin.sklearn.gaussian_process import (_gradient_constant, _gradient_kernel, _gradient_product, _gradient_rbf,
                                               gradient_of_mean, gradient_of_std)


def _reshapex(value):
    ''' Reshape an x value to include a second dimension '''
    assert len(value.shape) == 1
    return value.reshape((value.shape[0], 1))


class TestGaussianProcess(TestCase):
    ''' Tests for the gaussian_process module '''

    def setUp(self):
        super().setUp()
        numpy.random.seed(0)
        x = numpy.linspace(1, 8, 6)
        self.x = _reshapex(x)
        y = x * numpy.sin(x)
        x_eval = numpy.linspace(0, 10, 20)
        self.x_eval = _reshapex(x_eval)

        kernel = ConstantKernel(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.regressor = GaussianProcessRegressor(kernel, n_restarts_optimizer=20)
        self.regressor.fit(self.x, y)

    def _test_kernel_gradient_impl(self, kernel, gradient_func):
        ''' Common implementation for testing the gradient computation of a kernel '''
        gradient = gradient_func(kernel, self.x_eval, self.x)
        # _gradient_kernel should always work
        gradient_2 = _gradient_kernel(kernel, self.x_eval, self.x)

        # Numerically estimate the gradient. Note reshape we have to do, since here we're in the special
        # case of one input dimension
        delta = 1e-6
        expected_gradient = (kernel(self.x_eval + delta, self.x) - kernel(self.x_eval, self.x)) / delta
        expected_gradient = expected_gradient.reshape(expected_gradient.shape + (1,))

        self.assertTrue(numpy.allclose(gradient, expected_gradient))
        self.assertTrue(numpy.allclose(gradient_2, expected_gradient))

    def test_gradient_constant(self):
        ''' Test _gradient_constant '''
        kernel = self.regressor.kernel_.k1
        self.assertIsInstance(kernel, ConstantKernel)
        self._test_kernel_gradient_impl(kernel, _gradient_constant)

    def test_gradient_rbf(self):
        ''' Test _gradient_rbf '''
        kernel = self.regressor.kernel_.k2
        self.assertIsInstance(kernel, RBF)
        self._test_kernel_gradient_impl(kernel, _gradient_rbf)

    def test_gradient_product(self):
        ''' Test _gradient_product '''
        kernel = self.regressor.kernel_
        self.assertIsInstance(kernel, Product)
        self._test_kernel_gradient_impl(kernel, _gradient_product)

    def test_gradient_of_mean(self):
        ''' Test gradient_of_mean '''
        gradient = gradient_of_mean(self.regressor, self.x_eval)

        # Finite difference estimate. We need to do the reshaping, for the same reasons as in _test_kernel_gradient_impl
        delta = 1e-6
        expected_gradient = (self.regressor.predict(self.x_eval + delta) - self.regressor.predict(self.x_eval)) / delta
        expected_gradient = _reshapex(expected_gradient)
        self.assertTrue(numpy.allclose(gradient, expected_gradient))

    def test_gradient_of_std(self):
        ''' Test gradient_of_std '''
        gradient = gradient_of_std(self.regressor, self.x_eval)

        # Finite difference estimate. We need to do the reshaping, for the same reasons as in _test_kernel_gradient_impl
        delta = 1e-6
        _, std_x_dx = self.regressor.predict(self.x_eval + delta, return_std=True)
        _, std_x = self.regressor.predict(self.x_eval, return_std=True)
        expected_gradient = (std_x_dx - std_x) / delta
        expected_gradient = _reshapex(expected_gradient)

        # Have to relax tolerance slightly
        self.assertTrue(numpy.allclose(gradient, expected_gradient, atol=1e-6))
