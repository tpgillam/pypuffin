''' Extra functionality based on sklearn's Gaussian Process library '''

import numpy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF

from pypuffin.decorators import accepts


def _gradient_kernel(kernel, x_1, x_2):
    ''' Compute the gradient of a kernel covariance between x_1 and x_2
    
        Arguments:

            kernel:  kernel (possibly composite) instance
            x_1:  ndarray of shape (n_1, dim), where dim is the dimensionality of the input space.
            x_2:  ndarray of shape (n_2, dim) of the training points

        Returns an array of shape (n_1, n_2, dim), representing the gradient of each component of the covariance
        matrix.
    '''
    # In order to support a new type of kernel, create the necessary function below, and add it into this mapping.
    kernel_type_to_gradient_function = {
            ConstantKernel: _gradient_constant,
            Product: _gradient_product,
            RBF: _gradient_rbf}

    if type(kernel) not in kernel_type_to_gradient_function:
        raise ValueError("Unsupported kernel for gradient: {}".format(kernel))

    return kernel_type_to_gradient_function[type(kernel)](kernel, x_1, x_2)


@accepts(ConstantKernel, numpy.ndarray, numpy.ndarray)
def _gradient_constant(kernel, x_1, x_2):
    ''' Returns the gradient of a constant kernel. This will always be zero. '''
    return numpy.zeros((x_1.shape[0], x_2.shape[0], x_2.shape[1]))


@accepts(RBF, numpy.ndarray, numpy.ndarray)
def _gradient_rbf(kernel, x_1, x_2):
    ''' Compute the gradient of an RBF kernel covariance. '''
    x_twiddle = x_1[:, numpy.newaxis, :] - x_2[numpy.newaxis, :, :]
    return -(1 / kernel.length_scale ** 2) * x_twiddle * kernel(x_1, x_2)[:, :, numpy.newaxis]


@accepts(Product, numpy.ndarray, numpy.ndarray)
def _gradient_product(kernel, x_1, x_2):
    ''' Returns the gradient of the product kernel '''
    value_1 = kernel.k1(x_1, x_2)
    value_2 = kernel.k2(x_1, x_2)
    gradient_1 = _gradient_kernel(kernel.k1, x_1, x_2)
    gradient_2 = _gradient_kernel(kernel.k2, x_1, x_2)

    # Combine with the product rule
    # The values are of shape (n_eval, n_train), whereas the gradients are of dimension (n_eval, n_train, dim)
    # We must broadcast across the last dimension
    return value_1[:, :, numpy.newaxis] * gradient_2 + value_2[:, :, numpy.newaxis] * gradient_1


@accepts(GaussianProcessRegressor, numpy.ndarray)
def gradient_of_mean(regressor, x_eval):
    ''' Given a trained GP regressor instance, compute the gradient of the mean function at the given points
        in the input space.

        Arguments:

            regressor:  The trained GP regressor instance
            x_eval:     ndarray of shape (n_eval, dim)

        Returns array of shape (n_eval, dim), representing the gradient of the predicted surface at each point.
    '''
    # kernel_gradient is of shape (n_eval, n_train, dim)
    kernel_gradient = _gradient_kernel(regressor.kernel_, x_eval, regressor.X_train_)

    # alpha_ is of shape (n_train,). We need to contract along the matching indices, which in this case means axis 1
    # of kernel_gradient, and axis 0 of alpha_. This is what the following function call performs.
    return numpy.tensordot(kernel_gradient, regressor.alpha_, (1, 0))

