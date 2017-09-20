''' Implementation of GPHMC - Gaussian Process HMC '''

import numpy

from sklearn.gaussian_process import GaussianProcessRegressor

from pypuffin.decorators import accepts
from pypuffin.numeric.mcmc.base import MCMCBase
from pypuffin.sklearn.gaussian_process import gradient_of_mean, gradient_of_std
from pypuffin.types import Callable


# TODO need different f_construct_hmc methods for exploratory and sampling phases.
# TODO still not implementing all the heuristics for exploration in Rasumussen's paper


class GPHMC(MCMCBase):
    ''' An object to perform GPHMC sampling. Takes as arguments:

            f_target_log_prob: A callable mapping position x -> log of the target distribution.
            regressor:         A (non-trained) GaussianProcessRegressor instance, with appropriate kernel etc. This
                               will be used to approximate f_target_log_prob.
            f_construct_hmc:   A callable to construct an HMC sampler that takes the signature
                                (x_0, f_potential, f_grad_potential)
            x_start:           Position from which to start GP sampling
    '''

    @accepts(object, Callable, GaussianProcessRegressor, Callable, numpy.ndarray)
    def __init__(self, f_target_log_prob, regressor, f_construct_hmc, x_start):
        self._f_target_log_prob = f_target_log_prob
        self._regressor = regressor
        self._f_construct_hmc = f_construct_hmc
        self._x_start = x_start

        # Record training data for GP regressor; y values are from f_target_log_prob
        self._x_train = []
        self._y_train = []

        # The HMC sampler for using once training is complete.
        self._hmc_sampler = None

    @property
    def _started_sampling(self):
        ''' Return True iff we have started sampling from the non-training distribution '''
        return self._hmc_sampler is not None

    @property
    def dim(self):
        ''' The dimension of the sampling space '''
        return self._x_start.shape[0]

    def _fit_gp(self):
        ''' Perform fitting of the regressor to the current training data, taking into account the empirical mean of
            the training points.
            This follows the procedure given in Rasmussen GPHMC paper, section 4.
        '''
        x_train_array = numpy.asarray(self._x_train)
        y_train_array = numpy.asarray(self._y_train)
        mean = numpy.mean(y_train_array, axis=0)
        return self._regressor.fit(x_train_array, y_train_array - mean)

    def predict_gp(self, x, return_std=False):
        ''' Perform one or more predictions with the GP regression model, taking into account the training data mean
            which was used to train the current regressor.
        '''
        y_train_array = numpy.asarray(self._y_train)
        mean = numpy.mean(y_train_array, axis=0)

        single_prediction = False
        if len(x.shape) == 1:
            # If it looks like we have been passed a single x value, reshape appropriately
            x = x[numpy.newaxis, :]
            single_prediction = True

        if return_std:
            mus, stds = self._regressor.predict(x, return_std=True)
            if single_prediction:
                return mus[0] + mean, stds[0]
            else:
                return mus + mean, stds
        mus = self._regressor.predict(x, return_std=False)
        if single_prediction:
            return mus[0] + mean
        else:
            return mus + mean

    def sample_explore(self):
        ''' Perform an exploratory sample. This will perform HMC under the prior distribution, sample a point, and
            update the trained distribution accordingly. The sampled location will be returned.
            This should only be done before real sampling begins.
        '''
        if self._started_sampling:
            raise RuntimeError("Training should only be done prior to beginning real sampling!")

        if not self._x_train:
            # Our very first sampling point will be to evaluate x_start
            self._x_train.append(self._x_start)
            self._y_train.append(self._f_target_log_prob(self._x_start))
            x_1 = self._x_start
        else:
            # Our sampling is going to be for (mean - std), as per Rasmussen. Remember that our potential is -log(prob),
            # and our regressor is fitted to log(prob).
            # Since we want to sample single points, there's a bit of disgustingness in reshaping x
            # from (n_dim,) to (1, n_dim), and then only looking at the first element of the result.
            def f_potential(x):  # pylint: disable=invalid-name
                ''' (mu - std) '''
                mu, std = self.predict_gp(x, return_std=True)
                return -(mu - std)

            def f_grad_potential(x):  # pylint: disable=invalid-name
                ''' Gradient of (mu - std) '''
                reshaped_x = x[numpy.newaxis, :]
                mean_grad = gradient_of_mean(self._regressor, reshaped_x)[0]
                std_grad = gradient_of_std(self._regressor, reshaped_x)[0]
                # FIXME debugging
                if numpy.isnan(mean_grad).any() or numpy.isnan(std_grad).any():
                    print('plop', mean_grad, std_grad)
                return -(mean_grad - std_grad)

            # Construct a new sampler based on these operations, and take one sample.
            x_0 = self._x_train[-1]
            x_1 = self._f_construct_hmc(x_0, f_potential, f_grad_potential).sample()

            # Evaluate the target function. If we have proposed the same point again, there is no point in evaluating
            # the function, since it's expensive and we already know the answer. Moreover, adding duplicate points
            # to the training data seems to simply upset the GP, so don't do it.
            if x_1 != x_0:
                y_1 = self._f_target_log_prob(x_1)
                self._x_train.append(x_1)
                self._y_train.append(y_1)

        # Refit the distribution using all training data, and return our current location
        self._fit_gp()
        return x_1

    def sample(self):
        ''' Draw a new sample '''
        if not self._started_sampling:
            # We are now entering the sampling phase - construct an HMC sampler with our most up-to-date view of the
            # Gaussian Process. This sampler will be re-used henceforth.
            f_potential = lambda x: -self._regressor.predict(x[numpy.newaxis, :])[0]
            f_grad_potential = lambda x: -gradient_of_mean(self._regressor, x[numpy.newaxis, :])[0]

            self._hmc_sampler = self._f_construct_hmc(self._x_start, f_potential, f_grad_potential)

        return self._hmc_sampler.sample()
