''' Hamiltonian Monte Carlo '''

from numbers import Integral, Real

import numpy

from scipy.stats import norm

from pypuffin.decorators import accepts
from pypuffin.numeric.mcmc.base import MCMCBase
from pypuffin.types import Callable


@accepts(numpy.ndarray, numpy.ndarray, Callable, Callable, Real, Integral)
def _leapfrog_iteration(q, p, f_grad_potential, f_grad_kinetic, eps, num_steps):  # pylint: disable=invalid-name
    ''' Perform num_steps of leapfrog iteration in the kinetic energy and potential.

        Arguments:
            q: ndarray of shape (N,):    variables of interest ("position")
            p: ndarray of shape (N,) :   auxiliary variables ("momentum")
            f_grad_potential: q -> (N,): gradient of potential wrt. q
            f_grad_kinetic: p -> (N,):   gradient of potential wrt. p
            eps:                         step size
            num_steps:                   number of steps to take

        Returns q(t + num_steps * eps), p(t + num_steps * eps)
    '''
    # Copy arguments - we will subsequently perform in-place updates. Also cast to float
    q = q.astype(numpy.float64)
    p = p.copy(numpy.float64)

    # Initial step for momentum is a half-step
    p -= (eps / 2) * f_grad_potential(q)

    for _ in range(num_steps - 1):
        # Full steps for both position and momentum, except on the last step
        q += eps * f_grad_kinetic(p)
        p -= eps * f_grad_potential(q)

    # Finish with a full step for position, and half for momentum
    q += eps * f_grad_kinetic(p)
    p -= (eps / 2) * f_grad_potential(q)
    return q, p


class HMC(MCMCBase):
    ''' Hamiltonian Monte Carlo estimator.

        Arguments:
            q_0:                 (N,) Initial position
            f_potential:         q -> (N,) : potential wrt. q
            f_grad_potential:    q -> (N,) : gradient of potential wrt. q
            f_grad_kinetic:      p -> (N,) : gradient of potential wrt. p
            num_leapfrog_steps:  Number of leapfrog steps per HMC iteration
            eps:                 Size of each leapfrog step
    '''

    @accepts(object, numpy.ndarray, Callable, Callable, Callable, Callable, Integral, Real)
    def __init__(self, q_0, f_potential, f_kinetic, f_grad_potential, f_grad_kinetic, num_leapfrog_steps, eps):
        assert len(q_0.shape) == 1
        self._q = q_0  # pylint: disable=invalid-name
        self._f_potential = f_potential
        self._f_kinetic = f_kinetic
        self._f_grad_potential = f_grad_potential
        self._f_grad_kinetic = f_grad_kinetic
        self._num_leapfrog_steps = num_leapfrog_steps
        self._eps = eps

    @property
    def dim(self):
        ''' The dimension of the input space '''
        return self._q.shape[0]

    def sample(self):
        ''' Return a sample, given the current state. Also updates the internal state. '''
        # Start at the current position, and draw a sample for the momentum from a standard normal distribution
        q = self._q
        p = norm.rvs(size=self.dim)

        # Make a note of the initial values; we'll need these later
        q_0, p_0 = q, p

        # Perform the requisite number of leapfrog steps
        q, p = _leapfrog_iteration(q, p, self._f_grad_potential, self._f_grad_kinetic, self._eps,
                                   self._num_leapfrog_steps)

        # NB, we negative the momentum variable here as required to preserve detailed balance in the acceptance
        # criterion. However, in practice K(-p) = K(p), and therefore it doesn't matter.
        # See discussion on p12 of https://arxiv.org/pdf/1206.1901.pdf
        p = -p
        value = -self._f_potential(q) + self._f_potential(q_0) - self._f_kinetic(p) + self._f_kinetic(p_0)
        log_acceptance_probability = min(0, value)
        accept = numpy.random.random() < numpy.exp(log_acceptance_probability)
        q = q if accept else self._q

        # Update the internal state
        self._q = q
        return q


class ScalarMassHMC(HMC):
    ''' Simple HMC which is implemented in terms of a kinetic energy with a single tunable 'mass' parameter.
        That is, the kinetic energy takes the form p.p / 2m.
    '''

    @accepts(object, numpy.ndarray, Callable, Callable, Real, Integral, Real)
    def __init__(self, q_0, f_potential, f_grad_potential, mass, num_leapfrog_steps, eps):
        f_kinetic = lambda p: 0.5 * p.dot(p) / mass
        f_grad_kinetic = lambda p: p / mass
        super().__init__(q_0, f_potential, f_kinetic, f_grad_potential, f_grad_kinetic, num_leapfrog_steps, eps)
