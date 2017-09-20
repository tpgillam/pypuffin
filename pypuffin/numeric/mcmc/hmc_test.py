''' Tests for pypuffin.numeric.mcmc.hmc.py '''

from unittest import TestCase

import numpy

from pypuffin.numeric.mcmc.hmc import HMC


class TestHMC(TestCase):
    ''' Tests for the hmc module '''

    def test_hmc_1d_quadratic(self):
        ''' Test HMC sampling from a quadratic potential well. This is equivalent to sampling from a Gaussian '''
        numpy.random.seed(0)
        q_0 = numpy.asarray([0,])
        f_potential = lambda q: q.dot(q)  # pylint: disable=unnecessary-lambda
        f_kinetic = lambda p: p.dot(p)  # pylint: disable=unnecessary-lambda
        f_grad_potential = lambda q: 2 * q
        f_grad_kinetic = lambda p: 2 * p
        num_leapfrog_steps = 100
        eps = 1e-2
        hmc = HMC(q_0, f_potential, f_kinetic, f_grad_potential, f_grad_kinetic, num_leapfrog_steps, eps)

        samples = [hmc.sample() for _ in range(500)]
        mean = numpy.mean(samples)
        self.assertAlmostEqual(mean, 0, places=1)
