''' Base classes defining MCMC interface '''

from abc import ABCMeta, abstractproperty, abstractmethod


class MCMCBase(metaclass=ABCMeta):
    ''' An abstract class which defines methods for sampling '''

    @abstractproperty
    def dim(self):
        ''' The dimension of the sampling space '''

    @abstractmethod
    def sample(self):
        ''' Return a sample, given the current state. Also updates the internal state. '''
