''' Extra Keras-compatible layers for various purposes '''

from abc import abstractmethod

from keras import backend as K
from keras.layers import Layer


class VariationalBase(Layer):
    ''' Create Variational auto-encoder. It should be called with all of:

            training values
            final output (after going through encoder and decoder)
            mean of latent variables
            log variance of latent variables

        It manually adds a loss term to the model; therefore the model should subsequently
        be compiled with loss=None.
    '''

    def _vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):  # pylint: disable=invalid-name
        ''' Compute the variational loss -- this is the sum of a model-specific term, and a KL
            divergence term that is independent of any modelling choice.
        '''
        model_loss = self.model_nll(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(model_loss + kl_loss)

    @abstractmethod
    def model_nll(self, x, x_pred):  # pylint: disable=invalid-name
        ''' The negative log likelihood of x given a prediction x_pred. '''

    def call(self, inputs):  # pylint: disable=arguments-differ
        x, x_decoded_mean, z_mean, z_log_var = inputs
        loss = self._vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # This isn't used, but it doesn't seem to matter what is returned here, so long as it references the necessary
        # parts of the graph.
        return loss


class VariationalBinaryCrossEntropy(VariationalBase):
    ''' A variational layer based on binary crossentropy model loss '''

    def model_nll(self, x, x_pred):
        ''' Implement using binary_crossentropy '''
        # Summing over the last axis, which corresponds to the features on each training sample.
        # GARGH - Keras has recently changed the order of the arguments to binary_crossentropy.
        # Work around by explicitly specifying them as keyword arguments
        return K.sum(K.binary_crossentropy(output=x_pred, target=x), axis=-1)


class VariationalMSE(VariationalBase):
    ''' A variational layer with a fundamentally Gaussian model '''

    def model_nll(self, x, x_pred):
        ''' The negative log likelihood of x given a prediction x_pred. '''
        # Summing over the last axis, which corresponds to the features on each training sample.
        return K.sum(K.square(x_pred - x), axis=-1)
