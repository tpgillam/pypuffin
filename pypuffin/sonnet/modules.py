''' Helper methods for variational autoencoders, using sonnet '''

import sonnet as snt
import tensorflow as tf

from pypuffin.decorators import accepts
from pypuffin.types import Callable


# # TODO derive from AbstractModule to provide better-defined interface for encoder?
# class VAEEncoder(snt.AbstractModule):
#     ''' Encoder for VAEs - must provide a _build method that returns the mean of the latent variable, however
#         additionally can compute the standard deviation of the latent variables when training
#     '''
#
#     @property
#     def latent_mean(self):
#         ''' The mean of of the latent variables, given the input '''
#         self._ensure_is_connected()
#         return self._latent_mean
#
#     @property
#     def latent_log_var(self):
#         ''' The log variance of the latent variables '''
#         self._ensure_is_connected()
#         return self._latent_log_var


class VariationalAutoencoder(snt.AbstractModule):
    ''' Given encoder and decoder modules, this module constructs a network with the necessary KL divergence term in the
        loss. The _build operation returns the variational loss. Other information can be obtained from the module once
        it has been connected to the graph.

        Arguments:

            encoder_module:   Mapping from input (?, N,) -> mean & log variance of variational parameters (?, N, 2).
            decoder_module:   Mapping from variational parameters (?, M,) -> output (?, N,).
            f_model_nll:      Function mapping a pair of tensors (prediction, target) to the NLL of the prediction given
                              the true target value.
    '''

    @accepts(object, snt.AbstractModule, snt.AbstractModule, Callable, name=str)
    def __init__(self, encoder_module, decoder_module, f_model_nll, name='variational_autoencoder'):
        super().__init__(name=name)
        self._encoder_module = encoder_module
        self._decoder_module = decoder_module
        self._f_model_nll = f_model_nll

    def _build(self, input_):
        encoded_mean_var = self._encoder_module(input_)
        z_mean, z_log_var = tf.unstack(encoded_mean_var, axis=-1)

        mean_shape = tuple(z_mean.shape.as_list())
        log_var_shape = tuple(z_log_var.shape.as_list())
        if not mean_shape == log_var_shape:
            raise ValueError("Shapes of mean and log variance encodings are {} and {}, but should be equal".format(
                mean_shape, log_var_shape))

        # Samples from a standard normal distribution, which we then scale and move
        epsilon = tf.random_normal(tf.shape(z_mean), name='epsilon')
        latent = z_mean + epsilon * tf.exp(z_log_var / 2)

        output = self._decoder_module(latent)

        # KL divergence term
        kl_losses = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        model_nlls = self._f_model_nll(output, input_)

        self._kl_loss = tf.reduce_mean(kl_losses, name='kl_loss')
        self._model_nll = tf.reduce_mean(model_nlls, name='model_nll')

        return tf.add(self._kl_loss, self._model_nll, name='loss')

    @property
    def kl_loss(self):
        ''' The KL divergence term contributing to the loss. Only available after connection to the graph '''
        self._ensure_is_connected()
        return self._kl_loss

    @property
    def model_nll(self):
        ''' The model negative log-likelihood. Only available after connection to the graph '''
        self._ensure_is_connected()
        return self._model_nll
