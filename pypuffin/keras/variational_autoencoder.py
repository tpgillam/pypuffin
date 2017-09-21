''' Helper methods for creating variational autoencoders '''

from abc import ABCMeta, abstractmethod
from numbers import Integral

from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model
from keras.optimizers import Optimizer

from pypuffin.decorators import accepts, lazy
from pypuffin.keras.layers import VariationalBase


class VariationalAutoencoder(metaclass=ABCMeta):
    ''' Abstract wrapper class to assist in building a variational autoencoder. You probably want to use LayeredVAE
        below, which wraps this.

        Arguments:

            cls_vae:     Subclass of VariationalBase, determines what form of model likelihood is used.
            dim_latent:  The number of latent variables to use in the encoded representation.
            optimizer:   Any valid optimizer that can be passed to the compilation of a keras Model.
    '''

    @accepts(object, type, Integral, optimizer=(str, Optimizer))
    def __init__(self, cls_vae, dim_latent, optimizer='adam'):
        assert issubclass(cls_vae, VariationalBase), "A variational layer class must be provided"
        self._cls_vae = cls_vae
        self._dim_latent = dim_latent
        self._optimizer = optimizer

    # TODO refactor interface? Might be nicer just to return VAE model, and expose the others elsewhere.
    def build(self, input_):
        ''' Build the model, accepting a Keras layer or input. Returns three models:

                    (vae, encoder, decoder)

            The former is what one uses to train the whole model; following training, the encoder and decoder models
            are useful separately for sampling.
        '''
        z_mean, z_log_var = self.build_encoder(input_)

        def sample(args):
            ''' Given (mean, log_var), return samples from N(mean, std) '''
            mean, log_var = args
            # Construct appropriate number of samples from N(0, 1), depending on the runtime shape of mean.
            epsilon = K.random_normal(shape=(K.shape(mean)[0], self._dim_latent), mean=0, stddev=1)
            # Rescale by mean & variance
            return mean + K.exp(log_var / 2) * epsilon

        # This is a layer which generates samples from z, given appropriate parameter tensors
        training_z = Lambda(sample, output_shape=(self._dim_latent,))([z_mean, z_log_var])
        training_output = self.build_decoder(training_z)

        # The loss is really just a placeholder that gets put into the model - it isn't meaningful, since the
        # variational layer manually adds the necessary loss terms to our objective.
        loss = self._cls_vae()([input_, training_output, z_mean, z_log_var])
        vae = Model(input_, loss)
        # Now compile the model - note that loss is explicitly set to None
        vae.compile(optimizer=self._optimizer, loss=None)

        # An explicit encoder, which maps from the input to the mean of the variational layer
        encoder = Model(input_, z_mean)

        # An explicit decoder, which maps from the variational layer to the output
        input_variational = Input(shape=(self._dim_latent,))
        decoder = Model(input_variational, self.build_decoder(input_variational))

        # Return all models
        return vae, encoder, decoder

    @abstractmethod
    def build_encoder(self, input_):
        ''' Build the encoder network. Must return two layers of shape (dim_latent,) corresponding to the mean
            and log variance of the variational layer.

            NOTE: If invoked multiple times, weights for the encoder network should be shared.
        '''

    @abstractmethod
    def build_decoder(self, input_):
        ''' Given the output of the variational layer (i.e. tensor of shape (dim_latent,)), construct a model
            transforming into the output space.

            NOTE: It is important that, weights are shared between the return values of multiple invocations of this
            function. This is because it is employed both to construct the full VAE network, as well as en explicit
            decoder network.
        '''


class LayeredVAE(VariationalAutoencoder):
    ''' A simple wrapper around VariationalAutoencoder that implements build_en/decoder in terms of application of
        a series of layers. This class provides an abstract interface for constructing those layers, and contains
        the necessary logic to ensure weights are re-used when necessary.
    '''

    def build_encoder(self, input_):
        ''' Implement build_encoder '''
        pre_encoder = self._apply_layers(input_, self._pre_encoder_layers)
        mean = self._apply_layers(pre_encoder, self._encoder_mean_layers)
        variance = self._apply_layers(pre_encoder, self._encoder_variance_layers)
        return mean, variance

    def build_decoder(self, input_):
        ''' Implement build_decoder '''
        return self._apply_layers(input_, self._decoder_layers)

    @staticmethod
    def _apply_layers(input_, layers):
        ''' Apply all layers to the input sequentially, and return the result '''
        result = input_
        for layer in layers:
            result = layer(result)
        return result

    @lazy
    def _pre_encoder_layers(self):
        ''' Lazy construction of pre-encoder layers '''
        return self._build_pre_encoder_layers()

    @lazy
    def _encoder_mean_layers(self):
        ''' Lazy construction of encoder mean layers '''
        return self._build_encoder_mean_layers()

    @lazy
    def _encoder_variance_layers(self):
        ''' Lazy construction of encoder variance layers '''
        return self._build_encoder_variance_layers()

    @lazy
    def _decoder_layers(self):
        ''' Lazy construction of decoder layers '''
        return self._build_decoder_layers()

    @abstractmethod
    def _build_pre_encoder_layers(self):
        ''' Build the encoder layers, called exactly once. These layers build up to some point, after which the
            structure for variational mean and variance pathways diverge.
         '''

    @abstractmethod
    def _build_encoder_mean_layers(self):
        ''' Return a list of layers to map from the pre-encoder layer to the variational mean '''

    @abstractmethod
    def _build_encoder_variance_layers(self):
        ''' Return a list of layers to map from the pre-encoder layer to the variational variance '''

    @abstractmethod
    def _build_decoder_layers(self):
        ''' Build the decoder layers, called exactly once. Returns a list of layers. '''



class SimpleDenseVAE(LayeredVAE):
    ''' Example VAE that creates a dense network of depth that can be parameterised in a few ways.
        For more generality, create a subclass of LayeredVAE, or even VariationalAutoencoder.

        Arguments:

            n_layers:          The number of dense layers
            dim_intermediate:  The size of each dense layer
            activation:        Activation for layers, e.g. 'relu' or 'tanh'
            dropout:           None if no dropout is to be used, else float for dropout rate

        Other arguments as for VariationalAutoencoder.
    '''

    def __init__(self, n_layers, dim_intermediate, activation, dropout, cls_vae, dim_latent, optimizer='adam'):
        super().__init__(cls_vae, dim_latent, optimizer)
        self._n_layers = n_layers
        self._dim_intermediate = dim_intermediate
        self._activation = activation
        self._dropout = dropout

    def _build_pre_encoder_layers(self):
        ''' Build the encoder layers, called exactly once. These layers build up to some point, after which the
            structure for variational mean and variance pathways diverge.
         '''
        return self._make_layers()

    def _build_encoder_mean_layers(self):
        ''' Find the mean '''
        return [Dense(self._dim_latent)]

    def _build_encoder_variance_layers(self):
        ''' Find the variance '''
        return [Dense(self._dim_latent)]

    def _build_decoder_layers(self):
        ''' Build the decoder layers, called exactly once. Returns a list of layers. '''
        return self._make_layers()

    def _make_layers(self):
        ''' Both the pre-encoder, and decoder, use the same layer structure -- this implements that '''
        result = []
        for _ in range(self._n_layers):
            result.append(Dense(self._dim_intermediate, activation=self._activation))
            if self._dropout is not None:
                result.append(Dropout(self._dropout))
        return result
