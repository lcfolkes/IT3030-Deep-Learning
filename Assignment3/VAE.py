import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, MaxPooling2D, UpSampling2D, Multiply, Add
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import os

from Assignment3 import Help_functions


class VAE:
    def __init__(self, gen, learning_rate=0.005, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=15, latent_dim=8, force_learn=False):

        dir_name = './models/vae'
        os.makedirs(dir_name, exist_ok=True)
        self.file_name = os.path.join(dir_name, gen.get_gen_name() + ".h5")

        self.x_train, self.y_train = gen.get_full_data_set(training=True)
        self.x_test, self.y_test = gen.get_full_data_set(training=False)

        self.encoding_dim = latent_dim
        # input image dimensions
        img_shape = self.x_train.shape[1:]

        # Create folder to store plotted models
        model_name = 'vae'
        os.makedirs(model_name, exist_ok=True)

        # Encoder
        self.encoder = self.__encoder(input_shape=img_shape, latent_dim=latent_dim)
        filename = os.path.join(model_name, 'encoder_model.png')
        plot_model(self.encoder, to_file=filename, show_shapes=True)

        # Decoder
        self.decoder = self.__decoder(output_shape=img_shape, latent_dim=latent_dim)
        filename = os.path.join(model_name, 'decoder_model.png')
        plot_model(self.encoder, to_file=filename, show_shapes=True)

        # Variational Autoencoder
        encoder_input = self.encoder.get_layer('encoder_input').output
        eps = self.encoder.get_layer('eps').output
        x_pred = self.decoder(self.encoder.get_layer('z').output)

        self.model = Model(inputs=[encoder_input, eps], outputs=x_pred, name=model_name)
        filename = os.path.join(model_name, 'model.png')
        plot_model(self.model, to_file=filename, show_shapes=True)

        self.model.compile(optimizer='rmsprop', loss=self.__nll)

        print("Variational Autoencoder")
        if force_learn:
            self.model.fit(self.x_train, self.x_train, shuffle=True, epochs=epochs,
                           validation_data=(self.x_test, self.x_test))
            self.model.save_weights(filepath=self.file_name)
            print("Saved weights to: " + self.file_name)
        else:
            try:
                self.model.load_weights(filepath=self.file_name)
                print("Loaded weights from: " + self.file_name)

            except:
                self.model.fit(self.x_train, self.x_train, shuffle=True, epochs=epochs,
                               validation_data=(self.x_test, self.x_test))
                self.model.save_weights(filepath=self.file_name)
                print("Saved weights to: " + self.file_name)

    def __encoder(self, input_shape, latent_dim):
        # Encoder network, mapping inputs to our latent distribution parameters
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        h = Dense(128, activation='relu')(x)
        z_mu = Dense(latent_dim, name='z_mu')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

        z_sigma = Lambda(lambda t: K.exp(.5 * t), name='z_sigma')(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=1.0,
                                           shape=(K.shape(x)[0], latent_dim)), name='eps')
        z_eps = Multiply(name='z_eps')([z_sigma, eps])
        z = Add(name='z')([z_mu, z_eps])

        encoder = Model(inputs=[inputs, eps], outputs=z, name='encoder')
        encoder.summary()
        return encoder

    def __decoder(self, output_shape, latent_dim):
        inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(256, activation='relu')(inputs)
        dense_dim, conv_shape = Help_functions.get_dense_conv_shape(self.encoder)
        x = Dense(dense_dim, activation='relu')(x)
        x = Reshape(conv_shape)(x)
        #x = Dropout(0.25)(x)
        x = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2DTranspose(output_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=inputs, outputs=decoded, name='decoder')
        decoder.summary()
        return decoder

    def __vae_loss(self, inputs, outputs, z_log_var, z_mean):
        reconstruction_loss = K.binary_crossentropy(inputs, outputs)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = reconstruction_loss + kl_loss
        return vae_loss

    def __nll(self, y_true, y_pred):
        """ Negative log likelihood (Bernoulli). """
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
