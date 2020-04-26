from datetime import datetime
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, MaxPooling2D, UpSampling2D, Multiply, Add
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import Help_functions
import numpy as np
import os

import stacked_mnist


class VAE:
    def __init__(self, gen, learning_rate=0.001, optimizer='rmsprop',
                 epochs=30, force_learn=False):
        model_name = 'vae'
        dir_name = os.path.join('./models', model_name)
        os.makedirs(dir_name, exist_ok=True)
        gen_name = gen.get_gen_name()
        self.file_name = os.path.join(dir_name, gen_name + ".h5")

        self.x_train, self.y_train = gen.get_full_data_set(training=True)
        self.x_test, self.y_test = gen.get_full_data_set(training=False)
        # input image dimensions
        self.img_shape = self.x_train.shape[1:]
        self.channels = self.img_shape[-1]

        self.latent_dim = 16
        batch_size = 784

        # Clear previous sessions
        K.clear_session()

        # Encoder
        self.encoder = self.__encoder()
        filename = os.path.join(dir_name, 'encoder_model.png')
        plot_model(self.encoder, to_file=filename, show_shapes=True)

        # Decoder
        self.decoder = self.__decoder()
        filename = os.path.join(dir_name, 'decoder_model.png')
        plot_model(self.decoder, to_file=filename, show_shapes=True)

        # Variational Autoencoder
        encoder_input = self.encoder.get_layer('encoder_input').output
        eps = self.encoder.get_layer('eps').output
        x_pred = self.decoder(self.encoder.get_layer('z').output)

        self.model = Model(inputs=[encoder_input, eps], outputs=x_pred, name=model_name)
        filename = os.path.join(dir_name, 'model.png')
        plot_model(self.model, to_file=filename, show_shapes=True)

        optimizer = Help_functions.set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.nll)

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + model_name + "_" + gen_name + "_" + str(learning_rate) + "-" + \
                 datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        print("Variational Autoencoder")
        if force_learn:
            self.model.fit(self.x_train, self.x_train, shuffle=True, epochs=epochs, batch_size=batch_size,
                           validation_data=(self.x_test, self.x_test), verbose=1)#, callbacks=[tensorboard])
            self.model.save_weights(filepath=self.file_name)
            print("Saved weights to: " + self.file_name)
        else:
            try:
                self.model.load_weights(filepath=self.file_name)
                print("Loaded weights from: " + self.file_name)

            except:
                self.model.fit(self.x_train, self.x_train, shuffle=True, epochs=epochs, batch_size=batch_size,
                               validation_data=(self.x_test, self.x_test), verbose=1)#, callbacks=[tensorboard])
                self.model.save_weights(filepath=self.file_name)
                print("Saved weights to: " + self.file_name)


    def __encoder(self):
        # Encoder network, mapping inputs to our latent distribution parameters
        inputs = Input(shape=self.img_shape, name='encoder_input')
        x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(inputs)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = Flatten()(x)
        h = Dense(128, activation='relu')(x)
        z_mu = Dense(self.latent_dim, name='z_mu')(h)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

        z_sigma = Lambda(lambda t: K.exp(.5 * t), name='z_sigma')(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=1.0,
                                           shape=(K.shape(x)[0], self.latent_dim)), name='eps')
        z_eps = Multiply(name='z_eps')([z_sigma, eps])
        z = Add(name='z')([z_mu, z_eps])

        encoder = Model(inputs=[inputs, eps], outputs=z, name='encoder')
        #encoder.summary()
        return encoder

    def __decoder(self):
        inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(256, activation='relu')(inputs)
        dense_dim, conv_shape = Help_functions.get_dense_conv_shape(self.encoder)
        x = Dense(dense_dim, activation='relu')(x)
        x = Reshape(conv_shape)(x)
        #x = Dropout(0.25)(x)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        #x = UpSampling2D((2, 2))(x)
        decoded = Conv2DTranspose(self.channels, (3, 3), strides=(2, 2), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=inputs, outputs=decoded, name='decoder')
        #decoder.summary()
        return decoder

    def generate(self, n=60000):
        z = np.random.normal(0, 1, (n, self.latent_dim))
        return self.decoder.predict(z, verbose=2)

    def nll(self, x_true, x_pred):
        """ Negative log likelihood (Bernoulli). """
        x_true, x_pred = K.reshape(x_true, (-1, 784 * self.channels)), K.reshape(x_pred, (-1, 784 * self.channels))
        return K.sum(K.binary_crossentropy(x_true, x_pred), axis=-1)


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

if __name__ == "__main__":
    gen_stacked = stacked_mnist.StackedMNISTData(mode=stacked_mnist.DataMode.COLOR_FLOAT_COMPLETE, default_batch_size=2048)
    vae_stacked = VAE(gen_stacked, epochs=5, force_learn=True)