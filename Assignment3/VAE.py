from keras import backend as K
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.utils import plot_model
from keras.losses import binary_crossentropy

class VAE:
    def __init__(self, gen, learning_rate=0.005, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=15, latent_dim=8):
        self.x_train, self.y_train = gen.get_full_data_set(training=True)
        self.x_test, self.y_test = gen.get_full_data_set(training=False)

        self.encoding_dim = latent_dim
        data_shape = self.x_train.shape[1:]

        self.encoder = self.__encoder(input_shape=data_shape, latent_dim=latent_dim)
        self.decoder = self.__decoder(output_shape=data_shape, latent_dim=latent_dim)

        enc_input_layer = self.encoder.get_input_at(0)
        enc_output_layer = self.encoder.get_output_at(-1)
        z_mean, z_log_var = enc_output_layer[0], enc_output_layer[1]
        self.model = Model(inputs=enc_input_layer, outputs=self.decoder(enc_output_layer[2]),name='vae')

        vae_loss = self.__vae_loss(self.model.get_input_at(0), self.model.get_output_at(-1), z_log_var, z_mean)
        self.model.add_loss(vae_loss)
        self.model.compile(optimizer='adam')
        self.model.fit(self.x_train, self.x_train, validation_data=(self.x_test, self.x_test), batch_size=1000)

    # Sample new similar parameters
    def __sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(z_log_var) * epsilon

    def __encoder(self, input_shape, latent_dim):
        # Encoder network, mapping inputs to our latent distribution parameters
        inputs = Input(shape=input_shape)
        x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
		#what to do with sampling?
        z = Lambda(self.__sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, show_shapes=True)
        return encoder

    def __decoder(self, output_shape, latent_dim):
        inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(256, activation='relu')(inputs)
        dense_dim, conv_shape = self.__get_dense_conv_shape()
        x = Dense(dense_dim, activation='relu')(x)
        x = Reshape(conv_shape)(x)
        #x = Dropout(0.25)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        decoded = Conv2DTranspose(output_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=inputs, outputs=decoded, name='decoder')
        decoder.summary()
        plot_model(decoder, show_shapes=True)
        return decoder

    def __vae_loss(self, inputs, outputs, z_log_var, z_mean):
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = reconstruction_loss + kl_loss
        return vae_loss

    def __get_dense_conv_shape(self):
        for l in self.encoder.layers:
            if (len(l.input_shape) > len(l.output_shape)):
                return l.output_shape[1], l.input_shape[1:]
