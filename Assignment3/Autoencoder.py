from datetime import datetime
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose
from keras.utils import plot_model

import Help_functions
import numpy as np
import os

# This class creates an encoder model

class Autoencoder:
    def __init__(self, gen, learning_rate=0.001, epochs=15, force_learn=False):

        model_name = 'autoencoder'
        dir_name = os.path.join('./models', model_name)
        os.makedirs(dir_name, exist_ok=True)
        gen_name = gen.get_gen_name()
        file_name = os.path.join(dir_name, gen_name + ".h5")


        self.x_train, self.y_train = gen.get_full_data_set(training=True)
        self.x_test, self.y_test = gen.get_full_data_set(training=False)

        data_shape = self.x_train.shape[1:]
        loss_function = 'binary_crossentropy'
        optimizer = 'adam'
        batch_size = 256
        self.latent_dim = 16

        # Encoder
        self.encoder = self.__encoder(input_shape=data_shape, latent_dim=self.latent_dim)
        filename = os.path.join(dir_name, 'encoder_model.png')
        plot_model(self.encoder, to_file=filename, show_shapes=True)

        # Decoder
        self.decoder = self.__decoder(output_shape=data_shape, latent_dim=self.latent_dim)
        filename = os.path.join(dir_name, 'decoder_model.png')
        plot_model(self.decoder, to_file=filename, show_shapes=True)

        enc_input_layer = self.encoder.get_input_at(0)
        enc_output_layer = self.encoder.get_output_at(-1)

        self.model = Model(inputs=enc_input_layer, outputs=self.decoder(enc_output_layer), name=model_name)
        filename = os.path.join(dir_name, 'model.png')
        plot_model(self.model, to_file=filename, show_shapes=True)

        optimizer = Help_functions.set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function)

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + model_name + "_" + gen_name + "_" + str(learning_rate) + "-" + \
                 datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        print("Autoencoder")
        if force_learn:
            self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(self.x_train, self.x_train), callbacks=[tensorboard])
            self.model.save_weights(filepath=file_name)
            print("Saved weights to: " + file_name)
        else:
            try:
                self.model.load_weights(filepath=file_name)
                print("Loaded weights from: " + file_name)

            except Exception as e:
                print(e)
                self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(self.x_train, self.x_train), callbacks=[tensorboard])
                self.model.save_weights(filepath=file_name)
                print("Saved weights to: " + file_name)

    def __encoder(self, input_shape, latent_dim):
        inputs = Input(shape=input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        #x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        #x = Dropout(0.5)(x)
        encoded = Dense(latent_dim, activation='relu')(x)
        encoder = Model(inputs=inputs, outputs=encoded)
        # print(encoder.summary())
        return encoder

    def __decoder(self, output_shape, latent_dim):
        inputs = Input(shape=(latent_dim,))
        dense_dim, conv_shape = Help_functions.get_dense_conv_shape(self.encoder)
        x = Dense(dense_dim, activation='relu')(inputs)
        x = Reshape(conv_shape)(x)
        #x = Dropout(0.25)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2DTranspose(output_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=inputs, outputs=decoded)
        return decoder

    def generate(self, n=60000):
        z = np.random.rand(n, self.latent_dim)
        return self.decoder.predict(z)

