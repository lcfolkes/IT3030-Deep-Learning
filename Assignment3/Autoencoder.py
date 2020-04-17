from datetime import datetime
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose
from Assignment3 import Help_functions
import numpy as np

# This class creates an encoder model

class Autoencoder:
    def __init__(self, x_train, y_train, learning_rate=0.005, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=15, size_latent_vector=8):
        self.x_train = x_train
        self.y_train = y_train

        self.encoding_dim = size_latent_vector
        data_shape = self.x_train.shape[1:]

        self.encoder = self.__encoder(input_shape=data_shape, size_latent_vector=size_latent_vector)
        self.decoder = self.__decoder(output_shape=data_shape, size_latent_vector=size_latent_vector)

        enc_input_layer = self.encoder.get_input_at(0)
        enc_output_layer = self.encoder.get_output_at(-1)

        self.model = Model(inputs=enc_input_layer, outputs=self.decoder(enc_output_layer))
        self.optimizer = Help_functions.set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=['accuracy'])

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        print("Autoencoder training")
        self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000,
                       validation_data=(self.x_train, self.x_train), callbacks=[tensorboard])


    def __encoder(self, input_shape, size_latent_vector):
        inputs = Input(shape=input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        #x = Dropout(0.5)(x)
        encoded = Dense(size_latent_vector, activation='relu')(x)
        encoder = Model(inputs=inputs, outputs=encoded)
        # print(encoder.summary())
        return encoder

    def __decoder(self, output_shape, size_latent_vector):
        inputs = Input(shape=(size_latent_vector,))
        dense_dim, conv_shape = self.__get_dense_conv_shape()
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

    # def __autoencoder(self, input_shape, size_latent_vector):
    #
	# 	#Encoder
    #     inputs = Input(shape=input_shape)
    #     conv = Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same')(inputs)
    #     conv = MaxPooling2D((2,2),padding='same')(conv)
    #     conv = Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same')(conv)
    #     conv = MaxPooling2D((2,2),padding='same')(conv)
    #     #conv = Dropout(0.25)(conv)
    #     flatten_shape = np.prod(conv.shape[1:])
    #     flat = Reshape((flatten_shape,))(conv)
    #     #flat = Flatten()(conv)
    #     dense = Dense(128, activation='relu')(flat)
    #     #dense = Dropout(0.5)(dense)
    #     encoded = Dense(size_latent_vector, activation='relu')(dense)
    #
	# 	#Decoder
    #     dense = Dense(128, activation='relu')(dense)
    #     flat = Dense(flat.shape[-1], activation='relu')(encoded)
    #     conv = Reshape(conv.shape[1:])(flat)
    #     #conv = Dropout(0.25)(conv)
    #     conv = UpSampling2D((2, 2))(conv)
    #     conv = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', padding='same')(conv)
    #     conv = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', padding='same')(conv)
    #     conv = UpSampling2D((2, 2))(conv)
    #     decoded = Conv2DTranspose(inputs.shape[-1], (3, 3), activation='sigmoid', padding='same')(conv)
    #
    #     #Autoencoder
    #     autoencoder = Model(inputs=inputs, outputs=decoded)
    #     return autoencoder

    def generate(self, n=60000):
        z = np.random.rand(n, self.encoding_dim)
        return self.decoder.predict(z)


    def __get_dense_conv_shape(self):
        for l in self.encoder.layers:
            if (len(l.input_shape) > len(l.output_shape)):
                return l.output_shape[1], l.input_shape[1:]

    def get_data_predictions(self, n):
        return self.x_train[:n], self.model.predict(self.x_train[:n])

    def get_data_predictions_labels(self,n=None):
        if n is None:
            n = self.x_train.shape[0]
        return self.model.predict(self.x_train[:n]), self.y_train[:n]


