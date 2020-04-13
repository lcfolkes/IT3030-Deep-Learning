from datetime import datetime
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose
from Assignment2 import Help_functions
import numpy as np

# This class creates an encoder model

class Autoencoder_seq:
    def __init__(self, x_train, y_train, learning_rate=0.005, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=15, size_latent_vector=32):
        self.x_train = Help_functions.modify_input_shape(x_train)
        self.y_train = y_train
        input_shape = self.x_train.shape[1:]
        self.model = self.__autoencoder(input_shape, size_latent_vector)
        self.optimizer = Help_functions.set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=loss_function)

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        print("Autoencoder training")
        self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000,
                       validation_data=(self.x_train, self.x_train), callbacks=[tensorboard])

    def __autoencoder(self, input_shape, size_latent_vector):

		#Encoder
        inputs = Input(shape=input_shape)
        conv = Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same')(inputs)
        conv = MaxPooling2D((2,2),padding='same')(conv)
        conv = Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same')(conv)
        conv = MaxPooling2D((2,2),padding='same')(conv)
        conv = Dropout(0.25)(conv)
        flatten_shape = np.prod(conv.shape[1:])
        flat = Reshape((flatten_shape,))(conv)
        #flat = Flatten()(conv)
        dense = Dense(128, activation='relu')(flat)
        dense = Dropout(0.5)(dense)
        encoded = Dense(size_latent_vector, activation='relu')(dense)

		#Decoder
        dense = Dense(128, activation='relu')(dense)
        flat = Dense(flat.shape[-1], activation='relu')(encoded)
        conv = Reshape(conv.shape[1:])(flat)
        conv = Dropout(0.25)(conv)
        conv = UpSampling2D((2, 2))(conv)
        conv = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', padding='same')(conv)
        conv = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', padding='same')(conv)
        conv = UpSampling2D((2, 2))(conv)
        decoded = Conv2DTranspose(inputs.shape[-1], (3, 3), activation='sigmoid', padding='same')(conv)

        #Autoencoder
        autoencoder = Model(inputs=inputs, outputs=decoded)
        return autoencoder

    def get_data_predictions(self, n):
        return self.x_train[:n], self.model.predict(self.x_train[:n])

    def get_data_predictions_labels(self, n):
        return self.model.predict(self.x_train[:n]), self.y_train[:n]


