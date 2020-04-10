from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from Assignment2 import Help_functions
from datetime import datetime
from keras.models import Model
from keras.layers import Input, Dense, Dropout, UpSampling2D, Reshape, Conv2DTranspose
from keras.callbacks import TensorBoard
from Assignment2 import Help_functions

# This class creates an encoder model

class Encoder:
    def __init__(self, x_train, y_train, size_latent_vector=32):
        self.x_train = Help_functions.modify_input_shape(x_train)
        self.y_train = y_train
        input_shape = self.x_train.shape[1:]
        self.model = self.__encode(input_shape, size_latent_vector)
        self.channels = input_shape[-1]

    def __encode(self, input_shape, size_latent_vector):
        inputs = Input(shape=input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same')(inputs)
        x = MaxPooling2D((2,2),padding='same')(x)
        x = Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same')(x)
        x = MaxPooling2D((2,2),padding='same')(x)
        x = Dropout(0.25)(x)
        print(x.shape)
        x = Flatten()(x)
        print(x.shape)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        encoded = Dense(size_latent_vector, activation='relu')(x)
        encoder = Model(inputs=inputs, outputs=encoded)
        return encoder



