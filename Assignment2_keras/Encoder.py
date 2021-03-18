from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from Assignment2_keras import Help_functions

# This class creates an encoder model

class Encoder:
    def __init__(self, x_train, size_latent_vector):
        input_shape = Help_functions.modify_input_shape(x_train).shape[1:]
        self.model = self.__encode(input_shape, size_latent_vector)
        self.channels = input_shape[-1]

    def __encode(self, input_shape, size_latent_vector):
        inputs = Input(shape=input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same')(inputs)
        x = MaxPooling2D((2,2),padding='same')(x)
        x = Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same')(x)
        x = MaxPooling2D((2,2),padding='same')(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        encoded = Dense(size_latent_vector, activation='relu')(x)
        encoder = Model(inputs=inputs, outputs=encoded)
        return encoder



