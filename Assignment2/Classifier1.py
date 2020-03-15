from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np


class Classifier1:

    def __init__(self, x, y, optimizer, no_epochs, batch_size):
        self.x = self.__modify_input_shape(x)
        self.y = y
        self.size_latent_vector = 32
        self.no_classes = y.shape[1]
        self.model = self.__build_network(self.no_classes)
        self.model.compile(optimizer, loss="categorical_crossentropy")
        self.model.fit(self.x, y, epochs=no_epochs, batch_size=batch_size)

    def __modify_input_shape(self, input):
        if (len(input.shape) == 3):
            input = input.reshape(input.shape[0], input.shape[1],input.shape[2],1)
        return input

    def __build_network(self, no_classes):
        input = Input(shape=self.x.shape[1:])
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.x.shape[1:])(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.size_latent_vector, activation='relu')(x)
        x = Dense(no_classes, activation='relu')(x)
        output = Dense(no_classes, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)
        return model
