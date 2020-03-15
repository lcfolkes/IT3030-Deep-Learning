from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from Assignment2 import Encoder

class Classifier2:

    def __init__(self, x, y, optimizer, no_epochs, batch_size, encoder):
        self.x = self.__modify_input_shape(x)
        self.y = y
        self.size_latent_vector = 32
        self.no_classes = y.shape[1]
        self.encoder = encoder
        self.model = self.__build_network(self.no_classes, encoder)
        self.model.compile(optimizer, loss="categorical_crossentropy")
        self.model.fit(self.x, y, epochs=no_epochs, batch_size=batch_size)

    def __modify_input_shape(self, input):
        if (len(input.shape) == 3):
            input = input.reshape(input.shape[0], input.shape[1],input.shape[2],1)
        return input

    def __build_network(self, no_classes, encoder):
        input = Input(shape=self.x.shape[1:])
        x = encoder.encoder(input)
        x = Dense(no_classes, activation='relu')(x)
        output = Dense(no_classes, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)
        return model
