from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from Assignment2 import Encoder
from Assignment2.Help_functions import modify_input_shape


class Classifier2:

    def __init__(self, x, y, encoder, optimizer="adam", no_epochs=5, batch_size=1000):
        self.x = modify_input_shape(x)
        self.y = y
        self.size_latent_vector = 32
        self.no_classes = y.shape[1]
        self.encoder = encoder
        self.model = self.__build_network(self.no_classes, encoder)
        self.model.compile(optimizer, loss="categorical_crossentropy")
        self.model.fit(self.x, y, epochs=no_epochs, batch_size=batch_size)

    def __build_network(self, no_classes, encoder):
        inputs = Input(shape=self.x.shape[1:])
        x = encoder.encoder(inputs)
        x = Dense(no_classes, activation='relu')(x)
        output = Dense(no_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=output)
        return model





