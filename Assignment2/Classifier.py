from keras import Sequential
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.activations import softmax
from keras.optimizers import adam, SGD, rmsprop
#from optimizers import SGD, RMSprop
from keras import backend as K


class Classifier:

    def __init__(self, encoder, x, y, learning_rate_classifier, loss_function_classifier, optimizer_classifier, epochs_classifier, freeze=False):
        self.encoder = encoder
        self.x = x
        self.y = y
        self.learning_rate = learning_rate_classifier
        self.loss_function = loss_function_classifier
        self.epochs = epochs_classifier
        self.freeze = freeze
        self.optimizer_name = optimizer_classifier
        self.input_shape = self.encoder.encoder.get_layer(index=-1).output.shape[1:]
        self.no_classes = y.shape[1]
        self.model = self.__build_model()
        #self.loss = self.__compute_loss(y, self.loss_function)
        self.loss = categorical_crossentropy(K.constant(y), self.model.layers[-1].output)
        self.optimizer = self.__create_optimizer()
        self.compiled_model = self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.fitted_model = self.model.fit(self.x, self.y, epochs=self.epochs)

    def __build_model(self):
        input = Input(shape=self.input_shape)
        x = Dense(self.no_classes, activation='relu')(input)
        output = Dense(self.no_classes, activation='softmax')(x)
        built = Model(inputs=input, outputs=output)
        return built

    def __compute_loss(self, y, loss_function_classifier):
        if loss_function_classifier == 'categorical_crossentropy':
            loss = categorical_crossentropy(K.constant(y), self.model.layers[-1].output)
            return loss
        else:
            pass

    def __create_optimizer(self):
        if self.optimizer_name == "rmsprop":
            optimizer = rmsprop() #learning_rate=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = SGD() #learning_rate=self.learning_rate)
        return optimizer

   # def __compile_model(self):
    #    compiled_model = self.model.compile(self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #    return compiled_model

   # def __fit_model(self, x, y, epochs):
    #    fitted_model = self.model.fit(x, y, epochs=epochs)
    #    return fitted_model














        #self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, activation='relu', padding='same'))
        #self.model.add(Dropout(0.2))
        #self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.2))
        #self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        #self.model.add(Dropout(0.2))
        #self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Flatten())
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        #self.model.add(Dropout(0.2))
        # this is decode, where to stop encode?
        #self.model.add(Dense(self.num_classes, activation='softmax'))
