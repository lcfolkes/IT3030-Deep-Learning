from datetime import datetime
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard
from Assignment2_keras.Help_functions import modify_input_shape, set_optimizer

# This class combines an encoder model with a classifier head model to create a classifier model

class Classifier:
    def __init__(self, data, encoder, learning_rate=0.01, loss_function="categorical_crossentropy",
                 optimizer="adam", epochs=20, freeze=False):
        # Format data
        x_train = modify_input_shape(data.d2_x_train)
        x_val = modify_input_shape(data.d2_x_val)
        y_train = data.d2_y_train
        y_val = data.d2_y_val

        self.no_classes = y_train.shape[1]

        # Define and freeze layers of encoder
        self.encoder = encoder

        self.freeze_weights_of_encoder(freeze)

        # Define classifier model for accuracy and loss plots
        classifier_head = self.__classifier_head()
        enc_input_layer = self.encoder.model.get_input_at(0)
        enc_output_layer = self.encoder.model.get_output_at(-1)
        self.model = Model(inputs=enc_input_layer, outputs=classifier_head(enc_output_layer))
        self.optimizer = set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=['accuracy'])

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        # train in for-loop to get accuracy during training
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=500, validation_data=(x_val, y_val), callbacks=[tensorboard])

    def __classifier_head(self):
        # Create classifier head
        encoded_output_shape = self.encoder.model.get_output_shape_at(-1)[1:]
        inputs = Input(encoded_output_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(self.no_classes, activation='relu')(x)
        classified = Dense(self.no_classes, activation='softmax')(x)
        classifier = Model(inputs=inputs, outputs=classified)
        return classifier

    def freeze_weights_of_encoder(self, freeze):
        # Loop through layers of encoder and freeze weights
        if freeze:
            for l in self.encoder.model.layers:
                l.trainable=False

