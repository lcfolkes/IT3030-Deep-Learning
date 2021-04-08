from datetime import datetime
from Assignment2_pytorch import Help_functions

# This class combines an encoder model with a decoder model to create an autoencoder model

class Autoencoder:
    def __init__(self, data, encoder, learning_rate=0.01, loss_function='binary_crossentropy', optimizer='adam',
                 epochs=20):

        # Define encoder and decoder
        self.encoder = encoder
        decoder = self.__decode()

        # Define autoencoder
        self.model = Model(inputs=enc_input_layer, outputs=decoder(enc_output_layer))
        self.optimizer = Help_functions.set_optimizer(optimizer, learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=['accuracy'])
        self.x_train = Help_functions.modify_input_shape(data.d1_x)

        # Define Tensorboard for accuracy and loss plots
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

        print("Autoencoder training")
        self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000,
                       validation_data=(self.x_train, self.x_train), callbacks=[tensorboard])

    def __decode(self):
        # Create decoder model
        encoded_output_shape = self.encoder.model.get_output_shape_at(-1)[1:]
        inputs = Input(encoded_output_shape)
        dense_dim, conv_shape = self.__get_dense_conv_shape()
        x = Dense(dense_dim, activation='relu')(inputs)
        x = Reshape(conv_shape)(x)
        x = Dropout(0.25)(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        decoded = Conv2DTranspose(self.encoder.channels, (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=inputs, outputs=decoded)
        return decoder

    def __get_dense_conv_shape(self):
        for l in self.encoder.model.layers:
            if (len(l.input_shape) > len(l.output_shape)):
                return l.output_shape[1], l.input_shape[1:]

    def get_data_predictions(self, n):
        return self.x_train[:n], self.model.predict(self.x_train[:n])