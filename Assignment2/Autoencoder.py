import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose

from Assignment2 import Preprocessing
from Assignment2.Decoder import Decoder
from Assignment2.Encoder import Encoder


class Autoencoder:

	def __init__(self, x_train, encoder, learning_rate=0.01, loss_function='binary_crossentropy',
				 optimizer='adadelta', epochs=20,
				 encoded_layer=None):
		#Encoder(x, size_latent_vector).model
		# self.e.summary()

		decoder = Decoder(encoder).model
		#decoder.summary()

		enc_input_layer = encoder.get_input_at(0)
		enc_output_layer = encoder.get_output_at(-1)
		self.model = Model(inputs=enc_input_layer, outputs=decoder(enc_output_layer))
		#self.model.summary()
		self.model.compile(optimizer=optimizer, loss=loss_function)

		self.x_train = np.expand_dims(x_train, axis=len(x_train.shape))[:10000]
		# x_train1000 = x_train[:1000]

		self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000, shuffle=True,
							 validation_data=(self.x_train, self.x_train))

	def get_data_predictions(self, n):
		return self.x_train[:n], self.model.predict(self.x_train[:n])
