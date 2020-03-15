import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose

from Assignment2 import Preprocessing
from Assignment2.Decoder import Decoder
from Assignment2.Encoder import Encoder


class Autoencoder:

	def __init__(self, data, size_latent_vector, learning_rate=0.01, loss_function='binary_crossentropy',
				 optimizer='adadelta', epochs=20,
				 encoded_layer=None):
		self.e = Encoder(data, size_latent_vector).encoder
		# self.e.summary()

		self.d = Decoder(self.e).decoder
		# self.d.summary()

		enc_input_layer = self.e.get_input_at(0)
		enc_output_layer = self.e.get_output_at(-1)
		self.autoencoder = Model(enc_input_layer, self.d(enc_output_layer))
		# self.autoencoder.summary()
		self.autoencoder.compile(optimizer=optimizer, loss=loss_function)

		self.x_train = np.expand_dims(data, axis=len(data.shape))[:10000]
		# x_train1000 = x_train[:1000]

		self.autoencoder.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000, shuffle=True,
							 validation_data=(self.x_train, self.x_train))

	def get_data_predictions(self, n):
		return self.x_train[:n], self.autoencoder.predict(self.x_train[:n])
