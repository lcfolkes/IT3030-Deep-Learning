import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose

from Assignment2 import Preprocessing
from Assignment2.Decoder import Decoder


class Autoencoder:

	def __init__(self, encoder, learning_rate=0.01, loss_function='binary_crossentropy', optimizer='adadelta', epochs=50,
				 encoded_layer=None):
		self.e = encoder.encoder
		self.e.summary()

		self.d = Decoder(self.e).decoder
		self.d.summary()

		enc_input_layer = self.e.get_input_at(0)
		enc_output_layer = self.e.get_output_at(-1)
		self.a = Model(enc_input_layer, self.d(enc_output_layer))
		self.a.summary()
		#self.autoencoder.summary()
		#self.a.compile(optimizer=optimizer, loss = loss_function)

		#x_train = np.expand_dims(encoder.data, axis=len(encoder.data.shape))
		#print(x_train.shape)
		#self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=128, shuffle=True,
		#					 validation_data=(x_train, x_train))
		pass







