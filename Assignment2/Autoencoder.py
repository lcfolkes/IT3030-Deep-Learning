import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose


class Autoencoder:

	def __init__(self, encoder, learning_rate=0.01, loss_function='binary_crossentropy', optimizer='adadelta', epochs='500'):
		self.encoder = encoder.encoder
		self.decoded = self.__decode()
		input_layer = self.encoder.layers[0]
		self.autoencoder = Model(input_layer.output,self.decoded)
		self.autoencoder.summary()
		#self.autoencoder.compile(optimizer=optimizer, loss = loss_function)

		#x_train = encoder.data.d1_x
		#self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=256, shuffle=True,
		#					 validation_data=(x_train, x_train))
		pass

	def __get_flat_conv_shape(self):
		for l in self.encoder.layers:
			if(len(l.input_shape) > len(l.output_shape)):
				return l.output_shape[1], l.input_shape[1:]

	def __decode(self):
	 	# Decode
		#x = Dropout(0.5)(self.encoded)
		encoded = self.encoder.layers[-1]
		x = Dense(128, activation='relu')(encoded.output)
		flat_dim, conv_shape = self.__get_flat_conv_shape()
		x = Dense(flat_dim, activation='relu')(x)
		x = Reshape(conv_shape)(x)
		x = Dropout(0.25)(x)
		x = UpSampling2D((2, 2))(x)
		x = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu')(x)
		x = UpSampling2D((2, 2))(x)
		x = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu')(x)
		decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
		return decoded








