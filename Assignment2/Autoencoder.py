import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D

class Autoencoder:

	def __init__(self, encoder, learning_rate=0.01, loss_function='binary_crossentropy', optimizer='adadelta', epochs='500'):
		self.encoded = encoder.encoder
		self.decoded = self.__decode()
		input_layer = self.encoded.layers[0]
		self.autoencoder = Model(input_layer,self.decoded)

		#self.autoencoder.compile(optimizer=optimizer, loss = loss_function)

		#x_train = encoder.data.d1_x
		#self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=256, shuffle=True,
		#					 validation_data=(x_train, x_train))
		pass

	def __get_conv_shape(self):
		for l in self.encoded.layers:
			if(len(l.input_shape) > len(l.output_shape)):
				print(l.input_shape)
				return l.input_shape

	def __decode(self):
	 	# Decode
		#x = Dropout(0.5)(self.encoded)
		encoded = tf.convert_to_tensor(self.encoded.layers[-1])
		x = Dense(128, activation='relu')(encoded)
		conv_shape = self.__get_conv_shape()
		x = x.reshape(conv_shape)
		x = Dropout(0.25)(x)
		x = UpSampling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
		x = UpSampling2D(pool_size=(2, 2))(x)
		x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
		decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
		return decoded








