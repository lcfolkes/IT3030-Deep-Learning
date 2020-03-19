from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose

from Assignment2 import Preprocessing, Help_functions
from Assignment2.Encoder import Encoder
import copy
from time import time
from keras.callbacks import TensorBoard
from time import time



class Autoencoder:

	def __init__(self, data, encoder, learning_rate=0.01, loss_function='binary_crossentropy',
				 optimizer='adadelta', epochs=20,
				 encoded_layer=None):
		#Encoder(x, size_latent_vector).model
		self.encoder = encoder
		# self.e.summary()

		decoder = Decoder(self.encoder).model
		#decoder.summary()
		self.encoder = encoder
		# Decoder(self.encoder)
		decoder = self.__decode()
		enc_input_layer = encoder.model.get_input_at(0)
		enc_output_layer = encoder.model.get_output_at(-1)
		self.model = Model(inputs=enc_input_layer, outputs=decoder(enc_output_layer))
		self.model.compile(optimizer=optimizer, loss=loss_function)
		self.x_train = Help_functions.modify_input_shape(data.d1_x)

		# Define Tensorboard
		logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

		self.model.fit(self.x_train, self.x_train, epochs=epochs, batch_size=1000,
					   validation_data=(self.x_train, self.x_train),callbacks=[tensorboard])

	def __decode(self):
		# Decode
		encoded_output_shape = self.encoder.model.get_output_shape_at(-1)[1:]
		inputs = Input(encoded_output_shape)
		#x = Dense(128, activation='relu')(inputs)
		dense_dim, conv_shape = self.__get_dense_conv_shape()
		x = Dense(dense_dim, activation='relu')(inputs)
		x = Reshape(conv_shape)(x)
		x = Dropout(0.25)(x)
		x = UpSampling2D((2,2))(x)
		x = Conv2DTranspose(8, kernel_size=(3, 3), activation='relu',padding='same')(x)
		x = Conv2DTranspose(16, kernel_size=(3, 3), activation='relu',padding='same')(x)
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
