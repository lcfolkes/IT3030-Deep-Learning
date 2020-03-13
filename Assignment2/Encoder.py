import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class Encoder:
	def __init__(self, data, size_latent_vector):
		self.input_shape = self.__modify_input_shape((data.shape[1:]))
		self.encoder = self.__build_model(size_latent_vector)
		self.data = data

	# check add single channel to input_shape
	def __modify_input_shape(self, input):
		if (len(input) == 2):
			return input + (1,)
		return input

	def __build_model(self, size_latent_vector):
		# Encode
		visible = Input(shape=self.input_shape)
		conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(visible)
		conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
		dropout1 = Dropout(0.25)(pool1)
		flat = Flatten()(dropout1)
		hidden1 = Dense(128, activation='relu')(flat)
		dropout2 = Dropout(0.5)(hidden1)
		output = Dense(size_latent_vector, activation='relu')(dropout2)
		model = Model(inputs=visible, outputs=output)
		return model
