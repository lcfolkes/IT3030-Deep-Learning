import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class Encoder:

	def __init__(self, data, size_latent_vector):
		self.model = self.__build_model(data, size_latent_vector)

	# check add single channel to input_shape
	def __modify_input_shape(self, input):
		if (len(input) == 2):
			return input + (1,)
		return input

	def __build_model(self, data, size_latent_vector):
		input_shape = self.__modify_input_shape((data.shape[1:]))
		#Encode
		self.model = Sequential()
		self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(size_latent_vector, activation='relu'))
		return self.model