import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose

# check add single channel to input_shape
def modify_input_shape(input_shape):
	if len(input_shape) == 2:
		return input_shape + (1,)
	return input_shape

class Encoder:
	def __init__(self, data, size_latent_vector):
		self.input_shape = modify_input_shape((data.shape[1:]))
		self.encoder = self.__encode(size_latent_vector)

	def __encode(self, size_latent_vector):
		# Encode
		input = Input(shape=self.input_shape)
		x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input)
		#x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
		#x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.5)(x)
		encoded = Dense(size_latent_vector, activation='relu')(x)
		encoder = Model(inputs=input, outputs=encoded)
		return encoder

