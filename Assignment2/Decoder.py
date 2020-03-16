import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Conv2DTranspose, UpSampling2D


# get transition shapes dense to conv2d
def get_dense_conv_shape(encoder):
	for l in encoder.layers:
		if (len(l.input_shape) > len(l.output_shape)):
			return l.output_shape[1], l.input_shape[1:]


class Decoder:
	def __init__(self, encoder):
		self.model = self.__decode(encoder)

	def __decode(self, encoder):
		# Decode
		encoded_output_shape = encoder.get_output_shape_at(-1)[1:]
		inputs = Input(encoded_output_shape)
		x = Dense(128, activation='relu')(inputs)
		dense_dim, conv_shape = get_dense_conv_shape(encoder)
		x = Dense(dense_dim, activation='relu')(x)
		x = Reshape(conv_shape)(x)
		x = Dropout(0.25)(x)
		# x = UpSampling2D((2, 2))(x)
		x = Conv2DTranspose(32, kernel_size=(3, 3), activation='relu')(x)
		#x = UpSampling2D((2, 2))(x)
		x = Conv2DTranspose(64, kernel_size=(3, 3), activation='relu')(x)
		decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
		decoder = Model(inputs=inputs, outputs=decoded)
		return decoder
