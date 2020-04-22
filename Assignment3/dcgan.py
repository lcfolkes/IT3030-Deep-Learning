import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, MaxPooling2D, UpSampling2D, Multiply, Add, \
	BatchNormalization, LeakyReLU, Conv2DTranspose
from keras.models import Model

import time

from IPython import display

from Assignment3.stacked_mnist import StackedMNISTData, DataMode

class DCGAN:
	def __init__(self, gen, latent_dim=100):
		self.x_train, self.y_train = gen.get_full_data_set(training=True)
		self.generator = self.__generator(latent_dim)
		noise = np.random.normal(size=(1, latent_dim))
		generated_image = self.generator.predict(noise)
		plt.imshow(generated_image[0, :, :, 0], cmap='gray')
		plt.show()


	def __generator(self, latent_dim):
		inputs = Input(shape=(latent_dim,), name='z')
		x = Dense(7 * 7 * 256, use_bias=False)(inputs)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Reshape((7, 7, 256))(x)

		x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		generated = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
		generator = Model(inputs=inputs, outputs=generated, name='generator')
		generator.summary()
		return generator

if __name__ == "__main__":
	gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=9)
	generated_image = DCGAN(gen)