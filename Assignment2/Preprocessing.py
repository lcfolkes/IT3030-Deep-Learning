##balanced sample of cases
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils


class Data:

	# Initializer / Instance Attributes
	def __init__(self, dataset):
		if (dataset == 'mnist'):
			self.dataset = keras.datasets.mnist
		elif (dataset == 'fashion_mnist'):
			self.dataset = keras.datasets.fashion_mnist
		elif (dataset == 'cifar10'):
			self.dataset = keras.datasets.cifar10

		elif (dataset == 'cifar100'):
			self.dataset = keras.datasets.cifar100

		#Retrieve data
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset.load_data()

		#Normalize images for better predictions
		self.x_train = self.__normalize_image_data(self.x_train)
		self.x_test = self.__normalize_image_data(self.x_test)

		#one-hot encode
		self.y_train = self.one_hot_encode(self.y_train)
		self.y_test = self.one_hot_encode(self.y_test)


	def __normalize_image_data(self, images):
		images = images.astype('float32')
		return images / 255.0


	def one_hot_encode(self,images):
		return np_utils.to_categorical(images)

	def one_hot_decode(self,images):
		return np.argmax(images, axis=1)

	#def __unpickle(file):
	#		import _pickle as
	#		with open(file, 'rb') as fo:
	#			dict = _pickle.load(fo)
	#		return dict

	def get_data_description(self):
		print("x_train: {0} {1}\ny_train: {2} {3}\nx_test: {4} {5}\ny_train: {6} {7}".format(
			self.x_train.size, self.x_train.shape, self.y_train.size, self.y_train.shape,
			self.x_test.size, self.x_test.shape, self.y_test.size, self.y_test.shape))

	def print_image_example(self):
		plt.imshow(self.x_train[0])
		plt.show()
		#print(self.training_labels[0])
		#print(self.training_images[0])


