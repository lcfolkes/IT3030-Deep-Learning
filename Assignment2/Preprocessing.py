##balanced sample of cases
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils


class Data:

	# Initializer / Instance Attributes
	def __init__(self, dataset, dss_split=0.3, d2_train_frac=0.3, d2_val_frac=0.3):
		self.dss_split = dss_split
		self.d2_train_frac = d2_train_frac
		self.d2_val_frac = d2_val_frac

		if (dataset == 'mnist'):
			self.dataset = keras.datasets.mnist
		elif (dataset == 'fashion_mnist'):
			self.dataset = keras.datasets.fashion_mnist
		elif (dataset == 'cifar10'):
			self.dataset = keras.datasets.cifar10

		elif (dataset == 'cifar100'):
			self.dataset = keras.datasets.cifar100

		#Retrieve data
		self.d1_x, self.d1_y, self.d2_x_train, self.d2_y_train, self.d2_x_val, self.d2_y_val,\
		self.d2_x_test, self.d2_y_test = self._split_data()


	def _split_data(self):

		#load data
		(x_train, y_train), (x_test, y_test) = self.dataset.load_data()

		# Normalize images for better predictions
		x_train = self.__normalize_image_data(x_train)
		x_test = self.__normalize_image_data(x_test)

		# one-hot encode
		y_train = self.one_hot_encode(y_train)
		y_test = self.one_hot_encode(y_test)

		# concatenate dataset to data and labels
		data = np.concatenate([x_train,x_test])
		labels = np.concatenate([y_train, y_test])
		print(data.shape)


		# Split into D1 and D2
		d1_x, d2_x, d1_y, d2_y = train_test_split(data,labels,test_size=self.dss_split)

		# Split D2 in D2 into training, validation and testing sets
		d2_x_train, d2_x_val, d2_y_train, d2_y_val = train_test_split(d2_x,d2_y,test_size=self.d2_train_frac)
		d2_x_val, d2_x_test, d2_y_val, d2_y_test = train_test_split(d2_x_val,d2_y_val,test_size=self.d2_val_frac)

		return d1_x, d1_y, d2_x_train, d2_y_train, d2_x_val, d2_y_val, d2_x_test, d2_y_test


	def __normalize_image_data(self, images):
		images = images.astype('float32')
		return images / 255.0


	def one_hot_encode(self,images):
		return np_utils.to_categorical(images)

	def one_hot_decode(self,images):
		return np.argmax(images, axis=1)


	def get_data_description(self):
		print("d1_x: {0}\nd1_y: {1}\nd2_x_train: {2}\nd2_y_train: {3}\nd2_x_val: {4}\nd2_y_val: {5}\n"
			  "d2_y_test: {6}\nd2_y_test: {7}".format(self.d1_x.shape, self.d1_y.shape, self.d2_x_train.shape,
			self.d2_y_train.shape, self.d2_x_val.shape, self.d2_y_val.shape, self.d2_x_test.shape, self.d2_y_test.shape))

	#def print_image_example(self):
	#	plt.imshow(self.x_train[0])
	#	plt.show()
	#	#print(self.training_labels[0])
	#	#print(self.training_images[0])


