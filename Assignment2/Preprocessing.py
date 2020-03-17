import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from Assignment2 import Help_functions


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

		# Retrieve data
		self.d1_x, self.d1_y, self.d2_x_train, self.d2_y_train, self.d2_x_val, self.d2_y_val, \
		self.d2_x_test, self.d2_y_test = self.__split_data()

	def __split_data(self):

		# load data
		(x_train, y_train), (x_test, y_test) = self.dataset.load_data()

		# Normalize images for better predictions
		x_train = Help_functions.normalize_image_data(x_train)
		x_test = Help_functions.normalize_image_data(x_test)

		# one-hot encode
		y_train = Help_functions.one_hot_encode(y_train)
		y_test = Help_functions.one_hot_encode(y_test)

		# concatenate dataset to data and labels
		data = np.concatenate([x_train, x_test])
		labels = np.concatenate([y_train, y_test])

		#split data to make sample for faster training. Use only 30% of the data
		#stratify attribute ensures balanced dataset
		data_sample, data_throwaway, labels_sample, labels_throwaway = \
			train_test_split(data, labels, stratify=labels,test_size=0.7)

		# Split into D1 and D2
		d1_x, d2_x, d1_y, d2_y = train_test_split(data_sample, labels_sample, stratify=labels_sample,test_size=self.dss_split)

		# Split D2 in D2 into training, validation and testing sets
		d2_x_train, d2_x_val, d2_y_train, d2_y_val = train_test_split(d2_x, d2_y, stratify=d2_y, test_size=self.d2_train_frac)
		d2_x_val, d2_x_test, d2_y_val, d2_y_test = train_test_split(d2_x_val, d2_y_val, stratify=d2_y_val, test_size=self.d2_val_frac)

		return d1_x, d1_y, d2_x_train, d2_y_train, d2_x_val, d2_y_val, d2_x_test, d2_y_test

	def describe(self):
		print("d1_x: {0}\nd1_y: {1}\nd2_x_train: {2}\nd2_y_train: {3}\nd2_x_val: {4}\nd2_y_val: {5}\n"
			  "d2_y_test: {6}\nd2_y_test: {7}".format(self.d1_x.shape, self.d1_y.shape, self.d2_x_train.shape,
													  self.d2_y_train.shape, self.d2_x_val.shape, self.d2_y_val.shape,
													  self.d2_x_test.shape, self.d2_y_test.shape))
