import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from Assignment2 import Help_functions
from Assignment2.Help_functions import get_dataset


class Data:

	# Initializer / Instance Attributes
	def __init__(self, dataset_name, dss_frac=0.3, dss_d1_frac=0.4, d2_train_frac=0.5, d2_val_frac=0.7):
		self.dss_frac = dss_frac
		self.dss_d1_frac = dss_d1_frac
		self.d2_train_frac = d2_train_frac
		self.d2_val_frac = d2_val_frac
		self.dataset_name = dataset_name
		self.dataset = get_dataset(dataset_name)

		# Retrieve data
		self.d1_x, self.d1_y, self.d2_x_train, self.d2_y_train, self.d2_x_val, self.d2_y_val, \
		self.d2_x_test, self.d2_y_test = self.__split_data()

	def __split_data(self):

		# load data keras function load_data()
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

		# split data to make sample for faster training. Cut away (1 - self.dss_frac) as given in __init__.
		# stratify attribute ensures balanced dataset
		data_sample, data_throwaway, labels_sample, labels_throwaway = \
			train_test_split(data, labels, stratify=labels, test_size=1-self.dss_frac)

		# Split into D1 and D2
		d1_x, d2_x, d1_y, d2_y = train_test_split(data_sample, labels_sample, stratify=labels_sample,test_size=1-self.dss_d1_frac)

		# Split D2 in D2 into training, validation and testing sets
		d2_x_train, d2_x_val, d2_y_train, d2_y_val = train_test_split(d2_x, d2_y, stratify=d2_y, test_size=1-self.d2_train_frac)
		d2_x_val, d2_x_test, d2_y_val, d2_y_test = train_test_split(d2_x_val, d2_y_val, stratify=d2_y_val, test_size=1-self.d2_val_frac)

		return d1_x, d1_y, d2_x_train, d2_y_train, d2_x_val, d2_y_val, d2_x_test, d2_y_test

	def describe(self):
		dataset_size = 70000 if (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist') else 60000
		dss_size = dataset_size*self.dss_frac
		d1_size = self.d1_x.shape[0]
		d2_size = self.d2_x_train.shape[0] + self.d2_x_test.shape[0] + self.d2_x_val.shape[0]

		print("Total number of examples in data set {0} is {1}".format(self.dataset_name, round(dataset_size)))
		print("Now using {0}% of total data as data subset(DSS), with size {1}".format(self.dss_frac*100, round(dss_size)))
		print("The split between (simulated) unlabelled data, D1, and labelled data, D2, is {0:.3g}-{1:.3g}, or {2}-{3}".format(
			self.dss_d1_frac, 1-self.dss_d1_frac, d1_size, d2_size))
		print("The split of D2 into training, validation and testing data is {0:.3g}-{1:.3g}-{2:.3g}, or {3}-{4}-{5}\n".format(
			self.d2_train_frac, (1-self.d2_train_frac)*self.d2_val_frac,
			(1-self.d2_train_frac)*(1-self.d2_val_frac),
			self.d2_x_train.shape[0], self.d2_x_val.shape[0], self.d2_x_test.shape[0]))
		print("Shapes of all subsets:")
		print("d1_x: {1}\nd1_y: {1}\nd2_x_train: {2}\nd2_y_train: {3}\nd2_x_val: {4}\nd2_y_val: {5}\n"
			  "d2_y_test: {6}\nd2_y_test: {7}".format(self.d1_x.shape, self.d1_y.shape, self.d2_x_train.shape,
													  self.d2_y_train.shape, self.d2_x_val.shape, self.d2_y_val.shape,
													  self.d2_x_test.shape, self.d2_y_test.shape))