import numpy as np
from sklearn.model_selection import train_test_split
from Assignment2_pytorch import Help_functions
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


# This class takes in the specified dataset, formats it and splits it into
# D1 and D2 with its respective train, validate and test splits

class Data:

	# Initializer / Instance Attributes
	def __init__(self, dataset_name, dss_frac=0.3, dss_d1_frac=0.4, d2_train_frac=0.5, d2_val_frac=0.7):
		self.dss_frac = dss_frac
		self.dss_d1_frac = dss_d1_frac
		self.d2_train_frac = d2_train_frac
		self.d2_val_frac = d2_val_frac
		self.dataset_name = dataset_name
		self.data = Help_functions.get_dataset(dataset_name=dataset_name, train=True)
		#self.testing_data = Help_functions.get_dataset(dataset_name=dataset_name, train=False)

		# Retrieve data
		self.d1_x, self.d1_y, self.d2_x_train, self.d2_y_train, self.d2_x_val, self.d2_y_val, \
		self.d2_x_test, self.d2_y_test = self.__split_data()

	def __split_data(self):

		# Create data sample (DSS) by discarding (1-dss_frac)
		dss_idx, trow_away_idx = train_test_split(list(range(len(self.data))), test_size=1-self.dss_frac)
		dss = Subset(self.data, dss_idx)

		# Split DSS into D1 and D2
		d1_idx, d2_idx = train_test_split(list(range(len(dss))), test_size=1-self.dss_d1_frac)
		d1 = DataLoader(Subset(dss, d1_idx), batch_size=10, shuffle=True)
		d2 = Subset(dss, d2_idx)

		# Split D2 into training, validation and testing partitions
		d2_train_idx, d2_val_idx = train_test_split(list(range(len(d2))), test_size=1 - self.d2_train_frac)
		d2_train = DataLoader(Subset(d2, d2_train_idx), batch_size=10, shuffle=True)
		d2_val = Subset(d2, d2_val_idx)
		d2_val_idx, d2_test_idx = train_test_split(list(range(len(d2_val))), test_size=1 - self.d2_val_frac)
		d2_test = DataLoader(Subset(d2_val, d2_test_idx), batch_size=10, shuffle=True)
		d2_val = DataLoader(Subset(d2_val, d2_val_idx), batch_size=10, shuffle=True)

		# Separate input and labels
		(d1_x, d1_y) = next(iter(d1))
		(d2_train_x, d2_train_y) = next(iter(d2_train))
		(d2_val_x, d2_val_y) = next(iter(d2_val))
		(d2_test_x, d2_test_y) = next(iter(d2_test))


		# One-hot encode labels
		d1_y = Help_functions.one_hot_encode(d1_y)
		d2_train_y = Help_functions.one_hot_encode(d2_train_y)
		d2_val_y = Help_functions.one_hot_encode(d2_val_y)
		d2_test_y = Help_functions.one_hot_encode(d2_test_y)

		return d1_x, d1_y, d2_train_x, d2_train_y, d2_val_x, d2_val_y, d2_test_x, d2_test_y

	def describe(self):
		dataset_size = len(self.data)
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
		print("d1_x: {0}\nd1_y: {1}\nd2_x_train: {2}\nd2_y_train: {3}\nd2_x_val: {4}\nd2_y_val: {5}\n"
			  "d2_y_test: {6}\nd2_y_test: {7}".format(self.d1_x.shape, self.d1_y.shape, self.d2_x_train.shape,
													  self.d2_y_train.shape, self.d2_x_val.shape, self.d2_y_val.shape,
													  self.d2_x_test.shape, self.d2_y_test.shape))

		print(self.d1_x.shape)



if __name__ == "__main__":
	# Dataset parameters
	dataset_name = 'mnist'
	dss_frac = 0.4
	dss_d1_frac = 0.6
	d2_train_frac = 0.5
	d2_val_frac = 0.7

	# Create and split dataset
	data = Data(dataset_name=dataset_name, dss_frac=dss_frac, dss_d1_frac=dss_d1_frac,
				d2_train_frac=d2_train_frac, d2_val_frac=d2_val_frac)

	# Print data summary
	data.describe()







