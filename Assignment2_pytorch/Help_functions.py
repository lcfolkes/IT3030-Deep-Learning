from torchvision import transforms, datasets
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import optim



# This file constitutes a library of functions for reformatting and plotting etc.

# Modify the input shape by adding a channels-dimension in the end
def modify_input_shape(input):
	if len(input.shape) == 3:
		return input.reshape(input.shape + (1,))
	return input


# Calculate the accuracy of a classifier given examples and targets

def calc_accuracy_classifier(classifier, x_data, y_data):
	y_true = y_data
	y_pred = classifier.model.predict(modify_input_shape(x_data))
	cat_acc = categorical_accuracy(y_true, y_pred)
	acc = (sum(cat_acc) / len(cat_acc)) * 100
	print("Accuracy: ", acc.numpy(), "%")

def categorical_accuracy(y_true, y_pred):
	return torch.mean((y_true == torch.argmax(y_pred, dim=1)).float()).item()

def tsne_plot(encoder, data, title="", n_cases=250):
	# encoder and data must be objects/instances
	latent_vectors = encoder.model.predict(modify_input_shape(data.d1_x[:n_cases]))
	labels = one_hot_decode(data.d1_y[:n_cases])

	tsne_model = TSNE(n_components=2, random_state=0)
	reduced_data = tsne_model.fit_transform(latent_vectors)

	# creating a new data frame which help us in plotting the result data
	reduced_df = np.vstack((reduced_data.T, labels)).T
	reduced_df = pd.DataFrame(data=reduced_df, columns=('X', 'Y', 'label'))
	reduced_df.label = reduced_df.label.astype(np.int)

	# Plotting the result of tsne

	g = sns.FacetGrid(reduced_df, hue='label', height=6).map(plt.scatter, 'X', 'Y').add_legend()
	g.fig.subplots_adjust(top=.95)
	g.ax.set_title(title)
	plt.show()


def normalize_image_data(images):
	images = images.astype('float32')
	return images / 255.0


def one_hot_encode(images: object, num_classes: object = None, dtype: object = 'float32') -> object:
	#return np_utils.to_categorical(images)
	y = np.array(images, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical

def one_hot_decode(one_hot):
	return torch.argmax(one_hot, dim=1)


def display_reconstructions(autoencoder,n=16):
	x_test, decoded_imgs = autoencoder.get_data_predictions(n)
	if n > 8:
		plt.figure(figsize=(20, 6))
	else:
		plt.figure(figsize=(20, 10))
	plt.suptitle("Autoencoder reconstructions", fontsize=40)
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(reshape_img(x_test[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(reshape_img(decoded_imgs[i]))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def reshape_img(img):
	if(img.shape[-1]==1):
		return img.reshape(img.shape[:-1])
	return img


def set_optimizer(optimizer, learning_rate):
	if optimizer == "adadelta":
		optimizer = optim.Adadelta(lr=learning_rate)
	elif optimizer == "adagrad":
		optimizer = optim.Adagrad(lr=learning_rate)
	elif optimizer == "adam":
		optimizer = optim.Adam(lr=learning_rate)
	elif optimizer == "adamax":
		optimizer = optim.Adamax(lr=learning_rate)
	#elif optimizer == "nadam":
	#	optimizer = optim.Nadam(lr=learning_rate)
	elif optimizer == "rmsprop":
		optimizer = optim.RMSprop(lr=learning_rate)
	elif optimizer == "sgd":
		optimizer = optim.SGD(lr=learning_rate)

	return optimizer

def get_dataset(dataset_name, train=True):
	if dataset_name == 'mnist':
		dataset = datasets.MNIST("", train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
	elif dataset_name == 'fashion_mnist':
		dataset = datasets.FashionMNIST("", train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
	elif dataset_name == 'cifar10':
		dataset = datasets.CIFAR10("", train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
	elif dataset_name == 'cifar100':
		dataset = datasets.CIFAR100("", train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
	else:
		print("No such dataset")
	return dataset


def compare_accuracies(slearner, ss_learner, data):
	print("\nAccuracy summary")
	print("D2 testing data:")
	print("- Supervised learner:")
	calc_accuracy_classifier(slearner.classifier, data.d2_x_test, data.d2_y_test)
	print("- Semi-supervised learner:")
	calc_accuracy_classifier(ss_learner.classifier, data.d2_x_test, data.d2_y_test)

	print("D1 data:")
	print("- Supervised learner:")
	calc_accuracy_classifier(slearner.classifier, data.d1_x, data.d1_y)
	print("- Semi-supervised learner:")
	calc_accuracy_classifier(ss_learner.classifier, data.d1_x, data.d1_y)
	print("\n")
