from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import optimizers
from tensorflow import keras


# This file constitutes a library of functions for reformatting and plotting etc.

# Modify the input shape by adding a channels-dimension in the end
def modify_input_shape(input):
	if len(input.shape) == 3:
		return input.reshape(input.shape + (1,))
	return input


# Calculate the accuracy of a classifier given examples and targets
def calc_accuracy_classifier(classifier, x_data, y_data):
	cat_acc = categorical_accuracy(classifier.model.predict(modify_input_shape(x_data)), y_data)
	acc = (sum(cat_acc) / len(cat_acc)) * 100
	print("Accuracy: ", acc.numpy(), "%")


def tsne_plot(encoder, data, title="",n_cases=250):
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
	sns.FacetGrid(reduced_df, hue='label', height=6).map(plt.scatter, 'X', 'Y').add_legend()
	plt.title(title)
	plt.show()


def normalize_image_data(images):
	images = images.astype('float32')
	return images / 255.0


def one_hot_encode(images):
	return np_utils.to_categorical(images)


def one_hot_decode(images):
	return tf.argmax(images, axis=1)

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

def display_images(images,n=16):
	#images = autoencoder.generate(n)
	images = images[:n]
	no_images = images.shape[0]
	channels = images.shape[-1]
	# Do the plotting
	plt.Figure()
	no_rows = np.ceil(np.sqrt(no_images))
	no_cols = np.ceil(no_images / no_rows)
	for img_idx in range(no_images):
		plt.subplot(no_rows, no_cols, img_idx + 1)
		if channels == 1:
			plt.imshow(images[img_idx, :, :, 0], cmap="binary")
		else:
			plt.imshow(images[img_idx, :, :, :].astype(np.float))
		plt.xticks([])
		plt.yticks([])

	# Show the thing ...
	plt.show()

def reshape_img(img):
	if(img.shape[-1]==1):
		return img.reshape(img.shape[:-1])
	return img


def set_optimizer(optimizer, learning_rate):
	if optimizer == "adadelta":
		optimizer = optimizers.Adadelta(learning_rate=learning_rate)
	elif optimizer == "adagrad":
		optimizer = optimizers.Adagrad(learning_rate=learning_rate)
	elif optimizer == "adam":
		optimizer = optimizers.Adam(learning_rate=learning_rate)
	elif optimizer == "adamax":
		optimizer = optimizers.Adamax(learning_rate=learning_rate)
	elif optimizer == "nadam":
		optimizer = optimizers.Nadam(learning_rate=learning_rate)
	elif optimizer == "rmsprop":
		optimizer = optimizers.RMSprop(learning_rate=learning_rate)
	elif optimizer == "sgd":
		optimizer = optimizers.SGD(learning_rate=learning_rate)

	return optimizer

