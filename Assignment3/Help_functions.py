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
from Assignment3.Autoencoder import Autoencoder


# This file constitutes a library of functions for reformatting and plotting etc.

# Calculate the accuracy of a classifier given examples and targets
def calc_accuracy_classifier(classifier, x_data, y_data):
	cat_acc = categorical_accuracy(classifier.model.predict(x_data, y_data))
	acc = (sum(cat_acc) / len(cat_acc)) * 100
	print("Accuracy: ", acc.numpy(), "%")


def tsne_plot(autoencoder, title="",n_cases=250):
	encoder = autoencoder.encoder
	# encoder and data must be objects/instances
	latent_vectors = encoder.model.predict(autoencoder.x_train[:n_cases])
	labels = autoencoder.y_train[:n_cases]

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
	x_train, decoded_imgs = autoencoder.get_data_predictions(n)
	y_train = autoencoder.y_train[:n]
	if n > 8:
		plt.figure(figsize=(20, 6))
	else:
		plt.figure(figsize=(20, 10))
	plt.suptitle("Autoencoder reconstructions", fontsize=40)
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(reshape_img(x_train[i]))
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

def display_images(images,labels=None, n=16, title=None):
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
		if labels is not None:
			plt.title(f"Class is {str(labels[img_idx]).zfill(channels)}")

	plt.suptitle(title)
	# Show the thing ...
	plt.show()

def get_most_anomalous_images(data, autoencoder, n=16):
	df_loss = pd.DataFrame(columns=['index', 'loss'])
	i = 0
	for img in data:
		img = np.expand_dims(img, axis=0)
		loss, acc = autoencoder.model.evaluate(img, img, verbose=0)
		df_loss = df_loss.append({'index': i, 'loss': loss}, ignore_index=True)
		i += 1

	df_loss = df_loss.sort_values(['loss'], ascending=False)
	df_loss = df_loss.astype({'index': 'int32'})
	idx = df_loss['index'].values[:n]
	return np.array([data[i] for i in idx])


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

