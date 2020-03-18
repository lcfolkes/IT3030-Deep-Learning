import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from Assignment2 import Help_functions
from Assignment2.Classifier import Classifier
from Assignment2.Preprocessing import Data
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Encoder import Encoder




class Main:

	def __init__(self, dataset='mnist', learning_rate_autoencoder=0.001, learning_rate_classifier=0.001,
				 loss_function_autoencoder = 'binary_crossentropy', loss_function_classifier='categorical_crossentropy',
				 optimizer_autoencoder = 'adadelta', optimizer_classifier = 'adam', size_latent_vector='64',
				 epochs_autoencoder=20, epochs_classifier=20, dss_split = 0.3,d2_train_frac=0.3,d2_val_frac=0.3, freeze=False,
				 no_reconstructions=16, tSNE=False):

		#D1 = unlabeled, D2 = labeled
		data = Data(dataset, dss_split, d2_train_frac,d2_val_frac)
		encoder = Encoder(data.d1_x, size_latent_vector)
		autoencoder = Autoencoder(data.d1_x, size_latent_vector, learning_rate_autoencoder, loss_function_autoencoder, optimizer_autoencoder,
								  epochs_autoencoder)
		Help_functions.display_reconstructions(autoencoder)
		classifier_pretrained = Classifier(autoencoder.encoder, learning_rate_classifier, loss_function_classifier,
										   optimizer_classifier, epochs_classifier, size_latent_vector,freeze)

		classifier = Classifier(encoder, learning_rate_classifier, loss_function_classifier, optimizer_classifier,
								epochs_classifier, size_latent_vector, freeze)



