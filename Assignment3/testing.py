from Assignment3 import Help_functions
from Assignment3.Autoencoder import Autoencoder
from Assignment3.VAE import VAE
from Assignment3.stacked_mnist import StackedMNISTData, DataMode
from Assignment3.verification_net import VerificationNet
import Assignment3.Help_functions
import os
import numpy as np
import tensorflow as tf

### Functions

def train_ae(mode=DataMode.MONO_FLOAT_COMPLETE, learning_rate=0.001, epochs=10, training=True):
	gen = StackedMNISTData(mode=mode, default_batch_size=2048)
	x_train, y_train = gen.get_full_data_set(training=training)
	autoencoder = Autoencoder(x_train, y_train, learning_rate=learning_rate, epochs=epochs)
	return gen, autoencoder

def train_net_ae(mode=DataMode.MONO_FLOAT_COMPLETE, learning_rate=0.001, epochs=10, training=True, force_learn=False):
	gen, autoencoder = train_ae(mode=mode, learning_rate=learning_rate, epochs=epochs, training=training)
	net = VerificationNet(force_learn=force_learn)
	net.train(generator=gen, epochs=5)
	return gen, net, autoencoder


### AE-BASIC
def ae_results(net, autoencoder, n=16):
	### SHOW RECONSTRUCTION RESULTS
	img, labels = autoencoder.get_data_predictions_labels(n=60000)
	cov = net.check_class_coverage(data=img, tolerance=.8)
	pred, acc = net.check_predictability(data=img, correct_labels=labels)

	# Coverage
	print(f"Coverage: {100 * cov:.2f}%")
	# Quality
	print(f"Predictability: {100 * pred:.2f}%")
	# Accuracy
	print(f"Accuracy: {100 * acc:.2f}%")

	Help_functions.display_images(img, labels, n=n)
	Help_functions.display_reconstructions(autoencoder)

### AE-GEN
def ae_gen_results(net, autoencoder, n=16):
	### AE AS A GENERATIVE MODEL
	generated_img = autoencoder.generate(n=60000)
	cov_generated = net.check_class_coverage(data=generated_img, tolerance=.8)
	## why is pred so high??
	pred, acc = net.check_predictability(data=generated_img)

	# Coverage
	print(f"Coverage generated images: {100 * cov_generated:.2f}%")
	# Quality
	print(f"Predictability: {100 * pred:.2f}%")

	Help_functions.display_images(generated_img, n=n, title='Generated imgs')

def ae_anom(gen_complete, autoencoder_complete, mode=DataMode.MONO_FLOAT_MISSING):
	### AE AS AN ANOMALY DETECTOR
	gen_missing = StackedMNISTData(mode=mode, default_batch_size=2048)
	x_train_missing, y_train_missing = gen_missing.get_full_data_set(training=True)
	autoencoder_missing = Autoencoder(x_train_missing, y_train_missing, learning_rate=0.001, epochs=10)

	x_test_complete, y_test_complete = gen_complete.get_full_data_set(training=False)

	print('\n# Evaluate complete model on complete test data')
	results_complete = autoencoder_complete.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
	print('test loss:', results_complete[0])

	print('\n# Evaluate missing model on complete test data')
	results_missing = autoencoder_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
	print('test loss', results_missing[0])

	anomalous_imgs = Help_functions.get_most_anomalous_images(x_test_complete, autoencoder_missing, 16)
	Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs')

if __name__ == "__main__":

	# ##### AE-BASIC #####
	#
	# ### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST
	# print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST ###')
	# gen_standard, net_standard, autoencoder_standard = train_net_ae(mode=DataMode.MONO_FLOAT_COMPLETE, learning_rate=0.001, epochs=10, training=True,
	# 				 force_learn=False)
	#
	# ### SHOW RECONSTRUCTION RESULTS OF AE-BASIC
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-BASIC ###')
	# ae_results(net=net_standard, autoencoder=autoencoder_standard,)
	#
	# ##### AE-GEN #####
	# # Predictability is high because the verification net thinks all values are 1
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STANDARD MNIST DATA ###')
	# ae_gen_results(net=net_standard, autoencoder=autoencoder_standard)
	#
	# ##### AE-ANOM #####
	# print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STANDARD MNIST DATA ###')
	# ae_anom(gen_standard, autoencoder_standard, mode=DataMode.MONO_FLOAT_MISSING)

	# #### AE-STACK #####
	# '''Show the results for the AE-GEN and AE-ANOM tasks when learning from stackedMNIST data.
	# Be prepared to discuss how you adapted the model structure when going from one to three color channels.'''
	# gen_stacked, net_stacked, autoencoder_stacked = train_net_ae(mode=DataMode.COLOR_FLOAT_COMPLETE, force_learn=False)
	#
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STACKED MNIST DATA ###')
	# ae_gen_results(net=net_stacked, autoencoder=autoencoder_stacked)
	#
	# print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STACKED MNIST DATA ###')
	# ae_anom(gen_stacked, autoencoder_stacked, mode=DataMode.COLOR_FLOAT_MISSING)

	##### VAE #####
	gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=2048)
	vae = VAE(gen)

#os.system("tensorboard --logdir=logs/scalars")