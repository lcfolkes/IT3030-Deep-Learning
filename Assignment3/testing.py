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

def train_ae(mode=DataMode.MONO_FLOAT_COMPLETE, learning_rate=0.001, epochs=15, force_learn=True):
	gen = StackedMNISTData(mode=mode, default_batch_size=2048)
	autoencoder = Autoencoder(gen, learning_rate=learning_rate, epochs=epochs, force_learn=force_learn)
	return gen, autoencoder

def train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False):
	gen = StackedMNISTData(mode=mode, default_batch_size=2048)
	net = VerificationNet(force_learn=force_learn)
	net.train(generator=gen, epochs=5)
	return gen, net

def train_net_ae(mode=DataMode.MONO_FLOAT_COMPLETE, learning_rate=0.001, epochs=15, force_learn_net=False, force_learn_ae=False):
	gen, autoencoder = train_ae(mode=mode, learning_rate=learning_rate, epochs=epochs, force_learn=force_learn_ae)
	net = VerificationNet(force_learn=force_learn_net)
	net.train(generator=gen, epochs=5)
	return gen, net, autoencoder


### AE-BASIC
def ae_results(net, autoencoder, n=16):
	### SHOW RECONSTRUCTION RESULTS
	x, x_pred, labels = Help_functions.get_data_predictions_labels(autoencoder, n=60000)
	cov = net.check_class_coverage(data=x_pred, tolerance=.8)
	pred, acc = net.check_predictability(data=x_pred, correct_labels=labels)

	# Coverage
	print(f"Coverage: {100 * cov:.2f}%")
	# Quality
	print(f"Predictability: {100 * pred:.2f}%")
	# Accuracy
	print(f"Accuracy: {100 * acc:.2f}%")

	Help_functions.display_images(x_pred, labels, n=n)
	Help_functions.display_reconstructions(x, x_pred, n=n)

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

def ae_anom(gen_complete, autoencoder_complete, mode=DataMode.MONO_FLOAT_MISSING, epochs=15, force_learn=False):
	### AE AS AN ANOMALY DETECTOR
	gen_missing, autoencoder_missing = train_ae(mode=mode, learning_rate=0.001, epochs=epochs, force_learn=force_learn)
	x_test_complete, y_test_complete = gen_complete.get_full_data_set(training=False)

	print('\n# Evaluate complete model on complete test data')
	results_complete = autoencoder_complete.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
	print('test loss:', results_complete)

	print('\n# Evaluate missing model on complete test data')
	results_missing = autoencoder_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
	print('test loss', results_missing)

	anomalous_imgs = Help_functions.get_most_anomalous_images(x_test_complete, autoencoder_missing, 16)
	Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs')

if __name__ == "__main__":

	# ##### AE-BASIC #####
	#
	# ### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST
	# print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST ###')
	# gen_standard, net_standard, autoencoder_standard = train_net_ae(mode=DataMode.MONO_FLOAT_COMPLETE,
	# 																learning_rate=0.001,epochs=15, force_learn_ae=False)
	#
	# ### SHOW RECONSTRUCTION RESULTS OF AE-BASIC
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-BASIC ###')
	# ae_results(net=net_standard, autoencoder=autoencoder_standard)
	#
	# ##### AE-GEN #####
	# # Predictability is high because the verification net thinks all values are 1
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STANDARD MNIST DATA ###')
	# ae_gen_results(net=net_standard, autoencoder=autoencoder_standard)
	#
	# ##### AE-ANOM #####
	# print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STANDARD MNIST DATA ###')
	# ae_anom(gen_standard, autoencoder_standard, mode=DataMode.MONO_FLOAT_MISSING, epochs=15, force_learn=False)
	#
	# ##### AE-STACK #####
	# '''Show the results for the AE-GEN and AE-ANOM tasks when learning from stackedMNIST data.
	# Be prepared to discuss how you adapted the model structure when going from one to three color channels.'''
	# print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STACKED MNIST ###')
	# gen_stacked, net_stacked, autoencoder_stacked = train_net_ae(mode=DataMode.COLOR_FLOAT_COMPLETE, epochs=15,
	# 															 force_learn_ae=False)
	#
	# print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STACKED MNIST DATA ###')
	# ae_gen_results(net=net_stacked, autoencoder=autoencoder_stacked)
	#
	# print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STACKED MNIST DATA ###')
	# ae_anom(gen_stacked, autoencoder_stacked, mode=DataMode.COLOR_FLOAT_MISSING, epochs=15, force_learn=False)

	##### VAE #####
	gen, net = train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False)

	vae = VAE(gen, epochs=15)

	### SHOW RECONSTRUCTION RESULTS OF VAE
	print('\n### SHOW RECONSTRUCTION RESULTS OF AE-BASIC ###')
	ae_results(net=net, autoencoder=vae)

#os.system("tensorboard --logdir=logs/scalars")