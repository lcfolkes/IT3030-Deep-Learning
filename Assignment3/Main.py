import Help_functions
from Autoencoder import Autoencoder
from VAE import VAE
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
import Help_functions
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
import pandas as pd
from dcgan import DCGAN


### Functions


def train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False):
    gen = StackedMNISTData(mode=mode, default_batch_size=2048)
    net = VerificationNet(force_learn=force_learn)
    net.train(generator=gen, epochs=5)
    return gen, net

### AE-BASIC
def model_results(net, model, n=16):
    ### SHOW RECONSTRUCTION RESULTS
    x, x_pred, labels = Help_functions.get_data_predictions_labels(model, n=60000)
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

### AE-GEN/VAE-GEN/
def model_gen_results(net, model, n=16):
    ### AE AS A GENERATIVE MODEL
    generated_img = model.generate(n=60000)
    cov_generated = net.check_class_coverage(data=generated_img, tolerance=.8)
    ## why is pred so high??
    pred, acc = net.check_predictability(data=generated_img)

    # Coverage
    print(f"Coverage generated images: {100 * cov_generated:.2f}%")
    # Quality
    print(f"Predictability: {100 * pred:.2f}%")
    Help_functions.display_images(generated_img, n=n, title='Generated imgs')

def ae_anom(model_complete, model_missing):
    ### AE AS AN ANOMALY DETECTOR
    x_test_complete, y_test_complete = model_complete.x_test, model_complete.y_test

    print('\n# Evaluate complete model on complete test data')
    results_complete = model_complete.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
    print('test loss:', results_complete)
    anomalous_imgs = Help_functions.get_most_anomalous_images(x_test_complete, model=model_complete, n=16)
    Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs on complete model')

    print('\n# Evaluate missing model on complete test data')
    results_missing = model_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=0)
    print('test loss', results_missing)
    anomalous_imgs = Help_functions.get_most_anomalous_images(x_test_complete, model=model_missing, n=16)
    Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs on missing model')

  

if __name__ == "__main__":

    # ##### AE-BASIC #####
    #
    # ### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST
    # print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST ###')
    # gen_standard, net_standard = train_gen_net(mode=DataMode.MONO_BINARY_COMPLETE, force_learn=False)
    # autoencoder_standard = Autoencoder(gen_standard, force_learn=False)
    #
    # ### SHOW RECONSTRUCTION RESULTS OF AE-BASIC
    # print('\n### SHOW RECONSTRUCTION RESULTS OF AE-BASIC ###')
    # model_results(net=net_standard, model=autoencoder_standard)
    #
    # ##### AE-GEN #####
    # # Predictability is high because the verification net thinks all values are 1
    # print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STANDARD MNIST DATA ###')
    # model_gen_results(net=net_standard, model=autoencoder_standard)
    #
    # ##### AE-ANOM #####
    # print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STANDARD MNIST DATA ###')
    # gen_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    # autoencoder_missing = Autoencoder(gen_missing, force_learn=False)
    # ae_anom(model_complete=autoencoder_standard, model_missing=autoencoder_missing)
    #
    #
    # ##### AE-STACK #####
    # '''Show the results for the AE-GEN and AE-ANOM tasks when learning from stackedMNIST data.
    # Be prepared to discuss how you adapted the model structure when going from one to three color channels.'''
    # print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STACKED MNIST ###')
    # gen_stacked, net_stacked = train_gen_net(mode=DataMode.COLOR_FLOAT_COMPLETE, force_learn=False)
    # autoencoder_stacked_standard = Autoencoder(gen_stacked, force_learn=False)
    #
    # print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STACKED MNIST DATA ###')
    # model_gen_results(net=net_stacked, model=autoencoder_stacked_standard)
    #
    # print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STACKED MNIST DATA ###')
    # gen_stacked_missing = StackedMNISTData(mode=DataMode.COLOR_FLOAT_MISSING, default_batch_size=2048)
    # autoencoder_stacked_missing = Autoencoder(gen_stacked_missing, force_learn=False)
    #
    # ##### VAE-BASIC #####
    # ### TRAIN VERIFICATION NET AND VAE ON STANDARD MNIST
    #
    # print('### TRAIN VERIFICATION NET AND AUTOENCODER ON STANDARD MNIST ###')
    # gen_standard, net_standard = train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False)
    # vae_standard = VAE(gen_standard, force_learn=False)
    #
    # x_test_complete, y_test_complete = vae_standard.x_test, vae_standard.y_test
    #
    # ### SHOW RECONSTRUCTION RESULTS OF VAE
    # print('\n### SHOW RECONSTRUCTION RESULTS OF VAE-BASIC ###')
    # model_results(net=net_standard, model=vae_standard)
    #
    # ##### VAE-GEN #####
    # #Predictability is high because the verification net thinks all values are 1
    # print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STANDARD MNIST DATA ###')
    # model_gen_results(net=net_standard, model=vae_standard)
    #
    #
    # #### VAE-ANOM #####
    # ## Very important that mode.evaluate is called before a new model is instantiated
    #
    # print('\n### SHOW RESULTS FOR THE AE AS AN ANOMALY DETECTOR ON STANDARD MNIST DATA ###')
    # print('\n# Evaluate complete model on complete test data')
    # gen_standard = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=2048)
    # vae_standard = VAE(gen_standard, force_learn=False)
    # x_test_complete, y_test_complete = vae_standard.x_test, vae_standard.y_test
    # results_complete = vae_standard.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=1)
    # print('test loss:', results_complete)
    #
    # print('\n# Evaluate missing model on complete test data')
    # gen_missing = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=2048)
    # vae_missing = VAE(gen_missing, force_learn=False)
    # results_missing = vae_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=1024, verbose=1)
    # print('test loss', results_missing)
    #
    # # anomalous_imgs = Help_functions.get_most_anomalous_images(x_test_complete, model=vae_missing, n=16)
    # anomalous_imgs = Help_functions.vae_get_anomalous(data=x_test_complete, model=vae_missing, n=16)
    # Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs')
    #
    # ##### VAE-STACK #####
    #
    # print('### TRAIN VERIFICATION NET AND VAE ON STACKED MNIST ###')
    # gen_stacked, net_stacked = train_gen_net(mode=DataMode.COLOR_FLOAT_COMPLETE, force_learn=False)
    # vae_stacked = VAE(gen_stacked, force_learn=False)
    #
    # print('\n### SHOW RECONSTRUCTION RESULTS OF AE-GEN ON STACKED MNIST DATA ###')
    # model_gen_results(net=net_stacked, model=vae_stacked)
    #
    # print('\n### SHOW RESULTS FOR THE VAE AS AN ANOMALY DETECTOR ON STACKED MNIST DATA ###')
    #
    # print('\n# Evaluate complete model on complete test data')
    # x_test_complete, y_test_complete = vae_stacked.x_test, vae_stacked.y_test
    # results_complete = vae_stacked.model.evaluate(x_test_complete, x_test_complete, batch_size=784, verbose=1)
    # print('test loss:', results_complete)
    #
    # print('\n# Evaluate missing model on complete test data')
    # gen_stacked_missing = StackedMNISTData(mode=DataMode.COLOR_FLOAT_MISSING, default_batch_size=2048)
    # vae_stacked_missing = VAE(gen_stacked_missing, force_learn=False)
    # results_missing = vae_stacked_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=784, verbose=1)
    # print('test loss', results_missing)
    #
    # anomalous_imgs = Help_functions.vae_get_anomalous(data=x_test_complete, model=vae_stacked_missing, n=16)
    # Help_functions.display_images(anomalous_imgs, n=16, title='Anomalous imgs')


    ##### GAN-BASIC #####
    ### TRAIN VERIFICATION NET AND DCGAN ON STANDARD MNIST

    print('### TRAIN VERIFICATION NET AND DCGAN ON STANDARD MNIST ###')
    gen_standard, net_standard = train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False)
    dcgan_standard = DCGAN(gen_standard, force_learn=False)

    print('\n### SHOW RECONSTRUCTION RESULTS OF DCGAN-GEN ON STANDARD MNIST DATA ###')
    model_gen_results(net=net_standard, model=dcgan_standard)

    ##### GAN-STACK #####
    ### TRAIN VERIFICATION NET AND DCGAN ON STACKED MNIST

    print('### TRAIN VERIFICATION NET AND DCGAN ON STACKED MNIST ###')
    gen_stacked, net_stacked = train_gen_net(mode=DataMode.MONO_FLOAT_COMPLETE, force_learn=False)
    dcgan_stacked = DCGAN(gen_stacked, force_learn=False)

    print('\n### SHOW RECONSTRUCTION RESULTS OF DCGAN-GEN ON STANDARD MNIST DATA ###')
    model_gen_results(net=net_stacked, model=dcgan_stacked)

#os.system("tensorboard --logdir=logs/scalars")