from Assignment2.Help_functions import compare_accuracies
from Assignment2.Preprocessing import Data
from Assignment2.SemiSupervisedLearner import SemiSupervisedLearner
from Assignment2.SupervisedLearner import SupervisedLearner
import os

# This file is used to run the program

# PARAMETERS

# optimizers: adadelta, adagrad, adam, adamax, nadam, rmsprop, sgd

# Dataset parameters
dataset_name = 'mnist'
fraction_of_data_used = 0.4
fraction_d1 = 0.7
fraction_d2_training = 0.7
fraction_rest_of_d2_validation = 0.7

# General parameters
latent_vector_size = 64
freeze_encoder_flag = False
tsne_plots_flag = True
nr_of_autoencoder_reconstructions = 20

# Autoencoder parameters
autoencoder_learning_rate = 0.01
autoencoder_loss_function = "binary_crossentropy"
autoencoder_optimizer = "adam"
autoencoder_epochs = 10

# Semi-supervised classifier parameters
ss_classifier_learning_rate = 0.023
ss_classifier_loss_function = "categorical_crossentropy"
ss_classifier_optimizer = "adamax"
ss_classifier_epochs = 20

# Supervised classifier parameters
classifier_learning_rate = 0.0023
classifier_loss_function = "categorical_crossentropy"
classifier_optimizer = "adamax"
classifier_epochs = 20

# RUN SYSTEM

# Create and split dataset
data = Data(dataset_name=dataset_name, dss_frac=fraction_of_data_used, dss_d1_frac=fraction_d1,
            d2_train_frac=fraction_d2_training, d2_val_frac=fraction_rest_of_d2_validation)

# Print data summary
data.describe()

# Create and train semi-supervised learner, consisting of autoencoder and classifier with structure [encoder + classifier head]
semi_sup_learner = SemiSupervisedLearner(data, size_latent_vector=latent_vector_size,
                                         learning_rate_autoencoder=autoencoder_learning_rate,
                                         loss_function_autoencoder=autoencoder_loss_function,
                                         optimizer_autoencoder=autoencoder_optimizer,
                                         epochs_autoencoder=autoencoder_epochs,
                                         nr_of_reconstructions_autoencoder=nr_of_autoencoder_reconstructions,
                                         learning_rate_classifier=ss_classifier_learning_rate,
                                         loss_function_classifier=ss_classifier_loss_function,
                                         optimizer_classifier=ss_classifier_optimizer,
                                         epochs_classifier=ss_classifier_epochs, tsneplots_flag=tsne_plots_flag)

# Create and train supervised learner, consisting of classifier with structure [encoder + classifier head]
sup_learner = SupervisedLearner(data, size_latent_vector=latent_vector_size, learning_rate=classifier_learning_rate,
                                loss_function=classifier_loss_function, optimizer=classifier_optimizer,
                                epochs=classifier_epochs, tsneplots_flag=tsne_plots_flag)

compare_accuracies(sup_learner, semi_sup_learner, data)

# SHOW ACCURACY AND LOSS PLOTS

# Open a command window and navigate to the project directory (Assignment2)
# Type: 'tensorboard --logdir=logs/scalars'
os.system("tensorboard --logdir=logs/scalars")
