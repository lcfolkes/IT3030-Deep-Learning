from Assignment2.Classifier import Classifier
from Assignment2.Encoder import Encoder
from Assignment2.Help_functions import calc_accuracy_classifier
from Assignment2.Preprocessing import Data
from Assignment2.Autoencoder import Autoencoder

data = Data('mnist')
encoder_semi_supervised = Encoder(data.d2_x_train, 64)
autoencoder = Autoencoder(data.d1_x, encoder_semi_supervised.model)
classifier_semi_supervised = Classifier(data.d2_x_train, data.d2_y_train, autoencoder.encoder)

encoder_clean = Encoder(data.d2_x_train, 64)
classifier_supervised = Classifier(data.d2_x_train, data.d2_y_train, encoder_clean.model)

print("Semi_supervised: ")
result_with_auto = calc_accuracy_classifier(classifier_semi_supervised, data.d2_x_test, data.d2_y_test)

print("Supervised: ")
result_semi_supervised = calc_accuracy_classifier(classifier_supervised, data.d2_x_test, data.d2_y_test)
