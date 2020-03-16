from Assignment2.Autoencoder import Autoencoder
from Assignment2.Classifier import Classifier
from Assignment2.Help_functions import calc_accuracy_classifier
from Assignment2.Encoder import Encoder
from Assignment2.Main import display_reconstructions
from Assignment2.Preprocessing import Data



#a = Data('mnist')
#b = Encoder(a, 32)
#c = Classifier(a.d2_x_train, a.d2_y_train, b, no_epochs=20, batch_size=500)
#d = calc_accuracy_classifier(c, a.d2_x_test, a.d2_y_test)
#e = Autoencoder(a.d2_x_train, 32, epochs=5)

f = display_reconstructions(e)