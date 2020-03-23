from Assignment2 import Help_functions
from Assignment2.Classifier import Classifier
from Assignment2.Encoder import Encoder
from Assignment2.Preprocessing import Data


class SupervisedLearner:

    def __init__(self, data, size_latent_vector=64, learning_rate=0.023,
                 loss_function="categorical_crossentropy", optimizer="adamax",
                 epochs=20, tsneplots_flag=True):
        self.data = data
        self.encoder = Encoder(self.data.d2_x_test, size_latent_vector)
        if tsneplots_flag:
            Help_functions.tsne_plot(self.encoder, self.data, "T-SNE plot supervised encoder before classifier training")
        self.classifier = Classifier(self.data, self.encoder, learning_rate=learning_rate,
                                     loss_function=loss_function, optimizer=optimizer, epochs=epochs)
        if tsneplots_flag:
            Help_functions.tsne_plot(self.encoder, self.data, "T-SNE plot supervised encoder after classifier training")
        print("Supervised Classifier accuracy on D2 testing data")
        Help_functions.calc_accuracy_classifier(self.classifier, self.data.d2_x_test, self.data.d2_y_test)


