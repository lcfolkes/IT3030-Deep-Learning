from Assignment2 import Help_functions
from Assignment2.Classifier import Classifier
from Assignment2.Encoder import Encoder

# This class creates a supervised learner.
# First, an encoder object is created.
# Then, the encoder object is combined with a classifier head to create a supervised classifier object.
# Additionally, the class creates T-SNE plots for the learner if specified by the user

class SupervisedLearner:

    def __init__(self, data, size_latent_vector=64, learning_rate=0.023,
                 loss_function="categorical_crossentropy", optimizer="adamax",
                 epochs=20, tsneplots_flag=True):

        # Define encoder
        self.encoder = Encoder(data.d2_x_test, size_latent_vector)

        # Plot tsne plot
        if tsneplots_flag:
            Help_functions.tsne_plot(self.encoder, data, "T-SNE plot supervised encoder before classifier training")

        # Define classifier
        self.classifier = Classifier(data, self.encoder, learning_rate=learning_rate,
                                     loss_function=loss_function, optimizer=optimizer, epochs=epochs)

        # Plot tsne plot
        if tsneplots_flag:
            Help_functions.tsne_plot(self.encoder, data, "T-SNE plot supervised encoder after classifier training")

        # Accuracy
        print("Supervised Classifier accuracy on D2 testing data")
        Help_functions.calc_accuracy_classifier(self.classifier, data.d2_x_test, data.d2_y_test)


