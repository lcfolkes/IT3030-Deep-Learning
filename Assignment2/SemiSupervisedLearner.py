from Assignment2 import Help_functions
from Assignment2.Autoencoder import Autoencoder
from Assignment2.Classifier import Classifier
from Assignment2.Encoder import Encoder


class SemiSupervisedLearner:

    def __init__(self, data, size_latent_vector=64, learning_rate_autoencoder=0.01,
                 loss_function_autoencoder="binary_crossentropy", optimizer_autoencoder="adam",
                 epochs_autoencoder=20, nr_of_reconstructions_autoencoder=16, learning_rate_classifier=0.023,
                 loss_function_classifier="categorical_crossentropy", optimizer_classifier="adamax",
                 epochs_classifier=20, tsneplots_flag=True, freeze_encoder_flag=False):

        # Define encoder
        self.encoder = Encoder(data.d2_x_train, size_latent_vector)

        # Plot tsne plot
        if tsneplots_flag:
            Help_functions.tsne_plot(self.encoder, data, "T-SNE plot semi-supervised encoder before any training")

        # Define autoencoder
        self.autoencoder = Autoencoder(data, self.encoder, learning_rate=learning_rate_autoencoder,
                                       loss_function=loss_function_autoencoder, optimizer=optimizer_autoencoder,
                                       epochs=epochs_autoencoder)

        # Plot tsne plot
        if tsneplots_flag:
            Help_functions.tsne_plot(self.autoencoder.encoder, data, "T-SNE plot semi-supervised encoder after "
                                                                      "autoencoder training")

        # Plot reconstructions
        Help_functions.display_reconstructions(self.autoencoder, n=nr_of_reconstructions_autoencoder)

        # Define classifier
        self.classifier = Classifier(data, self.autoencoder.encoder, learning_rate=learning_rate_classifier,
                                     loss_function=loss_function_classifier, optimizer=optimizer_classifier,
                                     epochs=epochs_classifier, freeze=freeze_encoder_flag)

        # Plot tsne plot
        if tsneplots_flag:
            Help_functions.tsne_plot(self.classifier.encoder, data, "T-SNE plot semi-supervised encoder after "
                                                                     "autoencoder and classifier training")

        # Accuracy
        print("Semi-supervised classifier accuracy on D2 testing data")
        Help_functions.calc_accuracy_classifier(self.classifier, data.d2_x_test, data.d2_y_test)


