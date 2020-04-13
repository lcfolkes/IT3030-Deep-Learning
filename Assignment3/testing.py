from Assignment3 import Help_functions
from Assignment3.Autoencoder import Autoencoder
from Assignment3.Autoencoder_seq import Autoencoder_seq
from Assignment3.Encoder import Encoder
from Assignment3.stacked_mnist import StackedMNISTData, DataMode
import os

from Assignment3.verification_net import VerificationNet

gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)
#encoder = Encoder(x_train, y_train, size_latent_vector=32)
#autoencoder = Autoencoder(encoder, learning_rate=0.001)
autoencoder = Autoencoder_seq(x_train,y_train,learning_rate=0.001)
img, labels = autoencoder.get_data_predictions_labels(60000)
cov = net.check_class_coverage(data=img, tolerance=.8)
pred, acc = net.check_predictability(data=img, correct_labels=labels)
print(f"Coverage: {100 * cov:.2f}%")
print(f"Predictability: {100 * pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")
Help_functions.display_reconstructions(autoencoder)

#os.system("tensorboard --logdir=logs/scalars")