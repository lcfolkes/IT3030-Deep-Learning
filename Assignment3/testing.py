from Assignment3 import Help_functions
from Assignment3.Autoencoder import Autoencoder
from Assignment3.stacked_mnist import StackedMNISTData, DataMode
import os

from Assignment3.verification_net import VerificationNet

### GET DATA AND TRAIN VERIFICATION NET
gen_complete = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
x_train_complete, y_train_complete = gen_complete.get_full_data_set(training=True)
print(x_train_complete.shape)
x_test_complete, y_test_complete = gen_complete.get_full_data_set(training=False)
print(x_test_complete.shape)
net = VerificationNet(force_learn=False)
net.train(generator=gen_complete, epochs=5)

### DEFINE AND TRAIN AUTOENCODER
autoencoder_complete = Autoencoder(x_train_complete,y_train_complete,learning_rate=0.001, epochs=10)
img, labels = autoencoder_complete.get_data_predictions_labels(n=60000)
cov = net.check_class_coverage(data=img, tolerance=.8)
pred, acc = net.check_predictability(data=img, correct_labels=labels)
print(f"Coverage: {100 * cov:.2f}%")
print(f"Predictability: {100 * pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")
Help_functions.display_reconstructions(autoencoder_complete)


### AE AS A GENERATIVE MODEL
#generated_img = autoencoder_complete.generate(n=60000)
#cov_generated = net.check_class_coverage(data=generated_img, tolerance=.8)
#print(f"Coverage generated images: {100 * cov_generated:.2f}%")
#Help_functions.display_images(img, n=16)
#Help_functions.display_images(generated_img, n=16)

### AE AS AN ANOMALY DETECTOR@
gen_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
x_train_missing, y_train_missing = gen_missing.get_full_data_set(training=True)
x_test_missing, y_test_missing = gen_missing.get_full_data_set(training=False)
net = VerificationNet(force_learn=False)
net.train(generator=gen_missing, epochs=5)

autoencoder_missing = Autoencoder(x_train_missing,x_train_missing,learning_rate=0.001, epochs=10)
img, labels = autoencoder_missing.get_data_predictions_labels(n=60000)
cov = net.check_class_coverage(data=img, tolerance=.8)
pred, acc = net.check_predictability(data=img, correct_labels=labels)
print(f"Coverage: {100 * cov:.2f}%")
print(f"Predictability: {100 * pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")
Help_functions.display_reconstructions(autoencoder_missing)

print('\n# Evaluate complete model on complete test data')
results_complete = autoencoder_complete.model.evaluate(x_test_complete, x_test_complete, batch_size=128)
print('test loss, test acc:', results_complete)

print('\n# Evaluate missing model on complete test data')
results_missing = autoencoder_missing.model.evaluate(x_test_complete, x_test_complete, batch_size=128)
print('test loss, test acc:', results_missing)


#os.system("tensorboard --logdir=logs/scalars")