import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


#design model with one hidden layer and activation functions relu and softmax (output)
#model = keras.models.Sequential([keras.layers.Flatten(),
#                                    keras.layers.Dense(128, activation=tf.nn.relu),
#                                    keras.layers.Dense(10, activation=tf.nn.softmax)])

#build model and decide on optimizer and loss function
#model.compile(optimizer = 'adam',
#              loss = 'sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#train model
#model.fit(training_images, training_labels, epochs=5)

#evaluate model
#model.evaluate(test_images, test_labels)

#classifications = model.predict(test_images)

#print(classifications[0].argmax())
#print(test_labels[0])

