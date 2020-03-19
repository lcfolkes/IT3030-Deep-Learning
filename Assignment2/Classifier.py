from datetime import datetime

from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from Assignment2 import Encoder
from Assignment2.Help_functions import modify_input_shape
from keras.callbacks import TensorBoard

import copy


class Classifier:
	def __init__(self, data, encoder, learning_rate=0.01, loss_function="categorical_crossentropy",
				 optimizer="adam", epochs=20, freeze=False):
		x_train = modify_input_shape(data.d2_x_train)
		x_val = modify_input_shape(data.d2_x_val)
		y_train = data.d2_y_train
		y_val = data.d2_y_val
		self.no_classes = y_train.shape[1]
		self.encoder = encoder
		self.encoder.model.compile(optimizer=optimizer, loss=loss_function)
		classifier_head = self.__classifier_head()
		enc_input_layer = self.encoder.model.get_input_at(0)
		enc_output_layer = self.encoder.model.get_output_at(-1)
		self.model = Model(inputs=enc_input_layer, outputs=classifier_head(enc_output_layer))
		self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

		# Define Tensorboard
		logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=logdir, profile_batch=0)

		#train in for-lopp to get accuracy during training
		self.model.fit(x_train, y_train, epochs=epochs, batch_size=500, validation_data=(x_val,y_val), callbacks=[tensorboard])

	def __classifier_head(self):
		# Create classifier head
		encoded_output_shape = self.encoder.model.get_output_shape_at(-1)[1:]
		inputs = Input(encoded_output_shape)
		x = Dense(128, activation='relu')(inputs)
		x = Dense(self.no_classes, activation='relu')(x)
		classified = Dense(self.no_classes, activation='softmax')(x)
		classifier = Model(inputs=inputs, outputs=classified)
		return classifier
