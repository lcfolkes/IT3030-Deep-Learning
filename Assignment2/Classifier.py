from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
from Assignment2 import Encoder
from Assignment2.Help_functions import modify_input_shape


class Classifier:
	def __init__(self, x, y, encoder, learning_rate=0.01, loss="categorical_crossentropy",
				 optimizer="adam", epochs=20,freeze=False):
		self.x = modify_input_shape(x)
		self.y = y
		self.no_classes = y.shape[1]
		self.classifier_head = self.__classifier_head(encoder)
		enc_input_layer = encoder.get_input_at(0)
		enc_output_layer = encoder.get_output_at(-1)
		self.model = Model(enc_input_layer, self.classifier_head(enc_output_layer))
		self.model.compile(optimizer, loss="categorical_crossentropy")
		self.model.fit(self.x, y, epochs=epochs, batch_size=1000)

	def __classifier_head(self, encoder):
		# Create classifier head
		encoded_output_shape = encoder.get_output_shape_at(-1)[1:]
		inputs = Input(encoded_output_shape)
		# inputs = Input(shape=self.x.shape[1:])
		# x = encoder.encoder(inputs)
		x = Dense(128, activation='relu')(inputs)
		x = Dense(self.no_classes, activation='relu')(x)
		classified = Dense(self.no_classes, activation='softmax')(x)
		classifier = Model(inputs=inputs, outputs=classified)
		return classifier
