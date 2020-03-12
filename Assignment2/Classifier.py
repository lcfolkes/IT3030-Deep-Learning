from keras import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


class Classifier:

	def __init__(self, encoder, learning_rate_classifier, loss_function_classifier, optimizer_classifier,
							epochs_classifier,freeze):
		self.model = encoder


		def __build_model(self):
			self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, activation='relu', padding='same'))
			self.model.add(Dropout(0.2))
			self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
			self.model.add(Dropout(0.2))
			self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
			self.model.add(Dropout(0.2))
			self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(Flatten())
			self.model.add(Dropout(0.2))
			self.model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
			self.model.add(Dropout(0.2))
			self.model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
			self.model.add(Dropout(0.2))
			# this is decode, where to stop encode?
			self.model.add(Dense(self.num_classes, activation='softmax'))