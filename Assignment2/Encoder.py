from Assignment2.Preprocessing import Data

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils

class Encoder:

	def __init__(self,dataset):
			self.input_shape = self.__modify_input_shape((self.x_test.shape[1:]))
			self.model = Sequential()
			print(self.input_shape)
			self.__build_model()

	#check add single channel to input_shape
	def __modify_input_shape(self, input):
		if(len(input)==2):
			return input + (1,)
		return input

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
		#this is decode, where to stop encode?
		self.model.add(Dense(self.num_classes, activation='softmax'))

