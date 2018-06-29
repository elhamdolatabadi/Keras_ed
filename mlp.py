'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras.layers import AveragePooling2D, Input, Flatten, Concatenate,concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
import numpy as np

batch_size = 128
num_classes = 10
epochs = 3

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data(['/data'])

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Training parameters

# Input image dimensions.
input_shape = x_train.shape[1:]
inputs = Input(shape=input_shape)
x = inputs
#x = keras.layers.add([x, y])
#x = Activation('relu')(x)
h1 = Dense(128,activation='relu',kernel_initializer='he_normal')(x)
h2 = Dense(32,activation='relu',kernel_initializer='he_normal')(h1)
h22 = concatenate([h2,x])
outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(h22)

# Instantiate model.
model = Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#plot_model(model, to_file='multilayer_perceptron_graph.png')
#h_model = Model(inputs=model.input,outputs=model.layers[-2].output)
#z2 = h_model.predict(x_train)
#xx= np.concatenate(x_train,z2)



#input1 = keras.layers.Input(shape=(16,))
#x1 = keras.layers.Dense(8, activation='relu')(input1)
#input2 = keras.layers.Input(shape=(32,))
#x2 = keras.layers.Dense(8, activation='relu')(input2)
#added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])
#
#out = keras.layers.Dense(4)(added)
#model = keras.models.Model(inputs=[input1, input2], outputs=out)
