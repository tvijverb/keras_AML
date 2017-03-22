from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend as K

import numpy as np
import h5py

input_num_units = 125000
hidden_num_units = 500
output_num_units = 2

file_name = 'hdf5_AML6_H.h5'
file = h5py.File(file_name, 'r')
#nom
#dataset
#data
dataset = file['/data']
#labels
labels = file['/label']

print('Dataset dimension')
print(dataset.shape)
print('Labels dimension')
print(labels.shape)

dataset.astype('float32')
y_train = np_utils.to_categorical(labels)

# create model
model = Sequential()

model.add(Convolution2D(50, 3, 3, input_shape=(50,50,50), border_mode='same', activation='relu', W_constraint=maxnorm(3)))

model.add(Dropout(0.2)
)
model.add(Convolution2D(50, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(2, activation='softmax'))

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(dataset, y_train, nb_epoch=epochs, validation_split=0.5, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(dataset, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
K.clear_session()

