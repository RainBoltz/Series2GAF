from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from series2gaf import GenerateGAF

K.clear_session()

# -------------------------------------------------------------------
# Generate Gaf:
#
# create a random series with 200 numbers
# all numbers are in the range of 50.0 to 150.0
random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))

# set parameters
timeSeries = list(random_series)
windowSize = 50
rollingLength = 10
fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)

# generate GAF pickle file (output by Numpy.dump)
GenerateGAF(all_ts = timeSeries,
            window_size = windowSize,
            rolling_length = rollingLength,
            fname = fileName)

# -------------------------------------------------------------------
# CNN Example:
#
# data shape: (15, 50, 50)
gaf = np.load('%s_gaf.pkl'%fileName)
gaf = np.reshape(gaf, (gaf.shape[0], gaf.shape[1], gaf.shape[2], 1))
# fake data labels: (15,)
labels = np.zeros(15)
labels[:8] = 1
labels = np_utils.to_categorical(labels)

model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 padding='same',
                 input_shape=(gaf.shape[1], gaf.shape[2], 1),
                 activation='relu'))
# Create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2, 2)))
# Create CN layer 2
model.add(Conv2D(filters=36,
                 kernel_size=(5, 5),
                 padding='same',
                 input_shape=(gaf.shape[1], gaf.shape[2], 1),
                 activation='relu'))
# Create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add Dropout layer
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
# Define Compiler
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the Model
train_history = model.fit(x=gaf,
                          y=labels, validation_split=0.2,
                          epochs=10, batch_size=300, verbose=2)
