import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import keras
from keras.datasets import mnist
#from keras.layers import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

# importing dataset
mnist = tf.keras.datasets.mnist

# Splitting Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.show()

# Normalization
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.show()


# Training Model
def create_model():
    num_classes = 10;
    model = tf.keras.models.Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10 , batch_size=200, verbose=1)
print("The model is successfully trained")
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # Flattening (28,28) --> 28*28
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # Hidden Layer 1 with 256 units
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # Hidden Layer 2 with 256 units
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # Hidden Layer 3 with 256 units
# model.add(tf.keras.layers.Dense(256,activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(256,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Final Layer with softmax activation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
print("The model has successfully trained")
"""

# Saving Model
model.save('mnist.h5')
print("Saving the model as mnist.h5")
loss, acc = model.evaluate(X_test, y_test)
print(loss)
print(acc)
