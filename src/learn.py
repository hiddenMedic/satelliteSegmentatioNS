import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Input, Conv2DTranspose, concatenate, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = 100
BATCH_SIZE = 32
N_CLASSES = 9

_x = np.load("xy/ovp_x_slo.npy")
_y = np.load("xy/ovp_y_slo.npy")
print(_x.shape, _y.shape)
                                                                        
x_train, x_test, y_train, y_test = train_test_split(_x, _y, shuffle=True, random_state=6, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def build_unet(input_shape, num_classes):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path (downsampling)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottom layer
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Expanding path (upsampling)
    up4 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
    up4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    output = Conv2D(num_classes, 1, activation='softmax')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

model = build_unet(x_train.shape[1:], N_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs=60,
                    validation_data=(x_test, y_test), batch_size=BATCH_SIZE)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

y_pred = model.predict(x_test)
flat_y_pred = []
flat_y_test = []
for (it1, it2) in zip(y_pred, y_test):
  for (i1, i2) in zip(it1, it2):
    for (j1, j2) in zip(i1, i2):
      flat_y_pred.append(np.argmax(j1))
      flat_y_test.append(np.argmax(j2))
flat_y_pred = np.array(flat_y_pred)
flat_y_test = np.array(flat_y_test)
print(flat_y_pred.shape, flat_y_test.shape)

print(y_pred.shape, y_test.shape)

cm = confusion_matrix(flat_y_test, flat_y_pred)
print(cm)

cr = classification_report(flat_y_test, flat_y_pred)
print(cr)

model.save("models/model1")