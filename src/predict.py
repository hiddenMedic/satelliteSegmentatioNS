import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL
from PIL import Image

model = tf.keras.models.load_model('models/the_model_aug')

# model.summary()

pics = np.load("proc_data/in_ns_frames.npy")
proc_pics = model.predict(pics)

print(proc_pics.shape)

np.save("proc_data/ns_classes_frames.npy", proc_pics)