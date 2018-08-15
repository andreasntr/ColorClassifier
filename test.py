import tensorflow as tf
import numpy as np
import json

tf.enable_eager_execution()

with np.load("dataset.npz") as savedData:
    colors_test = tf.constant(savedData['test_x'])
    labels_test = tf.constant(savedData['test_y'])

model = tf.keras.models.model_from_json(json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")
predictions = model.predict(colors_test, batch_size=32, verbose=1)
predictions = tf.one_hot(np.argmax(predictions,1),9)
equals = np.sum(np.all(predictions.numpy()==labels_test.numpy(),axis=1))
print("Guess accuracy: {}".format(equals/len(colors_test.numpy())))
