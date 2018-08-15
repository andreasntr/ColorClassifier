import tensorflow as tf
import numpy as np
from random import randint
import json

colors = None
labels = None
data_size = 0

tf.enable_eager_execution()

with np.load("processedData.npz") as savedData:
    colors = np.array(savedData['colors'], dtype=np.float32)
    labels = tf.one_hot(savedData['labels'],9, dtype = tf.float32).numpy()
    data_size = len(savedData['colors'])

train_size = int(data_size*0.8)
test_size = validation_size = int((data_size - train_size)/2)

indexes = [randint(0, data_size-1) for i in range(train_size)]
colors_train = tf.constant([colors[i] for i in indexes])
labels_train = tf.constant([labels[i] for i in indexes])
test_indexes = []
for i in range(0, data_size):
    if not (i in indexes):
        test_indexes.append(i)
test_indexes = [test_indexes[randint(0, test_size-1)] for i in range(test_size)]
colors_test = tf.constant([colors[i] for i in test_indexes])
labels_test = tf.constant([labels[i] for i in test_indexes])
validation_indexes = []
for i in range(0, data_size):
    if not (i in test_indexes) and not (i in indexes):
        validation_indexes.append(i)
validation_indexes = [validation_indexes[randint(0, validation_size-1)]  for i in range(validation_size)]
colors_validation = tf.constant([colors[i] for i in validation_indexes])
labels_validation = tf.constant([labels[i] for i in validation_indexes])

np.savez_compressed("dataset", train_x = colors_train.numpy(), train_y = labels_train.numpy(), test_x = colors_test.numpy(),
        test_y = labels_test.numpy(), validation_x = colors_validation.numpy(), validation_y = labels_validation.numpy())

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(3,),activation=tf.nn.relu),
    tf.keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training:")
model.fit(colors_train, labels_train, epochs=10, batch_size=32)
print("Training ended. Validating:")
model.fit(colors_validation, labels_validation, epochs=10, batch_size=32)
json.dump({'model':model.to_json()}, open("model.json", "w"))
model.save_weights("model_weights.h5")
