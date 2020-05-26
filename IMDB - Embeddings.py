from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure.
    with_info=True)

encoder = info.features['text'].encoder

batch_size = 10
def batch_padder(data, batch_size):
    pdata = data.shuffle(1000).padded_batch(10)
    return pdata
train_batch = batch_padder(train_data, batch_size)
test_batch = batch_padder(test_data, batch_size)

model = keras.Sequential()
model.add(layers.Embedding(encoder.vocab_size, 12))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(12,activation='sigmoid'))
model.add(layers.Dense(1))
model.add(layers.)

model.compile(optimizer='Adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.accuracy,tf.keras.metrics.MeanAbsoluteError])

history = model.fit(
    train_batch,
    epochs=20,
    validation_data=test_batch, validation_steps=20)
import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()






