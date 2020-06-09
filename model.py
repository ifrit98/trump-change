import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import os
import yaml 

from numpy import load


# Load flags.yaml
stream = open("flags.yaml", 'r')
FLAGS = yaml.load(stream)


## Build model

BATCH_SIZE = FLAGS['batch_size'] #108
EPOCHS = FLAGS['epochs']

# Length of the vocabulary in chars 
# TODO: ensure vocab is in namespace or find way to get access
vocab = load(os.path.join('data', FLAGS['vocab_file']))
vocab_size = len(vocab) # FLAGS['vocab_size']  

# The embedding dimension
embedding_dim = FLAGS['embedding_dim'] #206 # divisor of examples_per_epoch

# Number of RNN units
rnn_units = FLAGS['rnn_units'] #412 # divisor **


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    # tf.keras.layers.Dense(rnn_units // 2, activation='relu'),
    tf.keras.layers.Dense(vocab_size*2, activation='relu'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

print(model.summary())


# Train the model
def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


min_lr = FLAGS['min_lr'] # 0.00001
max_lr = FLAGS['max_lr'] # 0.003

optimizer = tf.keras.optimizers.Adam(learning_rate=max_lr)


model.compile(optimizer=optimizer, loss=loss_fn)
