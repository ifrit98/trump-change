## TODO: RECREATE R WORKFLOW IN PYTHON!: MODEL.PY, TRAIN.PY, DATASET.PY, EXPORT.PY, ETC.


# Ref:  https://www.tensorflow.org/tutorials/text/text_generation
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

import numpy as np
import pandas as pd
import os
import time



# basedir = '/media/jason/Games/ml-data/trump-change'
basedir = '/home/jason/Documents/'


# Load saved_model
full_model_path = os.path.join(basedir, 'savedmodels', 'current')
model = tf.keras.models.load(full_model_path)
model.summary()

full_model = tf.function(lambda x: model(x))

full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))


from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2 

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

## Build model

# Length of the vocabulary in chars
vocab_size = 66 # 94 #len(vocab)

# The embedding dimension
embedding_dim = 206 # divisor of examples_per_epoch

# Number of RNN units
rnn_units = 412 # divisor **


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
    tf.keras.layers.Dense(vocab_size*2, activation='relu'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


# Configure checkpoints

# Directory where the checkpoints will be saved
checkpoint_dir = 'trump_training_checkpoints/current' 
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# If you want preds on CPU only
GENERATE_ON_CPU = True

if GENERATE_ON_CPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Check to see if GPU is not visible
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


checkpoint_dir = os.path.join(basedir, 'trump_training_checkpoints/archive/no_retweets')

tf.train.latest_checkpoint(checkpoint_dir) # './trump_training_checkpoints'

# Reload model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

 