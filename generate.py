#!/usr/bin/python3

# If you want preds on CPU only
GENERATE_ON_CPU = True

if GENERATE_ON_CPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Check to see if GPU is not visible
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


from __init__ import *
import tensorflow as tf
from model import build_model

# TODO: save preprocessed dataset, vocab, char2idx, idx2char for faster loading
from dataset import vocab, char2idx, idx2char


checkpoint_dir = os.path.join(os.getcwd(), 'trump_training_checkpoints/current')

tf.train.latest_checkpoint(checkpoint_dir) # './trump_training_checkpoints'

# Reload model

model = build_model(len(vocab), FLAGS['embedding_dim'], FLAGS['rnn_units'], batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

 
# Prediction

def generate_text(model, start_string, num_generate=256, temp=1.0):
  # Evaluation step (generating text using the learned model)

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = temp

  # Here batch size == 1
  model.reset_states()
  for _ in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


no_generate = 10

tweets = [
    generate_text(model, start_string="China ") for _ in range(no_generate)
]

print(tweets)

from datetime import datetime
time = str(datetime.now()).replace(' ', '_').replace(':', '_')

# Dump generated text to csv or txt file
outfile = os.path.join(
    os.getcwd(), 
    'generated/trump-tweets-{}-{}_epochs_'.format(no_generate, FLAGS['epochs']) + time[:-7] + '.txt')

import csv
with open(outfile, 'w') as f:
    writer = csv.writer(f, dialect='unix')
    writer.writerows([tweets])

print("Done generating!")