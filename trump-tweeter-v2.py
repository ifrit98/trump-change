
# TODO: Follow tensorflow tutorial?
# Ref:  https://www.tensorflow.org/tutorials/text/text_generation
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import numpy as np
import pandas as pd
import os
import time


ENCODING = "ISO-8859-1"

filepath = '/media/jason/Games/ml-data/trump-change/data/trump-tweets.csv'# 'data/trump-tweets.csv'

train = pd.read_csv(filepath,  encoding=ENCODING)
corpus = text = train['text'].values.astype(str)

# TODO: Remove special characters, URLs, etc to shrink vocab
chars_to_rm =  np.array(['\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
       '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
       '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
       '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
       '\xa0', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬',
       '\xad', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹',
       'º', '»', '¼', '½', '¾', '¿', 'Â', 'Ã', 'Ä', 'Å', 'É', 'Ê', '×',
       'Ø', 'Ù', 'Ú', 'Û', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è',
       'é', 'ê', 'ë', 'ì', 'í', 'ï', 'ð', 'ô'])


s = ''
for l in corpus:
    # if type(l) != str: continue
    s += (l + '\n')
# print(s[:256])

# Remove special chars (replace with <OOV>?)
s_clean = ''
for c in s:
    if c not in chars_to_rm:
        s_clean += c
# print(s_clean[:256])

# TODO: Remove URLs with regex

text = s_clean

# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])


# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Now we have an integer representation for each character. 
# Notice that we mapped the character as indexes from 0 to len(unique).
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# Print the first examples input and target values
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# Batch size
BATCH_SIZE = 128

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)


# Build model

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 512


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
    tf.keras.layers.Dense(vocab_size*2),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

print(model.summary())


# Try the model
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# First example in batch
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(sampled_indices)


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# Train the model
def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss_fn(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss_fn)


# Configure checkpoints

# Directory where the checkpoints will be saved
checkpoint_dir = './trump_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS = 250

# TODO: 
#  Callbacks: learning rate scheduler, early stopping
#  learning rate range test (use R package somehow?)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# Plot loss and accuracy from hist
import matplotlib.pyplot as plt

loss = history.history['loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()




tf.train.latest_checkpoint(checkpoint_dir)

# Reload model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

 
# Prediction

def generate_text(model, start_string, num_generate=256):
  # Evaluation step (generating text using the learned model)

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

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

for _ in range(10):
    print(generate_text(model, start_string=u"China "))


