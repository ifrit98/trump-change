# If you want preds on CPU only
GENERATE_ON_CPU = True

if GENERATE_ON_CPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Check to see if GPU is not visible
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


import tensorflow as tf
from numpy import load, array, random, float64
from datetime import datetime
import yaml
import os

from model import build_model, vocab, embedding_dim, rnn_units
from __init__ import basedir


vocab = load('data/vocab.npy')
vocab_size = len(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = array(vocab)

run_dir = 'trump_training_checkpoints/current'
checkpoint_dir = os.path.join(basedir, run_dir) #'trump_training_checkpoints/archive/no_retweets')

tf.train.latest_checkpoint(checkpoint_dir) # './trump_training_checkpoints'

# Reload model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()



class TrumpChange(tf.Module):
    def __init__(self):
        super(TrumpChange, self).__init__()
        self.model = model
        self.vocab = load('data/vocab.npy')
        self.vocab_size = len(vocab)
        self.char2idx = {u:i for i, u in enumerate(vocab)}
        self.idx2char = tf.convert_to_tensor(array(vocab), tf.string)
        self.num_generate = 280
        self.conditioning_string = 'China '
        self.temperature = 1.0

    @tf.function
    def __call__(self, x=None):
        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in self.conditioning_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = self.temperature

        # Here batch size == 1
        self.model.reset_states()
        for _ in range(self.num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0]

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        text_generated = tf.strings.join(text_generated, separator='', name=None)
        return tf.strings.join([tf.convert_to_tensor(self.conditioning_string, tf.string), text_generated])


    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def set_conditioning_str(self, new_string):
        self.conditioning_string = str(new_string)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def set_num_generate(self, new):
        self.num_generate = int(new)



tc = tc_model = TrumpChange()

tc.__call__.get_concrete_function()
tc.set_conditioning_str.get_concrete_function()
tc.set_num_generate.get_concrete_function()

# Save in saved_model format
tf.saved_model.save(tc, 'exported') 


if True:
    reloaded = tf.saved_model.load('exported')
    print(reloaded())