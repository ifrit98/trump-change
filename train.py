import os
import yaml
from datetime import datetime

import tensorflow as tf

start = datetime.now()
START_TIME = str(start).replace(' ', '_')[:-7]


# Load flags.yaml
stream = open("flags.yaml", 'r')
FLAGS = yaml.load(stream)
for key, value in FLAGS.items():
    print (key + " : " + str(value))

basedir = FLAGS['base_dir']
datafile = FLAGS['data_file']
datadir = os.path.join(basedir, 'data')
filepath = os.path.join(datadir, datafile) #'trump-tweets-no-retweets.json'



## Prepare and import dataset
from dataset import dataset, vocab, idx2char, char2idx


## Build and import model
from model import model, loss_fn


## Train model

# Configure checkpoints

# Directory where the checkpoints will be saved
checkpoint_dir = FLAGS['checkpoint_dir'] # 'trump_training_checkpoints/current' 

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# TODO: 
#  Callbacks: learning early stopping
#  Automatic learning rate range test (use R package somehow?)

# Callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto',
    min_delta=0.0005, cooldown=0, min_lr=FLAGS['min_lr']
)

EPOCHS = FLAGS['epochs']


history = model.fit(
    dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, reduce_lr_callback])

end = datetime.now()
END_TIME = str(end).replace(' ', '_')[:-7]

training_time = str(end - start)
print('Training took {} hour/min/sec'.format(training_time.split('.')[0]))

# Save final model weights for freezing and exporting later
save_model_path = os.path.join(basedir, 'savedmodels', 'final_{}'.format(END_TIME))
model.save_weights(save_model_path)



# Save history object, can't use pickle: .Rlock object error
import pickle

hist_file = os.getcwd() + '/history/history-{}_'.format(EPOCHS) + END_TIME
with open(hist_file, 'wb') as f:
    pickle.dump(dict(history.history), f)



# Plot loss and accuracy from hist
import matplotlib.pyplot as plt

loss = history.history['loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.savefig('plots/training_loss_{}_'.format(EPOCHS) + END_TIME + '.pdf')
plt.show()

