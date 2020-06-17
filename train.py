#!/usr/bin/python3

import os
import yaml
from datetime import datetime

import tensorflow as tf

start = datetime.now()
START_TIME = str(start).replace(' ', '_')[:-7]

from __init__ import *



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



## Callbacks
# TODO: reach into model or optimizer to grab current lr and compare with `min_lr` (via self?)
from math import floor
def exp_decay(epoch,
              logs=None,
              init_lr=FLAGS['max_lr'],
              min_lr=FLAGS['min_lr'],
              decay_steps=1, #FLAGS['steps_per_epoch'], 
              decay_epochs=FLAGS['decay_epochs'], 
              decay_rate=FLAGS['decay_rate'], 
              batch_size=FLAGS['batch_size'],
              staircase=True):
    # print("Logs:", logs)
    p = epoch / decay_steps #(decay_steps * decay_epochs * batch_size)
    # lr = logs['learning_rate']
    if staircase:
        p = floor(p)
    return init_lr * (decay_rate ** p)

# list(map(lambda x: exp_decay(x), range(50)))


def linear_decay(epoch):
    return 

lr_schedule_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=exp_decay)
# lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(exp_decay, verbose=1)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=FLAGS['lr_factor'], patience=FLAGS['patience'], verbose=0, mode='auto',
    min_delta=FLAGS['min_delta'], cooldown=0, min_lr=FLAGS['min_lr']
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'loss', patience = 10
)

EPOCHS = FLAGS['epochs']


history = model.fit(
    dataset, 
    epochs=EPOCHS, 
    callbacks=[checkpoint_callback, reduce_lr_callback])


end = datetime.now()
END_TIME = str(end).replace(' ', '_')[:-7]

training_time = str(end - start)
print('Training took {} hour/min/sec'.format(training_time.split('.')[0]))

# Save final model weights for freezing and exporting later
save_model_path = os.path.join(basedir, 'savedmodels', 'final_{}'.format(END_TIME))
model.save_weights(save_model_path)

