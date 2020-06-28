# Trump Change
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

`trump-change` is a character-level generative text model wrapped in a customized data science workflow written in `python` using `tensorflow` and the `pyruns` experiment manager.  This project contains all the necessary source code and data requred to train your own models, run hyperparameter tuning experiments, freeze models off in Tensorflow's `saved_model` format, and access pretrained models from a `python` API.

Note: You do not need `pyruns` to run any of this code.  `pyruns` is a simple port of Rstudio's `tfruns` for `python` that I wrote and can be found [here](https://github.com/ifrit98/pyruns).  It helps manage your expierments by creating a micro-hermetic build of a data science project in a unique run directory, making for easy evaluations and comparisons.

 To start generating with a pretrained model with an automated script, simply clone the repo and run `generate.py`:
 ```{bash}
 git clone www.github.com/ifrit98/trump-change.git
 cd trump-change
 ./generate.py
```

To have more control over knobs such as swapping out checkpoint files, setting the annealing `temperature`,`conditioning_string`s, and the number of chars to generate per tweet, use the python CLI provided in `CLI.py`:
```{bash}
./CLI.py
...
How many strings to generate?...
3

...

Would you like to update parameters? (y/n):
y

...
```

Note: This script will automatically load the latest checkpoint in the `trump_training_checkpoints` directory.  If you would like to use a different checkpoint directory, specify it upon calling `CLI.py`:
```{bash}
./CLI.py --checkpoint /path/to/checkpoint_dir
```

The CLI uses argparse to set values automatically so you only have to set the number of tweets to generate in the CLI:
```
./CLI.py --help

usage: CLI.py [-h] [-w CHECKPOINT] [-n NUM_GENERATE] [-c CONDITIONING]
              [-v VOCAB_PATH] [-t TEMPERATURE]

optional arguments:
  -h, --help            show this help message and exit
  -w CHECKPOINT, --checkpoint CHECKPOINT
                        Filepath to model checkpoint.
  -n NUM_GENERATE, --num_generate NUM_GENERATE
                        Set the number of characters to generate per tweet.
  -c CONDITIONING, --conditioning CONDITIONING
                        Set the conditioning string to use as input to the
                        model.
  -v VOCAB_PATH, --vocab_path VOCAB_PATH
                        Path to vocabulary file [.npy array].
  -t TEMPERATURE, --temperature TEMPERATURE
                        Set the temperature of the annealer during prediction.

```



To train your own model simply run the `train.py` script:
```{bash}
./train.py
```

You may wish to update hyperparameter values found in the `flags.yaml` file of the top level.
```
cat flags.yaml

--- 
 # Model
 epochs: 350
 rnn_units: 412
 embedding_dim: 206
 batch_size: 108
 vocab_file: vocab.npy

 # Optimizer
 min_lr: 0.0001
 max_lr: 0.004291934
 min_delta: 0.001 # for lr_scheduler
 lr_factor: 0.5
 patience: 5
 steps_per_epoch: 130
 decay_epochs: 5
 decay_rate: 0.96

 # Dataset
 keep_emojis: False
 data_file: trump-tweets-no-retweets-latest.json
 buffer_size: 100

 # Misc
 checkpoint_dir: trump_training_checkpoints/current
 verbose: False
 encoding: ISO-8859-2
 base_dir: /home/jason/internal/trump-change
 runs_dir: /home/jason/freya/runs # This gloab runs directory may be separate from your project directory

```
