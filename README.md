# Trump Change
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

`trump-change` is a character-level generative text model wrapped in a customized data science workflow written in `python` using `tensorflow`.  This project contains all the necessary source code and data requred to train your own models, run hyperparameter tuning experiments, freeze models off in Tensorflow's `saved_model` format, and access pretrained models from a `python` API.

 To start generating with a pretrained model, simply clone the repo and run the `generate.py` script:
 ```{bash}
 git clone www.github.com/ifrit98/trump-change.git
 cd trump-change
 ./generate.py
```

To train your own model simply run the `train.py` script:
```{bash}
./train.py
```
