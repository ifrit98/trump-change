#! /usr/bin/python3

from sys import argv
from pyruns import training_run

import os

if len(argv) == 1:
    raise ValueError('Must supply a runs_dir path.')


runs_dir = argv[1]
if not os.path.exists(runs_dir):
    print('Runs_dir path supplied does not exist.\nCreating it now...')
    try:
        os.mkdir(argv[1])
    except Exception as e:
        print(e)


print('Starting training run in top-level run directory: {}'.format(runs_dir))

training_run(runs_dir=runs_dir, exclude='*git')
