import os
import yaml

# Load flags.yaml
stream = open("flags.yaml", 'r')
FLAGS = yaml.load(stream)
for key, value in FLAGS.items():
    print (key + " : " + str(value))

basedir = FLAGS['base_dir']
datafile = FLAGS['data_file']
datadir = os.path.join(basedir, 'data')
filepath = os.path.join(datadir, datafile) #'trump-tweets-no-retweets.json'
