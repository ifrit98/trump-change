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


ENCODING = "ISO-8859-2"

basedir = '/media/jason/Games/ml-data/trump-change'
datadir = os.path.join(basedir, 'data')
filepath = os.path.join(datadir, 'trump-tweets-no-retweets.json') #csv'

df = train = pd.read_json(filepath)
corpus = train['text'].values # .astype(str) # conversion causing encoding issues?

# CHARSETS MUST BE TRACKED ACROSS MODELS.  INVALIDATES OLD MODELS IF CHANGED
# Remove unusual chars to decrease vocab size
chars_to_rm = np.array(['🇦', '🇧', '🇨', '🇪', '🇫', '🇬', '🇮', '🇯', '🇰', '🇱', '🇲', '🇳', '🇴',
       '🇵', '🇷', '🇸', '🇹', '🇺', '🇽', '🌊', '🌍', '🌐', '🌪', '🌲', '🍾', '🍿',
       '🎂', '🎄', '🎆', '🎉', '🎥', '🏁', '🏆', '🏈', '🏻', '🏼', '🏽', '👀', '👇',  '🤳', '🦅',
       '👈', '👉', '👊', '👍', '👏', '👷', '💜', '💤', '💥', '💪', '💫', '💬', '💯',
       '💰', '📈', '📉', '📌', '📍', '📱', '📸', '📺', '🔟', '🔥', '🔴', '🔹', '😂',
       '😆', '😉', '😜', '😝', '😳', '🙄', '🙋', '🙌', '🙏', '🚀', '🚒', '🚚', '🚛',
       '🚨', '🚫', '🛒', '🛰', '🤔', '🤗', '🤡', '🤣', '🤦', '€', '⃣', '™', '→', '↓', 
       '↴', '⇒', '⏰', '▪', '▶', '●', '◦', '☀', '★', '☑', '☘', '♀', '♂', '⚖', '⚙', 
       '⚠', '⚡', '⚽', '⚾', '⛈', '⛰', '⛳', '✅', '✈', '✍', '✓', '✔', '✖', '❌', 
       '❗', '❣', '❤', '➜', '➡', '⬆', '⬇', '⭐', '、', '。', '々', '《', '「', '」', 
       '【', '】', 'い', 'う', 'え', 'お', 'か', # New chars to remove above this line 
       'が', 'き', 'ぎ', 'く', 'け', 'こ', 'ご', 'さ', 'し', 'じ', 'す', 'そ', 'た',
       'だ', 'っ', 'つ', 'て', 'で', 'と', 'な', 'に', 'の', 'は', 'へ', 'ま', 'み',
       'め', 'も', 'よ', 'ら', 'り', 'る', 'を', 'ア', 'ゴ', 'サ', 'ジ', 'セ', 'ッ',
       'ト', 'ニ', 'フ', 'プ', 'ミ', 'メ', 'ラ', 'ル', 'ン', 'ー', '上', '下', '世',
       '両', '人', '代', '令', '以', '会', '倍', '共', '典', '出', '初', '励', '勢',
       '北', '千', '史', '同', '后', '和', '問', '国', '夕', '大', '天', '夫', '安',
       '対', '居', '席', '式', '応', '情', '投', '揃', '揺', '新', '日', '昨', '時',
       '更', '朝', '本', '様', '歓', '海', '済', '激', '理', '界', '皇', '盟', '相',
       '稿', '米', '経', '統', '続', '総', '考', '脳', '臨', '自', '至', '艦', '葉',
       '衛', '見', '訪', '話', '課', '談', '護', '賓', '軍', '迎', '過', '間', '阪',
       '陛', '隊', '領', '題', '食', '首', '鮮', '고', '국', '는', '대', '라', '렛',
       '리', '며', '모', '미', '바', '받', '보', '북', '브', '상', '서', '소', '습',
       '에', '오', '울', '을', '의', '쟁', '전', '정', '초', '측', '핑', '하', '한',
       '화', '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '–', '—',
       '―', '‘', '’', '“', '”', '•', '…', '\u202f', '′', '‼', '\u2060',
       '\u2063', '\u2066', '\u2069', '\U0001f928', '\U0001f92f',
       '\U0001f973', '\U0001f9d0', '\U0010fc00',
       '~', '¡', '£', '«', '®', 'º', '»', '½', 'É', '×', 'á', 'â', 'é',
       'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'ø', 'ú', 'ğ', 'ō', 'ɪ', 'ɴ', 'ʀ',
       'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל',
       'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'צ', 'ק', 'ר', 'ש', 'ת', '،', 'ء',
       'آ', 'أ', 'ؤ', 'إ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د',
       'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق',
       'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ً', 'چ', 'ژ', 'ک', 'گ', 'ی',
       '۰', '۴', 'ँ', 'ं', 'अ', 'आ', 'इ', 'उ', 'ए', 'औ', 'क', 'ख', 'ग',
       'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ', 'ड', 'ण', 'त', 'थ', 'द', 'ध',
       'न', 'प', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
       '़', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', '्', '।', 'ᴅ', 'ᴇ',
       'ᴍ', 'ễ', '️'])



import string, re # gruber html regex
regex = pat = r'\b(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^%s\s]|/)))'
regex = pat = pat % re.sub(r'([-\\\]])', r'\\\1', string.punctuation)

corpus_no_html = np.array(list(map(lambda x: re.sub(regex, ' ', x), corpus)))



s = ''
for l in corpus_no_html: 
    s += (l + ' ') # (l + '\n') or (l + ' \n ')
print(s[:280])



# Remove special chars
s_clean = ''
for c in s:
    if c not in chars_to_rm:
        s_clean += c + ''
print(s_clean[:280])


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
seq_length = max([len(x) for x in corpus_no_html]) # 144 OG tweet length
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
BATCH_SIZE = 136 # close to a divisor of examples_per_epoch

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print(dataset)



## Build model

# Length of the vocabulary in chars
vocab_size = len(vocab)

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
    # tf.keras.layers.Dense(rnn_units // 2, activation='relu'),
    tf.keras.layers.Dense(vocab_size*2, activation='relu'),
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
