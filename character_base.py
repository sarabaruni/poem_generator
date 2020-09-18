from pickle import dump
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
import keras.utils as ku
import numpy as np
from keras.models import load_model
from pickle import dump
from nltk.tokenize import sent_tokenize, word_tokenize
from pickle import load
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

#reading data
import os
e = os.listdir('drive/My Drive/shahname/')
data = ''
for file in e:
    data += open('drive/My Drive/shahname/'+file).read()
    data +='e'

raw_text = data
# print(data)
tokens = raw_text.split(' ')
# print(tokens)
raw_text = ' '.join(tokens)
# print(raw_text)
length = 10
sequences = list()

for i in range(length, len(raw_text)):
# select sequence of tokens
	seq = raw_text[i-length:i+1]
# store
	sequences.append(seq)


chars = sorted(list(set(raw_text)))
print(len(chars))
print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))


sequences1 = list()
for line in sequences:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences1.append(encoded_seq)

print(sequences1)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# split input and output
sequences1 = np.array(sequences1)

X,y = sequences1[:,:-1],sequences1[:,-1]
sequences1 = [ku.to_categorical(x, num_classes=vocab_size) for x in X]

X = np.array(sequences1)
y = ku.to_categorical(y, num_classes=vocab_size)


# define model
model = Sequential()
model.add(LSTM(75 ,return_sequences=True,input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.9))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=25 ,epochs=10, verbose=2)

# save the model to file
model.save('model.h5')

# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 10, 'رستم', 50))
