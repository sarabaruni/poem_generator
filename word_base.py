from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np


# reading data
import os
e = os.listdir('shah')
data = ''
for file in e:
    data += open('shah/'+file).read()


# reading all words
all_words = open('allShahnameWords.txt').read()

# invert captal corracters to small and get every line
corpus = data.lower().split('\n')
all_corpus = all_words.lower().split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
total_words = len(tokenizer.word_index) + 1

# training data
# input vector to feed into the network
input_sequences = []
for line in corpus:
    # get every sequence of every line
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)


# padding sequences
max_sequences_len = max([len(X) for X in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequences_len,padding='pre'))

# creat labels
predictors , label = input_sequences[:,:-1],input_sequences[:,-1]
# onehot labels
label = ku.to_categorical(label,num_classes=int(total_words))

# model building
model = Sequential()
model.add(Embedding(total_words,100,input_length=max_sequences_len-1))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation='softmax'))
model.compile(loss='categorical_crossentropy' , optimizer='adam',metrics=['accuracy'])
print(model.summary())

history = model.fit(predictors, label, epochs=200, verbose=1)

import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()