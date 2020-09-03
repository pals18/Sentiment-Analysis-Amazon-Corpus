import os
import sys
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import ast
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten

f1=sys.argv[1]
data=f1
x_train_with_stopwords = []
y_train_with_stopwords = []
x_validate_with_stopwords = []
y_validate_with_stopwords = []
x_test_with_stopwords = []
y_test_with_stopwords = []

def readfile(path,lis):
    data1=open(path,'r')
    for line in data1:
        temp = line[1:-2].replace("'", '').replace(' ','').split(',')
        lis.append(temp)
    return lis
        
       
x_train_with_stopwords=readfile(os.path.join(data,'train.csv'), x_train_with_stopwords)
y_train_with_stopwords=[label.strip('\n') for label in open(os.path.join(data,'train_label.csv'), 'r')]

x_validate_with_stopwords=readfile(os.path.join(data,'val.csv'), x_validate_with_stopwords)
y_validate_with_stopwords=[label.strip('\n') for label in open(os.path.join(data,'val_label.csv'), 'r')]

x_test_with_stopwords=readfile(os.path.join(data,'test.csv'),x_test_with_stopwords)
y_test_with_stopwords=[label.strip('\n') for label in open(os.path.join(data,'test_label.csv'), 'r')]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train_with_stopwords+x_validate_with_stopwords+x_test_with_stopwords)
train_tok = tokenizer.texts_to_sequences(x_train_with_stopwords)
val_tok   = tokenizer.texts_to_sequences(x_validate_with_stopwords)
test_tok  = tokenizer.texts_to_sequences(x_test_with_stopwords)

x_train = pad_sequences(train_tok, maxlen=26, padding='post', truncating='post')
x_test = pad_sequences(test_tok, maxlen=26, padding='post', truncating='post')
x_val = pad_sequences(val_tok, maxlen=26, padding='post', truncating='post')

embeddings_index = {};

f = open(os.path.join('', 'w2vm.txt'), encoding = "utf-8")
for line in f:
    vals = line.split()
    word = vals[0]
    coeffs = np.asarray(vals[1:])
    embeddings_index[word] = coeffs
    
f.close()


embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, len(coeffs)))

for word, i in tokenizer.word_index.items(): 
    try:
        embeddings_vector = embeddings_index[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector
        
y_train = to_categorical(np.asarray(y_train_with_stopwords))
y_val = to_categorical(np.asarray(y_validate_with_stopwords))
y_test = to_categorical(np.asarray(y_test_with_stopwords))


###RELU MODEL

model0 = Sequential()
model0.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
model0.add(Dense(64, activation='relu'))
model0.add(Flatten())
model0.add(Dense(2, activation = 'softmax'))
model0.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model0.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
model0.save("nn_relu_wo.model")

modelw0 = Sequential()
modelw0.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
modelw0.add(Dense(64, activation='relu', kernel_regularizer= l2(0.001)))
modelw0.add(Flatten())
modelw0.add(Dropout(0.2))
modelw0.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001)))
modelw0.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
modelw0.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
modelw0.save("nn_relu.model")

###SIGMOID MODEL

model1 = Sequential()
model1.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
model1.add(Dense(64, activation='sigmoid'))
model1.add(Flatten())
model1.add(Dense(2, activation = 'softmax'))
model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model1.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
model1.save("nn_sigmoid_wo.model")

modelw1 = Sequential()
modelw1.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
modelw1.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
modelw1.add(Flatten())
modelw1.add(Dropout(0.2))
modelw1.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
modelw1.summary()
modelw1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
modelw1.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
modelw1.save("nn_sigmoid.model")


###TANH MODEL

model2 = Sequential()
model2.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
model2.add(Dense(64, activation='tanh'))
model2.add(Flatten())
model2.add(Dense(2, activation = 'softmax'))
model2.summary()
model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model2.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
model2.save("nn_tanh_wo.model")

modelw2 = Sequential()
modelw2.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=26,
                          weights = [embeddings_matrix], trainable=False))
modelw2.add(Dense(64, activation='tanh', kernel_regularizer= l2(0.001)))
modelw2.add(Flatten())
modelw2.add(Dropout(0.2))
modelw2.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001)))
modelw2.summary()
modelw2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
modelw2.fit(x_train, y_train, batch_size=1024, epochs=10,validation_data=(x_val, y_val))
modelw2.save("nn_tanh.model")

