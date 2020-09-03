import os
import sys
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

f1=sys.argv[1]
f2=sys.argv[2]
f="nn_"+f2+".model"
pos_data=open(f1).readlines()
print("\n")
sentence1=[]
for line in pos_data:
    #print(line)
    sentence = line.strip().split()
    sentence1.append(sentence)
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence1)
test_tok = tokenizer.texts_to_sequences(sentence1)
x_test = pad_sequences(test_tok, maxlen=26, padding='post', truncating='post')
model = keras.models.load_model(f)
preds=model.predict_classes(x_test)
print(preds)