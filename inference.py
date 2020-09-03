import os
import sys
import pickle
import dill

rf=sys.argv[1]
m1=sys.argv[2]

pos_data=open(rf).readlines()
print("\n")
sentence1=[]
for line in pos_data:
    print(line)
    sentence = line.strip().split()
    sentence1.append(sentence)

mod1=pickle.load(open(m1,'rb'))
count_vect=dill.load(open(m1+"_cv",'rb'))

from sklearn.feature_extraction.text import TfidfTransformer
x_train_count = count_vect.transform(sentence1)
tfidf_transformer=TfidfTransformer()
x_tfidf=tfidf_transformer.fit_transform(x_train_count)
print("\n")
print(mod1.predict(x_tfidf))
print("\n")
print("0 stands for negative")
print("1 stands for positive")