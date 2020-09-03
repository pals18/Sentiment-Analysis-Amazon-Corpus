from gensim.models import Word2Vec
import os
import sys

rf=sys.argv[1]
w2v=Word2Vec.load('w2v.model')

pos_data=open(rf).readlines()
sentence1=[]
for line in pos_data:
    sentence = line.strip().split()
    sentence1.append(sentence)

for word in sentence1:
    print("The most similar words to",''.join(word),"are:")
    similar1=w2v.wv.most_similar(word,topn=20)
    for word in similar1:
        print(word)
    print("\n")


