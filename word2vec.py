from gensim.models import Word2Vec
import re
import os
import sys
import nltk

f1=sys.argv[1]

with open(os.path.join(f1,'pos.txt')) as f:
        pos_lines = f.read()
with open(os.path.join(f1,'neg.txt')) as f:
        neg_lines = f.read()
all_lines = pos_lines + neg_lines

def specialchar_remove(arg):
    special_characters=['!','-','"','_','.',',',"#",'$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','\n.','~','\t']
    text_nochar = "".join([char for char in arg if char not in special_characters])
    return text_nochar.lower()
x=specialchar_remove(all_lines)


text      =  all_lines.lower()
sentences =  nltk.sent_tokenize(text)
sentences =  [nltk.word_tokenize(sentence) for sentence in sentences]
w2vmodel=Word2Vec(sentences,min_count=1)

w2vmodel.save("w2v.model")

