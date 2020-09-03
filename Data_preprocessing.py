#Importing the libraries random and regex
import random
import re
import os
import sys

f1=sys.argv[1]
#Loading the positive and negative datasets using open
pos_data=open(os.path.join(f1,"pos.txt")).read()
neg_data=open(os.path.join(f1,"neg.txt")).read()

#Joining the postive and negative datasets
total_data=pos_data+neg_data

#Function to remove the special characters from our dataset
#The special characters are stored in special_characters 
#The loop checks if the character is not present in special_characters and returns if not present
def specialchar_remove(arg):
    special_characters=['!','-','"','_','.',',',"#",'$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','\n.','~','\t']
    text_nochar = "".join([char for char in arg if char not in special_characters])
    return text_nochar.lower()
x=specialchar_remove(total_data)
final_data_nochar=x.splitlines()


#Function to tokenize the sentences
#Fetches a line from final_data_nochar and splits it based on white spaces
#Returns a tokenized list of list
text_tokenized=list()
def tokenize(arg):
    for line in arg:
        text_split = re.split(' ',line)
        text_tokenized.append(text_split)
    return text_tokenized

#tokenized data
tokenized_data=tokenize(final_data_nochar)    

len(pos_data)
len(neg_data)
l1 = []
x1 = 1
for i in range(400000):
    l1.append(x1)
l2 = []
x2 = 0
for i in range(400000):
    l2.append(x2)
#l2
ltest=[]
ltest1=l1+l2
ltest2=l1+l2

c = list(zip(tokenized_data, ltest1))
random.seed(1)
random.shuffle(c)

tokenized_data, ltest1 = zip(*c)

#function to write to a file
def fn(path,sets):
    with open(path, 'w+', newline='') as f:
        for item in sets:
            f.write(str(item)+'\n')
            
train_len  = int(0.80 * len(tokenized_data))
validate_len = int(0.10 * len(tokenized_data))+train_len
train_c    = ltest1[:train_len]
validate_c = ltest1[train_len:validate_len]
test_c     = ltest1[validate_len:]
train_tk    = tokenized_data[:train_len]
validate_tk = tokenized_data[train_len:validate_len]
test_tk     = tokenized_data[validate_len:]

fn(os.path.join(f1,'train_label.csv'),train_c)
fn(os.path.join(f1,'val_label.csv'),validate_c)
fn(os.path.join(f1,'test_label.csv'),test_c)
fn(os.path.join(f1,'train.csv'),train_tk)
fn(os.path.join(f1,'val.csv'),validate_tk)
fn(os.path.join(f1,'test.csv'),test_tk)
fn(os.path.join(f1,'out.csv'),tokenized_data)

#Function to remove the stopwords
#The stopwords are stored in a variable stopwords
#The loop checks if the word is not present in stopwords and returns if not present
text_no_stop=list()
def stopwords_remove(arg):
    for line in arg:
        text_split = line.split()
        stopwords  = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        stop_w     = [word for word in text_split if word not in stopwords]
        text_no_stop.append(stop_w)
    return text_no_stop
#Variable data_no_stop without stopwords
data_no_stop=stopwords_remove(final_data_nochar)

c1 = list(zip(data_no_stop, ltest2))
random.seed(1)
random.shuffle(c1)

data_no_stop, ltest2 = zip(*c1)

train_len_nostop    = int(0.80 * len(data_no_stop))
validate_len_nostop = int(0.10 * len(data_no_stop))+train_len_nostop

train_label_len_nostop    = int(0.80 * len(data_no_stop))
validate_label_len_nostop = int(0.10 * len(data_no_stop))+train_len_nostop


train_nostop          =   data_no_stop[:train_len_nostop]
validate_nostop       =   data_no_stop[train_len_nostop:validate_len_nostop]
test_nostop           =   data_no_stop[validate_len_nostop:]
train_label_nostop    =   ltest2[:train_label_len_nostop]
validate_label_nostop =   ltest2[train_label_len_nostop:validate_len_nostop]
test_label_nostop     =   ltest2[validate_len_nostop:]

fn(os.path.join(f1,'train_ns.csv'),train_nostop)
fn(os.path.join(f1,'val_ns.csv'),validate_nostop)
fn(os.path.join(f1,'test_ns.csv'),test_nostop)
fn(os.path.join(f1,'train_label_ns.csv'),train_label_nostop)
fn(os.path.join(f1,'val_label_ns.csv'),validate_label_nostop)
fn(os.path.join(f1,'test_label_ns.csv'),test_label_nostop)
fn(os.path.join(f1,'out_ns.csv'),data_no_stop)
