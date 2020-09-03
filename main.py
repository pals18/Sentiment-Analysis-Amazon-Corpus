import pickle
import dill
import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#### Reading files with stopwords ####
data=sys.argv[1]
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


#### Reading files without stopwords ####
x_train_no_stopwords = []
y_train_no_stopwords = []
x_validate_no_stopwords = []
y_validate_no_stopwords = []
x_test_no_stopwords = []
y_test_no_stopwords = []

x_train_no_stopwords=readfile(os.path.join(data,'train_ns.csv'), x_train_no_stopwords)
y_train_no_stopwords=[label.strip('\n') for label in open(os.path.join(data,'train_label_ns.csv'), 'r')]

x_validate_no_stopwords=readfile(os.path.join(data,'val_ns.csv'), x_validate_no_stopwords)
y_validate_no_stopwords=[label.strip('\n') for label in open(os.path.join(data,'val_label_ns.csv'), 'r')]

x_test_no_stopwords=readfile(os.path.join(data,'test_ns.csv'),x_test_no_stopwords)
y_test_no_stopwords=[label.strip('\n') for label in open(os.path.join(data,'test_label_ns.csv'), 'r')]

###### With stopwords ########

#unigrams
def unigram_model(x_train_model, y_train_model):
    count_vector = CountVectorizer(ngram_range=(1,1),tokenizer=lambda doc: doc, lowercase=False)
    count_x_train = count_vector.fit_transform(x_train_model)
    tfidf_transformer = TfidfTransformer()
    tfidf_x_train = tfidf_transformer.fit_transform(count_x_train)
    clf = MultinomialNB().fit(tfidf_x_train, y_train_model)
    return count_vector, tfidf_transformer, clf


#NaiveBayessModel_bigrams
def bigram_model(x_train_model, y_train_model):
    count_vector_bi = CountVectorizer(ngram_range=(2,2),tokenizer=lambda doc: doc, lowercase=False)
    count_x_train_bi = count_vector_bi.fit_transform(x_train_model)
    tfidf_transformer_bi = TfidfTransformer()
    tfidf_x_train_bi = tfidf_transformer_bi.fit_transform(count_x_train_bi)
    clf_bi = MultinomialNB().fit(tfidf_x_train_bi, y_train_model)
    return count_vector_bi, tfidf_transformer_bi, clf_bi


#NaiveBayessModel_Unigrams&bigrams_with_stopwords
def unibi_model(x_train_model, y_train_model):
    count_vector_n = CountVectorizer(ngram_range=(1,2),tokenizer=lambda doc: doc, lowercase=False)
    count_x_train_n = count_vector_n.fit_transform(x_train_model)
    tfidf_transformer_n = TfidfTransformer()
    tfidf_x_train_n = tfidf_transformer_n.fit_transform(count_x_train_n)
    clf_n = MultinomialNB().fit(tfidf_x_train_n, y_train_model)
    return count_vector_n, tfidf_transformer_n, clf_n



def model_evaluation(x_set, y_set, clf, count_vector, tfidf_transformer):
    count_x = count_vector.transform(x_set)
    tfidf_x = tfidf_transformer.transform(count_x)
    prediction = clf.predict(tfidf_x)
    accuracy_sco= accuracy_score(y_set, prediction)
    print("Accuracy=",accuracy_sco)
    precision_sco= precision_score(y_set, prediction,pos_label='1')
    print("precision=",precision_sco)
    recall_sco= recall_score(y_set, prediction,pos_label='1')
    print("recall=",recall_sco)
    f1_sco= f1_score(y_set, prediction,pos_label='1')
    print("f1=",f1_sco)
    return accuracy_sco,prediction,precision_sco,recall_sco,f1_sco



#NaiveBayessModel_unigrams_with_stopwords Training
count_vector, tfidf_transformer,clf = unigram_model(x_train_with_stopwords, y_train_with_stopwords)

dill.dump(count_vector,open('mnb_uni_cv','wb'))
#pickle.dump(tfidf_transformer,open('tfid_transformer','wb'))
pickle.dump(clf,open('mnb_uni','wb'))


##NaiveBayessModel_unigrams_with_stopwords Validation
print('Model with unigrams and with stopwords: Validation')
accuracy, prediction, precision, recall, f1 = model_evaluation(x_validate_with_stopwords, y_validate_with_stopwords, clf, count_vector, tfidf_transformer)

#NaiveBayessModel_unigrams_with_stopwords: Testing
print('Model with unigrams and with stopwords: Testing')
accuracy_uni_with_stop, prediction, precision, recall, f1 = model_evaluation(x_test_with_stopwords, y_test_with_stopwords, clf, count_vector, tfidf_transformer)


#NaiveBayessModel_bigrams_with_stopwords Training 
count_vector_bi, tfidf_transformer_bi, clf_bi = bigram_model(x_train_with_stopwords, y_train_with_stopwords)

dill.dump(count_vector_bi,open('mnb_bi_cv','wb'))
#pickle.dump(tfidf_transformer_bi,open('tfid_transformer_bi','wb'))
pickle.dump(clf_bi,open('mnb_bi','wb'))

#NaiveBayessModel_bigrams_with_stopwords Validation
print('Model with bigrams and with stopwords: Validation')
accuracy,preds,precision,recall,f1 = model_evaluation(x_validate_with_stopwords, y_validate_with_stopwords, clf_bi, count_vector_bi, tfidf_transformer_bi)

#NaiveBayessModel_bigram_with_stopwordss Testing
print('Model with bigrams and with stopwords: Testing')
accuracy_bi_with_stop,preds,precision,recall,f1 = model_evaluation(x_test_with_stopwords, y_test_with_stopwords, clf_bi, count_vector_bi, tfidf_transformer_bi)



#NaiveBayessModel_Unigrams&bigrams_with_stopwords Training
count_vector_n, tfidf_transformer_n, clf_n = unibi_model(x_train_with_stopwords, y_train_with_stopwords)

dill.dump(count_vector_n,open('mnb_uni_bi_cv','wb'))
pickle.dump(clf_n,open('mnb_uni_bi','wb'))


#NaiveBayessModel_Unigrams&bigrams_with_stopwords Validation
print('Model with Unigrams & bigrams and with stopwords: Validation')
accuracy,prediction,precision,recall,f1= model_evaluation(x_validate_with_stopwords, y_validate_with_stopwords, clf_n, count_vector_n, tfidf_transformer_n)


#NaiveBayessModel_Unigrams&bigrams_with_stopwords Validation Confusion matrix
#confusion_matrix(y_validate_with_stopwords,prediction)


#NaiveBayessModel_Unigrams&bigrams_with_stopwords Testing
print('Model with Unigrams & bigrams and with stopwords: Testing')
accuracy_n_with_stop,preds,precision,recall,f1= model_evaluation(x_test_with_stopwords, y_test_with_stopwords, clf_n, count_vector_n, tfidf_transformer_n)

#NaiveBayessModel_bigrams_with_stopwords Testing Confusion matrix
#confusion_matrix(y_test_with_stopwords,prediction)


#NaiveBayessModel_unigramss_no_stopwords Training
count_vector, tfidf_transformer,clf  = unigram_model(x_train_no_stopwords, y_train_no_stopwords)

dill.dump(count_vector,open('mnb_uni_ns_cv','wb'))
pickle.dump(clf,open('mnb_uni_ns','wb'))

##NaiveBayessModel_unigrams_no_stopwords Validation
print('Model with Unigrams and no stopwords: Validation')
accuracy, prediction, precision, recall, f1 = model_evaluation(x_validate_no_stopwords, y_validate_no_stopwords, clf, count_vector, tfidf_transformer)


#NaiveBayessModel_unigrams_no_stopwords Testing
print('Model with Unigrams and no stopwords: Testing')
accuracy_u_no_stop, prediction, precision, recall, f1 = model_evaluation(x_test_no_stopwords, y_test_no_stopwords, clf, count_vector, tfidf_transformer)

#NaiveBayessModel_unigrams_no_stopwords Validation Confusion Matrix
#confusion_matrix(y_validate_no_stopwords,prediction)


##NaiveBayessModel_unigrams_no_stopwords Testing Confusion Matrix
#confusion_matrix(y_test_no_stopwords,prediction)

#NaiveBayessModel_bigrams_no_stopwords Training 
count_vector_bi, tfidf_transformer_bi,clf_bi = bigram_model(x_train_no_stopwords, y_train_no_stopwords)

dill.dump(count_vector_bi,open('mnb_bi_ns_cv','wb'))
pickle.dump(clf_bi,open('mnb_bi_ns','wb'))

#NaiveBayessModel_bigrams_no_stopwords Validation
print('Model with bigrams and no stopwords: Validation')
accuracy,prediction,precision,recall,f1 = model_evaluation(x_validate_no_stopwords, y_validate_no_stopwords, clf_bi, count_vector_bi, tfidf_transformer_bi)
#NaiveBayessModel_bigrams_no_stopwords Validation Confusion matrix
#confusion_matrix(y_validate_no_stopwords,prediction)

#NaiveBayessModel_bigram_no_stopwords Testing
print('Model with bigrams and no stopwords: Testing')
accuracy_bi_no_stop,prediction,precision,recall,f1 = model_evaluation(x_test_no_stopwords, y_test_no_stopwords, clf_bi, count_vector_bi, tfidf_transformer_bi)

#NaiveBayessModel_bigrams_no_stopwords Testing Confusion matrix
#confusion_matrix(y_test_no_stopwords,prediction)


#NaiveBayessModel_Unigrams&bigrams_no_stopwords Training
count_vector_n, tfidf_transformer_n, clf_n = unibi_model(x_train_no_stopwords, y_train_no_stopwords)

dill.dump(count_vector_n,open('mnb_uni_bi_ns_cv','wb'))
pickle.dump(clf_n,open('mnb_uni_bi_ns','wb'))


#NaiveBayessModel_Unigrams&bigrams_no_stopwords Validation
print('Model with unigrams & bigrams and no stopwords: Validation')
accuracy,prediction,precision,recall,f1= model_evaluation(x_validate_no_stopwords, y_validate_no_stopwords, clf_n, count_vector_n, tfidf_transformer_n)

#NaiveBayessModel_Unigrams&bigrams_no_stopwords Validation Confusion matrix
confusion_matrix(y_validate_no_stopwords,prediction)
#NaiveBayessModel_Unigrams&bigrams_no_stopwords Testing
print('Model with unigrams & bigrams and no stopwords: Testing')
accuracy_n_no_stop,prediction,precision,recall,f1= model_evaluation(x_test_no_stopwords, y_test_no_stopwords, clf_n, count_vector_n, tfidf_transformer_n)


#NaiveBayessModel_bigrams_no_stopwords Testing Confusion matrix
#confusion_matrix(y_test_no_stopwords,preds)

print('Stopwords_removed text_features Accuracy(test set)')
print('Yes \t \t  unigrams \t    ',accuracy_uni_with_stop)
print('Yes \t \t  bigrams \t    ',accuracy_bi_with_stop)
print('Yes \t \t  unigrams+bigrams   ',accuracy_n_with_stop)
print('No \t \t  unigrams \t    ',accuracy_u_no_stop)
print('No \t \t  bigrams \t    ',accuracy_bi_no_stop)
print('No \t \t  unigrams+bigrams   ',accuracy_n_no_stop)


