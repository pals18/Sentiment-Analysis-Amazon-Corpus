1. The condition with stopwords performed better than without stopwords in all the three cases.
 The model without stopwords may have more rare words which may have a huge weightage assigned by 
 Tfidf even when that word is not related to that particular sentence or document and this may result
 in less accuracy. For instance: [‘not’,'a',’good’,’feature’] expresses -ve vs [‘good’,’feature’] without stopwords
 expresses +ve which completely changes the sentiment of the sentence and may result in incorrect predictions.

2. The condition with unigrams+bigrams performed better as compared to unigrams and bigrams alone.
 This is mainly because the model can now be trained on two different text features rather than one and with these 
 features, it can be trained on different combinations of words. The unigram and bigram features has more distinct 
 types of fusion of words which may result in a better prediction. For eg, ['not','a','good','feature','not a',
 'a good','good feature'] may predict better than the unigram -['not','a','good','feature'] and bigram ['not a',
 'a good','good feature']
 

|Stopwords removed|	 Text Features	 | Accuracy(test set) |
|-----------------|------------------|--------------------|
|Yes	          |    unigrams	     |     0.808325		  |
|Yes			  |	   bigrams	     |     0.822475		  |
|Yes	          | unigrams+bigrams |     0.8316375      |
|No	              |    unigrams	     |     0.805275       |
|No	              |    bigrams	     |     0.7870875      |
|No	              | unigrams+bigrams |     0.8233625      |

STEPS TO RUN

1. Run main.py in order to test build all the models and to test the accuracy. 
The pickle files for all the models are already built. so it's not necessary
to run the main.py to run the models again. In order to run, pass an argument 
of the folder which has pos.txt and neg.txt
     
	 python main.py data 

2. In order to test and predict a sentence,run inference.py file which predicts 
the sentence whether it is positive or negative. You are required to pass two arguments
i)name of text file - test_sentences.txt 
ii)Type of Naive Bayes classifier that you want to use :-
mnb_uni, mnb_bi, mnb_uni_bi, mnb_uni_ns, mnb_bi_ns, mnb_uni_bi_ns

python inference.py test_sentences.txt mnb_uni