# Reasoning

Yes, most of the words are similar to good positive and bad negative except bad in good and vice versa. This is because
the words are first one hot encoded and they act as the input layer of the neural network, these words are then converted to 
vectors and passed to hidden layer and the output is the context with respect to other words. The words in
the hidden layers which have **similar context with respect to other words have a similar vector values**. Then
based on their coordinates,the distance between the vectors are calculated. The 20 **words which have the smallest
distance or closest vector values to good/bad are evaluated and printed.** eg, quality of the item is good and
quality of the product is bad, have a similar context  *quality of the product*

# 20 most similar words to good 
1. decent
2. great
3. nice
4. terrific
5. fantastic
6. wonderful
7. superb
8. bad
9. fabulous
10. excellent
11. lovely
12. reasonable
13. impressive
14. terrible
15. poor
16. perfect
17. cool
18. awesome
19. clever
20. lousy

# 20 most similar words to bad
1. horrible
2. good
3. terrible
4. awful
5. strange
6. weird
7. funny
8. obvious
9. poor
10. weak
11. lame
12. lousy
13. crappy
14. nasty
15. harsh
16. scary
17. stupid
18. gross
19. fake
20. disappointing

# STEPS TO RUN:

1. The main.py file consists of the implementation of the word2vec model. A w2v model is already generated
   but can be generated again using: 
	python main.py data
2. The 20 most similar words can be found out by running the inference.py file. It takes a text file "test.txt" 
   (which contains two words, good and bad) as an argument.
    python inference.py test.txt 
   
