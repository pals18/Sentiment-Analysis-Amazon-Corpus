## REASONING

The model with relu as the activation function and without l2 norm regularization and dropout has the highest accuracy(79.54%).
This is because there is no vanishing gradient in relu whereas tanh and sigmoid are more likely to saturate and will be difficult to 
know the direction in which the parameters should move in order to improve the loss function as the activation function considers 
the effects of different parameters and transforms the data by squashing it after which it to decides the neuron it will forward 
to the next layer.L2 norm regularization term is the sum of square of all weights of the features. Although the accuracy has reduced after
regularization, l2 norm regularization helps reducing overfitting and variance by penalizing for large weights and to generalize
the model on the data it has not seen before by adding a term to the loss function.After adding the dropout layer there was a slight
drop in the accuracy as the dropout layer at the time of training drops out certain units from the layers for each of the differnt 
models which helps in preventing overfitting of the model As the during the training process the neruons may develop dependency on
each other which may reduce the individual power of each neuron thus leading to further overfitting and getting less accuracy on unseen data.



| Activation Fn  |Without l2 & dropout  |  With l2=0.001 & dropout=0.2  |  With l2=0.01 & dropout=0.4  |With L2 =0.001 | With dropout =0.4|
|----------------|----------------------|-------------------------------|------------------------------|---------------|------------------|
|Relu            |    79.54             |           78.28               |               73.48          |      76.19    |       78.18      |
|Sigmoid		 |    77.87             |           74.83               |               61.28          |      75.42    |       76.85      |
|tanh		 	 |     78.56            |           76.25               |               72.69          |      77.13    |       77.93      |

## STEPS TO RUN

1. Run main.py in order to test build all the models and to test the accuracy. 
The model files for all the models are already built. so it's not necessary
to run the main.py to run the models again. In order to run, pass an argument 
of the folder which has pos.txt and neg.txt
     
	 python main.py data 

2. In order to test and predict a sentence,run inference.py file which predicts 
the sentence whether it is positive or negative. You are required to pass two arguments
i)name of text file - test_sentences.txt 
ii)Type of activation function to be used without specifying the path for eg: relu, sigmoid, tanh

python inference.py test_sentences.txt relu

NOTE: The word embeddings are present in a text file w2vm.txt
