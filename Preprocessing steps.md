  
# MSCI 641 Assignment 1

## STEPS TO RUN

1. Simply Run main.py data(The path is not hard coded, pass data as a parameter)

2. All the output, train, test and validation files are stored in a folder called data.

3. The following files are created in a folder called "data" uploaded on learn under Assignment 2 data along with label files

   Out_.csv     -> Tokenized data with stopwords
   
   Out_ns.csv       -> Tokenized data without stopwords
   
   test.csv    -> Test data with stopwords
   
   train.csv   -> Train data with stopwords
   
   val.csv-> Validation data with stopwords
   
   test_ns.csv      -> Test data without stopwords
   
   train_ns.csv     -> Train data without stopwords
   
   val_ns.csv  -> Validation data without stopwords
   

## Implementation of Main.py 

1. The first step is to import the libraries. For this assignment, regex and random have been used.

2. The next step is to load the datasets namely pos.txt and neg.txt which are present in our RawDatasets which been done using open and reading.

3. A function to remove the special characters has been defined which contains a list of special characters and checks from the given list of characters if that character is present in our line and removes it if present and is further splitted based on new lines.

4. A function is created in order to tokenize the sentence, which splits the line based on the white spaces and stores them in the form of a list.

5. The text obtained after tokenization is then shuffled using random.shuffle.

6. The three files train, validate and test are then created and the tokenized data is split into these three files based on the splitting percentages mentioned.

7. A function to remove the stopwords was defined which contains a list of stopwords and checks from the given list of stopwords if that word is present in our line and removes it if present. 

8. The text obtained after tokenization and removing the stopwords is then shuffled using random.shuffle.

9. The three files train, validate and test are then created and the data after removing the stopwords is split into these three files based on the splitting percentages mentioned.
