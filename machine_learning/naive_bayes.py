import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

class BagOfWords(object):
    """
    Class for implementing Bag of Words
     for Q1.1
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary_size = vocabulary_size

    def preprocess(self, text):
        """
        text: a string of words from Review Text feature 
        
        Preprocessing of one Review Text
            - convert to lowercase
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        #covnert to lowercase
        lc_text = text.lower()
        #remove punctuation
        no_punc_text = re.sub(r'[^\w\s]', '', lc_text)
        #empty spaces
        no_sp_text = re.sub(' +', ' ', no_punc_text)
        #remove 1 letter words
        no_one_letter = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', no_sp_text)
        split_text = no_one_letter.split()
        return split_text

    def fit(self, X_train):
    #for testing
    #def fit(X_train):
        """
        Building the vocabulary using X_train
        """
        rows = len(X_train)
        #create an empty vocabulary list
        vocab_list = {}
        #loop over all rows
        for row in range(1, rows):
            #assign review text string
            text = X_train[row]
            #split the text
            split_text = self.preprocess(text)
            #loop over all words in the current review text
            for word in split_text:
                #check if the word is in the vocabulary list. If it is not, add to list
                if word not in vocab_list:
                    vocab_list[word] = 1
                #and if it is, add increase the count of that word by 1
                else:
                    vocab_list[word] += 1
        #Now we have to extract the top 10 most frequent words in the vocabulary list
        sorted_vocab_list = sorted(vocab_list, key=vocab_list.get, reverse = True)[:10]
        #put the top 10 most frequent words in alphabetical order
        alph_vocab_list = sorted(sorted_vocab_list)
        #convert from a list to a dictionary, which will make it easy to get indices of strings
        self.alphabetized_vocab_list = dict(zip(alph_vocab_list,range(len(alph_vocab_list))))
        pass
        
    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        #create an empty matrix to store vocab words in strings of X
        representation = np.zeros((100,10))
        #loop over rows of X
        for row in range(len(X)):
            #peprocess row of x
            split_text = self.preprocess(X[row])
            #loop over words in current row, 
            for word in split_text:
                if word in self.alphabetized_vocab_list:
                    word_index = self.alphabetized_vocab_list[word]
                    #add a count to the representation matrix
                    representation[row, word_index] +=1
        #add up the counts of vocabulary in the X array
        summed_representation = sum(representation).reshape((1,-1))
        return self.alphabetized_vocab_list.keys(), summed_representation 

class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.n_classes = n_classes

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """
        #build a bag of words from X_train, which will contain all vocabulary present in the training set
        #create vectorizer
        X_train_array = X_train.toarray()
        #calcualate the priors
        self.prior_negative, self.prior_positive = np.bincount(y_train)/len(y_train)
        
        #caculate the number of words that correspond to each class 
        negative_word_counts = sum(X_train_array[y_train==0])
        positive_word_counts = sum(X_train_array[y_train==1])
        #sum up all of the words in the positive and negative classes
        total_negative_word_count = np.sum(negative_word_counts)   
        total_positive_word_count = np.sum(positive_word_counts)
        #calculate the liklihoods
        negative_probability = (negative_word_counts + self.beta)/(total_negative_word_count + (X_train_array.shape[1] * self.beta))
        positive_probability = (positive_word_counts + self.beta)/(total_positive_word_count + (X_train_array.shape[1] * self.beta))
        
        #concatenate liklihood estimates into one array, where column 0 represents the negative class and column 1 represents the positive class
        self.cond_probabilities = np.stack((negative_probability, positive_probability))
        #cond_probabilities = np.stack((negative_probability, positive_probability))
        return 

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        #create array of the bag of words for X_test
        X_test_array = X_test.toarray()
        #initialize a list of prediction values
        y_pred = []
        #loop over the the array representation of X_test
        for row in X_test_array:
            #get the nonzero indices of the row of X_test array
            non_zero_idx = np.array(np.nonzero(row))
            #and the values of those indices
            non_zero_vals = row[non_zero_idx]
            #calculate the probability that the row belongs to each class, 0 or 1
            neg_post_prob = self.prior_negative * np.prod(self.cond_probabilities[0][non_zero_idx]**non_zero_vals)
            pos_post_prob = self.prior_positive * np.prod(self.cond_probabilities[1][non_zero_idx]**non_zero_vals)
            #get the argument of the higher probability 
            y_pred_instance = np.argmax([neg_post_prob, pos_post_prob])
            #and append the prediction vector
            y_pred.append(y_pred_instance)
         
        y_pred = np.array(y_pred)  
        
        return y_pred

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    #creating a confusion matrix
    y_true = pd.Series(y_true, name = 'Actual')
    y_pred = pd.Series(y_pred, name = 'Predicted')
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return confusion_matrix

def load_data(return_numpy=False):
    """
    Load data

    Params
    ------
    return_numpy:   when true return the representation of Review Text
                    using the CountVectorizer or BagOfWords
                    when false return the Review Text
    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    X_train = pd.read_csv("Data/X_train.csv")['Review Text'].values
    X_valid = pd.read_csv("Data/X_val.csv")['Review Text'].values
    X_test  = pd.read_csv("Data/X_test.csv")['Review Text'].values
    y_train = (pd.read_csv("Data/Y_train.csv")['Sentiment'] == 'Positive').astype(int).values
    y_valid = (pd.read_csv("Data/Y_val.csv")['Sentiment'] == 'Positive').astype(int).values

    if return_numpy:
        # To do (not for Q1.1, used in Q1.3)
        # transform the Review Text into bag of word representation using vectorizer
        # process X_train, X_valid, X_test
        vectorizer = CountVectorizer()
        #fit the vectorizer so that we can get a list of all strings in the vocab list
        global_vectorizer = vectorizer.fit(X_train)
        X_train = global_vectorizer.transform(X_train)
        X_valid = global_vectorizer.transform(X_valid)
        X_test = global_vectorizer.transform(X_test)
        pass

    return X_train, y_train, X_valid, y_valid, X_test


def main():
    #create a dictionary to store the metrics
    metrics = ['Beta','ROC_AUC','F1 score', 'Accuracy', 'Precision', 'Recall']
    results = dict([(key, []) for key in metrics])
    
    #for tuning beta hyperparameter
    #beta_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    beta_list = [1]
    
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
        
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    vocab_list, summed_representation = bow.transform(X_train[100:200])
    
    print('Words in Vocabulary List for Question 1.1:\n', vocab_list)
    print('Word Count of Vocabulary List for Question 1.1:\n', summed_representation)

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)

    # Fit the Naive Bayes model for Q1.3
    for n in beta_list:
    
        nb = NaiveBayes(beta=n)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_valid)
        #for predicting on the test set
        #y_pred = nb.predict(X_test)
        print("Beta: ", n)
        print(confusion_matrix(y_valid, y_pred))

        calculate and store performance metrics
        results['Beta'].append(n)
        results['F1 score'].append(f1_score(y_valid, y_pred))
        results['Precision'].append(precision_score(y_valid, y_pred))
        results['Recall'].append(recall_score(y_valid, y_pred))
        results['ROC_AUC'].append(roc_auc_score(y_valid, y_pred))
        results['Accuracy'].append(accuracy_score(y_valid,y_pred))
    
    print(results)
    
    #Plot beta and ROC_AUC score
    plt.plot(results['Beta'], results['ROC_AUC'])
    plt.scatter(results['Beta'], results['ROC_AUC'])
    plt.xlabel("Beta Values")
    plt.ylabel("ROC AUC Score")
    plt.title("ROC AUC Score for Various\nValues of Beta")
    plt.savefig("Submission/Figures/roc_auc_scores_for_param_combos.png")
    plt.show() 

    return y_pred

if __name__ == '__main__':
     y_pred = main()
