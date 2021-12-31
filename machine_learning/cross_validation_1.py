#This script is for Question 3.1 of homework 3. The reason this script is different
#than part 2 is because Question 3.2 is a multiclass classification problem on a 
#separate dataset, so I decided to keep the scripts separate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing math to use floor() function
import math
#import metrics
from sklearn import metrics
#import svm classifier
from sklearn import svm
#import logistic regression classifier
from sklearn.linear_model import LogisticRegression
#import adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
#import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#import the one hot endcoder
from sklearn.preprocessing import OneHotEncoder


#create a function to generate k blocks of random samples from training data
def cross_val_split(x_train, k):
    """
    Split the training data into k blocks
    
    Params
    ------
    x_train     :   (ndarray, shape = (n_samples, n_features)):
                    Training input matrix where each row is a feature vector.
    k           :   number of cross-folds in which the data is split into
    
    Return
    ------
    blocks     :   an list containing 5 arrays with randomly selected blocks
    """
    #randomly partition the data into k blocks
    #get the number of samples in the training data
    n_samples, n_features = x_train.shape
    #calculate the number of samples in each block, rounding the number of samples per block down to the nearest integer
    block_samples = math.floor(n_samples/k)
    
    #generate an array with the indices of x_train
    x_train_indices = list(range(n_samples))
    
    #create a list to store the blocks in
    blocks = list()
    
    #divide the training data into k number of blocks
    for i in range(k):
        #create a list of the current block indices
        current_block = list()
        #randomly select a number of samples for the current block
        current_block = list(np.random.choice(x_train_indices, size=block_samples, replace=False))
        #append the blocks list with the indices of the current block
        blocks.append(current_block)
        #delete the indices of the current block from the 
        x_train_indices = list(set(x_train_indices) - set(current_block))
    
    return blocks

def get_val_metrics(x_train, y_train, blocks, k, clf):
    #create a dictionary to store the validation set metrics
    metrics_keys = ['Precision', 'Recall', 'F1_Score', 'AUC']
    results = dict([(key, []) for key in metrics_keys])
    
    #loop over all combinations of training and validation set, calculate metrics, and store metrics in dictionary
    for fold in range(k):
        #assign the validation set to the nth iteration in the loop
        validation_set = x_train[blocks[fold]]
        #based on which iteration of the loop, assign the training indices
        train_indices = np.hstack(blocks[:fold] + blocks[fold+1:])
        #assign the training set using the training indices
        train_set = x_train[train_indices]
        #assign the validation set labels
        y_val_true = y_train[blocks[fold]]
        #assign the training set labels
        y_train_set = y_train[train_indices]
        #fit with training data
        clf.fit(train_set, y_train_set)
        #predict the labels of the training set
        validation_preds = clf.predict(validation_set)
        #append the dictionary with the scores
        results['Precision'].append(metrics.precision_score(y_val_true, validation_preds))
        results['Recall'].append(metrics.recall_score(y_val_true, validation_preds))
        results['F1_Score'].append(metrics.f1_score(y_val_true, validation_preds))
        results['AUC'].append(metrics.roc_auc_score(y_val_true, validation_preds))

    #calculate the mean of the validation data metrics for each cross fold set    
    validation_precision = np.mean(results['Precision'])
    validation_recall = np.mean(results['Recall'])
    validation_f1_score = np.mean(results['F1_Score'])
    validation_auc = np.mean(results['AUC'])
    
    ###On full training set###
    #fit model on full training data
    clf.fit(x_train, y_train)
    #predict on full training data
    train_predictions = clf.predict(x_train)
    #calculate metrics on full training set
    training_precision = metrics.precision_score(y_train, train_predictions)
    training_recall = metrics.recall_score(y_train, train_predictions)
    training_f1_score = metrics.f1_score(y_train, train_predictions)
    training_auc = metrics.roc_auc_score(y_train, train_predictions)

    return [[validation_precision, validation_recall, validation_f1_score, validation_auc], [training_precision, training_recall, training_f1_score, training_auc]]
        

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    train_X = np.load(f"../../Data/breast_cancer_data/train_x.npy")
    train_y = np.load(f"../../Data/breast_cancer_data/train_y.npy")
    test_X = np.load(f"../../Data/breast_cancer_data/test_x.npy")
    test_y = np.load(f"../../Data/breast_cancer_data/test_y.npy")

    return train_X, train_y, test_X, test_y
        
def main():
    # Set the seed for numpy random number generator
    # so we'll have consistent results at each run
    np.random.seed(0)
    
    #load the breast cancer data
    train_X, train_y, test_X, test_y = load_data()
    
    #build a classification model - SVM
    #Model 1 - SVM
    #binary classification
    #clf = svm.SVC(kernel = 'linear')
   
    #Model 2 - Logistic Regression
    #binary classification
    clf = LogisticRegression(random_state=0, max_iter=10000, C=0.8)
    
    #Model 3 - Adaboost
    #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate=0.1)

    #set the number of cross folds
    k = 5
    #generate the array of blocks
    blocks = cross_val_split(train_X, k)
    #call the function to generate the validation and training set metrics
    cv_metrics = get_val_metrics(train_X, train_y, blocks, k, clf)
    
    print('Question 3.1 using random sampling method to generate folds:')
    print('')
    print('Cross Validation and Training Set Evaluation Metrics')
    print(cv_metrics)
    print('')
    
    #predict the test set labels
    clf.fit(train_X, train_y)
    y_preds = clf.predict(test_X)
    #for a sanity check, calculate the evaluation metrics on the testing set
    test_precision = metrics.precision_score(test_y, y_preds)
    test_recall = metrics.recall_score(test_y, y_preds)
    test_f1_score = metrics.f1_score(test_y, y_preds)
    test_auc = metrics.roc_auc_score(test_y, y_preds)
    #and print the metrics
    print('Evaluation Metrics for Breast Cancer Test Set:')
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1_score)
    print("Test ROC_AUC:", test_auc)
    print('cross_validation script for Q3.1 (random sampling) completed')
    print('')
    
    
    return y_preds, cv_metrics

if __name__ == '__main__':
   y_preds, cv_metrics = main()