#This script is for Question 3.2 of homework 3. The reason this script is different
#than part 2 is because Question 3.2 is a multiclass classification problem on a 
#separate dataset, so I decided to keep the scripts separate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
#import StratifiedKFold to ensure that we are getting every class in every k fold
from sklearn.model_selection import StratifiedKFold


def get_val_metrics(x_train, y_train, k, clf):
    #create a dictionary to store the validation set metrics
    metrics_keys = ['Precision', 'Recall', 'F1_Score', 'AUC']
    results = dict([(key, []) for key in metrics_keys])
    kf = StratifiedKFold(n_splits=5)

    for train_index, test_index in kf.split(x_train, y_train):
        #assign the training set using the training indices
        train_set = x_train[train_index]
        #assign the validation set using test indices
        validation_set = x_train[test_index]
        #assign the training set labels
        y_train_set = y_train[train_index]
        #assign the validation set labels
        y_val_true = y_train[test_index]
        #fit with training data
        clf.fit(train_set, y_train_set)
        #predict the labels of the training set
        validation_preds = clf.predict(validation_set)
        #append the results dictionary with the evaluation metrics
        results['Precision'].append(metrics.precision_score(y_val_true, validation_preds))
        results['Recall'].append(metrics.recall_score(y_val_true, validation_preds))
        results['F1_Score'].append(metrics.f1_score(y_val_true, validation_preds))
        results['AUC'].append(metrics.roc_auc_score(y_val_true, validation_preds))
        
        #display the confusion matrix, if needed
        #disp = metrics.plot_confusion_matrix(clf, validation_set, validation_preds)
        #disp.figure_.suptitle("Confusion Matrix")
        #plt.show()
        #plt.clf()
        
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
    #calculate evaluation metrics on the training set predictions
    training_precision = metrics.precision_score(y_train, train_predictions, average='weighted')
    training_recall = metrics.recall_score(y_train, train_predictions, average='weighted')
    training_f1_score = metrics.f1_score(y_train, train_predictions, average='weighted')
    training_auc = metrics.roc_auc_score(y_train, train_predictions, average='weighted')
    
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
    
    #load the gene expression data
    train_X, train_y, test_X, test_y = load_data()
    
    #Build multi-class classification models
    #Model 1 - SVM
    #clf = svm.SVC(kernel = 'linear')
    #clf = svm.SVC(kernel = 'poly', degree=3)
    #clf = svm.SVC(kernel = 'rbf')
    
    #Model 2 - Logistic Regression
    clf = LogisticRegression(random_state=0, max_iter=10000, C=0.8)
    
    #Model 3 - Adaboost
    #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate=0.1)
    
    #Model 4 - Random Forest
    #clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')

    #set the number of cross folds
    k = 5
    #call the function to generate the validation and training set metrics
    cv_metrics = get_val_metrics(train_X, train_y, k, clf)
    
    print('Question 3.1 using StratifiedKFolds method to generate folds:')
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
    print('cross_validation script for Q3.1 (StratifiedKFolds completed')
    print('')
    
    
    return cv_metrics, y_preds

if __name__ == '__main__':
   cv_metrics, y_preds = main()



