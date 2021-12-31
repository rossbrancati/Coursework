#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:50:39 2021

@author: rossbrancati
"""

#import packages
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score



def main():
    #import training and testing data
    train_X = np.load('../../Data/breast_cancer_data/train_x.npy')
    train_Y = np.load('../../Data/breast_cancer_data/train_y.npy')
    test_X = np.load('../../Data/breast_cancer_data/test_x.npy')
    test_Y = np.load('../../Data/breast_cancer_data/test_y.npy')
    
    ###BASE ESTIMATORS###
    #create a list of the base estimator models to use with the StackingClassifier function
    #Model 1 base estimators: random forest and linear support vector machine
    #estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
    
    #Model 2 base estimators: Random Forest and KNN
    #estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('knn', KNeighborsClassifier(n_neighbors=5))]

    #Model 3: Random Forest and Logistic Regression with final estimator as SVM
    #estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('lr', KNeighborsClassifier(n_neighbors=5))]

    #Model 4: Random Forest and SVM (testing different kernels), final estimator = LinearRegression
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(), SVC(kernel='rbf',random_state=42)))]
    
    
    
    ###STACKED CLASSIFIERS###
    #stack the base estimators with the final estimator
    #Final estimator as logistic regression
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    
    #fit the model with the training data and labels
    clf.fit(train_X, train_Y)
    
    #generate predictions on the test data set
    y_pred = clf.predict(test_X)
    
    #calculate the f1 score
    score = f1_score(test_Y, y_pred)
    
    #print the f1 score
    print('F1 Score:')
    print(score)
    

if __name__ == '__main__':
    main()


#Results from different models
#Model 1: Base: Random Forest and LinearSVC, Final: Logistic Regression
#F1 Score: 0.9589

#Model 2: Base: Random Forest and KNN, Final: Logistic Regression
#F1 Score: 0.9445


#Model 4: Base: Random Forest and linear kernel SVC, final: Logistic Regression
#F1 Score: RBF Kernel: 0.9859, Linear Kernel:0.9722, Polynomial Kernel: 0.9859
