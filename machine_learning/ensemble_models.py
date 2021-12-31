import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 63
    # of testing samples: 20
    ------
    """
    train_X = np.genfromtxt("../../Data/gene_data/gene_train_x.csv", delimiter= ",")
    train_y = np.genfromtxt("../../Data/gene_data/gene_train_y.csv", delimiter= ",")
    test_X = np.genfromtxt("../../Data/gene_data/gene_test_x.csv", delimiter= ",")
    test_y = np.genfromtxt("../../Data/gene_data/gene_test_y.csv", delimiter= ",")

    return train_X, train_y, test_X, test_y



def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    
    #get the number of features and samples in training data
    train_samples, train_features = train_X.shape
    
    ###Question 2.4###
    #Create empty arrays to store the error
    sqrt_error = np.array([])
    n_features_error = np.array([])
    n_features_d10_error = np.array([])
    N = 150 # Each part will be tried with 1 to 150 estimators
    #loop over the number of estimators
    x_axis = np.array(range(N))
    for i in range(N):
        
        #Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)
        clf_sqrt = RandomForestClassifier(n_estimators=i+1, max_features='sqrt')
        #Fit the model with training data
        clf_sqrt.fit(train_X, train_y)
        #Use built in method to get the accuracy (or score)
        sqrt_accuracy = clf_sqrt.score(test_X, test_y)
        #append the error vector with the error for the current model
        sqrt_error = np.append(sqrt_error, 1-sqrt_accuracy)
        
        #Train RF with m = n_features recording the errors (errors will be of size 150)
        clf_n_features = RandomForestClassifier(n_estimators=i+1, max_features=int(train_features))
        #Fit the model with training data
        clf_n_features.fit(train_X, train_y)
        #Use built in method to get the accuracy (or score)
        n_features_accuracy = clf_n_features.score(test_X, test_y)
        #append the error vector with the error for the current model
        n_features_error = np.append(n_features_error, 1-n_features_accuracy)
        
        #Train RF with m = n_features/10 recording the errors (errors will be of size 150)
        clf_n_features_d10 = RandomForestClassifier(n_estimators=i+1, max_features=int(train_features/10))
        #Fit the model with training data
        clf_n_features_d10.fit(train_X, train_y)
        #Use built in method to get the accuracy (or score)
        n_features_d10_accuracy = clf_n_features_d10.score(test_X, test_y)
        #append the error vector with the error for the current model
        n_features_d10_error = np.append(n_features_d10_error, 1-n_features_d10_accuracy)
    
    #plot the Random Forest results
    #plot error for n_estimars = sqrt(n_features)
    plt.plot(x_axis, sqrt_error, label = 'Square Root of N_Features')
    #plot error for n_estimators = n_features
    plt.plot(x_axis, n_features_error, label = 'N_Features')
    #plot error for n_estimators = n_features/10
    plt.plot(x_axis, n_features_d10_error, label = 'N_Features/10')
    plt.title('Random Forest Classifier\nError for Various Numbers of Trees Ranging from 1 to 150')
    plt.xlabel('Number of Trees (count)')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("../Figures/random_forest_error.png")
    plt.show()
    plt.clf()
    
    ###Question 2.6###
    #create empty vectors to store error in
    clf_1_error = np.array([])
    clf_3_error = np.array([])
    clf_5_error = np.array([])
    N = 150 # Each part will be tried with 1 to 150 estimators
    #loop over the number of estimators
    x_axis = np.array(range(N))
    for i in range(N):
    
        # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)
        clf_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=i+1, learning_rate=0.1)
        #Fit the model with training data
        clf_1.fit(train_X, train_y)
        #Use built in method to calculate accuracy (or score)
        clf_1_accuracy = clf_1.score(test_X, test_y)
        #append the error vector with the error for the current model
        clf_1_error = np.append(clf_1_error, 1-clf_1_accuracy)
        
        # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)
        clf_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=i+1, learning_rate=0.1)
        #Fit the model with training data
        clf_3.fit(train_X, train_y)
        #Use built in method to calculate accuracy (or score)
        clf_3_accuracy = clf_3.score(test_X, test_y)
        #append the error vector with the error for the current model
        clf_3_error = np.append(clf_3_error, 1-clf_3_accuracy)
    
        # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)
        clf_5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=i+1, learning_rate=0.1)
        #Fit the model with training data
        clf_5.fit(train_X, train_y)
        #Use built in method to calculate accuracy (or score)
        clf_5_accuracy = clf_5.score(test_X, test_y)
        #append the error vector with the error for the current model
        clf_5_error = np.append(clf_5_error, 1-clf_5_accuracy)
        
    # plot the adaboost results
    #plot error for DecisionTreeClassifier max_depth = 1
    plt.plot(x_axis, clf_1_error, label = 'max_depth = 1')
    #plot error for DecisionTreeClassifier max_depth = 3
    plt.plot(x_axis, clf_3_error, label = 'max_depth = 3')
    #plot error for DecisionTreeClassifier max_depth = 5
    plt.plot(x_axis, clf_5_error, label = 'max_depth = 5')
    plt.title('AdaBoost Classifier\nError for Various Numbers of Trees Ranging from 1 to 150')
    plt.xlabel('Number of Trees (count)')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("../Figures/ada_boost_error.png")
    plt.show()
    
    print('Template_Ensembles script for Q2 completed.\nSee Report or Submission/Figures for error plots')
    print('----------')
    print('')

if __name__ == '__main__':
    main()
