import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import BallTree
from collections import Counter

# define F Score function
def f1_score(y_true, y_pred, epsilon=1e-6):
    """
    Function for calculating the F1 score

    Params
    ------
    y_true  : the true labels shaped (N, C), 
              N is the number of datapoints
              C is the number of classes
    y_pred  : the predicted labels, same shape
              as y_true

    Return
    ------
    score   : the F1 score, shaped (N,)

    """
    #calculate TP, TN, FP, FN
    true_positives = sum((y_true == 1) & (y_pred ==1))
    true_negatives = sum((y_true == 0) & (y_pred ==0))
    false_positives = sum((y_true == 0) & (y_pred ==1))
    false_negatives = sum((y_true == 1) & (y_pred ==0))
    
    #calcualte accuracy
    accuracy = ((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives))
    
    #calculate precision, recall, f1_score
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f1_score = 2*((precision*recall)/(precision+recall))
    
    return f1_score, accuracy

class KNN(object):
    """
    The KNN classifier
    """
    #init initializes the class for a specific object
    def __init__(self, n_neighbors):
        #Initialize the KNN with n_neighbors
        self.n_neighbors = n_neighbors
        return 

    def fit(self, x_train, y_train):
        #print("start fitting")
        """
        Fitting the KNN classifier

        Hint:   Build a tree to get neighbors 
                faster at test time
        """
        #set the leaf size to 26 because at most we search for 25 neighbors
        self.y_train = y_train
        self.tree = BallTree(x_train, leaf_size = 100) 
        #print("stop fitting")
        return

    def getKNeighbors(self, x_instance):
        #print('getting kNeighbors')
        """
        Locating the K nearest neighbors of 
        the instance and return
        """
        neighbors = self.tree.query(x_instance, k=self.n_neighbors, return_distance=False, sort_results=True)
        return neighbors
        
    def getResponse(self, neighbors):
        #print("getting Response")
        """
        Helper function to count/vote neighbors' labels
        """
        return Counter(neighbors).most_common(1)[0][0]

    def predict(self, x_test):
        """
        Predicting the test data
        Hint:   Get the K-Neighbors, then generate
                predictions using the labels of the
                neighbors
        """
        #rows,__ = x_test.shape
        y_pred = []
        #print('begin predicting')
        #for i in range(rows):
            #pull first row of test data set to use as query in tree class
            #test_point = x_test[i-1,:]
            #nearest_neighbors = self.getKNeighbors(test_point)
        nearest_neighbors = self.getKNeighbors(x_test)
        length = len(nearest_neighbors)
        
        for rows in range(length):
            X_indices = nearest_neighbors[rows]
            instance_labels = []
            
            for i in X_indices:
                current_label = int(self.y_train[i])
                instance_labels.append(current_label)
            
            pred_value = self.getResponse(instance_labels)
            #store the predicted value in the y_pred vector
            y_pred.append(pred_value)
        
        #print('done predicting')
        return np.asarray(y_pred)


def load_data(data_dir='Data'):
    
    """
    Function for loading the dataset from data_dir
        default: data_dir='Data'
    """
    # Load the data
    X_train = pd.read_csv(f'{data_dir}/X_Train.csv', header=None).values
    y_train = pd.read_csv(f'{data_dir}/Y_Train.csv', header=None).values[:,0]

    X_test = pd.read_csv(f'{data_dir}/X_Test.csv', header=None).values
    y_test = pd.read_csv(f'{data_dir}/Y_Test.csv', header=None).values[:,0]


    return {"train": dict(X=X_train, y=y_train),
            "test": dict(X=X_test, y=y_test)}


def main():
    # Set up the neighbors list
    n_neighbors_list = [3,5,10,20,25]
    time_per_neighbor = []
    metrics = ['accuracy', 'f1_score', 'time']
    results = dict([(key, []) for key in metrics])

    print("\n\n")
    print('Running knn.py')
    
    # Load in data
    data = load_data('/Users/rossbrancati/Documents/Fall_2021_Classes/cs_589/assignments/hw1/Data')
    
    X_train = data['train']['X']
    y_train = data['train']['y']
    print(f"Training data loaded: X shape: {X_train.shape}, y shape: {y_train.shape}")

    X_test = data['test']['X']
    y_test = data['test']['y']
    print(f"Test data loaded: X shape: {X_train.shape}, y shape: {y_train.shape}")

    print(f"\t Class [0/1] ratio: Train [{(y_train==0).sum()}/{(y_train==1).sum()}]")
    print(f"\t Class [0/1] ratio: Test [{(y_test==0).sum()}/{(y_test==1).sum()}]")

    # loop over neighbors_list
    for n_neighbors in n_neighbors_list:
        print("n_neighbors: ", n_neighbors)

        # start timer
        t0 = time.time()

        # instantiate KNN instance
        knn = KNN(n_neighbors=n_neighbors)
        
        # fit the KNN model with X_train, y_train
        knn.fit(X_train, y_train)
        
        # generate predictions on the test set X_test
        y_pred = knn.predict(X_test)
                
        # compute metrics F1_score on the test set with y_test
        F1_score, Accuracy = f1_score(y_test, y_pred, epsilon=1e-6)
        
        # stop timer
        t1 = time.time()
        
        # record time and report metric
        total_time = (t1-t0)*1000
        
        time_per_neighbor.append(total_time)
        results['accuracy'].append(Accuracy)
        results['f1_score'].append(F1_score)
        results['time'].append(total_time)
    
    print(results)
    print("\n Completed!")

    #Plot the time used 
    plt.plot(n_neighbors_list, time_per_neighbor)
    plt.scatter(n_neighbors_list, time_per_neighbor)
    plt.xlabel("K: Number of Neighbors (count)")
    plt.ylabel("Time (milliseconds)")
    plt.title("Time to Complete KNN Algorithm at\nVarious Neighbor Values")
    plt.show() 
    #or plt.savefig()


if __name__ == '__main__':
    main()
