import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import svm


class SVM(object):
    """
    SVM Support Vector Machine
    with sub-gradient descent training.
    """
    def __init__(self, C=1):
        """
        Params
        ------
        n_features  : number of features (D)
        C           : regularization parameter (default: 1)
        """
        self.C = C

    def fit(self, X, y, lr=0.002, iterations=10000):
        """
        Fit the model using the training data.

        Params
        ------
        X           :   (ndarray, shape = (n_samples, n_features)):
                        Training input matrix where each row is a feature vector.
        y           :   (ndarray, shape = (n_samples,)):
                        Training target. Each entry is either -1 or 1.
        lr          :   learning rate
        iterations  :   number of iterations
        """
        n_samples, n_features = X.shape

        # Initialize the parameters wb
        self.wb =np.zeros((n_features+1))
        # initialize any container needed for save results during training
        obj = 0
        #initialize an objective list to see if model is converging
        obj_list = []
        for i in range(iterations):
            # calculate learning rate with iteration number i
            lr_t = lr/(np.sqrt(i+1))
            # calculate the subgradients
            subgrad = self.subgradient(self.wb, X, y)
            # update the parameter wb with the gradients
            self.wb = self.wb - (lr_t*subgrad)
            # calculate the new objective function value
            current_obj = self.objective(self.wb, X, y)
            # compare the current objective function value with the saved best value
            # update the best value if the current one is better
            if current_obj < obj:
                obj = current_obj
            # Logging
            #if (i+1)%1000 == 0:
            #    print(f"Training step {i+1:6d}: LearningRate[{lr_t:.7f}], Objective[{f:.7f}]")
            #append the objective list with the current_obj value
            obj_list.append(current_obj)
        # Save the best parameter found during training 
        #I'm not really sure why we have to save the parameters again if the most previous update of the parameters
        #was out last version of them
        self.wb = self.wb
        # optinally return recorded objective function values (for debugging purpose) to see
        # if the model is converging
        return obj_list

    @staticmethod
    def unpack_wb(wb, n_features):
        """
        Unpack wb into w and b
        """
        w = wb[:n_features]
        b = wb[-1]

        return (w,b)

    def g(self, X, wb):
        """
        Helper function for g(x) = WX+b
        """
        n_samples, n_features = X.shape

        w,b = self.unpack_wb(wb, n_features)
        gx = np.dot(w, X.T) + b

        return gx

    def hinge_loss(self, X, y, wb):
        """
        Hinge loss for max(0, 1 - y(Wx+b))

        Params
        ------

        Return
        ------
        hinge_loss
        hinge_loss_mask
        """
        hinge = 1 - y*(self.g(X, wb))
        hinge_mask = (hinge > 0).astype(np.int)
        hinge = hinge * hinge_mask

        return hinge, hinge_mask

    def objective(self, wb, X, y):
        """
        Compute the objective function for the SVM.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        obj (float): value of the objective function evaluated on X and y.
        """
        n_samples, n_features = X.shape
        #Hyperparameter C can be multiplied by objective function at different step
        
        # Calculate the objective function value 
        # Be careful with the hinge loss function
        hinge_loss, hinge_mask = self.hinge_loss(X, y, wb)
        #calcualte w
        w = wb[:n_features]
        #calculate the objective function
        obj = np.sum(hinge_loss) + np.dot(w,w)

        return obj

    def subgradient(self, wb, X, y):
        """
        Compute the subgradient of the objective function.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model 
        """
        n_samples, n_features = X.shape
        w, b = self.unpack_wb(wb, n_features)

        # Retrieve the hinge mask
        hinge, hinge_mask = self.hinge_loss(X, y, wb)

        ## Cast hinge_mask on y to make y to be 0 where hinge loss is 0
        cast_y = - hinge_mask * y

        # Cast the X with an addtional feature with 1s for b gradients: -y
        cast_X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

        # Calculate the gradients for w and b in hinge loss term
        grad = self.C * np.dot(cast_y, cast_X)

        # Calculate the gradients for regularization term
        grad_add = np.append(2*w,0)
        
        # Add the two terms together
        subgrad = grad+grad_add

        return subgrad

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Params
        ------
        X   :    (ndarray, shape = (n_samples, n_features)): test data

        Return
        ------
        y   :   (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        n_samples, n_features = X.shape
        
        #initialize y_pred list
        y_pred = np.array([])
        # retrieve the parameters wb
        params_wb = self.unpack_wb(self.wb, n_features)
        w = params_wb[0]
        b = params_wb[1]
        # calculate the predictions
        for i in range(n_samples):
            current_y = np.sign(np.dot(w,X[i]))
            y_pred = np.append(y_pred, current_y)
        # return the predictions
        return y_pred

    def get_params(self):
        """
        Get the model parameters.

        Params
        ------
        None

        Return
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        return (self.w, self.b)

    def set_params(self, w, b):
        """
        Set the model parameters.

        Params
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        self.w = w
        self.b = b

    

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


def plot_decision_boundary(clf, X, y, title):
    """
    Helper function for plotting the decision boundary

    Params
    ------
    clf     :   The trained SVM classifier
    X       :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
    y       :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
    title   :   The title of the plot

    Return
    ------
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    
    meshed_data = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(meshed_data)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.savefig("../Figures/"+title)
    plt.show()



def main():
    # Set the seed for numpy random number generator
    # so we'll have consistent results at each run
    np.random.seed(0)
    
    ###QUESTION 1.3###
    #For Question 1.3 and 1.4
    #create a dictionary to store the f1 score, precision, and recall
    metrics_keys = ['Train_F1_Score', 'Train_Precision', 'Train_Recall', 'Test_F1_Score', 'Test_Precision', 'Test_Recall']
    results = dict([(key, []) for key in metrics_keys])
    
    train_X, train_y, test_X, test_y = load_data()

    #For Question 1.3
    clf = SVM(C=1)
    objs = clf.fit(train_X, train_y, lr=0.002, iterations=10000)

    train_preds  = clf.predict(train_X)
    test_preds  = clf.predict(test_X)

    #For Question 1.3
    print('Question 1:')
    print('Results for Question 1.3:\n')
    print('1st column is testing set scores & 2nd column is training set scores')
    print(f"F1_score: [{metrics.f1_score(test_y, test_preds)}, {metrics.f1_score(train_y, train_preds)}]")
    print(f"Precision: [{metrics.precision_score(test_y, test_preds)}, {metrics.precision_score(train_y, train_preds)}]")
    print(f"Recall: [{metrics.recall_score(test_y, test_preds)}, {metrics.recall_score(train_y, train_preds)}]")
    print('')
    
    ###QUESTION 1.4###
    #Linear Kernel
    #generate SVM class for training data
    clf = svm.SVC(kernel = 'linear')
    #fit with training data
    clf.fit(train_X, train_y)
    #predict with training data
    train_preds = clf.predict(train_X)
    #predict with test data
    test_preds = clf.predict(test_X)
    print('Results for Question 1.4\n')
    print('Linear Kernel')
    print('1st column is testing set scores & 2nd column is training set scores')
    print(f"F1_score: [{metrics.f1_score(test_y, test_preds)}, {metrics.f1_score(train_y, train_preds)}]")
    print(f"Precision: [{metrics.precision_score(test_y, test_preds)}, {metrics.precision_score(train_y, train_preds)}]")
    print(f"Recall: [{metrics.recall_score(test_y, test_preds)}, {metrics.recall_score(train_y, train_preds)}]")
    print('')
    
    #3rd Degree Polynomial Kernel
    #generate SVM class for training data
    clf = svm.SVC(kernel = 'poly', degree=3)
    #fit with training data
    clf.fit(train_X, train_y)
    #predict with training data
    train_preds = clf.predict(train_X)
    #predict with test data
    test_preds = clf.predict(test_X)
    print('Polynomial Kernel with Degree=3')
    print('1st column is testing set scores & 2nd column is training set scores')
    print(f"F1_score: [{metrics.f1_score(test_y, test_preds)}, {metrics.f1_score(train_y, train_preds)}]")
    print(f"Precision: [{metrics.precision_score(test_y, test_preds)}, {metrics.precision_score(train_y, train_preds)}]")
    print(f"Recall: [{metrics.recall_score(test_y, test_preds)}, {metrics.recall_score(train_y, train_preds)}]")
    print('')
    
    #RBF Kernel
    #generate SVM class for training data
    clf = svm.SVC(kernel = 'rbf')
    #fit with training data
    clf.fit(train_X, train_y)
    #predict with training data
    train_preds = clf.predict(train_X)
    #predict with test data
    test_preds = clf.predict(test_X)
    print('RBF Kernel')
    print('1st column is testing set scores & 2nd column is training set scores')
    print(f"F1_score: [{metrics.f1_score(test_y, test_preds)}, {metrics.f1_score(train_y, train_preds)}]")
    print(f"Precision: [{metrics.precision_score(test_y, test_preds)}, {metrics.precision_score(train_y, train_preds)}]")
    print(f"Recall: [{metrics.recall_score(test_y, test_preds)}, {metrics.recall_score(train_y, train_preds)}]")
    print('')
    
    #For Question 1.3 and 1.4
    #calculate metrics and store in dictionary
    results['Train_F1_Score'].append(metrics.f1_score(train_y, train_preds))
    results['Train_Precision'].append(metrics.precision_score(train_y, train_preds))
    results['Train_Recall'].append(metrics.recall_score(train_y, train_preds))
    results['Test_F1_Score'].append(metrics.f1_score(test_y, test_preds))
    results['Test_Precision'].append(metrics.precision_score(test_y, test_preds))
    results['Test_Recall'].append(metrics.recall_score(test_y, test_preds))
    
    #For Question 1.5
    #For using the first two dimensions of the data - Question 1.5
    train_X = train_X[:,:2]
    test_X = test_X[:,:2]
    
    print('Question 1.5:')
    print('For question 1.5, see report or Submissions/Figures for decision boundaries\n')
    
    #Plotting RBF Kernel Decision Boundary
    clf = svm.SVC(kernel = 'linear')
    clf.fit(train_X, train_y)
    plot_decision_boundary(clf, train_X, train_y, title='SVM Decision Boundary for Linear Kernel')
    
    #Plotting RBF Kernel Decision Boundary
    clf = svm.SVC(kernel = 'poly', degree=3)
    clf.fit(train_X, train_y)
    plot_decision_boundary(clf, train_X, train_y, title='SVM Decision Boundary for 3rd Degree Polynomial Kernel')
    
    #Plotting RBF Kernel Decision Boundary
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(train_X, train_y)
    plot_decision_boundary(clf, train_X, train_y, title='SVM Decision Boundary for RBF Kernel')
    
    print('Template_SVM script completed')
    print('----------')
    
    return

if __name__ == '__main__':
   main()
