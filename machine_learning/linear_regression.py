import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

def load_data(dataset):
    """
    Load a pair of data X,y 

    Params
    ------
    dataset:    train/valid/test

    Return
    ------
    X:          shape (N, 240)
    y:          shape (N, 1)
    """
    X = pd.read_csv(f"../../Data/housing_data/{dataset}_x.csv", header=None).to_numpy()
    y = pd.read_csv(f"../../Data/housing_data/{dataset}_y.csv", header=None).to_numpy()

    return X,y

def score(model, X, y):
    """
    Score the model with X, y

    Params
    ------
    model:  the model to predict with
    X:      the data to score on
    y:      the true value y

    Return
    ------
    mae:    the mean absolute error
    """
    #generate predictions on X 
    y_pred = model.predict(X)
    #calculate the mean_absolute_error
    mae = mean_absolute_error(y, y_pred)
    #return the mean absolute error
    return mae

#function that will run the ordinary least square linear regression model
def ols_lin_reg(train, valid, test):
    """
    Fit OLS model and calculate mean absolute error

    Params
    ------
    train:    training dataset
    valid:    validation dataset
    test:     test data set

    Return
    ------
    mae:    the mean absolute error of OLS model
    """
    #concatenate training and validation data. There are no hyperparameters to tune
    #with ordinary least squares because there is no regularization term
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)
    
    #generate matrices for test data    
    test_X, test_y = test
    
    #fit the model with training and test set data
    model = LinearRegression().fit(train_X, train_y)
    
    #get mean absolute error on test dataset
    mae_ols = score(model, test_X, test_y)
    
    return mae_ols

def hyper_parameter_tuning(model_class, param_grid, train, valid):
    """
    Tune the hyper-parameter using training and validation data

    Params
    ------
    model_class:    the model class
    param_grid:     the hyper-parameter grid, dict
    train:          the training data (train_X, train_y)
    valid:          the validatation data (valid_X, valid_y)

    Return
    ------
    model:          model fit with best params
    best_param:     the best params
    """
    train_X, train_y = train
    valid_X, valid_y = valid
    
    #initialize an arbitrarily large value of mean absolute error
    best_mae = 1000000
    
    # Set up the parameter grid
    param_grid = list(ParameterGrid(param_grid))

    # train the model with each parameter setting in the grid
    #loop over each combination of the parameters, alpha and max_iter
    for i in param_grid:
        #fit the model with the parameters and training data
        model = model_class(alpha=i.get('alpha'), max_iter=i.get('max_iter'), tol=i.get('tol')).fit(train_X, train_y)
        #calculate the mean absolute error on the validation set
        mae = score(model, valid_X, valid_y)
        # choose the model with lowest MAE on validation set
        #if the current mae is lower then the best_mae, overwrite the best_mae
        if mae < best_mae:
            #overwrite the best_mae score
            best_mae = mae
            #overwrite the best_params
            best_params = i 
    
    # then fit the model with the training and validation set (refit)
    #concatenate training data and labels into new training set
    new_train_X = np.concatenate((train_X, valid_X))
    new_train_y = np.concatenate((train_y,valid_y))
    #refit the model with the best parameters
    final_model = model_class(alpha=best_params.get('alpha'), max_iter=best_params.get('max_iter')).fit(new_train_X, new_train_y)

    # return the fitted model, the best parameter setting, and best mae
    return final_model, best_params, best_mae
    
def plot_mae_alpha(model_class, params, train, valid, test, title="Model"):
    """
    Plot the model MAE vs Alpha (regularization constant)

    Params
    ------
    model_class:    The model class to fit and plot
    params:         The best params found 
    train:          The training dataset
    valid:          The validation dataest
    test:           The testing dataset
    title:          The plot title

    Return
    ------
    None
    """
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)

    #generate matrices for test data    
    test_X, test_y = test

    # set up the list of alphas to train on
    alphas = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 10, 100]

    #initialize an empty list for storing MAE values
    mae_vals = []

    # train the model with each alpha, log MAE
    for n in range(len(alphas)):
        #fit the model with each alpha value, and the best max_iter from hyper_parameter_tuning
        model = model_class(alpha=alphas[n], max_iter=params.get('max_iter')).fit(train_X, train_y)
        #calculate the mean absolute error
        mae = score(model, test_X, test_y)
        #append the list of mean absolute errors
        mae_vals.append(mae)

    # plot the MAE - Alpha
    plt.plot(alphas, mae_vals)
    #plot title
    plt.title('MAE for Various Values of the Regularization\nConstant, Alpha, for the '+title+' Model')
    #label x and y axes
    plt.xlabel('Regularization Constant (Alpha)')
    plt.ylabel('Mean Absolute Error')
    #save to Submission/Figures folder with appropriate name
    plt.savefig(('../Figures/'+title+'.png'))
    #clear figure
    plt.clf()


def main():
    """
    Load in data
    """
    train = load_data('train')
    valid = load_data('valid')
    test = load_data('test')

    """
    Define the parameter grid for each classifier
    e.g. lasso_grid = dict(alpha=[0.1, 0.2, 0.4],
                           max_iter=[1000, 2000, 5000])
    """
    
    #create a dictioanry to store MAE values
    models = ['OLS', 'Lasso', 'Ridge']
    models_best_mae = dict([(key, []) for key in models])

    #calculate MAE of ordinary least squares model
    ols_mae = ols_lin_reg(train, valid, test)
    models_best_mae['OLS'].append(ols_mae)
    
    # Tune the hyper-paramter by calling the hyper-parameter tuning function
    # e.g. lasso_model, lasso_param = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    lasso_grid = dict(alpha=[0.1, 0.2, 0.4, 0.8, 1.0, 10], max_iter=[1000, 2000, 5000], tol=[0.001, 0.01, 0.1, 1, 10, 100])
    lasso_model, lasso_params, best_lasso_mae = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    print('\nBest Lasso Params:')
    print(lasso_params)
    models_best_mae['Lasso'].append(best_lasso_mae)
    
    ridge_grid = dict(alpha=[0.1, 0.2, 0.4, 0.8, 1.0, 10], max_iter=[1000, 2000, 5000], tol=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge_model, ridge_params, best_ridge_mae = hyper_parameter_tuning(Ridge, ridge_grid, train, valid)
    print('\nBest Ridge Params:')
    print(ridge_params)
    models_best_mae['Ridge'].append(best_ridge_mae)
    
    # Plot the MAE - Alpha plot by calling the plot_mae_alpha function
    # e.g. plot_mae_alpha(Lasso, lasso_param, train, valid, test, "Lasso")
    plot_mae_alpha(Lasso, lasso_params, train, valid, test, "Lasso")
    plot_mae_alpha(Ridge, ridge_params, train, valid, test, "Ridge")
    
    print('\nBest MAEs for each model')
    print(models_best_mae)
    
if __name__ == '__main__':
    main()
