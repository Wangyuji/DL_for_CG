import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import ConfusionMatrixDisplay
import os

def _init_KNN_clf():
    #intialize paramter grid for grid search
    param_grid = [{
        'n_neighbors': np.arange(1, 50),
        'weights': ['uniform', 'distance']
        
    }]
    
    #initialize cross validation generator
    kfd = KFold(n_splits = 10, shuffle = True, random_state = 1)
    
    #initialize knn classifier with default params (n_neighbors=5 by default) 
    iter_clf = KNeighborsClassifier()
    
    #run grid search to find best hyperparams for classifier
    clf_search = GridSearchCV(iter_clf, param_grid, scoring = 'accuracy', cv=kfd)
    
    return clf_search

def _init_KNN_reg():
    #intialize paramter grid for grid search
    param_grid = [{
        'n_neighbors': np.arange(1,10),
        'weights': ['uniform', 'distance']
    }]
    #initialize cross validation generator
    kfd = KFold(n_splits = 10, shuffle = True, random_state = 1)
    
    #initialize knn regressor with default params (n_neighbors=5 by default)
    iter_reg = KNeighborsRegressor()
    
    #prepare grid search to find best hyperparams for regressor
    reg = GridSearchCV(iter_reg, param_grid, cv=kfd)
    
    return reg

def KNN_classifier(x_train, y_train, x_test, y_test, dataset_name):
    clf_search = _init_KNN_clf()
    
    #run grid search on training data
    clf_search.fit(x_train, y_train.ravel())
    
    best_score = clf_search.best_score_
    best_params = clf_search.best_params_
    clf = clf_search.best_estimator_
    y_pred = clf.predict(x_test)
    results = pd.DataFrame(clf_search.cv_results_)
    
 
    #print(results.columns)
    results_test = pd.DataFrame(results[['mean_test_score','param_weights','param_n_neighbors']])
    uniform_results = results_test[results_test['param_weights'] == 'uniform']
    distance_results = results_test[results_test['param_weights'] == 'distance']
    
    plt.title("KNNTest" + dataset_name)
    line1, = plt.plot(uniform_results['param_n_neighbors'], uniform_results['mean_test_score'], 'b', label='uniform')
    line2, = plt.plot(distance_results['param_n_neighbors'], distance_results['mean_test_score'], 'g', label='distance')
    #plt.plot(np.arange(1,10), scores[9:], 'go')
    plt.xlabel("value of k")
    plt.ylabel("testing accuracy")
    plt.legend(handles=[line1,line2])
    plt.savefig(dataset_name + 'testaccuracy.png')
    plt.draw()
    
    plt.clf()
    
    
    print("The best parameters are: ")
    print(best_params)

    
    
    test_acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', test_acc)
   
    
    
    #generate confusion matrix from testing set predictions, set normalize to true so rows will sum to 1
    cm = confusion_matrix(y_test, y_pred, normalize = 'true')
    #cm = cm/cm.astype(np.float).sum(axis = 1)
    
    print("Confusion Matrix: ")
    print(cm)
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap='BuGn', normalize='true', values_format = '.2f')
    plt.savefig(dataset_name + 'confusionmatrixnormalized.png')
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap='BuGn')
    plt.savefig(dataset_name + 'confusionmatrix.png')
    plt.show()

def KNN_regressor(x_train, y_train, x_test, y_test):
    reg = _init_KNN_reg()
    reg.fit(x_train, y_train)
    
    best_params = reg.best_params_
   
    

    print("The best parameters are: ")
    print(best_params)
    optimalreg = KNeighborsRegressor(n_neighbors = best_params['n_neighbors'], weights = best_params['weights'])
    optimalreg.fit(x_train,y_train)
    
    y_pred = optimalreg.predict(x_test) #generate predictions on testing set
    y_pred = np.where(y_pred > 0.5, 1, 0) #round values to either 0 or 1
    
    total_acc = np.zeros(9)
    for i in range(9):
        total_acc[i] = accuracy_score(y_test[:,i], y_pred[:,i], normalize=False)

    
    acc = np.sum(total_acc) / (np.shape(y_test)[0] * 9)
    print ("Accuracy for knn regressor: {0}".format(acc))


def exe_knn_clf(x_train,y_train, x_test,y_test,title):
    KNN_classifier(x_train,y_train,x_test,y_test,title)

def exe_knn_reg(x_train,y_train, x_test,y_test):
    KNN_regressor(x_train, y_train, x_test, y_test)