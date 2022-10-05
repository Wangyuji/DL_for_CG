import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

def _init_MLP():
    param_setting =[{
        'hidden_layer_sizes': [(150,100,50,10)],
        'activation':['relu', 'sigmoid'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant']
    }]
    
    kfd = KFold(n_splits = 10, shuffle=True, random_state = 10)
    
    iter_clf = MLPClassifier(max_iter = 1000)
    #class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, 
    #scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, 
    #return_train_score=False)[source]
    clf = GridSearchCV(iter_clf, param_setting, n_jobs =-1, cv=kfd, scoring="accuracy")
    
    iter_reg = MLPRegressor(random_state=20, max_iter=1000)
    
    reg = GridSearchCV(iter_reg, param_setting, cv=kfd, n_jobs=-1)
    
    return clf, reg

def MLP_classifier(x_train, y_train, x_test, y_test):
    clf, reg = _init_MLP()
    clf.fit(x_train, y_train)
    
    best_score = clf.best_score_
    best_params = clf.best_params_
    best_esti = clf.best_estimator_
    y_pred = clf.predict(x_test)
    
    print("The best socre is: ")
    print(best_score)
    print("The best params is: ")
    print(best_params)
    #print("The best estimator is: ")
    #print(best_esti)
    #print("The best prediction for y is: ")
    #print(y_pred)
    #print(y_test)
    
    #nomalize default is true here
    test_acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', test_acc)
    test_acc = accuracy_score(y_test, y_pred, normalize=False)
    print('Test accuracy with normalize flase: \n', test_acc)  

    
    cm = confusion_matrix(y_test, y_pred)
    cm = cm/cm.astype(np.float).sum(axis = 1)
    print("Confusion Matrix is: ")
    print(cm)
    
    plot_confusion_matrix(clf, x_test, y_pred)  
    plt.show()
    
    
def MLP_regressor(x_train, y_train, x_test, y_test):
    clf, reg = _init_MLP()
    reg.fit(x_train, y_train)
     
    best_score = reg.best_score_
    best_params = reg.best_params_
    best_esti = reg.best_estimator_
    y_pred1 = reg.predict(x_test)
    
    print("The best socre is: ")
    print(best_score)
    print("The best params is: ")
    print(best_params)
    
    y_pred = np.where(y_pred1 > 0.5, 1, 0)
    #print(y_pred1)
    # print(y_test)
    
    #nomalize default is true here
    test_acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', test_acc)
    test_acc = accuracy_score(y_test, y_pred, normalize=False)
    print('Test accuracy with normalize flase: \n', test_acc)  

    total_acc = np.zeros(9)
    acc = 0
    for i in range(9):
        total_acc[i] = accuracy_score(y_test[:, i],y_pred[:, i], normalize=False)
        acc += total_acc[i]
    
    size = np.shape(y_test)[0] * 9
    acc /= size
    print("The accuracy calculate via MLP is: "+ str(acc))
    
    filename = 'mlp_model.pkl'
    pickle.dump(reg, open(filename, 'wb'))
  
    
def exe_mlp_clf(x_train, y_train, x_test, y_test):
    MLP_classifier(x_train, y_train, x_test, y_test)

def exe_mlp_reg(x_train, y_train, x_test, y_test):
    MLP_regressor(x_train, y_train, x_test, y_test)