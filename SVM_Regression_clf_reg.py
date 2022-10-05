
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import sklearn.svm as svm
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


def _init_SVC_Reg():
    
    kfd = KFold(n_splits = 10, shuffle=True, random_state = 10)
    param_setting ={
        'C': [0.01,0.1,1,2,5]
    }
    
    svc = svm.SVC(kernel='linear')
    
    init_reg = LinearRegression()
    
    clf = GridSearchCV(svc, param_setting, cv=kfd)
    reg = GridSearchCV(init_reg, param_setting, cv=kfd)
    
    return clf




def SVC_Classifier(x_train,y_train, x_test, y_test):

    clf = _init_SVC_Reg()
    clf.fit(x_train,y_train)
    
    best_score = clf.best_score_
    best_params = clf.best_params_
    y_pred = clf.predict(x_test)
    
    print('the best score is: ')
    print(best_score)
    
    print('the best parameter is: ')
    print(best_params)
    
    test_acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', test_acc)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is: ")
    print(cm)
    
    plot_confusion_matrix(clf, x_test, y_pred)  
    plt.show()




from sklearn.metrics import mean_squared_error
def Linear_Regressor(x_train,y_train, x_test, y_test):
    from sklearn.metrics import mean_squared_error
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def Linear_Regressor(x_train,y_train, x_test, y_test):
    print("Linear Regression")
    #print(np.shape(y_test)[0])
    #print(np.shape(y_test)[1])
    Y_tilde = np.empty((np.shape(y_train)[0], np.shape(y_train)[1]))
    Y_pred = np.empty((np.shape(y_test)[0], np.shape(y_test)[1]))

    bias = 1
    for i in range(np.shape(y_test)[1]):
        #y = y_train[:, i]
        #W = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y
        beta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train[:, i])
        #W = [weight + bias for weight in W]
        y_pred = x_test @ beta
        y_tilde = x_train @ beta
        Y_pred[:, i] = y_pred
        Y_tilde[:, i] = y_tilde

    Y_pred = (Y_pred == Y_pred.max(axis=1)[:, None]).astype(int)
    Y_tilde = (Y_tilde == Y_tilde.max(axis=1)[:, None]).astype(int)
    
    print("Train R2")
    print(R2(y_train,Y_tilde))
    print("The train MSE is: ")
    print(MSE(y_train,Y_tilde))
    
    print("Test R2")
    print(R2(y_test, Y_pred))
    print("The test MSE is: ")
    print(MSE(y_test,Y_pred))

    total_acc = np.empty(9)
    for i in range(9):
         total_acc[i] = accuracy_score(y_test[:, i], Y_pred[:, i], normalize=False)

    acc = np.sum(total_acc) / (np.shape(y_test)[0] * 9)
    print("Accuracy LR: {0}".format(acc))


def exe_svm_clf(x_train, y_train, x_test, y_test):
    SVC_Classifier(x_train, y_train, x_test, y_test)

def exe_linReg_reg(x_train, y_train, x_test, y_test):
    Linear_Regressor(x_train, y_train, x_test, y_test)





