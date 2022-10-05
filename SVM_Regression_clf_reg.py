
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
    
    kfd = KFold(n_splits = 20, shuffle=True, random_state = 10)
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


print('final dataset: ')
SVC_Classifier(x_train,y_train, x_test, y_test)
print('single dataset: ')
SVC_Classifier(x2_train,y2_train, x2_test, y2_test)



from sklearn.metrics import mean_squared_error
def Linear_Regressor(x_train,y_train, x_test, y_test):
    
    reg = _init_SVC_Reg()
    reg.fit(x_train,y_train)

    best_score = reg.best_score_
    best_params = reg.best_params_
    best_esti = reg.best_estimator_
    
    y_pred = reg.predict(x_test)
    #y_pred = np.where(y_pred > 0.5,1, 0)
    
    test_acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', test_acc)
    
    mse = mean_squared_error(y_test,y_pred,squared=False)
    print('mean squared error: ',mse)
    print('the best score: ')
    print(best_score)
    
    print('the best parameter is: ')
    print(best_params)



print('final')
Linear_Regressor(x_train,y_train, x_test, y_test)
print('single')
Linear_Regressor(x2_train,y2_train, x2_test, y2_test)





