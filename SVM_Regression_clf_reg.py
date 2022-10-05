#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[14]:


file_path = '/Users/juyeonpark/Desktop/titactoedatasets/tictactoedatasets/tictac_final.txt'
print('File exists:', os.path.exists(file_path))

np.arr = np.loadtxt(file_path)

X = np.arr[:, :9 ]
Y = np.arr[:, 9:]

print(X.shape)
print(Y.shape)


# In[15]:


file_path2 = '/Users/juyeonpark/Desktop/titactoedatasets/tictactoedatasets/tictac_single.txt'
print('File exists:', os.path.exists(file_path2))

np.arr2 = np.loadtxt(file_path2)

X2 = np.arr2[:, : 9 ]
Y2 = np.arr2[:, 9:]

print(X2.shape)
print(Y2.shape)


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.7)
print ("from tictac_final.txt:")
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[17]:


x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.7)
print ("from tictac_single.txt:")
print (x2_train.shape, y2_train.shape)
print (x2_test.shape, y2_test.shape)


# In[18]:



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


# In[19]:



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


# In[20]:


print('final dataset: ')
SVC_Classifier(x_train,y_train, x_test, y_test)
print('single dataset: ')
SVC_Classifier(x2_train,y2_train, x2_test, y2_test)


# In[10]:


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


# In[11]:


print('final')
Linear_Regressor(x_train,y_train, x_test, y_test)
print('single')
Linear_Regressor(x2_train,y2_train, x2_test, y2_test)


# In[ ]:




