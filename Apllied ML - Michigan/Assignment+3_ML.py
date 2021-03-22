
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 3 - Evaluation
# 
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[45]:

import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[46]:

def answer_one():
    df=pd.read_csv('fraud_data.csv')
    s=df['Class'].value_counts()
    ans=s.iloc[1]/len(df)
    
    # Your code here
    
    return ans


# In[47]:

answer_one()


# In[48]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[49]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    d_m=DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
    pred=d_m.predict(X_test)
    #print(pred)
    
    # Your code here
    
    return (d_m.score(X_test,y_test),recall_score(y_test,pred))


# In[50]:

answer_two()


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[51]:

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    modl=SVC(kernel='rbf',C=1).fit(X_train , y_train)
    pred=modl.predict(X_test)

    # Your code here
    
    return (modl.score(X_test,y_test),recall_score(y_test,pred),precision_score(y_test,pred))


# In[52]:

answer_three()


# In[53]:

X_test


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[54]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    clf=SVC(kernel='rbf',C=1e9,gamma=1e-07).fit(X_train,y_train)
    pred=clf.decision_function(X_test) > -220
    ans=confusion_matrix(y_test,pred)
    

    # Your code here
    
    return ans


# In[55]:

answer_four()


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[56]:

def answer_five():
    from sklearn.linear_model import LogisticRegression
    
    #from sklearn.metrics import plot_precision_recall_curve
    from sklearn.metrics import precision_recall_curve
    #lr=LogisticRegression().fit(X_train,y_train)
    #ans=plot_precision_recall_curve(lr,X_test,y_test)
    
    
        
    # Your code here
    
    return 


# In[57]:

answer_five()


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# In[58]:

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    param={'penalty': ['l1', 'l2'],'C':[0.01, 0.1, 1, 10, 100]}
    new=GridSearchCV(lr,param_grid=param,scoring='recall',cv=3)
    new.fit(X_train,y_train)

    # Your code here
    
    return new.cv_results_['mean_test_score'].reshape(5,2)


# In[59]:

answer_six()


# In[ ]:

# Use the following function to help visualize results from the grid search
#def GridSearch_Heatmap(scores):
    #%matplotlib notebook
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    #plt.figure()
    #sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    #plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())

