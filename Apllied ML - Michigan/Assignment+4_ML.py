
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[45]:

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt


# In[46]:

train = pd.read_csv('train.csv',encoding='ISO-8859-1')
test = pd.read_csv('test.csv',encoding='ISO-8859-1')
#train.head(1)


# In[47]:

train.drop(['balance_due',
 'collection_status',
 'compliance_detail',
 'payment_amount',
 'payment_date',
 'payment_status'],axis=1,inplace=True)


# In[48]:

train.dropna(subset=["compliance"],inplace=True)
pd.set_option('display.max_columns', None)


# In[49]:

#train.head(1)


# In[50]:

train.drop(['grafitti_status'],axis=1,inplace=True)


# In[51]:

test.drop('grafitti_status',axis=1,inplace=True)


# In[52]:

y_train=train['compliance']
train.drop('compliance',axis=1,inplace=True)


# In[53]:

test.set_index('ticket_id',inplace=True)
train.set_index('ticket_id',inplace=True)


# In[54]:

ans=pd.concat([train,test],axis=0)


# In[55]:

#ans.head(1)


# In[56]:

#lst=[]
#for i in ans.columns:
    #if(ans[i].dtype == "O"):
        #lst.append((i,train[i].nunique()))
#lst


# In[57]:

drp=['violator_name','ticket_issued_date','mailing_address_str_name','mailing_address_str_number']


# In[58]:

ans.drop(drp,axis=1,inplace=True)


# In[59]:

#nl=[]
#for i in ans.columns:
    #if(ans[i].dtype == "O"):
        #nl.append((i,train[i].nunique()))


# In[60]:

#nl


# In[61]:

# since 99% of values are 0
ans.drop('clean_up_cost',axis=1,inplace=True)


# In[62]:

ans.head()
ans.drop('discount_amount',axis=1,inplace=True)


# In[63]:

ans.drop('violation_zip_code',axis=1,inplace=True)


# In[64]:

ans.drop('non_us_str_code',axis=1,inplace=True)


# In[65]:

ans.drop('hearing_date',axis=1,inplace=True)


# In[66]:

#sns.heatmap(ans.corr())
#plt.show()
ans['violation_description'].nunique()
ans.drop(['zip_code','violation_street_name','city','violation_description','violation_code'],axis=1,inplace=True)


# In[67]:

ans.head(4)
df=pd.get_dummies(ans[['agency_name','country','disposition']],drop_first=True)


# In[68]:

#df.head(1)


# In[69]:

ans.drop(['inspector_name','state'],axis=1,inplace=True)


# In[70]:

#sns.heatmap(ans.corr())
#plt.show()
# highly correlated
ans.drop(['judgment_amount','late_fee'],axis=1,inplace=True)


# In[71]:

ans=pd.concat([df,ans],axis=1)


# In[72]:

ans.drop(['agency_name','country','disposition'],axis=1,inplace=True)


# In[73]:

X_train=ans.iloc[:len(train),:]
X_test=ans.iloc[len(train):,:]
drpp=['agency_name_Neighborhood City Halls','country_Cana','country_Egyp','country_USA','disposition_Responsible - Compl/Adj by Default']
X_train.drop(drpp,axis=1,inplace=True)
X_test.drop(drpp,axis=1,inplace=True)
X_train.drop(['country_Germ','disposition_Responsible - Compl/Adj by Determi','disposition_Responsible by Dismissal','admin_fee','state_fee'],axis=1,inplace=True)
X_test.drop(['country_Germ','disposition_Responsible - Compl/Adj by Determi','disposition_Responsible by Dismissal','admin_fee','state_fee'],axis=1,inplace=True)


# In[37]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 4, 5]}
clf=GradientBoostingClassifier(random_state=0)
final=GridSearchCV(clf,param_grid=values,scoring='roc_auc')
final.fit(X_train,y_train)


# In[38]:


#print('Grid best score (AUC): ', final.best_score_)
results=final.predict_proba(X_test)


# In[39]:

results
finalres = pd.Series(results[:,1], index=X_test.index)


# In[41]:



def blight_model():
    
    # Your code here
    
    return finalres


# In[42]:

blight_model()


# In[ ]:



