
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    
    
    return (747/len(spam_data))*100#spam_data['target'].value_counts()


# In[4]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    cvv=CountVectorizer()
    cvv.fit(X_train)
    lst=cvv.vocabulary_.keys()
    nl=sorted(lst,key=len,reverse=True)
    
    
    
    return nl[0]


# In[6]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns',2000)

def answer_three():
    clf=MultinomialNB(alpha=0.1)
    vect=CountVectorizer()
    vect.fit(X_train)
    newtrain=vect.transform(X_train)
    newtest=vect.transform(X_test)
    clf.fit(newtrain,y_train)
    pred=clf.predict(newtest)
    #print(pred)
    
    
    
    
    return roc_auc_score(y_test,pred)


# In[8]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_rows',2000)

def answer_four():
    vect=TfidfVectorizer().fit(X_train)
    test=vect.transform(X_test)
    train=vect.transform(X_train)
    features=np.array(vect.get_feature_names()) #list of all features
    maxx=train.max(0).toarray()[0] # additional [0] shapes (1,7345) to (7345) # maening of max(0) ??
    neww=maxx.argsort() # index after sorting
    values=maxx[neww] # contains values of tfidf in increasing order
    neww=features[neww] # features names in increasing order
    #maxx=maxx.argsort()
    s1=pd.Series(values[:20],index=neww[:20])
    s2=pd.Series(values[-20:][::-1],index=neww[-20:][::-1])
    
    
    
    return (s1,s2)  #Your answer here


# In[10]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[11]:


def answer_five():
    vect=TfidfVectorizer(min_df=3).fit(X_train)
    clf=MultinomialNB(alpha=0.1)
    train=vect.transform(X_train)
    test=vect.transform(X_test)
    clf.fit(train,y_train)
    pred=clf.predict(test)
    
    
    return roc_auc_score(y_test,pred)#Your answer here


# In[12]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[13]:


def answer_six():
    import nltk
    nltk.download('punkt')
    spam_data.groupby('target',axis=1)
    df1=spam_data[spam_data['target']==0]
    df2=spam_data[spam_data['target']==1]
    s=0
    s1=0
    for sntcs in df1['text']:
        s=s+len(sntcs)
    for sntcs in df2['text']:
        s1=s1+len(sntcs)
        
        
        
    
    return (s/len(df1),s1/len(df2))#spam_data.groupby('target',axis=1)['text]#Your answer here


# In[14]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[15]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[16]:


from sklearn.svm import SVC

def answer_seven():
    
    len_t = [len(x) for x in X_train]
    len_tt = [len(x) for x in X_test]
    vect=TfidfVectorizer(min_df=5).fit(X_train)
    X_train_1=vect.transform(X_train)
    X_test_1=vect.transform(X_test)
    X_train_1=add_feature(X_train_1,len_t)
    X_test_1=add_feature(X_test_1,len_tt)
    clf=SVC(C=10000)
    clf.fit(X_train_1,y_train)
    pred=clf.predict(X_test_1)
    #nw=lent-tt
    
    
    return roc_auc_score(y_test,pred)#roc_auc_score(y_test,pred)#Your answer here


# In[17]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[18]:


def answer_eight():
    import nltk
    #dig_spam = [sum(char.isnumeric() for char in x) for x in spam_data.loc[spam_data['target']==1,'text']]
    df1=spam_data[spam_data['target']==0]
    df2=spam_data[spam_data['target']==1]
    tc=[]
    tcc=[]
    tl=0
    cnt=0
    for i in df1['text']:
        #tl=tl+len(nltk.word_tokenize(i))
        for c in i:
            if c.isnumeric():
                cnt=cnt+1
        tc.append(cnt)
        cnt=0
    cnt=0
    for i in df2['text']:
        #tl=tl+len(nltk.word_tokenize(i))
        for c in i:
            if c.isnumeric():
                cnt=cnt+1
        tcc.append(cnt)
        cnt=0
    
            
            
    
    
    return (np.array(tc).mean(),np.array(tcc).mean()) #Your answer here


# In[19]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[20]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    vect=TfidfVectorizer(min_df=5,ngram_range=(1,3)).fit(X_train)
    len_t = [len(x) for x in X_train]
    len_tt = [len(x) for x in X_test]
    X_tr=vect.transform(X_train)
    X_ts=vect.transform(X_test)
    X_tr=add_feature(X_tr,len_t)
    X_ts=add_feature(X_ts,len_tt)
    df1=spam_data[spam_data['target']==0]
    df2=spam_data[spam_data['target']==1]
    tc=[ sum( c.isnumeric() for c in x ) for x in X_train]
    tcc=[ sum( c.isnumeric() for c in x) for x in X_test]
    X_tr=add_feature(X_tr,tc)
    X_ts=add_feature(X_ts,tcc)
    clf=LogisticRegression(C=100).fit(X_tr,y_train)
    pred=clf.predict(X_ts)
    
    
    
    
    
    return roc_auc_score(y_test,pred)#Your answer here


# In[21]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[42]:


def answer_ten():
    import re
    df1=spam_data[spam_data['target']==0]
    df2=spam_data[spam_data['target']==1]
    cnt=0
    spm=[]
    for c in df1['text']:
        for w in c:
            if( w!='_' and w.isalpha()==False and w.isnumeric()==False):
                cnt=cnt+1
        spm.append(cnt)
        cnt=0
    cnt=0
    notspm=[]
    for c in df2['text']:
        for w in c:
            if( w!='_' and w.isalpha()==False and w.isnumeric()==False):
                cnt=cnt+1
        notspm.append(cnt)
        cnt=0
    
                
                
    
    
    
    
    return (np.array(spm).mean(),np.array(notspm).mean())#Your answer here


# In[43]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[85]:


def answer_eleven():
    cv=CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb').fit(X_train)
    X_train_1=cv.transform(X_train)
    X_test_1=cv.transform(X_test)
    len_train= [sum( w!='_' and w.isalpha()==False and w.isnumeric()==False for w in x) for x in X_train ]
    len_test=[ sum( w!='_' and w.isalpha()==False and w.isnumeric()==False for w in x) for x in X_test ]
    len_t= [ sum( w.isnumeric() for w in x) for x in X_train]
    len_tt=[ sum(w.isnumeric() for w in x) for x in X_test]
    lentr=[ len(x) for x in X_train]
    lentt=[ len(x) for x in X_test]
    X_train_1=add_feature(X_train_1,[lentr,len_t,len_train])
    X_test_1=add_feature(X_test_1,[lentt,len_tt,len_test])
    lr=LogisticRegression(C=100).fit(X_train_1,y_train)
    pred=lr.predict(X_test_1)
    names=np.array(cv.get_feature_names()+['length_of_doc', 'digit_count', 'non_word_char_count'] )
    #X_test_1=add_feature(X_test_1)
    indices=lr.coef_[0].argsort()
    sorted_names=names[indices]
    
    
    return (roc_auc_score(y_test,pred),sorted_names[0:10],sorted_names[-11:-1])#Your answer here [0] converts 2-d to 1-d
    


# In[86]:


answer_eleven()


# In[ ]:




