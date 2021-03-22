
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[3]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df=pd.read_csv('City_Zhvi_AllHomes.csv')
#dff=pd.read_csv('university_towns.txt',header= None)
df1=open("university_towns.txt")
data=[]
for line in df1:
    data.append(line[:-1]) #removes backslash n
pair=[]
for line in data:
    if(line[-6:]=='[edit]'):
        state=line[:-6]
    elif '(' in line:
        ind= line.index('(')
        town=line[:ind-1]
        pair.append([state,town])
    else:
        town=line
        pair.append([state,town])
    fa=pd.DataFrame(pair,columns=['State','RegionName'])
        
        
df2=pd.read_excel('gdplev.xls',skiprows=7)
df2=df2[['Unnamed: 4', 'Unnamed: 5']]
df2.columns=[['quarter','gdp']]


# In[6]:


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[7]:


def get_list_of_university_towns():
    df=pd.read_csv('City_Zhvi_AllHomes.csv')
#dff=pd.read_csv('university_towns.txt',header= None)
    df1=open("university_towns.txt")
    data=[]
    for line in df1:
        data.append(line[:-1]) #removes backslash n
    pair=[]
    for line in data:
        if(line[-6:]=='[edit]'):
            state=line[:-6]
        elif '(' in line:
            ind= line.index('(')
            town=line[:ind-1]
            pair.append([state,town])
        else:
            town=line
            pair.append([state,town])
    fa=pd.DataFrame(pair,columns=['State','RegionName'])
    
    return fa


# In[8]:


def get_recession_start():
    df=pd.read_csv('City_Zhvi_AllHomes.csv')
    df1=open("university_towns.txt")
    df2=pd.read_excel('gdplev.xls',skiprows=7)
    df2=df2[['Unnamed: 4', 'Unnamed: 5']]
    df2.columns=[['quarter','gdp']]
    lst=[]
    last=len(df2)
    for x in range(212,last):
        if((df2.iloc[x][1]<df2.iloc[x-1][1]) and (df2.loc[x-1][1]<df2.iloc[x-2][1])):
            ans=df2.iloc[x-2][0]
            break
    
    return ans


# In[9]:


def get_recession_end():
    df3=df2[df2['quarter']> '2008q3' ]
    for x in range( 0, len(df3)-2 ):
        if((df3.iloc[x+1][1]>df3.iloc[x][1]) and (df3.iloc[x+2][1]>df3.iloc[x+1][1])):
            anss=df3.iloc[x+2][0]
            break
        
    return anss


# In[10]:


def get_recession_bottom():
    df=pd.read_csv('City_Zhvi_AllHomes.csv')
    df1=open("university_towns.txt")
    df2=pd.read_excel('gdplev.xls',skiprows=7)
    df2=df2[['Unnamed: 4', 'Unnamed: 5']]
    df2.columns=[['quarter','gdp']]
    st=get_recession_start()
    end=get_recession_end()
    dff=df2[(df2['quarter']>=st) & (df2['quarter']<=end)]
    ind=dff['gdp'].argmin()
    fa=dff.iloc[3]
    
    return fa[0]


# In[11]:


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df['State'].replace(states,inplace=True)
    dfa=df[['State','RegionName']]
    for i in range(2000,2016):
        dfa[ str(i) + 'q1']= ((df[str(i) + '-01'] +df[str(i) + '-02'] + df[ str(i) + '-03' ])/3)#.mean(axis = 1)
        dfa[ str(i) + 'q2']= ((df[str(i) + '-04'] +df[str(i) + '-05'] + df[ str(i) + '-06' ])/3)#.mean(axis = 1)
        dfa[ str(i) + 'q3']= ((df[str(i) + '-07'] +df[str(i) + '-08'] + df[ str(i) + '-09' ])/3)#.mean(axis = 1)
        dfa[ str(i) + 'q4']= ((df[str(i) + '-10'] +df[str(i) + '-11'] + df[ str(i) + '-12' ])/3)#.mean(axis = 1)
    chck=str(2016)
    dfa[ chck + 'q1']= ((df[chck + '-01'] +df[chck + '-02'] + df[ chck + '-03' ])/3)#.mean(axis=1)
    dfa[ chck + 'q2']= ((df[chck + '-04'] +df[chck + '-05'] + df[ chck + '-06' ])/3)#.mean(axis=1)
    dfa[ chck + 'q3']= ((df[chck+ '-07'] +df[chck + '-08'] )/2)#.mean(axis=1)
    ffa=dfa.set_index(['State','RegionName'])
    
    return ffa


# In[12]:


convert_housing_data_to_quarters()


# In[31]:


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    uni= get_list_of_university_towns()
    new=convert_housing_data_to_quarters()
    new=new.reset_index()
    dft=pd.read_excel('gdplev.xls',skiprows=7)
    dft=dft[['Unnamed: 4', 'Unnamed: 5']]
    dft.columns=[['quarter','gdp']]
    uni['univ']=True
    st=get_recession_start()
    end=get_recession_end()
    bottom=get_recession_bottom()
    new=new[["2008q2" ,  "2009q2",'State','RegionName' ]]
    new['ratio']=(new['2008q2']-new['2009q2'])/new['2008q2']
    un=pd.merge(new,uni,how='inner',on=['State','RegionName'])#ratio even negative
    un['ratio'].dropna(inplace=True)
    #un['univ']=True
    nun=pd.merge(new,uni,how='outer',on=['State','RegionName'])
    nun['univ'].fillna(False,inplace=True)
    nun=nun[nun.univ != True]
    t,p = ttest_ind(un['ratio'].dropna(), nun['ratio'].dropna())
    if(p<0.01):
        different=True
    else:
        differnet=False
    #better = "university town" if un['ratio'].mean() < nun['ratio'].mean() else "non-university town"
    c1= un['ratio'].mean()
    c2= nun['ratio'].mean()
    if( c1 < c2):
        better='university town'
    else:
        better='non-university town'
    #for i,chck in nun['univ']:
        #if(chck==True):
            #nun
    #dftt=dft[(dft['quarter'] >= st) & (dft['quarter'] <= end)]
    #fat['RegionName']=new['RegionName']
    #alltowns=df['RegionName']
    #newdf=pd.merge(uni,fat,how='inner',left_on='RegionName',right_on='RegionName')
    #newdf.dropna(inplace=True)
    #nwdf=fat-newdf
    return (different,p,better)


# In[32]:


run_ttest()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




