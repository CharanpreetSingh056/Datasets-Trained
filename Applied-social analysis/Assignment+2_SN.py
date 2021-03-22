
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[1]:


import networkx as nx
import numpy as np
import pandas as pd
# This line must be commented out when submitting to the autograder
#!head email_network.txt


# In[141]:


#[m for m in nx.__dir__() if 'pandas' in m]


# ### Question 1
# 
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[2]:


def answer_one():
    df=pd.read_csv('email_network.txt',sep="\t",names=['sender','recipient','time'],skiprows=1)
    G=nx.from_pandas_dataframe(df,'sender','recipient',create_using=nx.MultiDiGraph())
    #df['time']=df['time'].map(lambda x:str(x))
    #df_new=df.drop('time',axis=1)
    #G=nx.from_pandas_dataframe(df_new,'sender','recipient')
    #G=nx.read_edgelist('email_network.txt',delimiter="\t")
    #G_new=nx.MultiDiGraph(G)
    # Your Code Here
    
    return G#G_new# Your Answer Here


# In[3]:


answer_one()


# 
# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[4]:


def answer_two():
    G=answer_one()
    sett=set()
    edge=G.edges(data=True)
    for i in edge:
        sett.add(i[0])
        sett.add(i[1])
        
    
    # Your Code Here
    
    return (len(sett),len(edge)) # Your Answer Here


# In[5]:


answer_two()


# 
# 
# 
# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[6]:


def answer_three():
    G=answer_one()
    f1=nx.is_strongly_connected(G)
    df=pd.read_csv('email_network.txt',sep="\t",skiprows=1,names=['sender','receiver','time'])
    G2=nx.from_pandas_dataframe(df,'sender','receiver')   
    f2=nx.is_connected(G2)
    # Your Code Here
    
    return (f1,f2)# Your Answer Here


# In[7]:


answer_three()


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[8]:


def answer_four():
    G=answer_one()
    mx=-1
    all_connected=nx.weakly_connected_components(G)
    for i in all_connected:
        mx=max(mx,len(i))
        
    # Your Code Here
    
    return mx# Your Answer Here


# In[9]:


answer_four()


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[10]:


def answer_five():
    G=answer_one()
    mx=-1
    all_connected=nx.strongly_connected_components(G)
    for i in all_connected:
        mx=max(mx,len(i))
        
    # Your Code Here
    
    return mx# Your Answer Here


# In[11]:


answer_five()


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[12]:


def answer_six():
    G=answer_one()
    mx=-1
    chck=answer_five()
    al=nx.strongly_connected_component_subgraphs(G)
    for i in al:
        if len(i.nodes())==chck :
            G_sc=i
            break
    #G_sc=[ len(graph.nodes()) for graph in al ] #if len(graph.nodes())==answer_five() ]
        
        
    # Your Code Here
    
    return G_sc# Your Answer Here


# In[13]:


answer_six()


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[14]:


def answer_seven():
    G=answer_six()
    ans=nx.average_shortest_path_length(G)    
    # Your Code Here
    
    return ans# Your Answer Here


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[15]:


def answer_eight():
    G=answer_six()
    ans=nx.diameter(G)
    
    
    return ans# Your Answer Here


# In[16]:


answer_eight()


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[17]:


def answer_nine():
    G = answer_six()
    ans=set(nx.periphery(G))
    return set([ str(i) for i in ans])


# In[18]:


answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[19]:


def answer_ten():
    G=answer_six()
    ans=nx.center(G)
        
    # Your Code Here
    
    return set([ str(i) for i in ans])# Your Answer Here


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[20]:


def answer_eleven():
    G=answer_six()
    al=nx.periphery(G)
    dia=nx.diameter(G)
    fa=[]
    for i in al:
        temp=nx.shortest_path_length(G, source=i)
        cnt=0
        vls=temp.values()
        for val in vls:
            if val == dia:
                cnt=cnt+1
        fa.append(cnt)
        #if min(vls) < dia:
            #continue
        #fa.append(i)
        #f=0
        #for j in temp:
            #if( j < dia):
                #f=1
                #break
        #if f==0:
            #fa.append(i)
        
    new=np.array(fa)
    indmax=new.argmax()
    
        
    
    # Your Code Here
    
    return (str(al[indmax]),max(fa)) #Your Answer Here


# In[21]:


answer_eleven()


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[27]:


def answer_twelve():
    G=answer_six() 
    cn=nx.center(G)[0] 
    p=answer_eleven() 
    n1 = int(p[0]) 
    #ans=nx.minimum_node_cut(G,n1,cn)
    #fa=[i for i in ans]
    # Your Code Here 
    return len(nx.minimum_node_cut(G,cn,n1))# Your Answer Here


# In[28]:


answer_twelve()


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[23]:


def answer_thirteen():
    G=answer_six()
    edgess=G.edges(data=True)
    G_new=nx.Graph(data=edgess)
        
    # Your Code Here
    
    return G_new# Your Answer Here


# In[24]:


answer_thirteen()


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[25]:


def answer_fourteen():
    G=answer_thirteen()
    a1=nx.average_clustering(G)
    a2=nx.transitivity(G)
        
    # Your Code Here
    
    return (a2,a1)# Your Answer Here


# In[ ]:





# In[ ]:




