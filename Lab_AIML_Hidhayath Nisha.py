#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


cd C:\AIML_LAB


# In[3]:


df = pd.read_csv('bottle.csv')
df_binary = df[['Salnty', 'T_degC']]


# In[4]:


df_binary.columns = ['Sal', 'Temp']


# In[5]:


df_binary.head()


# In[6]:


sns.lmplot(x ="Sal", y ="Temp", data = df_binary, order = 2, ci = None)


# In[7]:


# Eliminating NaN or missing input numbers
df_binary.fillna(method ='ffill', inplace = True)


# In[8]:


X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary['Temp']).reshape(-1, 1)


# In[9]:


df_binary.dropna(inplace = True)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[11]:


regr = LinearRegression()
  
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# In[12]:


y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()
# Data scatter of predicted values


# In[13]:


df_binary500 = df_binary[:][:500]


# In[14]:


sns.lmplot(x ="Sal", y ="Temp", data = df_binary500,order = 2, ci = None)


# In[15]:


df_binary500.fillna(method ='ffill', inplace = True)

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500['Temp']).reshape(-1, 1)

df_binary500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# In[16]:


y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()


# In[ ]:




