#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[4]:


df=pd.read_csv('datasets.csv')


# In[5]:


df.head(10)


# In[6]:


#column names
df.columns


# In[7]:


#type of data
df.info()


# In[8]:


#rows which filled with 1
df['step'].unique()


# In[9]:


#null values
df.isnull().sum()


# In[10]:


#rows and columns
df.shape


# In[11]:


#type of object
df['type'].unique()


# In[12]:


#counts of 
df.type.value_counts()


# In[13]:


type=df['type'].value_counts()


# In[14]:


type.index


# In[15]:


transaction=type.index


# In[16]:


quantity=type.values


# In[17]:


import plotly.express as px


# In[18]:


px.pie(df,values=quantity,names=transaction,hole=0.4,title="Distribution of Transaction Type")


# In[19]:


#to drop null values
df=df.dropna()


# In[20]:


#to replace values
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],value=[2,4,1,5,3],inplace=True)


# In[21]:


df['isFraud']=df['isFraud'].map({0:'No fraud',1:'fraud'})
df


# In[22]:


x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]


# In[23]:


y=df.iloc[:,-2]


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[26]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# In[27]:


LR = LogisticRegression(random_state=42)
KN = KNeighborsClassifier()
DC = DecisionTreeClassifier(random_state=42)
RF = RandomForestClassifier(random_state=42)


# In[28]:


#create list of your model names
models = [LR,KN,DC,RF]


# In[29]:


def plot_confusion_matrix(y_test,prediction):
    cm_ = confusion_matrix(y_test,prediction)
    plt.figure(figsize = (6,4))
    sns.heatmap(cm_, cmap ='coolwarm', linecolor = 'white', linewidths = 1, annot = True, fmt = 'd')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


#create function to train a model and evaluate accuracy
def trainer(model,X_train,y_train,X_test,y_test):
    #fit your model
    model.fit(X_train,y_train)
    #predict on the fitted model
    prediction = model.predict(X_test)
    #print evaluation metric
    print('\nFor {}, Accuracy score is {} \n'.format(model.__class__.__name__,accuracy_score(prediction,y_test)))
    print(classification_report(y_test, prediction)) #use this later
    plot_confusion_matrix(y_test,prediction)


# In[33]:


#loop through each model, training in the process
for model in models:
    trainer(model,xtrain,ytrain,xtest,ytest)


# In[34]:


model.fit(xtrain,ytrain)


# In[35]:


model.score(xtest,ytest)


# In[36]:


x


# In[41]:


model.predict([[2,9800,170136,160296]])


# In[42]:


model.predict([[1,132557.35,479803.00,347245.65]])


# In[43]:


model.predict([[1,181.00,181.00,0.00]])


# In[44]:


model=DecisionTreeClassifier()


# In[45]:


model.fit(xtrain,ytrain)


# In[46]:


model.score(xtest,ytest)


# In[47]:


x


# In[48]:


model.predict([[4,9800,170136,160296]])


# In[49]:


model= KNeighborsClassifier()


# In[50]:


model.fit(xtrain,ytrain)


# In[51]:


model.score(xtest,ytest)


# In[52]:


model.predict([[3,14140,90545,80627]])


# In[ ]:




