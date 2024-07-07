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


# In[1]:


import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df = pd.read_csv('datasets.csv')
df = df.dropna()
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[2, 4, 1, 5, 3], inplace=True)
X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)


# In[ ]:


def predict_fraud():
    type = int(type_var.get())
    amount = float(amount_var.get())
    old_balance = float(old_balance_var.get())
    new_balance = float(new_balance_var.get())

    prediction = model.predict([[type, amount, old_balance, new_balance]])[0]
    result.set("Fraud" if prediction == 1 else "Not Fraud")

# Set up the main window
root = tk.Tk()
root.title("Online Payment Fraud Detection")

# Set up input fields
tk.Label(root, text="Type (1-5):").grid(row=0, column=0)
type_var = tk.StringVar()
tk.Entry(root, textvariable=type_var).grid(row=0, column=1)

tk.Label(root, text="Amount:").grid(row=1, column=0)
amount_var = tk.StringVar()
tk.Entry(root, textvariable=amount_var).grid(row=1, column=1)

tk.Label(root, text="Old Balance:").grid(row=2, column=0)
old_balance_var = tk.StringVar()
tk.Entry(root, textvariable=old_balance_var).grid(row=2, column=1)

tk.Label(root, text="New Balance:").grid(row=3, column=0)
new_balance_var = tk.StringVar()
tk.Entry(root, textvariable=new_balance_var).grid(row=3, column=1)

# Prediction button
tk.Button(root, text="Predict", command=predict_fraud).grid(row=4, column=0, columnspan=2)

# Result display
result = tk.StringVar()
tk.Label(root, textvariable=result).grid(row=5, column=0, columnspan=2)

root.mainloop()


# In[ ]:


import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    file_path = filedialog.askopenfilename()
    if file_path:
        data = pd.read_csv(file_path)
        return data

def preprocess_data(df):
    df = df.dropna()
    df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[2, 4, 1, 5, 3], inplace=True)
    return df

def train_model(df):
    X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
    y = df['isFraud']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    return model, xtest, ytest

def predict_fraud():
    try:
        type = int(type_var.get())
        amount = float(amount_var.get())
        old_balance = float(old_balance_var.get())
        new_balance = float(new_balance_var.get())

        prediction = model.predict([[type, amount, old_balance, new_balance]])[0]
        result.set("Fraud" if prediction == 1 else "Not Fraud")
    except Exception as e:
        result.set(f"Error: {e}")

# Set up the main window
root = tk.Tk()
root.title("Online Payment Fraud Detection")

# Load dataset button
tk.Button(root, text="Load Dataset", command=lambda: preprocess_and_train()).grid(row=0, column=0, columnspan=2)

def preprocess_and_train():
    global model, xtest, ytest
    df = load_data()
    df = preprocess_data(df)
    model, xtest, ytest = train_model(df)
    result.set("Model trained successfully")

# Set up input fields
tk.Label(root, text="Type (1-5):").grid(row=1, column=0)
type_var = tk.StringVar()
tk.Entry(root, textvariable=type_var).grid(row=1, column=1)

tk.Label(root, text="Amount:").grid(row=2, column=0)
amount_var = tk.StringVar()
tk.Entry(root, textvariable=amount_var).grid(row=2, column=1)

tk.Label(root, text="Old Balance:").grid(row=3, column=0)
old_balance_var = tk.StringVar()
tk.Entry(root, textvariable=old_balance_var).grid(row=3, column=1)

tk.Label(root, text="New Balance:").grid(row=4, column=0)
new_balance_var = tk.StringVar()
tk.Entry(root, textvariable=new_balance_var).grid(row=4, column=1)

# Prediction button
tk.Button(root, text="Predict", command=predict_fraud).grid(row=5, column=0, columnspan=2)

# Result display
result = tk.StringVar()
tk.Label(root, textvariable=result).grid(row=6, column=0, columnspan=2)

root.mainloop()


# In[ ]:




