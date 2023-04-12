#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Import the necessary libraries: Streamlit, sklearn.datasets, and sklearn.ensemble.
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import streamlit as st


# In[2]:


#2.Load the iris dataset using the "datasets.load_iris()" function and assign the data and target variables to "X" and "Y", respectively.
iris = datasets.load_iris()
X = iris.data
y= iris.target


# In[6]:


print(X)
print(y)


# In[12]:


iris


# In[3]:


#3.Set up a Random Forest Classifier and fit the model using the "RandomForestClassifier()" and "fit()" functions
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2) #splitting data with test size of 20%


# In[5]:


model=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 10 decision trees
model.fit(x_train, y_train)  #Training our model
y_pred=model.predict(x_test)  #testing our model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  #Measuring the accuracy of our model


# In[8]:


#4.Create a Streamlit app using the "streamlit.title()" and "streamlit.header()" functions to add a title and header to the app.
st.title("Iris Dataset Predicator") #Create title
st.header("App that predicts the type of iris flower") #Create header


# In[25]:


#5.Add input fields for sepal length, sepal width, petal length, and petal width using the "streamlit.slider()" function. Use the minimum, maximum, and mean values of each feature as the arguments for the function
sepal_length = st.slider("Sepal Length", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal Length", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.slider("Petal Width", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))


# In[32]:


#6.Define a prediction button using the "streamlit.button()" function that takes in the input values and uses the classifier to predict the type of iris flower

prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    


# In[35]:


#7.Use the "streamlit.write()" function to display the predicted type of iris flower on the app.
if st.button("Predict"):
    if prediction[0] == 0:
        st.write("The predicted type of iris flower is Setosa")
    elif prediction[0] == 1:
        st.write("The predicted type of iris flower is Versicolor")
    else:
        st.write("The predicted type of iris flower is Virginica")


# In[ ]:





# In[ ]:




