
# coding: utf-8

# # Ad Clicks Prediction
# 
# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user will click on an Advertisement. I will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style='ggplot'
import seaborn as sns
sns.set_style('white')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[3]:


ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[4]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[5]:


ad_data.info()


# In[6]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# ** Create a histogram of the Age**

# In[7]:


plt.figure(figsize=(10,4))
sns.distplot(ad_data['Age'],kde=False,bins=30)


# **Create a jointplot showing Area Income versus Age.**

# In[8]:


sns.jointplot(x='Age',y='Area Income',data= ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[9]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data= ad_data,kind='kde',cmap='inferno')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[10]:


sns.jointplot(y='Daily Internet Usage',x='Daily Time Spent on Site',data= ad_data,cmap='inferno')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[11]:


sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='inferno')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train the model!

# ** Split the data into training set and testing set using train_test_split**

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]

y = ad_data['Clicked on Ad']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ** Train and fit a logistic regression model on the training set.**

# In[15]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()


# In[16]:


logreg.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[17]:


predictions = logreg.predict(X_test)


# ** Create a classification report for the model.**

# In[18]:


from sklearn import metrics


# In[19]:


class_rep = metrics.classification_report(y_test,predictions)
print(class_rep)


# In[20]:


conf_matrix = metrics.confusion_matrix(y_test,predictions)
print(conf_matrix)


# ## End! 87% accuracy is not bad for such small dataset. :)
