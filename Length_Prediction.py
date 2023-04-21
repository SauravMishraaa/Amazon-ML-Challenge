#!/usr/bin/env python
# coding: utf-8

# In[78]:


#importing the desired libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[79]:


#loading the dataset
dataset=pd.read_csv("train.csv")


# In[80]:


#splitting the training and testing data
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[81]:


#to view the original dataset
print(dataset)


# In[82]:


#to view the dataset
print(x)


# In[83]:


#to view the result 
print(y)


# In[84]:


#to count the number of missing data in our dataset
print(dataset.isnull().sum())


# In[85]:


# replacing the missing data with the averege value of the column
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer = imputer.fit(dataset)
dataset.iloc[:,:] = imputer.transform(dataset)
print(dataset)


# In[86]:


#to count the number of missing data in our dataset
print(dataset.isnull().sum())


# In[87]:


#using nlp to clean the text the bulletpoints
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  bp = re.sub('[^a-zA-Z]', ' ', dataset['BULLET_POINTS'][i])
  bp = bp.lower()
  bp = bp.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  bp = [ps.stem(word) for word in bp if not word in set(all_stopwords)]
  review = ' '.join(bp)
  corpus.append(bp)


# In[88]:


print(corpus)


# In[89]:


#using nlp to clean the decsription
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus1 = []
for i in range(0, 1000):
  dp = re.sub('[^a-zA-Z]', ' ', dataset['DESCRIPTION'][i])
  dp = dp.lower()
  dp = dp.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  dp = [ps.stem(word) for word in dp if not word in set(all_stopwords)]
  review = ' '.join(dp)
  corpus1.append(dp)


# In[90]:


print(corpus1)


# In[91]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


# In[ ]:




