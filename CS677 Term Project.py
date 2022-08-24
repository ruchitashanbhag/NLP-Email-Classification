#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from math import log, sqrt
import re # for handling string
import string # for handling mathematical operations
import math
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[ ]:


conda install -c conda-forge wordcloud=1.6.0 


# In[2]:


df= pd.read_csv("emails.csv")
df.shape#(48076, 5)
df.head()


# In[3]:


df.info() # 48076, object(4)
df.describe()
df.isnull().sum() # no null values


# In[ ]:


df.columns


# In[4]:


df['Class'].unique() # abusive, non abusive
df['Class'].value_counts() # abusive (3410), non-abusive (44666)


# In[5]:


df['content'].unique()
df['content'].value_counts()


# In[6]:


df1= df.iloc[:,3:5]
df1.head(5)


# In[ ]:


#df1.loc[df1.Class=="Abusive","Class"] = 1
#df1.loc[df1.Class=="Non Abusive","Class"] = 0
#df1['Class'].value_counts()


# In[ ]:


duplicate= df[df1.duplicated()] 
df1= df1.drop_duplicates() 
df1.shape #(24656, 2)
df1['Class'].value_counts()# 0:23014, 1:1642


# In[7]:


# text cleaning
df1['cleaned']=df1['content'].apply(lambda x: x.lower()) # convert to lower cases
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x)) # remove digits and words with digits
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)) # remove punctuation
df1['cleaned']=df1['cleaned'].apply(lambda x: re.sub(' +',' ',x)) # remove extra spaces
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n\n')[0])
df1['cleaned']=df1['cleaned'].apply(lambda x: x.split('\n')[0])
df1['cleaned'].head()


# In[8]:


# tokenize one sentence from the dataframe
sample= df1.iloc[0]
print(sample['cleaned'])
print (nltk.word_tokenize(sample['cleaned']))


# In[9]:


# tokenise entire df
def identify_tokens(row):
    new = row['cleaned']
    tokens = nltk.word_tokenize(new)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df1['cleaned'] = df1.apply(identify_tokens, axis=1)
df1['cleaned'].head()


# In[10]:


# stemming
stemming = PorterStemmer()
def stem_list(row):
    my_list = row['cleaned']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

df1['stemmed_words'] = df1.apply(stem_list, axis=1)
df1['stemmed_words'].head()


# In[11]:


# remove stopwords
stop_words = []
with open('stop-2.txt') as f:
    stop_words = f.read()


# In[12]:


# getting list of stop words
stop_words = stop_words.split("\n")               

def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

df1['stem_meaningful'] = df1.apply(remove_stops, axis=1)
df1['stem_meaningful'].head()


# In[13]:


# rejoin meaningful stem words in single string like a sentence
def rejoin_words(row):
    my_list = row['stem_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

df1['final'] = df1.apply(rejoin_words, axis=1)


# In[14]:


# check the cleaned mails
for index,text in enumerate(df1['final'][50:55]):
  print('Mail %d:\n'%(index+1),text) 


# In[15]:


# 1= abusive and 0= non abusive wordcloud
spam= ' '.join(list(df1[df1['Class'] == "Abusive"]['final']))
spam_cloud = WordCloud(width = 512, height = 512).generate(spam)
plt.figure(figsize = (10,8), facecolor = 'k')
plt.imshow(spam_cloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[16]:


ham= ' '.join(list(df1[df1['Class'] == "Non Abusive"]['final']))
ham_cloud = WordCloud(width = 512, height = 512).generate(ham)
plt.figure(figsize = (10,8), facecolor = 'k')
plt.imshow(ham_cloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[17]:


# Preparing email texts into word count matrix format 
mail= df1.loc[:,['final','Class']]
# removing empty rows 
mail.shape #(24656,2)
mail['final'].replace('', np.nan, inplace=True)
mail.dropna(subset=['final'], inplace=True)
mail.shape #(20164,2)


# In[18]:


def split_into_words(i):
    return (i.split(" "))

#create vectors from words
from sklearn.feature_extraction.text import CountVectorizer

# Preparing email texts into word count matrix format 
mail_vector = CountVectorizer(analyzer=split_into_words).fit(mail.final)


# In[19]:


# vectorising all mails
all_emails_matrix = mail_vector.transform(mail['final'])
all_emails_matrix.shape # (20164, 9698)
type(all_emails_matrix)


# In[20]:


# splitting data into train and test data sets 

from sklearn.model_selection import train_test_split
train,test = train_test_split(mail,test_size=0.3)


# In[21]:


# For training messages
train_matrix = mail_vector.transform(train.final)
train_matrix.shape # (14114, 9698)
type(train_matrix)


# In[22]:


# For testing messages
test_matrix = mail_vector.transform(test.final)
test_matrix.shape # (6050, 9698)
type(test_matrix)


# In[23]:


####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_matrix,train['Class'])


# In[24]:


train_pred_m = classifier_mb.predict(train_matrix)
accuracy_train_m = np.mean(train_pred_m==train['Class']) # 96.1%
accuracy_train_m


# In[25]:


test_pred_m = classifier_mb.predict(test_matrix)
accuracy_test_m = np.mean(test_pred_m==test['Class']) # 93.3%
accuracy_test_m


# In[26]:


# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_matrix.toarray(),train['Class'].values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==train['Class']) # 75.9%
accuracy_train_g


# In[27]:


test_pred_g = classifier_gb.predict(test_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==test['Class']) # 68.57%
accuracy_test_g


# In[28]:


# Learning Term weighting and normalizing on entire emails
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_matrix)
train_tfidf.shape # (14114, 9698)


# In[29]:


# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_matrix)
test_tfidf.shape #  (6050, 9698)


# In[30]:


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,train['Class'])
train_pred_m_tfidf = classifier_mb.predict(train_tfidf)
accuracy_train_m_tfidf = np.mean(train_pred_m_tfidf==train['Class']) # 94.45%
accuracy_train_m_tfidf


# In[31]:


test_pred_m_tfidf = classifier_mb.predict(test_tfidf)
accuracy_test_m_tfidf = np.mean(test_pred_m_tfidf==test['Class']) # 94.06%
accuracy_test_m_tfidf
confusion_matrix = confusion_matrix(train_pred_m_tfidf,train['Class'])
print (confusion_matrix)
print(classification_report(train_pred_m_tfidf,train['Class']))




# In[32]:


# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),train['Class'].values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g_tfidf = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g_tfidf = np.mean(train_pred_g_tfidf == train['Class']) # 73.59%
accuracy_train_g_tfidf


# In[33]:


test_pred_g_tfidf = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g_tfidf = np.mean(test_pred_g_tfidf==test['Class']) # 65.06%
accuracy_test_g_tfidf


# In[34]:


import pickle

import joblib
from joblib import dump
joblib.dump(classifier_mb,'nlp_model1.pkl')
joblib.dump(mail_vector,'vector1.pkl')


# In[35]:


print(type(mail_vector))


# In[36]:


f = open('mail_vector.pkl', 'wb')
pickle.dump(mail_vector, f)


# In[ ]:




