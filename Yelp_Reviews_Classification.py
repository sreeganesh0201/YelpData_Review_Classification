
# coding: utf-8

# In[ ]:


code 1
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing The Dataset
dataset = pd.read_csv('review.csv')
dataset.keys()
sample=dataset[0:10000]
sample['Binary'] = np.where(sample['stars']>=3, 1, 0)


# In[ ]:


#We have to clean all the text in the reviews dataset since we are creating one independent variable for each word. 
#This is done using Natural Language Processing techniques


# In[ ]:


#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,10000):
text = re.sub('[^a-zA-Z]', ' ', sample['text'][i])
text = text.lower()
text = text.split()
ps = PorterStemmer()
text = [ps.stem(word) for word in text if not word in (set(stopwords.words('english')) - set(['no', 'not']))]
text = ' '.join(text)
corpus.append(text)


# In[ ]:


#I used a tool called ‘CountVectorizer’ for creating Bag of words model, 
#which is imported from sklearn through feature extraction


# In[ ]:


#Creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 15000)
X = cv.fit_transform(corpus).toarray()
y = sample.iloc[:, 10].values


# In[ ]:


#So, now we created the matrix of independent variables with 15,000 columns and a dependent variable ‘Binary’
#We will now implement the Naive Bayes model and Logistic Regression model on our independent and dependent variables.


# In[ ]:


#Naive Bayes Model
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred)


# In[ ]:


Nb_Mod_Acc


# In[ ]:


#Logistic Regression
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_log = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test, y_pred_log)


# In[ ]:


Log_Mod_Acc

