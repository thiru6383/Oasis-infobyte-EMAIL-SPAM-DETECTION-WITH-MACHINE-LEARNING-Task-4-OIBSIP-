#!/usr/bin/env python
# coding: utf-8

# # EMAIL SPAM DETECTION WITH MACHINE LEARNING:
#   

# In[1]:


#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #It convert text document into numeric representation
from sklearn.linear_model import LogisticRegression   # Logistic Regression Algorithm
from sklearn.metrics import accuracy_score


# In[2]:


data_set=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\oasis infobytes\spam.csv",encoding='ISO-8859-1')


# In[3]:


data_set.head()


# In[ ]:





# In[ ]:





# # Data cleaning:

# In[4]:


dataset=data_set.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5',
       'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
       'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
       'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
       'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22',
       'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'],axis=1)


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset.duplicated().sum() 


# In[9]:


dataset=dataset.drop_duplicates(keep="first") #Removing the duplicate


# In[10]:


dataset.duplicated().sum()   #rechecking Duplicate values


# In[11]:


dataset.shape


# In[12]:


#rename the columns
dataset.rename(columns={'v1':'Category','v2':'Message'},inplace=True)
dataset.head()


# In[ ]:





# In[ ]:





# # Data Visualization

# In[13]:


plt.pie(dataset['Category'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[ ]:





# In[ ]:





# # Machine learning model

# In[14]:


dataset.loc[dataset['Category']=='spam','Category',]=0    # spam = 0 and Ham = 1
dataset.loc[dataset['Category']=='ham','Category',]=1


# In[15]:


X=dataset['Message']
Y=dataset['Category']


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[19]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[20]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[21]:


feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)


# In[22]:


Y_train=Y_train.astype(int)
Y_test=Y_test.astype(int)


# In[23]:


Y_train


# In[24]:


Y_test


# In[25]:


X_train


# In[26]:


X_test


# In[27]:


print(X_test_features)


# In[28]:


model = LogisticRegression()
model.fit(X_train_features, Y_train)


# In[29]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[30]:


print("Accuracy on training data: ",accuracy_on_training_data)


# In[31]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[32]:


print("Accuracy on test data: ",accuracy_on_test_data)


# In[ ]:





# In[ ]:





# # Final result or prediction:

# In[40]:


input_mail=[" #spam Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
input_data_features = feature_extraction.transform(input_mail)
prediction=model.predict(input_data_features)
print("Our Mail is: ",prediction)
if(prediction[0]==1):
    print("Ham mail")
else:
    print("Spam mail")
    
    
# some sample messages:
    #spam Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
    #Even my brother is not like to speak with me. They treat me like aids patent.
    #WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
    #I HAVE A DATE ON SUNDAY WITH WILL!!
    #Nah I don't think he goes to usf, he lives around here though
    #FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv
    


# In[ ]:





# In[ ]:


#Thanking You...


#                                                                                                   -Thiruvalluvan G
