# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:23:09 2019

@author: SURAJ BHADHORIYA
"""
#load libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#load dataset
df=pd.read_csv("C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/amazon_baby_subset.csv")

#load set of important words
import json
important_word=json.loads(open("C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/important_words.json").read())
print(important_word)

#remove punctuation
df['review']=df['review'].str.replace('[^\w\s]','')
df['review'].head()
print(df['review'])

#fillna values
df=df.fillna({'review':''})
#create colounm to all importtant_word how many time they repeat
for word in important_word:
    df[word]=df['review'].apply(lambda s: s.split().count(word))
    
# no pf +ive and -ive 1's  
pos_1=len(df[df['sentiment']==1]) 
neg_1=len(df[df['sentiment']==-1])  
print(pos_1)
print(neg_1)     
    
#convert df to multi dimension array
feature1=important_word
label=["sentiment"]
        
    
# using matrix to form bag of words
def get_numpy_data(dataframe,features,label):
    dataframe['constant']=1
    features=['constant']+features
    features_frame=dataframe[features]
    feature_matrix=features_frame.as_matrix()
    label_sarray=dataframe[label]
    label_array=label_sarray.as_matrix()
    return(feature_matrix,label_array)

x,y=get_numpy_data(df,feature1,label) 

print(x)
print(y)
#split data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#apply stocastic gradient descent with logestic regression         
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(alpha=0.0005,loss="hinge", penalty="l2" ,max_iter=10)
clf.fit(X_train,y_train)
score1=clf.score(X_test,y_test)
#accuracy
print("accuracy",score1)































    