#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re, math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error


WORD = re.compile(r'\w+')

#applying cosine similarity for finding similarities between user interests and places
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

#remove spaces from the category column of dataset
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#calulating weighted rating of places



metadata = pd.read_csv('data_content.csv',nrows=160, low_memory=False)
#print(metadata.head())
print("Select your preferred category: \n1.heritage \n2.pilgirmage\n3.park\n4.museum")
text1 = input("Enter User Interests: ")   #user preference
vector1 = text_to_vector(text1)
C = metadata['p_rating'].mean()
m = metadata['count'].quantile(0.75)

def weighted_rating(x, m=m, C=C):
    v = x['count']
    R = x['p_rating']
    # Calculation based on the Bayesian Rating Formula
    return (v/(v+m) * R) + (m/(m+v) * C)

metadata['category'] = metadata['category'].apply(clean_data)
metadata['score'] = metadata.apply(weighted_rating, axis=1)
#print(metadata.head())
cos=[]
for i in list(metadata['category']):
    #print(type(i))
    text2 = i
    vector2 = text_to_vector(text2)
    #print(vector2)
    cosine = get_cosine(vector1, vector2)
    cos.append(cosine)
metadata['cosine']=cos
x=metadata['cosine']>0.0
rec=pd.DataFrame(metadata[x])
pre_final=rec.sort_values('score',ascending=False)[:4]
dest=list(rec['title'])
#print(type(dest))


final=pd.DataFrame(pre_final,index=None,columns=['title','category','score'])
print(final)




#RMSE Calculation
rec['counts']=rec['count']
rec = rec._convert(numeric=True)
rec.distance=rec.distance.fillna(0)
rec.p_rating=rec.p_rating.fillna(0)
rec.itemId=rec.itemId.fillna(0)
rec.counts=rec.counts.fillna(0)
X = rec.iloc[:,[1,6,7,8]].values
y = rec.loc[:,['score']].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

model2 = linear_model.LinearRegression()
model2.fit(X_train,y_train)
prediction2=model2.predict(X_test)
MSE = mean_squared_error(y_test,prediction2)
RMSE = math.sqrt(MSE)
print("RMSE :"+str(round(RMSE,2)))
MAE=mean_absolute_error(y_test,prediction2)
print("MAE :"+str(round(MAE,2)))










