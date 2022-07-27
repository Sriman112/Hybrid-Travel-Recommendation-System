#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import re, math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv('data_collaborative.csv')
placeInfo=pd.read_csv('data_content.csv')

data=pd.merge(data,placeInfo,left_on='itemId',right_on="itemId")
userIds=data.userId
userIds2=data[['userId']]
data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])



userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['itemId'])


def similarity(user1,user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")



def nearestNeighbourRatings(activeUser,K):
    try:
        similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
        for i in userItemRatingMatrix.index:
            similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],userItemRatingMatrix.loc[i])
        
       
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        #print(similarityMatrix)
        nearestNeighbours=similarityMatrix[:K]
    
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
        
        predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                if predictedRating>0:
                   predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")
    #print(predictItemRating)
    return predictItemRating
activeUser=int(input("Enter userid: "))
predictItemRating=nearestNeighbourRatings(activeUser,10)


#print("The user's favorite places are: ")
#print(favoritePlace(activeUser,5))
#print("The recommended places for you are: ")
#print(topNRecommendations(activeUser,4))






#Content Based System

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
metadata = pd.read_csv('data_content.csv', low_memory=False)
print("Select your preferred category:\n1.heritage \n2.pilgirmage\n3.park\n4.museum")
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
cos=[]
for i in list(metadata['category']):
    #print(type(i))
    text2 = i
    vector2 = text_to_vector(text2)
    cosine = get_cosine(vector1, vector2)
    cos.append(cosine)
metadata['cosine']=cos
metadata['rating']=predictItemRating['Rating']
x=metadata['cosine']>0.0
rec=pd.DataFrame(metadata[x])
final=pd.DataFrame(rec,index=None,columns=['title','category','score'])



#Hybrid approach


scaling=MinMaxScaler()
hyb_scaled_df=scaling.fit_transform(rec[['score','rating']])
hyb_normalized_df=pd.DataFrame(hyb_scaled_df,columns=['score','rating'])
y = hyb_normalized_df.loc[:,['score']].values
x = hyb_normalized_df.loc[:,['rating']].values
rec['normalized_score']= y
rec['normalized_rating']= x
rec['hybrid_score'] = y * 0.5 + x * 0.5
pre_fin = rec.sort_values(['hybrid_score'], ascending=False)[:4]
final=pd.DataFrame(pre_fin,index=None,columns=['title','category','score','hybrid_score'])
print(final)



#RMSE MAE CALCULATION
rec['counts']=rec['count']
rec = rec._convert(numeric=True)
rec.hybrid_score=rec.hybrid_score.fillna(0) 
rec.distance=rec.distance.fillna(0)
rec.p_rating=rec.p_rating.fillna(0)
rec.counts=rec.counts.fillna(0)
rec.itemId=rec.itemId.fillna(0)
#rec.timestamp=rec.timestamp.fillna(0)

#vectorizer=CountVectorizer()

X = rec.iloc[:,[1,6,7,8]].values
Y = rec.loc[:,['hybrid_score']].values

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.30,random_state=0)


model2 = linear_model.LinearRegression()
model2.fit(X_train,Y_train)
prediction2=model2.predict(X_test)
MSE = mean_squared_error(Y_test,prediction2)
RMSE = math.sqrt(MSE)
print("RMSE :"+str(round(RMSE,2)))
MAE=mean_absolute_error(Y_test,prediction2)
print("MAE :"+str(round(MAE,2)))






# In[ ]:





# In[ ]:





# In[ ]:




