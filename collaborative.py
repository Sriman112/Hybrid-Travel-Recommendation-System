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
import math


data=pd.read_csv('data_collaborative.csv')
placeInfo=pd.read_csv('data_content.csv')
data.head()
placeInfo.head()
data=pd.merge(data,placeInfo,left_on='itemId',right_on="itemId")
data.head()
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
        nearestNeighbours=similarityMatrix[:K]
    
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
        

        predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                   #print(predictedRating)
                if predictedRating>0:
                   predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")    
    return predictItemRating


def topNRecommendations(activeUser,N):
    try:
        predictItemRating=nearestNeighbourRatings(activeUser,10)
        placeAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
        predictItemRating=predictItemRating.drop(placeAlreadyWatched)
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
        topRecommendationTitles=(placeInfo.loc[placeInfo.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return list(topRecommendationTitles.title)


activeUser=int(input("Enter userid: "))
print("The recommended places for you are: ")
print(topNRecommendations(activeUser,4))


#RSME MAE CALULATION 

predictItemRating=nearestNeighbourRatings(activeUser,10)
d=pd.read_csv('data_collaborative.csv',names=['userId','itemId','rating','timestamp'])
d['predicted_rating']=predictItemRating['Rating']
d = d._convert(numeric=True)
d.userId=d.userId.fillna(0)
d.itemId=d.itemId.fillna(0)
d.rating=d.rating.fillna(0)
d.timestamp=d.timestamp.fillna(0)
d.predicted_rating=d.predicted_rating.fillna(0)
X = d.loc[:,['userId','itemId','rating','timestamp']].values
y = d.loc[:,['predicted_rating']].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)
model2 = linear_model.LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=True)
model2.fit(X_train,y_train)
prediction2=model2.predict(X_test)
MSE = mean_squared_error(y_test,prediction2)
RMSE = np.sqrt(MSE)
MAE=mean_absolute_error(y_test,prediction2)
print("RMSE :"+str(round(RMSE,2)))
print("MAE :"+str(round(MAE,2)))



# In[ ]:





# In[ ]:





# In[ ]:




