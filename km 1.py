
#Required libraries
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset

dataset = pd.read_excel("C:/Users/rohika/OneDrive/Desktop/360digiTMG assignment/Datasets_Kmeans/EastWestAirlines (1).xlsx" , sheet_name ='data')
dataset.head()

#EDA calculation

dataset= dataset.rename(columns={'ID#':'ID', 'Award?':'Award'})

dataset1 =  dataset.drop(['ID','Award'], axis=1)
dataset1.head()

#Normalization function

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

#normalised data
df_norm = norm_func(dataset1.iloc[:,0:])
df_norm.describe()

###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
dataset['clust'] = mb # creating a  new column and assigning it to new column 

dataset.head()
df_norm.head()

dataset = dataset.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
dataset.head()

dataset.iloc[:, 2:12].groupby(dataset.clust).mean()

dataset.to_csv("East.csv", encoding = "utf-8")

import os
os.getcwd()
