

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

Ins = pd.read_csv("C:/Users/rohika/OneDrive/Desktop/360digiTMG assignment/Datasets_Kmeans/Insurance Dataset.csv")

Ins.describe()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Ins.iloc[:, :])

###### scree plot or elbow curve ############
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
Ins['clust'] = mb # creating a  new column and assigning it to new column 

Ins.head()
df_norm.head()
#Rearranging the columns

Ins = Ins.iloc[:,[5,0,1,2,3,4]]
Ins.head()

Ins.iloc[:, 2:8].groupby(Ins.clust).mean()

Ins.to_csv("Insurance.csv", encoding = "utf-8")
#Saving the file to my working directory

import os
os.getcwd()
