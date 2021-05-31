


import pandas as pd
#<comp = matplotlib,library for visualization is imported, matplotlib not imported,seaborn
import matplotlib.pylab as plt 

#<comp=read_csv,input file is being read,read_csv() not used;read_excel
crime = pd.read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\h clustering data sets\\crime_data.csv")
#<optional = .columns, Display columns, Columns not displayed
crime.columns
#<optional = .isna, Ckecking for NAs, NAs not checked;optional = .sum, total NAs count
crime.isna().sum()
#optional = .isnull, Checking for NULL values, NULL values not checked; Optional = .sum, total NULL count
crime.isnull().sum()


# Normalization function suing z std. all are continuous data , not considering city variable.
#<comp= function, Normalization fucntion is defined, Function is not Normalised
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
	  
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
#<optional = .describe, analysing data informations like count,

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
crime['clust'] = mb # creating a  new column and assigning it to new column 

crime.head()
df_norm.head()
#Rearranging the columns

crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

crime.iloc[:, 2:5].groupby(crime.clust).mean()

crime.to_csv("crime.csv", encoding = "utf-8")

#To save in my working directory
import os
os.getcwd()
