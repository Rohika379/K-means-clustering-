
#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the Dataset
data=pd.read_excel("C://Users//rohika/OneDrive//Desktop//360digiTMG assignment//h clustering data sets//Telco_customer_churn.xlsx")
data.head()

data.shape

#Customer ID and the target variable column are not to be included in the data

data.drop(['Customer ID','Count','Quarter'],axis=1,inplace=True)
data.head()

#Checking for null values

data.isnull().sum()
np.isnan(data.any())
data['Payment Method'].unique()

data.dtypes

cols=data.columns
cols
#Changing the column types

cat_cols=data.select_dtypes(exclude=['int','float']).columns
cat_cols

#Checking the data types
data.dtypes

enc_data=list(cat_cols)
enc_data=enc_data[:-1]
enc_data

#Importing  library for encoding
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

#Defining the function for column transformentation

data[enc_data]=data[enc_data].apply(lambda col:le.fit_transform(col))
data[enc_data].head()

data.head()


# Normalization function suing z std. all are continuous data , not considering city variable.
#<comp= function, Normalization fucntion is defined, Function is not Normalised
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame 
df_norm = norm_func(data.iloc[:, 0:])
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
data['clust'] = mb # creating a  new column and assigning it to new column 

data.head()
df_norm.head()




