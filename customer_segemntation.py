#This project applies unsupervised learning (clustering) to identify distinct customer segments
#based on demographic and behavioral features.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.decomposition import PCA

#-----------------------------------
# load data
#-----------------------------------
Dataset = pd.read_csv('Train.csv')

#----------------------------------
# Remove space from columns name
#----------------------------------
Dataset.columns = Dataset.columns.str.strip()

## drop segmentation because there is no target in clustering its only leak our data
Dataset = Dataset.drop(columns=['Segmentation', 'ID'])

#-----------------------------------
# view some data
#-----------------------------------
print(" First 5 Dataset :\n", Dataset.head())
print(" Last 5 Dataset :\n", Dataset.tail())
print("Null columns :\n", Dataset.isnull().sum())

#-----------------------------------
# dataset overall describe (to check outlier , scaling and normalization also )
#-----------------------------------
print("\n Summary about dataset:\n", Dataset.describe())

#-----------------------------------------
# Filling missing value in respective columns
#-----------------------------------------
for col in ['Ever_Married', 'Graduated', 'Profession', 'Var_1']:
    Dataset[col] = Dataset[col].fillna(Dataset[col].mode()[0])

print("Null columns :\n", Dataset.isnull().sum())

for col in ['Work_Experience', 'Age']:
    Dataset[col] = Dataset[col].fillna(Dataset[col].median())

Dataset['Family_Size'] = Dataset['Family_Size'].fillna(Dataset['Family_Size'].mean())

print("Null columns :\n", Dataset.isnull().sum())

#-----------------------------------------------
# ordinal encoder
#-----------------------------------------------
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
Dataset['Spending_Score'] = oe.fit_transform(Dataset[['Spending_Score']])

#-------------------------------------------------
# outliers remove
#-------------------------------------------------
for i in ['Work_Experience', 'Age', 'Family_Size']:
    q1 = Dataset[i].quantile(0.25)
    q3 = Dataset[i].quantile(0.75)
    IQR = q3 - q1
    min_range = q1 - (1.5 * IQR)
    max_range = q3 + (1.5 * IQR)
    Dataset = Dataset[Dataset[i] <= max_range]

#--------------------------------------------------
# remove duplicated data
#--------------------------------------------------
Dataset = Dataset.drop_duplicates()
print(" duplicated data  :\n", Dataset.duplicated().value_counts())

#---------------------------------------------------
# split in numerical and categorical data
#---------------------------------------------------
num_features = ['Work_Experience', 'Age', 'Family_Size']
nominal_features = ['Ever_Married', 'Graduated', 'Profession', 'Var_1', 'Gender']
ordinal_features = ['Spending_Score']

#---------------------------------------------
# numerical pipeline
#---------------------------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('square', FunctionTransformer(func=lambda x: x**2))  # square all numeric values
])

#---------------------------------------------
# categorical pipeline
#---------------------------------------------
nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('hot', OneHotEncoder(drop='first'))
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

#--------------------------------------------------
# combine pipeline into 1
#--------------------------------------------------
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('nom', nominal_pipeline, nominal_features),
    ('ord', ordinal_pipeline, ordinal_features)
])

# To find the k (centroid /cluster ) we use both silhotte and elbow method
X_processed = preprocessor.fit_transform(Dataset)
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_processed)
    wcss.append(kmeans.inertia_)

# Visualization using elbow method
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), wcss, label='WCSS')
plt.title("Elbow method")
plt.xlabel("Number of cluster(k)")
plt.ylabel("WCSS")
plt.legend()
plt.grid()
plt.show()

#------------------------------------------
# Cluster model
#-------------------------------------------
cluster_model = Pipeline([
    ('processor', preprocessor),
    ('cluster', KMeans(
        n_clusters=4,
        init='k-means++',
        n_init=100,
        verbose=0,
        max_iter=500,
        random_state=42
    ))
])

cluster_model.fit(Dataset)
Dataset['Cluster'] = cluster_model.predict(Dataset)

# NOTE:
# PCA removes noise and redundant dimensions created by OneHotEncoding.
# KMeans works better in compact, low-dimensional space.
# PCA + KMeans below is used ONLY to improve cluster separation
# and silhouette score. Final cluster interpretation is based
# on PCA-based KMeans results.

#--------------------------------
# transform dataset and apply PCA
#--------------------------------
X_processed = cluster_model.named_steps['processor'].transform(Dataset)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_processed)

# Use PCA-transformed data for improved KMeans
kmeans = KMeans(n_clusters=4, n_init=100, max_iter=500, random_state=42)
labels = kmeans.fit_predict(X_pca)
Dataset['Cluster'] = labels

# Silhouette score after PCA
sil_score = silhouette_score(X_pca, Dataset['Cluster'])
print(" \n Silhouette after PCA:", round(sil_score, 3))

cluster_counts = Dataset['Cluster'].value_counts().sort_index()
print("\n Count the cluster and sorting them in an order :\n", cluster_counts)
print("Percentage to represent each cluster :\n",
      (cluster_counts / len(Dataset) * 100).round(2))

#----------------------------------------------
# Find average numeric behaviour for each cluster
#----------------------------------------------
num_colls = ['Age', 'Work_Experience', 'Family_Size']
cluster_numeric_profile = Dataset.groupby('Cluster')[num_colls].mean().round(2)
print("\n Numeric behaviour for cluster :\n", cluster_numeric_profile)

#----------------------------------------------
# Check spending level for each cluster
#----------------------------------------------
cluster_spending = Dataset.groupby('Cluster')['Spending_Score'].mean().round(2)
print("\n Spending behaviour in cluster :\n", cluster_spending)

#------------------------------
# categorical dominance
#--------------------------------
pd.crosstab(Dataset['Gender'], Dataset['Cluster'], normalize='columns').round(2)
pd.crosstab(Dataset['Ever_Married'], Dataset['Cluster'], normalize='columns').round(2)
pd.crosstab(Dataset['Profession'], Dataset['Cluster'])

# Summary Table
cluster_summary = Dataset.groupby('Cluster').agg({
    'Age': 'mean',
    'Work_Experience': 'mean',
    'Family_Size': 'mean',
    'Spending_Score': 'mean'
})
print(" \n Summary table for cluster :\n", cluster_summary)

# Cluster naming
cluster_names = {
    0: 'Older Moderate Spenders',
    1: 'Experienced Professionals',
    2: 'Mainstream Customers',
    3: 'Young Large-Family Spenders'
}
Dataset['Cluster_Name'] = Dataset['Cluster'].map(cluster_names)

# visualization using PCA
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=Dataset['Cluster'], palette='Set2')
plt.title("Customer Segments (PCA)")
plt.show()


# PCA showed overlapping clusters, and one-hot encoding increased the number of features,
# introducing sparsity and noise. This reduced DBSCAN’s effectiveness in identifying dense regions.
# Dbscan
dbscan = DBSCAN(eps=0.5, min_samples=20)  # tune eps & min_samples
clusters = dbscan.fit_predict(X_pca)  # use PCA-transformed data
print("\n DBScan for my dataset :\n", clusters)


#--------------------------------------------------
# To test data 
#-------------------------------------------------
# Load test data
test_data = pd.read_csv('Test.csv')
test_data.columns = test_data.columns.str.strip()

# Drop unnecessary columns
test_data = test_data.drop(columns=['ID'], errors='ignore')

#-----------------------------------------
# Filling missing value in respective columns
#-----------------------------------------
for col in ['Ever_Married', 'Graduated', 'Profession', 'Var_1']:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])


for col in ['Work_Experience', 'Age']:
    test_data[col] = test_data[col].fillna(test_data[col].median())

test_data['Family_Size'] = test_data['Family_Size'].fillna(test_data['Family_Size'].mean())


#-----------------------------------------------
# ordinal encoder
#-----------------------------------------------
# Apply the trained encoder to test data
test_data['Spending_Score'] = oe.transform(test_data[['Spending_Score']])


# Predict clusters directly using the trained pipeline
# The pipeline takes care of preprocessing (scaling, encoding)
test_data['Cluster'] = cluster_model.predict(test_data)
test_data['Cluster_Name'] = test_data['Cluster'].map(cluster_names)


print( "Datset for test data :\n",test_data)

 # Business Insights:
#- Young large-family customers may be targeted with family-oriented promotions.
#- Experienced professionals may be suitable for premium or loyalty programs.
#- Older customers may respond better to budget-friendly offers.
