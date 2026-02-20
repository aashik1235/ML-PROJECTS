#Goal: Segment customers to identify Champions, Loyal, and At-Risk customers
#for targeted marketing and retention.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler ,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


dataset=pd.read_csv('online_retail.csv')
#remove the spaces from the column names
dataset.columns=dataset.columns.str.strip()

# Display the first few rows of the dataset
print("First 5 rows of the dataset:\n",dataset.head(5))
# Display the last few rows of the dataset
print("Last 5 rows of the dataset:\n",dataset.tail(5))

#check null values
print("Null values in each columns :\n",dataset.isnull().sum())

#check null values for column (%)
print("Percentage of null values in each column:\n", (dataset.isnull().sum() /(dataset.shape [0])* 100))

#check null values for overall dataset(%)
print("Percentage of null values in overall dataset:\n", (dataset.isnull().sum().sum() /(dataset.shape [0]*dataset.shape[1])* 100))

#Split the customer id into two columns
rfm_dataset =dataset.dropna(subset=['CustomerID'])
rfm_dataset = rfm_dataset[(rfm_dataset['Quantity'] > 0) & (rfm_dataset['UnitPrice'] > 0)]


#Remove the columns which are not required for analysis
rfm_dataset = rfm_dataset.drop(columns=['StockCode','Description'])

#Change data type
for col in ['UnitPrice' ,'CustomerID','InvoiceNo']:
    rfm_dataset[col] = pd.to_numeric(rfm_dataset[col], errors='coerce')


rfm_dataset['InvoiceDate'] = pd.to_datetime(rfm_dataset['InvoiceDate'], errors='coerce')


print('Dataset data type :\n',rfm_dataset.info())


#Feature engineeering 
#Calculate the total price for each transaction
rfm_dataset['Total_price'] =rfm_dataset['UnitPrice']*rfm_dataset['Quantity']

monetary =rfm_dataset.groupby('CustomerID')['Total_price'].sum()

#calcukate the frequency of transactions for each customer
frequency = rfm_dataset.groupby('CustomerID')['InvoiceNo'].nunique()

#Calculate the recency of transactions for each customer
reference_date =rfm_dataset['InvoiceDate'].max() 
recency= rfm_dataset.groupby('CustomerID')['InvoiceDate'].max()
recency =(reference_date - recency).dt.days


#create new dataset for model
rfm =pd .DataFrame(
    {
        'Recency' : recency,
        'Frequency' :frequency,
        'Monetary' :monetary
    }
)

print('RFM dataset :\n',rfm.head())

#Visualize the distribution of RFM features
#Histogram  of RFM features
plt.figure(figsize=(13,4))

plt.subplot(1,3,1)
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Recency Distribution')
plt.xlabel('Recency (days)')
plt.ylabel('Count')

plt.subplot(1,3,2)
sns.histplot(rfm['Frequency'], bins=30, kde=True)
plt.title('Frequency Distribution')
plt.xlabel('Frequency (transactions)')
plt.ylabel('Count')

plt.subplot(1,3,3)
sns.histplot(rfm['Monetary'], bins=30, kde=True)
plt.title('Monetary Distribution')
plt.xlabel('Monetary Value')
plt.xticks(rotation=45)
plt.ylabel('Count')

plt.tight_layout()
plt.show()

#Boxplot for RFM features
plt.figure(figsize=(13,4))

plt.subplot(1,3,1)
sns.boxplot(y=rfm['Recency'])
plt.title('Recency Outliers')
plt.ylabel('Recency (days)')

plt.subplot(1,3,2)
sns.boxplot(y=rfm['Frequency'])
plt.title('Frequency Outliers')
plt.ylabel('Frequency')

plt.subplot(1,3,3)
sns.boxplot(y=rfm['Monetary'])
plt.title('Monetary Outliers')
plt.ylabel('Monetary Value')

plt.tight_layout()
plt.show()

#scatter plot for RFM features
plt.figure(figsize=(13,4))

plt.subplot(1,3,1)
sns.scatterplot(x='Recency', y='Monetary', data=rfm)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

plt.subplot(1,3,2)
sns.scatterplot(x='Frequency', y='Monetary', data=rfm)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

plt.subplot(1,3,3)
sns.scatterplot(x='Recency', y='Frequency', data=rfm)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

plt.tight_layout()
plt.show()


#Heatmap for correlation
cor=rfm[['Recency','Frequency','Monetary']].corr()
sns.heatmap(cor ,annot = True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#Split into  numeric features 
num_features =rfm[['Frequency','Monetary']].columns
recency_feature =rfm[['Recency']].columns

#pipelining 
num_pipeline =Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('Function' ,FunctionTransformer(lambda x: np.log1p(x))), #help to make the cluster separation better by compressing the scale of the data and reducing the impact of outliers.
    ('scale',StandardScaler())
    
])

recency_pipeline =Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale',StandardScaler())
    
])

preprocessor =ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('recency', recency_pipeline, recency_feature)
])




#elbow method to find the optimal number of clusters
rfm_scaled = preprocessor.fit_transform(rfm)
wcss=[]

print("\nSilhouette Scores for different K:")

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

#visualization of elbow method
plt.figure(figsize=(8,5))
plt.plot((range(1,11)),wcss ,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

#Different value of K and their silhouette scores
for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    print(f"K = {k}, Silhouette Score = {round(score,3)}")


cluster_model =Pipeline([
    ('processing', preprocessor),
    ('clustering', KMeans(n_clusters=3,
                          init='k-means++',
                          n_init=10,
                          verbose=0,
                          max_iter=300,
                          random_state=42))

])
cluster_model.fit(rfm)
rfm['labels']=cluster_model.predict(rfm)

#score of silhouette
sil_score = silhouette_score(rfm_scaled, rfm['labels'])
print("Silhouette Score:", round(sil_score, 3))


#Count the label of RFM dataset
rfm_counts=rfm['labels'].value_counts().sort_index()
print("Label counts:\n",rfm_counts)
#label count in percentage 
print("Label percentages:\n",rfm_counts/len(rfm)*100)

#Numeric behaviour into business analysis
rfm_list =['Recency','Frequency','Monetary']
cluster_numeric_profile =rfm.groupby('labels')[rfm_list].mean()
print("Cluster numeric profile:\n",cluster_numeric_profile)


rfm_summary =rfm.groupby('labels')[rfm_list].agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':'mean'
}
)
print("Cluster summary:\n",rfm_summary)
 
cluster_names = {
    0: "Regular Customers",
    1: "VIP / Champions",
    2: "At-Risk Customers"
}


rfm['Customer_Segment'] = rfm['labels'].map(cluster_names)

print("RFM dataset with customer segments:\n",rfm.head())


# check the silhouette score for the clustering
pca=PCA(n_components=2)
x_pca= pca.fit_transform(rfm_scaled)


# visualization using PCA
sns.scatterplot(
    x=x_pca[:, 0],
    y=x_pca[:, 1],
    hue=rfm['Customer_Segment'],
    palette={
        'VIP / Champions': 'red',
        'Regular Customers': 'blue',
        'At-Risk Customers': 'green'
    }
)

plt.title("Customer Segments (PCA)")
plt.legend(title='Customer Segment')
plt.show()

#Visualize the distribution of RFM features by customer segments
sns.scatterplot(
    data=rfm,
    x='Recency',
    y='Monetary',
    hue='Customer_Segment'
)
plt.title('Customer Segmentation')
plt.show()


# Add Country back for post-cluster analysis
country = rfm_dataset.groupby('CustomerID')['Country'].first()
rfm['Country'] = country

print("RFM dataset with Country added:\n", rfm.head())

# -------------------------------
# TOP 10 COUNTRY SEGMENT ANALYSIS
# ------------------------------

# Find top 10 countries by total number of customers
top10_countries = rfm['Country'].value_counts().head(10).index


# Filter dataset to include only those top 10 countries
rfm_top10 = rfm[rfm['Country'].isin(top10_countries)]


# Create segment distribution table
country_segment_top10 = pd.crosstab(
    rfm_top10['Country'],
    rfm_top10['Customer_Segment']
)

country_segment_top10 = country_segment_top10.reset_index()

country_segment_top10 = country_segment_top10.loc[
    country_segment_top10.iloc[:, 1:].sum(axis=1)
    .sort_values(ascending=False).index
]

country_segment_top10 = country_segment_top10.set_index('Country')

# Plot stacked bar chart
country_segment_top10.plot(
    kind='bar',
    stacked=True,
    figsize=(12,6)
)

plt.title("Top 10 Countries - Customer Segments")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
