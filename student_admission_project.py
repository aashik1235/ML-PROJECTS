import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn. impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from imblearn.over_sampling import RandomOverSampler

#Load dataset
dataset=pd.read_csv("student_admission_record_dirty.csv")

#Remove leading space in the column names

dataset.columns = dataset.columns.str.strip()

#Check small dataset
print(" First 5 dataset :\n",dataset.head())
print(" Last 5 dataset :\n",dataset.tail())


#Add new features
dataset['Score_percentage']=(dataset['Admission Test Score']*0.6) + (dataset['High School Percentage']*0.4)




# Replace -1 or other placeholders with np.nan
for col in ['Age', 'Admission Test Score', 'High School Percentage','Score_percentage']:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')   # convert to numeric
    dataset[col] = dataset[col].replace(-1, np.nan)  


#convert categorical data into string

dataset['Gender']=dataset['Gender'].astype(str).str.strip()
dataset['City']=dataset['City'].astype(str).str.strip()


#visualization wether we can separate it or not 
sns.pairplot(data=dataset , hue="Admission Status" , vars=['Age','Admission Test Score','High School Percentage','Score_percentage'],diag_kind='kde')
plt.show()


#Split the target and input features 
dataset['Admission Status_numeric']=dataset["Admission Status"].map({'Rejected':0,'Accepted':1})
mask=dataset['Admission Status_numeric'].notna()
x=dataset.loc[mask ,["Age","Gender","Admission Test Score","High School Percentage","City" ,'Score_percentage']]
y=dataset.loc[mask ,'Admission Status_numeric']

print(x.shape, y.shape)
print(y.value_counts())

#visualization wether we can separate it or not 
sns.pairplot(data=dataset.loc[mask] , hue="Admission Status" , vars=['Age','Admission Test Score','High School Percentage','Score_percentage'],diag_kind='kde')
plt.show()


#Visualization for categorical data 
clean_data=dataset.loc[mask].copy()
clean_data['Admission Status']=clean_data["Admission Status"].map({'Rejected':0,'Accepted':1})
#For Gender
sns.barplot(x='Gender',y='Admission Status' , data=clean_data , color='blue' ,linestyle='--')
#For City
sns.barplot(x='City',y='Admission Status' , data=clean_data , color='blue' ,linestyle='--')



#Split in train and test data
x_train,x_test,y_train,y_test=train_test_split(x ,y, test_size=0.2 ,random_state=42)
x_test['City'] = x_test['City'].apply(lambda x: x if x in x_train['City'].unique() else 'Other')


#Random over sampler
ro =RandomOverSampler(random_state=42)
x_train , y_train=ro.fit_resample(x_train , y_train)
print(y_train.value_counts())


#Split into numeric and categorical features
num_features=['Age','Admission Test Score','High School Percentage','Score_percentage']
cat_features=['Gender','City']


#numeric pipeline 
num_pipeline= Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#Categorical pipeline
cat_pipeline= Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(drop='first' ,handle_unknown='ignore'))
])

#Combine different features
preprocessor= ColumnTransformer([
    ('num',num_pipeline,num_features),
    ('cat',cat_pipeline,cat_features)
])
model= Pipeline([
    ('processing', preprocessor),
    ('classifier',RandomForestClassifier(random_state=42))
])
params_grid = {
    'classifier__n_estimators': [100, 200],  # fewer trees
    'classifier__max_depth': [10, 20],       # skip None
    'classifier__min_samples_split': [2, 3],
    'classifier__min_samples_leaf': [1]
}


grid_search=GridSearchCV(model ,param_grid=params_grid ,cv=5)
grid_search.fit(x_train,y_train)
print("\nBest Hyperparameters:", grid_search.best_params_)
print("Best accuracy score  is :\n",grid_search.best_score_)
y_pred=grid_search.best_estimator_.predict(x_test)
print("\nAccuracy on Test set:", accuracy_score(y_test,y_pred))

# sns.scatterplot(x=y_test, y=y_pred, color='red', label='Predicted vs Actual')
# # Line showing perfect predictions
# sns.lineplot(x=y_test, y=y_test, color='blue', label='Perfect Prediction')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Prediction Plot')
# plt.legend()
# plt.show()

print("\n Classification matrix :\n",classification_report(y_test,y_pred))

# confusion Matrix
cf=confusion_matrix(y_test,y_pred)
print("Confusion matrix :\n",cf)
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected','Accepted'], yticklabels=['Rejected','Accepted'])
plt.title('Confusion Matrix')
plt.show()

#for testing
data_test =pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank"],
    "Age": [18, 20, 19, np.nan, 21, 22],
    "Gender": ["Female", "Male", "Male", "Male", "Female", "Male"],
    "Admission Test Score": [75, 80, 65, 90, np.nan, 70],
    "High School Percentage": [85, 78, 88, 92, 80, 76],
    "City": ["Quetta", "Lahore", "Karachi", "Islamabad", "Quetta", "Lahore"]
})
# Add the Score_percentage column to match your trained model
data_test['Score_percentage'] = (data_test['Admission Test Score'] * 0.6 + data_test['High School Percentage'] * 0.4)

print("Output for unseen data to model :\n",grid_search.best_estimator_.predict(data_test))