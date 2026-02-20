import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder ,LabelEncoder ,FunctionTransformer,PolynomialFeatures
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn. impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score ,mean_absolute_error ,mean_squared_error ,root_mean_squared_error
dataset=pd.read_csv("student_data_with_issues.csv")
#Remove leading space in column names 
dataset.columns=dataset.columns.str.strip()

print("First  5 dataset :\n",dataset.head())
print("Last  5 dataset :\n",dataset.tail())

#Null data present in columns 
print("Null dataset check per columns :\n",dataset.isnull().sum())

#Null data present in columns with percentage 
print("Null dataset check per columns in percentage :\n",dataset.isnull().sum()/dataset.shape[0])

#Null data present in whole dataset with percentage 
print("Null dataset check in whole dataset in percentage :\n",(dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1]))*100)

#check them in summary
print("Dataset describe :\n", dataset.describe())
#check their type
print("Dataset type :\n", dataset.info())


print(dataset['Age'].unique())
dataset['Age']=pd.to_numeric(dataset['Age'] , errors='coerce')
dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())

#Fill missing value to categorical data 
for col in ['Gender', 'Class']:
   si=SimpleImputer(strategy='most_frequent')
   dataset[col]=si.fit_transform(dataset[[col]]).flatten()

# #convert into numeric for Gender and Class
# ohe=OneHotEncoder(drop='first')
# ohe_encod=ohe.fit_transform(dataset[['Gender']]).toarray()
# dataset['Gender']=pd.DataFrame(ohe_encod ,columns=['Gender'])


#convert into numeric for Class
le=LabelEncoder()
lab_encoded=le.fit_transform(dataset['Class'])
dataset['Class']=pd.DataFrame(lab_encoded ,columns=['Class'])


#Combine in 1 column
subject=['Math Score' , 'English Score', 'Computer Score', 'Sport Score', 'Attendance Score' , 'Art Score']

#fill data with mean in missing value
dataset[subject]=dataset[subject].fillna(dataset[subject].mean())

#Total score of student 
dataset['Total Score']=dataset[subject].sum(axis=1)
#Average score of student 
dataset['Average Score']=dataset['Total Score']/len(subject)

def grade(num):
    if num >= 90:
        return 'A+'
    elif 80 <= num < 90:
        return 'A'
    elif 70 <= num < 80:
        return 'B+'
    elif 60 <= num < 70:
        return 'B'
    elif 50 <= num < 60:
        return 'C+'
    elif 40 <= num < 50:
        return 'C'
    elif 30 <= num < 40:
        return 'D+'
    else:
        return 'F'


 #For grade  of each student
dataset['Grade']= dataset['Average Score'].apply(grade)
#For result of each student 
# Assign 'Pass' if Total Score >= 40 else 'Fail'
dataset['Result'] = np.where(dataset['Average Score'] >= 40, 'Pass', 'Fail')
#Top and weak student data 
Top_student=dataset.loc[[dataset['Total Score'].idxmax()]]
print("Top student marks :\n",Top_student)

weak_student=dataset.loc[dataset['Result']=='Fail']
print("Weak Student :\n",weak_student)
#Display dataset
print("Clean dataset :\n",dataset)

#Null data present in columns 
print("Null dataset check per columns :\n",dataset.isnull().sum())

print("Result:\n",dataset['Result'].unique())


#check duplicated value
print(dataset.drop_duplicates(inplace=True))
print(dataset.duplicated().sum())


#For visualization :
#visualization for dataset
plt.figure(figsize=(10 ,7))
sns.pairplot(data=dataset , hue='Result' ,vars=subject ,palette='coolwarm' )
plt.suptitle("Pairwise relationships between scores")
plt.show()

#distribution for each subject 
plt.figure(figsize=(12,8))
for i, col in enumerate(subject):
    plt.subplot(2 ,3 ,i+1)
    sns.histplot(dataset[col], kde=True , color='skyblue')
    plt.title(f"distribution for {col}")

plt.tight_layout()
plt.show()

#grade Distribution 
plt.figure(figsize=(10,8))
sns.countplot(x='Grade', data=dataset,hue='Result', order=['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'F'], palette='coolwarm')
plt.title("Distribution of Grades")
plt.show()

#Heatmap
corr=dataset[subject+['Age','Class']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(data=corr ,annot=True)
plt.title("Correlation heatmap of features ")#Helps visually justify SFS feature selection (higher correlation with target → more likely selected).
plt.show()

#split the target and input features
#Get all subject except math score
subject_features=[col for col in subject if col != 'Math Score']
x=dataset[['Age','Gender','Class']+subject_features]
y=dataset['Total Score']

#split into train test features
x_train,x_test,y_train,y_test=train_test_split(x ,y ,test_size=0.2 ,random_state=42)
#Split numeric data
num_features=x.select_dtypes(include=['int64','float64']).columns.tolist()
cat_features=x.select_dtypes(include=['object']).columns.tolist()

#numeric pipeline
num_pipeline=Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('Poly',PolynomialFeatures(degree=2 ,include_bias=False)),
    ('log',FunctionTransformer(func=np.log1p ,validate=False)),
    ('scaler',StandardScaler()),

])

cat_pipeline=Pipeline([
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

#combine pipeline 
preprocessor=ColumnTransformer([
    ('num',num_pipeline,num_features),
    ('cat',cat_pipeline,cat_features)
])

#For features selection
sfs=SequentialFeatureSelector( LinearRegression() ,k_features=5,forward=True , floating=False , scoring='r2' ,cv=5)

 #linear regression pipeline 
model=Pipeline([
    ('preprocessing',preprocessor),
    ('feature_selection', sfs),
    ('Regression',LinearRegression())

])

#model use
print("Linear Model :\n")
lr=model.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("MSE :\n",mean_squared_error(y_test,y_pred))
print("MAE :\n",mean_absolute_error(y_test,y_pred))
print("RMSE :\n",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 score :\n",r2_score(y_test ,y_pred))

#actual Vs predicted linear model 
plt.scatter(y_test,y_pred )
plt.plot([y_test.min() ,y_test.max()],[y_test.min() ,y_test.max()], 'r--' ,lw=2)
plt.xlabel("Actual Total Score")
plt.ylabel("Predicted Total Score")
plt.title("Actual vs Predicted - Linear Regression")
plt.show()

#Visualization for linear regression 
residual=y_test-y_pred
sns.scatterplot(x=y_pred ,y=residual)
plt.axhline(0, color='red', linestyle='--')#Confirms model fit and assumptions of linear regression.
plt.xlabel("Predicted Total Score")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


#lasso regression pipeline 
lasso_model=Pipeline([
    ('preprocessing',preprocessor),
    ('Regression',Lasso(max_iter=1000))

])
param_grid={
    'Regression__alpha':[1e-10 ,0.0001 ,0.01 ,1 ,10 ,50 ,100]
}
grid_search =GridSearchCV(lasso_model ,param_grid=param_grid ,cv=5)
print("For lasso regression :\n")
grid_search.fit(x_train,y_train)
print("Best hyperparamter for lasso  :\n",grid_search.best_estimator_)
print("Best score by using that particular parameter for lasso :\n",grid_search.best_score_)
y_pred1=grid_search.best_estimator_.predict(x_test)
print("MSE for lasso :\n",mean_squared_error(y_test,y_pred1))
print("MAE  for lasso:\n",mean_absolute_error(y_test,y_pred1))
print("RMSE  for lasso:\n",np.sqrt(mean_squared_error(y_test,y_pred1)))
print("R2 score  for lasso:\n",r2_score(y_test ,y_pred1))



#ridge regression pipeline 
ridge_model=Pipeline([
    ('preprocessing',preprocessor),
    ('Regression',Ridge(max_iter=1000))

])

param1_grid={
    'Regression__alpha':[1e-10 ,0.0001 ,0.01 ,1 ,10 ,50 ,100]
}
grid_search2 =GridSearchCV(ridge_model ,param_grid=param1_grid ,cv=5)
print("For ridge  regression :\n")
grid_search2.fit(x_train,y_train)
print("Best hyperparamter for ridge :\n",grid_search2.best_estimator_)
print("Best score by using that particular parameter for ridge :\n",grid_search2.best_score_)
y_pred2=grid_search2.best_estimator_.predict(x_test)
print("MSE for ridge :\n",mean_squared_error(y_test,y_pred2))
print("MAE  for ridge:\n",mean_absolute_error(y_test,y_pred2))
print("RMSE  for ridge:\n",np.sqrt(mean_squared_error(y_test,y_pred2)))
print("R2 score  for ridge:\n",r2_score(y_test ,y_pred2))

# Example test data
test_data = pd.DataFrame({
    'Age': [15, 17, 16, 14],
    'Gender': ['Male', 'Female', 'Male', 'Female'],
      'Class': [0, 1, 2, 3], # encoded class from your LabelEncoder
    'English Score': [20, 75, 65, 55],
    'Computer Score': [18, 80, 60, 50],
    'Sport Score': [12, 70, 58, 48],
    'Attendance Score': [30.5, 85.2, 78.3, 68.2],
    'Art Score': [10, 90, 62, 53]
})
print("Using linear regression for test data:\n",lr.predict(test_data))
# lasso regression
print("Using lasso regression for test data:\n",grid_search.best_estimator_.predict(test_data))
# ridge regression
print("Using ridge regression for test data:\n",grid_search2.best_estimator_.predict(test_data))