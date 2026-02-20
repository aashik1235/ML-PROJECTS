import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

dataset=pd.read_csv('Titanic-Dataset.csv')
print("5 dataset :\n",dataset.head())

#Visualization for numerical data
sns.pairplot(dataset, hue='Survived', vars=['Age', 'Fare'])
plt.show()


# Pclass vs Survival
sns.barplot(data=dataset, x='Pclass', y='Survived')  # ci=None to remove confidence interval
plt.title('Survival Rate by Pclass')
plt.show()

# Sex vs Survival
sns.barplot(data=dataset, x='Sex', y='Survived')
plt.title('Survival Rate by Sex')
plt.show()

# Embarked vs Survival
sns.barplot(data=dataset, x='Embarked', y='Survived')
plt.title('Survival Rate by Embarked Port')
plt.show()

#Split dependent and independent features
x=dataset[['Pclass','Sex','Age','Fare','Embarked']]
y=dataset['Survived']

#split train and test
x_train,x_test,y_train,y_test=train_test_split(x ,y ,test_size=0.2,random_state=42)

#Separate numeric and Categorical data
num_features=['Age', 'Fare']
cat_features=['Sex', 'Pclass','Embarked']

#Numeric pipeline
num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('Scaler',StandardScaler())
])

#Categorical pipeline
cat_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

#combine both using columnTransformer

preprocessor= ColumnTransformer([
    ('num',num_pipeline ,num_features),
    ('cat',cat_pipeline ,cat_features)
])

#create full ml pipeline 
model=Pipeline([
    ('preprocessing', preprocessor),
    ('Classifier', RandomForestClassifier(random_state=42 , class_weight='balanced'))
])

params_grid={
    'Classifier__n_estimators':[100 ,200 ,300],
    'Classifier__max_depth':[5 ,10]

}
grid_search= GridSearchCV(model ,param_grid=params_grid, cv=5 ,scoring='recall')
#Train
grid_search.fit(x_train,y_train)
print("Best hyperparameter :\n ",grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
y_pred = grid_search.best_estimator_.predict(x_test)

#make human understandable output 
# label mapping
label_map = {0: 'Did Not Survive', 1: 'Survived'}

#Label for train data too
y_train_pred = grid_search.best_estimator_.predict(x_train)
train_results = x_train.copy()
train_results['Actual'] = y_train.values
train_results['Predicted'] = y_train_pred

train_results['Actual_Label'] = train_results['Actual'].map(label_map)
train_results['Predicted_Label'] = train_results['Predicted'].map(label_map)

#no data leakage
X_test_transformed = grid_search.best_estimator_.named_steps['preprocessing'].transform(x_test)
feature_names = grid_search.best_estimator_.named_steps['preprocessing'].get_feature_names_out()
X_test_model_view = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=x_test.index
)
X_test_model_view['Actual'] = y_test.values
X_test_model_view['Predicted'] = y_pred

#label test here 
label_map = {0: 'Did Not Survive', 1: 'Survived'}

X_test_model_view['Actual_Label'] = (
    X_test_model_view['Actual'].map(label_map)
)
X_test_model_view['Predicted_Label'] = (
    X_test_model_view['Predicted'].map(label_map)
)


#set survival probabilities from the trained model
y_proba = grid_search.best_estimator_.predict_proba(x_test)[:, 1]  # P(Survived)

#Set a lower threshold to reduce FN
threshold = 0.35  # lower than 0.5 to catch more survivors
y_pred_low_fn = (y_proba >= threshold).astype(int)

#calculate metrics
precision = precision_score(y_test, y_pred_low_fn)
recall = recall_score(y_test, y_pred_low_fn)
cf = confusion_matrix(y_test, y_pred_low_fn)

#Print results
print(f"Precision score with threshold {threshold}:\n", precision)
print(f"Recall score with threshold {threshold}:\n", recall)
print("Confusion matrix:\n", cf)

#Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cf, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold={threshold})")
plt.show()


#output in label form 
print("\nTrain labeled data:")
print(train_results.head())
 
print("\nTest labeled data:")
print(X_test_model_view.head())



# #Train the model 
# model.fit(x_train , y_train)
# #prediction and accuracy 
# y_pred=model.predict(x_test)
# print("Accuracy :",accuracy_score(y_test,y_pred))
# cf=confusion_matrix(y_test,y_pred)
# print("Confusion matrix ", cf)
# print("Precision :\n",precision_score(y_test,y_pred))
# print("Recall :\n",recall_score(y_test,y_pred))
# print("f1-score :\n",f1_score(y_test,y_pred))

# sns.heatmap(data=cf ,annot=True)
# plt.show()


# Predict new passengers (at least 5)
new_passengers = pd.DataFrame({
    'Age': [23, 49, 35, 8, 60],
    'Fare': [7.23, 56.23, 12.50, 21.07, 30.00],
    'Pclass': [3, 1, 2, 3, 1],
    'Sex': ['female', 'male', 'female', 'female', 'male'],
    'Embarked': ['S', 'C', 'S', 'Q', 'S']
})

# Numeric prediction
new_pred = grid_search.best_estimator_.predict(new_passengers)

# Label mapping
label_map = {0: 'Did Not Survive', 1: 'Survived'}

# Add predictions to dataframe
new_passengers['Prediction'] = new_pred
new_passengers['Survival_Label'] = new_passengers['Prediction'].map(label_map)

print("\nNew passenger predictions:")
print(new_passengers)
