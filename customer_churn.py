import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score , precision_recall_curve,average_precision_score
from imblearn.pipeline import Pipeline as imPipeline

#----------------------------------
# Load dataset
#----------------------------------
dataset = pd.read_csv('customer_churn_dataset-training-master.csv')

# Remove spaces from column names
dataset.columns = dataset.columns.str.strip()

# Preview dataset
print("5 Dataset:\n", dataset.head(5))
print(dataset.shape)

# Null value analysis
print("Dataset of null value in column:\n", dataset.isnull().sum())
print("Dataset of null value in column (%):\n", (dataset.isnull().sum() / dataset.shape[0]) * 100)
print("Dataset of null value in overall dataset (%):\n",
      (dataset.isnull().sum().sum()) / (dataset.shape[0] * dataset.shape[1]) * 100)

# Drop unnecessary column
dataset = dataset.drop(columns=['CustomerID'])

# Dataset info & description
print("Info of dataset:\n", dataset.info())
print("Behaviour of dataset:\n", dataset.describe())

#----------------------------------
# Visualization: Categorical vs Churn
#----------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Churn', data=dataset)
plt.title('Gender effect on churn (Count Plot)')
plt.xlabel('Gender')
plt.ylabel('Number of customers')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Churn', hue='Gender', data=dataset)
plt.title('Gender effect on churn (Churn Rate)')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Contract Length', hue='Churn', data=dataset)
plt.title('Contract Length effect on churn (Count Plot)')
plt.xlabel('Contract Length')
plt.ylabel('Number of customers')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Contract Length', y='Churn', hue='Gender', data=dataset)
plt.title('Contract Length effect on churn (Churn Rate)')
plt.xlabel('Contract Length')
plt.ylabel('Churn Rate')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Subscription Type', hue='Churn', data=dataset)
plt.title('Subscription Type effect on churn (Count Plot)')
plt.xlabel('Subscription Type')
plt.ylabel('Number of customers')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Subscription Type', y='Churn', hue='Gender', data=dataset)
plt.title('Subscription Type effect on churn (Churn Rate)')
plt.xlabel('Subscription Type')
plt.ylabel('Churn Rate')
plt.show()

#----------------------------------
# Visualization: Numeric vs Churn (Box Plots)
#----------------------------------
numeric_features = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay']

for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Churn', y=feature, data=dataset)
    plt.title(f"{feature} vs Churn")
    plt.show()

#----------------------------------
# Scatter Plots
#----------------------------------
# Tenure vs Total Spend
plt.figure(figsize=(8, 6))
plt.hexbin(
    dataset['Tenure'],
    dataset['Total Spend'],
    gridsize=40,
    cmap='viridis'
)
plt.colorbar(label='Customer Count')
plt.title("Tenure vs Total Spend (Hexbin Plot)")
plt.xlabel("Tenure (months)")
plt.ylabel("Total Spend ($)")
plt.show()

# Usage Frequency vs Payment Delay
plt.figure(figsize=(8, 6))
plt.hexbin(
    dataset['Usage Frequency'],
    dataset['Payment Delay'],
    gridsize=40,
    cmap='viridis'
)
plt.colorbar(label='Customer Count')
plt.title("Usage Frequency vs Payment Delay (Hexbin Plot)")
plt.xlabel("Usage Frequency")
plt.ylabel("Payment Delay (days)")
plt.show()


# Select numeric columns
numeric_data = dataset.select_dtypes(include=['int64', 'float64'])

# Compute correlation
corr_matrix = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

#----------------------------------
# Prepare data for modeling
#----------------------------------
mask = dataset['Churn'].notna()
x = dataset.loc[mask, ['Age', 'Tenure', 'Usage Frequency','Gender',
                        'Subscription Type', 'Contract Length']]
y = dataset.loc[mask, 'Churn']

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Separate feature types
num_features = x.select_dtypes(include=['int64', 'float64']).columns
nom_features = ['Gender']
ord_features = ['Contract Length', 'Subscription Type']

#----------------------------------
# Pipelines
#----------------------------------
# Numeric pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Nominal categorical pipeline
nom_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Ordinal categorical pipeline
ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[['Monthly', 'Quarterly', 'Annual'],
                                           ['Basic', 'Standard', 'Premium']]))
])

# Combine all pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('nomm', nom_pipeline, nom_features),
    ('ordin', ord_pipeline, ord_features)
])

# Full pipeline with undersampling and classifier
model = Pipeline([
    ('processing', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=200,
        class_weight='balanced'
    ))
])


# Hyperparameter grid
param_grid = {
    'classifier__C': [0.1, 1]
}

# Grid search
grid_search = GridSearchCV(model, param_grid=param_grid, scoring='recall', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Best estimator
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

# Predict and evaluate
y_pred = grid_search.best_estimator_.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
cf =confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cf)
print("Classification Report:\n", classification_report(y_test, y_pred))

#confusion matrix plot
sns.heatmap(data =cf ,annot=True, fmt='d')
plt.title("Confusion Matrix for Logistic Regression Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#Apply random classifier
model2 = Pipeline([
    ('processing', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])


param_grid2 ={
    'classifier__n_estimators':[100],
    'classifier__max_depth':[None,10],
    'classifier__min_samples_split':[2]

}
# Grid search
grid_search2 = GridSearchCV(model2, param_grid=param_grid2, scoring='recall', cv=5, n_jobs=-1)
grid_search2.fit(x_train, y_train)

# Best estimator
print(grid_search2.best_estimator_)
print(grid_search2.best_params_)
print(grid_search2.best_score_)

# Predict and evaluate
y_pred2 = grid_search2.best_estimator_.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2))
cf2 =confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:\n",cf2)
print("Classification Report:\n", classification_report(y_test, y_pred2))

#confusion matrix plot
sns.heatmap(data =cf2 ,annot=True, fmt='d')
plt.title("Confusion Matrix for Random Forest Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ---- Precision-Recall Curve for Random Forest ----

# Get predicted probabilities for the positive class (churn)
y_probs = grid_search2.best_estimator_.predict_proba(x_test)[:, 1]

# Compute precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Average precision score (optional, overall metric)
avg_precision = average_precision_score(y_test, y_probs)
print(f"Average Precision Score: {avg_precision:.3f}")

# Plot Precision-Recall curve
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='b', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Random Forest)')
plt.grid(True)

# Highlight your business threshold (0.35)
# Find nearest threshold in thresholds array
threshold_idx = (np.abs(thresholds - 0.35)).argmin()
plt.scatter(recall[threshold_idx], precision[threshold_idx],
            color='red', label='Threshold = 0.35', s=100)

plt.legend()
plt.show()




# ---- Business-oriented threshold tuning (reduce FN) ----

# Lower threshold to reduce False Negatives
threshold = 0.35 # try 0.4, 0.35, 0.3
y_pred2_business = (y_probs >= threshold).astype(int)
# Evaluate again
print(f"\nBusiness Threshold = {threshold}")
print("Accuracy:", accuracy_score(y_test, y_pred2_business))
print("F1 Score:", f1_score(y_test, y_pred2_business))

cf2_business = confusion_matrix(y_test, y_pred2_business)
print("Confusion Matrix:\n", cf2_business)
print("Classification Report:\n", classification_report(y_test, y_pred2_business))

sns.heatmap(cf2_business, annot=True, fmt='d')
plt.title(f"Confusion Matrix (Threshold={threshold})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Extract feature importances from Random Forest
rf_model = grid_search2.best_estimator_.named_steps['classifier']
feature_names = (
    list(x.select_dtypes(include=['int64', 'float64']).columns) +
    list(grid_search2.best_estimator_.named_steps['processing']
         .named_transformers_['nomm']
         .named_steps['one_hot']
         .get_feature_names_out(['Gender'])) +
    list(['Contract Length', 'Subscription Type'])
)

importances = rf_model.feature_importances_

# Sort and plot
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.show()

#For test data prediction
# Load your test data
test_data = pd.read_csv('customer_churn_dataset-testing-master.csv')


# Clean column names
test_data.columns = test_data.columns.str.strip()

# Separate features (drop CustomerID and Churn if present)
x_test_final = test_data.drop(columns=['CustomerID', 'Churn'], errors='ignore')

# Predict probabilities for positive class (Churn=1)
y_probs = grid_search2.best_estimator_.predict_proba(x_test_final)[:, 1]

# Apply business threshold
threshold = 0.35
y_pred_business = (y_probs >= threshold).astype(int)

# Add predictions to test dataframe
test_data['Churn_Predicted'] = y_pred_business
test_data['Churn_Probability'] = y_probs

print(test_data.head())