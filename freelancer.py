import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------- Load Dataset -------------------------------
dataset = pd.read_csv('global_freelancers_raw.csv')
dataset.columns = dataset.columns.str.strip()

print("First 5 rows:\n", dataset.head())
print("Last 5 rows:\n", dataset.tail())

# ------------------------------- Missing Data Overview -------------------------------
print("Null values per column:\n", dataset.isnull().sum())
print("Null values per column (%) :\n", (dataset.isnull().sum() / dataset.shape[0]) * 100)
print("Overall null percentage:", (dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1])) * 100)

# ------------------------------- Drop unnecessary columns -------------------------------
dataset = dataset.drop(columns=['freelancer_ID', 'name'])

#-------------------------------- hourly_rate cleaning data --------------------------------
dataset['hourly_rate (USD)'] = (
    dataset['hourly_rate (USD)']
    .str.replace('USD','')
    .str.replace('$','')
    .str.strip()
    .astype(float)
)
dataset['hourly_rate (USD)'] = dataset['hourly_rate (USD)'].fillna(dataset['hourly_rate (USD)'].median())
print(dataset['hourly_rate (USD)'].isnull().sum())

#-------------------------------- rating cleaning data --------------------------------
dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
dataset['rating'] = dataset['rating'].fillna(dataset['rating'].mean())

print("Dataset info:\n", dataset.info())
print("\nDataset description:\n", dataset.describe())

# ------------------------------- Fill missing numerical data for age and convert to categorical -------------------------------
age_bins = [0, 30, 45, 60, 100]
age_labels = ['Young', 'Mid', 'Senior', 'Old']
dataset['age'] = dataset['age'].fillna(dataset['age'].mean()).astype(int)
dataset['age_group'] = pd.cut(dataset['age'], bins=age_bins, labels=age_labels)

exp_bins = [0, 5, 15, 25, 50]
exp_labels = ['Junior', 'Intermediate', 'Senior', 'Expert']
dataset['years_of_experience'] = dataset['years_of_experience'].fillna(dataset['years_of_experience'].mean())
dataset['experience_level'] = pd.cut(dataset['years_of_experience'], bins=exp_bins, labels=exp_labels)

# ------------------------------- Standardize binary column: is_active -------------------------------
active_map = {'1': 1, 'true': 1, 'false': 0, 'n': 0, 'y': 1}
dataset['is_active'] = dataset['is_active'].astype(str).str.lower()
dataset['is_active'] = dataset['is_active'].map(active_map)
dataset['is_active'] = dataset['is_active'].fillna(dataset['is_active'].mode()[0]).astype(int)

# ------------------------------- Standardize categorical columns -------------------------------
dataset['gender'] = dataset['gender'].astype(str).str.lower()
dataset['gender'] = dataset['gender'].replace({'male': 'm', 'female': 'f'})

dataset['country'] = dataset['country'].replace(
    dataset['country'].value_counts()[dataset['country'].value_counts() < 10].index,
    'Other'
)

# ------------------------------- Client satisfaction preprocessing -------------------------------
dataset['client_satisfaction'] = dataset['client_satisfaction'].str.replace('%', '', regex=False).astype('float64')
dataset = dataset.dropna(subset=['client_satisfaction'])

#------------------------------- Client satisfaction categorical------------------------------------
bins = [0, 70, 90, 100]
labels = ['Low', 'Medium', 'High']
dataset['client_satisfaction_cat'] = pd.cut(dataset['client_satisfaction'], bins=bins, labels=labels)

# ------------------------------- Visualizations -------------------------------
sns.pairplot(data=dataset, vars=['rating','hourly_rate (USD)'])
plt.show()

# ------------------------------- Prepare Features & Target -------------------------------
mask = dataset['client_satisfaction'].notna()
x = dataset.loc[mask, ['primary_skill', 'country', 'language', 'is_active', 'rating','hourly_rate (USD)','experience_level','age_group']]
y = dataset.loc[mask, 'client_satisfaction_cat']
y_reg = dataset.loc[mask, 'client_satisfaction']

# ------------------------------- Split Data -------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.2, random_state=42)

# ------------------------------- Preprocessing Pipelines -------------------------------
cat_features = ['primary_skill', 'country', 'language', 'experience_level', 'age_group']
num_features = ['rating','hourly_rate (USD)']
binary_features = ['is_active']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

binary_pipeline = Pipeline([('passthrough', 'passthrough')])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('binary', binary_pipeline, binary_features),
    ('cat', cat_pipeline, cat_features)
])

# ------------------------------- Encode the target labels -------------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# ----------------------------- XGBoost Classifier Pipeline -----------------------------
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        scale_pos_weight=None
    ))
])

classes = np.unique(y_train_enc)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_enc)
class_weights = dict(zip(classes, weights))
sample_weights = np.array([class_weights[label] for label in y_train_enc])

param_grid = {
    'classifier__n_estimators': [300, 500],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01,0.03, 0.05],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb_pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train_enc, classifier__sample_weight=sample_weights)
best_xgb = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred_enc = best_xgb.predict(x_test)
y_pred = le.inverse_transform(y_pred_enc)

# ------------------------------- Classifier Evaluation -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()


# ----------------------------- XGBoost Regressor Pipeline -----------------------------
xgb_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    ))
])

param_grid_reg = {
    'regressor__n_estimators': [300, 500],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.03, 0.05],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

grid_search_reg = GridSearchCV(
    xgb_reg_pipeline,
    param_grid_reg,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search_reg.fit(x_train_reg, y_train_reg)
best_xgb_reg = grid_search_reg.best_estimator_
y_pred_reg = best_xgb_reg.predict(x_test_reg)

# ------------------------------- Regression Evaluation -------------------------------
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nXGB REGRESSION RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Clip predictions and convert to categorical using quantiles
y_pred_reg = np.clip(y_pred_reg, 0, 100)
quantile_bins = np.quantile(y_train_reg, [0, 0.33, 0.66, 1.0])

y_pred_reg_cat = pd.cut(
    y_pred_reg,
    bins=quantile_bins,
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

print("\nREGRESSION → CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_reg_cat))
print(classification_report(y_test, y_pred_reg_cat))

sns.heatmap(confusion_matrix(y_test, y_pred_reg_cat), annot=True, fmt='d')
plt.title("Regression-based Classification Confusion Matrix")
plt.show()


# ------------------------------- Feature Importance -------------------------------
xgb_model = grid_search.best_estimator_.named_steps['classifier'] 
importances = xgb_model.feature_importances_

cat_features_encoded = grid_search.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'].named_steps['one_hot'].get_feature_names_out(cat_features)
all_features = num_features + binary_features + cat_features_encoded.tolist()

importances_df = pd.DataFrame({'feature': all_features, 'importance': importances})
top_features = importances_df.sort_values(by='importance', ascending=False).head(10)

plt.figure(figsize=(8,5))
plt.barh(top_features['feature'], top_features['importance'])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Top 10 Features in XGBoost')
plt.show()
 