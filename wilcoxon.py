import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('bank-full.csv', sep=';')

#clean data 
df.isna().sum() #no null values found
df.duplicated().sum() #no duplicated values found

X = df.drop(columns=['y', 'duration'])
y = df['y'].map({'yes': 1, 'no': 0}) #encode target variable

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
def make_preprocessor(categorical_features, numerical_features): 
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

print('Categorical features:', categorical_features)
print('Numerical features:', numerical_features)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#top 5 features fromn each path 
top_features_borda = ['poutcome', 'month', 'pdays', 'contact', 'housing']
top_features_stack = ['month', 'contact', 'day', 'pdays', 'poutcome']

#training random classifier with top features from both paths
clf = RandomForestClassifier(random_state=42)

scores_borda = []
scores_stack = []

for i in range(1, 6):
    scores_borda.append(cross_val_score(
        clf, X[top_features_borda[:i]], 
        y, cv=5, scoring='roc_auc').mean()
    )
    scores_stack.append(cross_val_score(
        clf, X[top_features_stack[:i]], 
        y, cv=5, scoring='roc_auc').mean()
    )
print(f'\nBorda scores: {scores_borda}')
print(f'\nStack scores: {scores_stack}')

#wilcoxon test
stat, p_value = wilcoxon(scores_borda, scores_stack)

print(f'\nWilcoxon statistic: {stat:.4f}')
print(f'\nP-value: {p_value:.4f}')
