import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from collections import defaultdict

df = pd.read_csv('bank-full.csv', sep=';')

#clean data 
df.isna().sum() #no null values found
df.duplicated().sum() #no duplicated values found

X = df.drop(columns='y')
y = df['y'].map({'yes': 1, 'no': 0}) #encode target variable

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
print('Categorical features:', categorical_features)
print('Numerical features:', numerical_features)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#


#create a pipeline that combines preprocessing and the Random Forest model
rf_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_classifier.fit(X_train, y_train) #train the model
y_pred_rf = rf_classifier.predict(X_test) #make predictions on the test set

#evaluate the random forest model 
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
f1_rf = class_report_rf['1']['f1-score']
precision_rf = class_report_rf['1']['precision']
auc_rf = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])

#permutation feature importance random forest
pfi_rf = permutation_importance(rf_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_rf_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_rf.importances_mean,
    'importance_std': pfi_rf.importances_std
}).sort_values(by='importance_mean', ascending=False)


#create a pipeline that combines preprocessing and the SVM model
svm_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

svm_classifier.fit(X_train, y_train) #train the model
y_pred_svm = svm_classifier.predict(X_test) #make predictions on the test set

#evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
f1_svm = class_report_svm['1']['f1-score']
precision_svm = class_report_svm['1']['precision']
auc_svm = roc_auc_score(y_test, svm_classifier.predict_proba(X_test)[:, 1])

#permutation feature importance SVM
pfi_svm = permutation_importance(svm_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_svm_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_svm.importances_mean,
    'importance_std': pfi_svm.importances_std
}).sort_values(by='importance_mean', ascending=False)


#create a pipeline that combines preprocessing and the KNN model
knn_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn_classifier.fit(X_train, y_train) #train the model
y_pred_knn = knn_classifier.predict(X_test) #make predictions on the test set

#evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
f1_knn = class_report_knn['1']['f1-score']
precision_knn = class_report_knn['1']['precision']
auc_knn = roc_auc_score(y_test, knn_classifier.predict_proba(X_test)[:, 1])

#permutation feature importance KNN
pfi_knn = permutation_importance(knn_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_knn_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_knn.importances_mean,
    'importance_std': pfi_knn.importances_std
}).sort_values(by='importance_mean', ascending=False)


#BORDA COUNT
model_dfs = {
    'Random Forest': pfi_rf_df,
    'SVM': pfi_svm_df,
    'KNN': pfi_knn_df
}
borda_scores = defaultdict(float) #score dictionary

for model_name, df in model_dfs.items():
    #sort by importance descending
    df_stored = df.sort_values(by='importance_mean', ascending=False)

    #get ordered list of features 
    feature_list = df_stored['feature'].tolist()
    n = len(feature_list)

    #assign Borda scores
    for rank, feature in enumerate(feature_list): 
        points = n-rank
        borda_scores[feature] += points

final_rank = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)

print("Final Borda Count Rankings:")
for feature, score in final_rank:
    print(f"{feature}: {score:.2f}")

#using the highest performing features to train new models 
top_features = [feat for feat, _ in final_rank]
X_top = X[top_features[:5]] #select top 5 features

categorical_featues_top = [feat for feat in categorical_features if feat in X_top.columns] #update categorical features list
numerical_features_top = [feat for feat in numerical_features if feat in X_top.columns] #update numerical features list

preprocessor_xai = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_top),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_featues_top)
    ])

X_train_xai, X_test_xai, y_train_xai, y_test_xai = train_test_split(X_top, y, test_size=0.2, random_state=42)


#train new random forest model with top features
rf_classifier_xai = Pipeline(steps=[
    ('preprocessor', preprocessor_xai),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_classifier_xai.fit(X_train_xai, y_train_xai) #train the model
y_pred_rf_xai = rf_classifier_xai.predict(X_test_xai) #make predictions on the test set

#evaluate the new random forest model
accuracy_rf_xai = accuracy_score(y_test_xai, y_pred_rf_xai)
cm_rf_xai = confusion_matrix(y_test_xai, y_pred_rf_xai)
class_report_rf_xai = classification_report(y_test_xai, y_pred_rf_xai, output_dict=True)
f1_rf_xai = class_report_rf_xai['1']['f1-score']
precision_rf_xai = class_report_rf_xai['1']['precision']
auc_rf_xai = roc_auc_score(y_test_xai, rf_classifier_xai.predict_proba(X_test_xai)[:, 1])


#train new SVM model with top features
svm_classifier_xai = Pipeline(steps=[
    ('preprocessor', preprocessor_xai),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

svm_classifier_xai.fit(X_train_xai, y_train_xai) #train the model
y_pred_svm_xai = svm_classifier_xai.predict(X_test_xai) #make predictions on the test set

#evaluate the new SVM model
accuracy_svm_xai = accuracy_score(y_test_xai, y_pred_svm_xai)
cm_svm_xai = confusion_matrix(y_test_xai, y_pred_svm_xai)
class_report_svm_xai = classification_report(y_test_xai, y_pred_svm_xai, output_dict=True)
f1_svm_xai = class_report_svm_xai['1']['f1-score']
precision_svm_xai = class_report_svm_xai['1']['precision']
auc_svm_xai = roc_auc_score(y_test_xai, svm_classifier_xai.predict_proba(X_test_xai)[:, 1])


#train new KNN model with top features
knn_classifier_xai = Pipeline(steps=[
    ('preprocessor', preprocessor_xai),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn_classifier_xai.fit(X_train_xai, y_train_xai) #train the model
y_pred_knn_xai = knn_classifier_xai.predict(X_test_xai) #make predictions on the test set

#evaluate the new KNN model
accuracy_knn_xai = accuracy_score(y_test_xai, y_pred_knn_xai)
cm_knn_xai = confusion_matrix(y_test_xai, y_pred_knn_xai)
class_report_knn_xai = classification_report(y_test_xai, y_pred_knn_xai, output_dict=True)
f1_knn_xai = class_report_knn_xai['1']['f1-score']
precision_knn_xai = class_report_knn_xai['1']['precision']
auc_knn_xai = roc_auc_score(y_test_xai, knn_classifier_xai.predict_proba(X_test_xai)[:, 1])


#fianl results 
table = pd.DataFrame({
    'Model': ['XAI SVM', 'XAI RF', 'XAI KNN', 'XAI Ensemble', 'ML SVM', 'ML RF', 'ML KNN', 'ML Ensemble'], 
    'Accuracy': [accuracy_svm_xai, accuracy_rf_xai, accuracy_knn_xai, None, None, None, None, None],
    'F1 Score': [f1_svm_xai, f1_rf_xai, f1_knn_xai, None, None, None, None, None],
    'Precision': [precision_svm_xai, precision_rf_xai, precision_knn_xai, None, None, None, None, None], 
    'AUC': [auc_svm_xai, auc_rf_xai, auc_knn_xai, None, None, None, None, None]
})

print(table)