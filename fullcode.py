import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from collections import defaultdict

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


#create a pipeline that combines preprocessing and the Random Forest model
rf_classifier = Pipeline(steps=[
    ('preprocessor', make_preprocessor(categorical_features, numerical_features)),
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
recall_rf = class_report_rf['1']['recall']
auc_rf = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])


#permutation feature importance random forest
pfi_rf = permutation_importance(rf_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_rf_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_rf.importances_mean,
    'importance_std': pfi_rf.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(f'\nPermutation Feature Importance for Random Forest:\n{pfi_rf_df}')

#bar plot for visualization of feature importance RF
plt.figure(figsize=(15, 10))
plt.bar(pfi_rf_df['feature'], pfi_rf_df['importance_mean'], yerr=pfi_rf_df['importance_std'], capsize=4)
plt.ylabel('Mean importance (performance drop)')
plt.title('Permutation Feature Importance - RF')
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()


#create a pipeline that combines preprocessing and the SVM model
svm_classifier = Pipeline(steps=[
    ('preprocessor', make_preprocessor(categorical_features, numerical_features)),
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
recall_svm = class_report_svm['1']['recall']
auc_svm = roc_auc_score(y_test, svm_classifier.predict_proba(X_test)[:, 1])


#permutation feature importance SVM
pfi_svm = permutation_importance(svm_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_svm_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_svm.importances_mean,
    'importance_std': pfi_svm.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(f'\nPermutation Feature Importance for SVM:\n{pfi_svm_df}')

#bar plot for visualization of feature importance SVM
plt.figure(figsize=(15, 10))
plt.bar(pfi_svm_df['feature'], pfi_svm_df['importance_mean'], yerr=pfi_svm_df['importance_std'], capsize=4)
plt.ylabel('Mean importance (performance drop)')
plt.title('Permutation Feature Importance - SVM')
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()


#KNN - finding the best k value to use 
k_range = range(1, 21)
cv_scores = []

for k in k_range: #evaluating each k using 5fold cross-validation
    knn_classifer = Pipeline(steps=[
        ('preprocessor', make_preprocessor(categorical_features, numerical_features)), 
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(knn_classifer, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

#plot for visualization optimal k value 
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.title('KNN cross-validation accuracy vs k')
plt.xlabel('number of neighbors: k')
plt.ylabel('cross-validated accuracy')
plt.grid(True)
plt.show()

best_k = k_range[np.argmax(cv_scores)]
print(f'best k form coss-validation: {best_k}')

#create a pipeline that combines preprocessing and the KNN model
knn_classifier = Pipeline(steps=[
    ('preprocessor', make_preprocessor(categorical_features, numerical_features)),
    ('knn', KNeighborsClassifier(n_neighbors=best_k )) #metric - used for distance computations, default: minkowski
])

knn_classifier.fit(X_train, y_train) #train the model
y_pred_knn = knn_classifier.predict(X_test) #make predictions on the test set

#evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
f1_knn = class_report_knn['1']['f1-score']
precision_knn = class_report_knn['1']['precision']
recall_knn = class_report_knn['1']['recall']
auc_knn = roc_auc_score(y_test, knn_classifier.predict_proba(X_test)[:, 1])

#permutation feature importance KNN
pfi_knn = permutation_importance(knn_classifier, X_test, y_test, n_repeats=10, random_state=42)
pfi_knn_df = pd.DataFrame({
    'feature': X.columns, 
    'importance_mean': pfi_knn.importances_mean,
    'importance_std': pfi_knn.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(f'\nPermutation Feature Importance for KNN:\n{pfi_knn_df}')

#bar plot for visualization of feature importance KNN
plt.figure(figsize=(15, 10))
plt.bar(pfi_knn_df['feature'], pfi_knn_df['importance_mean'], yerr=pfi_knn_df['importance_std'], capsize=4)
plt.ylabel('Mean importance (performance drop)')
plt.title('Permutation Feature Importance - KNN')
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()


#BORDA COUNT
model_dfs = {
    'Random Forest': pfi_rf_df,
    'SVM': pfi_svm_df,
    'KNN': pfi_knn_df
}
borda_scores = defaultdict(float) #score dictionary
#borda_per_model = defaultdict(list) #track individual model points

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
        #borda_per_model[feature].append(points)

final_rank = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)

print("Final Borda Count Rankings:")
for feature, score in final_rank:
    print(f"{feature}: {score:.2f}")


#bar chart for visualization of Borda Count feature importance

plt.figure(figsize=(15, 10))
plt.bar([f for f, _ in final_rank], df_stored['importance_mean'], yerr=df_stored['importance_std'], capsize=4)
plt.ylabel('Mean importance (performance drop)')
plt.title('Borda Count Feature Rankings (RF + SVM + KNN)')
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()

"""features_ordered = [f for f, _ in final_rank]

importance_per_model = {f: [] for f in features_ordered}

for model_name, df in model_dfs.items():
    for f in features_ordered: 
        val = df.loc[df['feature'] == f, 'importance_mean'].values[0]
        importance_per_model[f].append(val)

avg_importance = [np.mean(importance_per_model[f]) for f in features_ordered]
std_importance = [np.std(importance_per_model[f]) for f in features_ordered]

fig, ax = plt.subplots(figsize=(15, 10))

ax.bar(features_ordered, avg_importance, color='steelblue', width=0.6, yerr=std_importance, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': 'black'})

ax.set_ylabel('Mean importance (performance drop)')
ax.set_title('Borda Count Feature Rankings (RF + SVM + KNN)')
ax.set_xticks(range(len(features_ordered)))
ax.set_xticklabels(features_ordered, rotation=45, ha='right', fontsize=10)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()"""

#using the highest performing features to train new models 
top_features = [feat for feat, _ in final_rank]
X_top = X[top_features[:5]] #select top 5 features
print("Top features selected for XAI models:", X_top.columns.tolist())

categorical_featues_top = [feat for feat in categorical_features if feat in X_top.columns] #update categorical features list
numerical_features_top = [feat for feat in numerical_features if feat in X_top.columns] #update numerical features list

preprocessor_xai = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_top),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_featues_top)
    ])

#splitting the top features data into training and testing sets
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
recall_rf_xai = class_report_rf_xai['1']['recall']
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
recall_svm_xai = class_report_svm_xai['1']['recall']
auc_svm_xai = roc_auc_score(y_test_xai, svm_classifier_xai.predict_proba(X_test_xai)[:, 1])


#train new KNN model with top features
knn_classifier_xai = Pipeline(steps=[
    ('preprocessor', preprocessor_xai),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

knn_classifier_xai.fit(X_train_xai, y_train_xai) #train the model
y_pred_knn_xai = knn_classifier_xai.predict(X_test_xai) #make predictions on the test set

#evaluate the new KNN model
accuracy_knn_xai = accuracy_score(y_test_xai, y_pred_knn_xai)
cm_knn_xai = confusion_matrix(y_test_xai, y_pred_knn_xai)
class_report_knn_xai = classification_report(y_test_xai, y_pred_knn_xai, output_dict=True)
f1_knn_xai = class_report_knn_xai['1']['f1-score']
precision_knn_xai = class_report_knn_xai['1']['precision']
recall_knn_xai = class_report_knn_xai['1']['recall']
auc_knn_xai = roc_auc_score(y_test_xai, knn_classifier_xai.predict_proba(X_test_xai)[:, 1])


#creating xai ensemble model using stacking 
estimators_xai = [
    ('rf', rf_classifier_xai), 
    ('svm', svm_classifier_xai), 
    ('knn', knn_classifier_xai)
]
final_estimator_xai = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

ensamble_xai = StackingClassifier(
    estimators=estimators_xai, 
    final_estimator=final_estimator_xai,
    cv=5
)
ensamble_xai.fit(X_train_xai, y_train_xai) #train the ensemble model
y_pred_ensemble_xai = ensamble_xai.predict(X_test_xai) #make predictions on the test set

#evaluate the xai ensemble model
accuracy_ensemble_xai = accuracy_score(y_test_xai, y_pred_ensemble_xai)
cm_ensemble_xai = confusion_matrix(y_test_xai, y_pred_ensemble_xai)
class_report_ensemble_xai = classification_report(y_test_xai, y_pred_ensemble_xai, output_dict=True)
f1_ensemble_xai = class_report_ensemble_xai['1']['f1-score']
precision_ensemble_xai = class_report_ensemble_xai['1']['precision']
recall_ensemble_xai = class_report_ensemble_xai['1']['recall']
auc_ensemble_xai = roc_auc_score(y_test_xai, ensamble_xai.predict_proba(X_test_xai)[:, 1])


#fianl results 
table = pd.DataFrame({
    'Model': ['XAI SVM', 'XAI RF', 'XAI KNN', 'XAI Ensemble', 'ML SVM', 'ML RF', 'ML KNN', 'ML Ensemble'], 
    'Accuracy': [accuracy_svm_xai, accuracy_rf_xai, accuracy_knn_xai ,accuracy_ensemble_xai, None, None, None, None],
    'F1 Score': [f1_svm_xai, f1_rf_xai, f1_knn_xai, f1_ensemble_xai, None, None, None, None],
    'Precision': [precision_svm_xai, precision_rf_xai, precision_knn_xai, precision_ensemble_xai, None, None, None, None], 
    'Recall': [recall_svm_xai, recall_rf_xai, recall_knn_xai, recall_ensemble_xai, None, None, None, None],
    'AUC': [auc_svm_xai, auc_rf_xai, auc_knn_xai, auc_ensemble_xai, None, None, None, None]
})

print(table)