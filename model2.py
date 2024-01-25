import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, roc_curve, f1_score,
    precision_score, recall_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Load your dataset (replace 'your_dataset.csv' with the actual file path and column names)
df = pd.read_csv('tic-tac-toe.data', header=None)

# x -> 2 , o -> 1, b -> 0, for classes - > negative -> 0 and positive -> 1

# Label encode categorical values
label_encoder = LabelEncoder()
for col in range(9):
    df[col] = label_encoder.fit_transform(df[col])

# Encode the target variable
df[9] = label_encoder.fit_transform(df[9])

# Split the data into features (X) and target variable (y)
# 10 columns
# first 9 columns for features
X = df.iloc[:, :9]
# last column for target
y = df[9]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.01, 0.1, 1, 10, 100]}

# Create an SVM model
svm_model = SVC(probability=True)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_svm_model = grid_search.best_estimator_

# best_svm_model = SVC(kernel='poly', C=10, probability=True)
# best_svm_model.fit(X_train, y_train)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data points
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')

# Plot the decision boundaries using contourf
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = best_svm_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(np.unique(y)) + 1) - 0.5, cmap='viridis')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('SVM Decision Boundaries (PCA)')
plt.legend()
plt.show()

# Get the decision function values
dec_function_values = best_svm_model.decision_function(X_test)

# Get the probability scores
y_scores = best_svm_model.predict_proba(X_test)
# print(y_scores)
# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
# Print the classification report
print("\nClassification Report:")
print(classification_report_str)
# Print F1 score
print(f'F1 Score: {f1:.4f}')

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Precision-Recall curve
precision = dict()
recall = dict()
threshold = dict()
for i in range(2):
    precision[i] ,recall[i], threshold[i] = precision_recall_curve(y_test == i, y_scores[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=f"class {i}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()

# Plot Precision-Recall curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f"class {i} (area = {roc_auc[i]:0.2f})")
# # Calculate TPR and FPR for ROC curve
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

