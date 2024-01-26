import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

#KNN

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset from a .data file
data = pd.read_csv('tic-tac-toe.data', header=None)

# Assuming the target column is the 10th column (index 9)
X = data.iloc[:, :9]  # Features
y = data.iloc[:, 9]  # Target

# Perform one-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=X.columns)

# Encode the target labels for visualization
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Define and train the k-NN model with k=1
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)

# Define and train the k-NN model with k=3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)

# Create a meshgrid for visualization
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict class for each point in the meshgrid for k=1
Z_1 = knn_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

# Predict class for each point in the meshgrid for k=3
Z_3 = knn_3.predict(np.c_[xx.ravel(), yy.ravel()])
Z_3 = Z_3.reshape(xx.shape)

# Plot decision boundaries along with training data
plt.figure(figsize=(12, 6))

# Decision boundary for k=1
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_1, alpha=0.8, cmap='viridis')
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, edgecolors='k', cmap='viridis')
plt.title('Decision Boundary for k=1')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Classes")

# Decision boundary for k=3
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_3, alpha=0.8, cmap='viridis')
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, edgecolors='k', cmap='viridis')
plt.title('Decision Boundary for k=3')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Classes")

plt.tight_layout()
plt.show()

# Define and train the k-NN model with k=1
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)

# Predict and evaluate for k=1
y_pred_1 = knn_1.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred_1)
precision_1 = precision_score(y_test, y_pred_1, pos_label=1)
recall_1 = recall_score(y_test, y_pred_1, pos_label=1)
conf_matrix_1 = confusion_matrix(y_test, y_pred_1, labels=[1, 0])
classification_rep_1 = classification_report(y_test, y_pred_1)

# Calculate ROC curve and AUC for k=1
fpr_1, tpr_1, thresholds_1 = roc_curve(y_test, knn_1.predict_proba(X_test)[:, 1])
roc_auc_1 = roc_auc_score(y_test, knn_1.predict_proba(X_test)[:, 1])

# Calculate precision-recall curve and AUC for k=1
precision_recall_curve_1, recall_precision_curve_1, _ = precision_recall_curve(y_test, knn_1.predict_proba(X_test)[:, 1])
pr_auc_1 = auc(recall_precision_curve_1, precision_recall_curve_1)

# Print results for k=1
print("Results for k=1:")
print(f'Accuracy: {accuracy_1}')
print(f'Precision: {precision_1}')
print(f'Recall: {recall_1}')
print('Confusion Matrix:')
print(conf_matrix_1)
print('Classification Report:')
print(classification_rep_1)
print(f'ROC AUC: {roc_auc_1}')
print(f'PR AUC: {pr_auc_1}')

# Define and train the k-NN model with k=3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)

# Predict and evaluate for k=3
y_pred_3 = knn_3.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred_3)
precision_3 = precision_score(y_test, y_pred_3, pos_label=1)
recall_3 = recall_score(y_test, y_pred_3, pos_label=1)
conf_matrix_3 = confusion_matrix(y_test, y_pred_3, labels=[1, 0])
classification_rep_3 = classification_report(y_test, y_pred_3)

# Calculate ROC curve and AUC for k=3
fpr_3, tpr_3, thresholds_3 = roc_curve(y_test, knn_3.predict_proba(X_test)[:, 1])
roc_auc_3 = roc_auc_score(y_test, knn_3.predict_proba(X_test)[:, 1])

# Calculate precision-recall curve and AUC for k=3
precision_recall_curve_3, recall_precision_curve_3, _ = precision_recall_curve(y_test, knn_3.predict_proba(X_test)[:, 1])
pr_auc_3 = auc(recall_precision_curve_3, precision_recall_curve_3)

# Print results for k=3
print("\nResults for k=3:")
print(f'Accuracy: {accuracy_3}')
print(f'Precision: {precision_3}')
print(f'Recall: {recall_3}')
print('Confusion Matrix:')
print(conf_matrix_3)
print('Classification Report:')
print(classification_rep_3)
print(f'ROC AUC: {roc_auc_3}')
print(f'PR AUC: {pr_auc_3}')

# Plot ROC curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr_1, tpr_1, label=f'k=1 (AUC = {roc_auc_1:.2f})')
plt.plot(fpr_3, tpr_3, label=f'k=3 (AUC = {roc_auc_3:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall_precision_curve_1, precision_recall_curve_1, label=f'k=1 (AUC = {pr_auc_1:.2f})')
plt.plot(recall_precision_curve_3, precision_recall_curve_3, label=f'k=3 (AUC = {pr_auc_3:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()
