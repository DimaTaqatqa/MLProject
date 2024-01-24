import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, \
    precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with the actual file path and column names)
df = pd.read_csv('tic-tac-toe.data', header=None)

# Label encode categorical values
label_encoder = LabelEncoder()
for col in range(9):
    df[col] = label_encoder.fit_transform(df[col])

# Encode the target variable
df[9] = label_encoder.fit_transform(df[9])

# Split the data into features (X) and target variable (y)
X = df.iloc[:, :9]
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

# Get the probability scores
y_scores = best_svm_model.predict_proba(X_test)

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred, zero_division=1)

# Print results
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report_str)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

recall = dict()
precision = dict()
threshold = dict()
fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(len(label_encoder.classes_)):
    precision[i], recall[i], threshold[i] = precision_recall_curve(y_test == i, y_scores[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {label_encoder.classes_[i]}')

# Plot Precision-Recall curves for each class
plt.figure(figsize=(10, 7))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Each Class")

# Explicitly set the labels for the legend
legend_labels = [f'Class {label_encoder.classes_[i]}' for i in range(len(label_encoder.classes_))]
plt.legend(loc="best", title='Class', labels=legend_labels)

plt.show()

for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()