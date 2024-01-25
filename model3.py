import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

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

# Define and train the Decision Tree model
dt_params = {'max_depth': [None, 5, 10, 15, 20]}  # Example hyperparameters to tune
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
# grid_search_dt = GridSearchCV(dt, dt_params, cv=3)
# grid_search_dt.fit(X_train, y_train)
# best_dt = grid_search_dt.best_estimator_
# print(grid_search_dt.best_params_)

# Predict and evaluate for Decision Tree
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, pos_label=1)
recall_dt = recall_score(y_test, y_pred_dt, pos_label=1)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print(f'Decision Tree Metrics:')
print(f'Accuracy: {accuracy_dt}')
print(f'Precision: {precision_dt}')
print(f'Recall: {recall_dt}')
print(f'F1-Score: {f1_dt}')
print(f'Confusion Matrix:\n{conf_matrix_dt}\n')


# Define and train the Random Forest model
rf_params = {'n_estimators': [50, 100, 150, 200]}  # Example hyperparameters to tune
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, rf_params, cv=3)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Predict and evaluate for Random Forest
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, pos_label=1)
recall_rf = recall_score(y_test, y_pred_rf, pos_label=1)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(f'Random Forest Metrics:')
print(f'Accuracy: {accuracy_rf}')
print(f'Precision: {precision_rf}')
print(f'Recall: {recall_rf}')
print(f'F1-Score: {f1_rf}')
print(f'Confusion Matrix:\n{conf_matrix_rf}\n')

# Function to plot decision boundaries in 2D after PCA
def plot_decision_boundaries_2d_pca(X_pca, y, model, title):
    h = .02  # Step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the decision boundaries
    plt.contour(xx, yy, Z, cmap=cmap_bold, alpha=0.3)

    # Plot the training points with colors corresponding to labels
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'{title}\nConfusion Matrix:\n{conf_matrix_rf}')

# Plot decision boundaries for Decision Tree in 2D after PCA
plot_decision_boundaries_2d_pca(X_pca, y_encoded, dt, "Decision Boundaries for Decision Tree after PCA")
plt.show()

# Plot decision boundaries for Random Forest in 2D after PCA
# plot_decision_boundaries_2d_pca(X_pca, y_encoded, best_rf, "Decision Boundaries for Random Forest after PCA")
# plt.show()
