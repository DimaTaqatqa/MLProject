import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

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

# Predict and evaluate for k=1
y_pred_1 = knn_1.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred_1)
print(f'Accuracy for k=1: {accuracy_1}')

# Precision for k=1
precision_1 = precision_score(y_test, y_pred_1, pos_label=1)
print(f'Precision for k=1: {precision_1}')

# Define and train the k-NN model with k=3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)

# Predict and evaluate for k=3
y_pred_3 = knn_3.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred_3)
print(f'Accuracy for k=3: {accuracy_3}')

# Precision for k=3
precision_3 = precision_score(y_test, y_pred_3, pos_label=1)
print(f'Precision for k=3: {precision_3}')

# True Positives, True Negatives, False Positives, False Negatives for k=1
conf_matrix_1 = confusion_matrix(y_test, y_pred_1, labels=[1, 0])
tp_1, fn_1, fp_1, tn_1 = conf_matrix_1.ravel()

# Precision for k=1
precision_1 = precision_score(y_test, y_pred_1, pos_label=1)
print(f'Precision for k=1: {precision_1}')
accuracy_1 = (tp_1 + tn_1) / (tp_1 + tn_1 + fp_1 + fn_1)
print(f'Accuracy for k=1: {accuracy_1}')

# True Positives, True Negatives, False Positives, False Negatives for k=3
conf_matrix_3 = confusion_matrix(y_test, y_pred_3, labels=[1, 0])
tp_3, fn_3, fp_3, tn_3 = conf_matrix_3.ravel()

# Precision for k=3
precision_3 = precision_score(y_test, y_pred_3, pos_label=1)
print(f'Precision for k=3: {precision_3}')
accuracy_3 = (tp_3 + tn_3) / (tp_3 + tn_3 + fp_3 + fn_3)
print(f'Accuracy for k=3: {accuracy_3}')

# Define and train the Decision Tree model
dt_params = {'max_depth': [None, 5, 10, 15]}  # Example hyperparameters to tune
dt = DecisionTreeClassifier()
grid_search_dt = GridSearchCV(dt, dt_params, cv=3)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_

# Predict and evaluate for Decision Tree
y_pred_dt = best_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, pos_label=1)
print(f'Accuracy for Decision Tree: {accuracy_dt}')
print(f'Precision for Decision Tree: {precision_dt}')

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
print(f'Accuracy for Random Forest: {accuracy_rf}')
print(f'Precision for Random Forest: {precision_rf}')

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
    plt.title(title)

# Plot decision boundaries for Decision Tree in 2D after PCA
plot_decision_boundaries_2d_pca(X_pca, y_encoded, best_dt, "Decision Boundaries for Decision Tree after PCA")
plt.show()

# Plot decision boundaries for Random Forest in 2D after PCA
plot_decision_boundaries_2d_pca(X_pca, y_encoded, best_rf, "Decision Boundaries for Random Forest after PCA")
plt.show()

