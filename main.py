# Maksim KOZLOV 20219332
# Imported libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("./wdbc.data", header=None)

# Drop the first column
dataset.drop(0, axis=1, inplace=True)

# Extract the second column as the class label and Extract the rest of the columns as features
y = dataset.iloc[:, 0]
X = dataset.iloc[:, 1:]

# Split the data into training and testing sets for KNN
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=169, train_size=400, random_state=42)

# Define feature dimensions for KNN
dimensions_knn = [5, 10, 15, 20]

# Define values of K for KNN
k_values_knn = [1, 3, 5, 7, 9]

# Initialise results for KNN
accuracy_results_knn = {}
precision_results_knn = {}
recall_results_knn = {}
f1_results_knn = {}

# Apply PCA to reduce the original input features into new feature vectors with
# different dimensions, 5, 10, 15, 20 for KNN
for n_components in dimensions_knn:
    # Create PCA instance
    pca_knn = PCA(n_components=n_components)

    # Fit and transform the training set for KNN
    X_train_pca_knn = pca_knn.fit_transform(X_train_knn)

    # Transform the testing set for KNN
    X_test_pca_knn = pca_knn.transform(X_test_knn)

    print(f"\nKNN with PCA Components: {n_components}")

    accuracy_results_knn[n_components] = []
    precision_results_knn[n_components] = []
    recall_results_knn[n_components] = []
    f1_results_knn[n_components] = []

    # Accuracy, precision, recall, and f1 values for KNN
    # Try different values of K: 1, 3, 5, 7, 9 for KNN
    for k_value in k_values_knn:
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k_value)

        # Fit the KNN model on the original or reduced input for KNN
        knn.fit(X_train_pca_knn, y_train_knn)

        # Predict on the testing set for KNN
        y_predict_knn = knn.predict(X_test_pca_knn)

        # Calculate accuracy for KNN
        accuracy_knn = accuracy_score(y_test_knn, y_predict_knn)
        accuracy_results_knn[n_components].append(accuracy_knn)

        # Calculate precision for KNN
        precision_knn = precision_score(y_test_knn, y_predict_knn, pos_label='M')
        precision_results_knn[n_components].append(precision_knn)

        # Calculate recall for KNN
        recall_knn = recall_score(y_test_knn, y_predict_knn, pos_label='M')
        recall_results_knn[n_components].append(recall_knn)

        # Calculate f1 for KNN
        f1_knn = f1_score(y_test_knn, y_predict_knn, pos_label='M')
        f1_results_knn[n_components].append(f1_knn)


# Plot the results
plt.figure(figsize=(12, 10))

for n_components in dimensions_knn:
    plt.plot(k_values_knn, accuracy_results_knn[n_components], label=f'Accuracy Dim={n_components}', color='blue')
    plt.plot(k_values_knn, precision_results_knn[n_components], label=f'Precision Dim={n_components}', color='red')
    plt.plot(k_values_knn, recall_results_knn[n_components], label=f'Recall Dim={n_components}', color='green')
    plt.plot(k_values_knn, f1_results_knn[n_components], label=f'Results Dim={n_components}', color='cyan')

plt.title('KNN Results of various feature dimension and Ks.')
plt.xlabel('K Value')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.show()

# Printing of the results
with open('output.txt', 'a') as file:
    print(f"Accuracy Results: {accuracy_results_knn}", file=file)
    print(f"Precision Results: {precision_results_knn}", file=file)
    print(f"Recall Results: {recall_results_knn}", file=file)
    print(f"F1 Results: {f1_results_knn}", file=file)

# Load the dataset for MLP
dataset_mlp = pd.read_csv("./wdbc.data", header=None)

# Drop the first column for MLP
dataset_mlp.drop(0, axis=1, inplace=True)

# Extract the second column as the class label and Extract the rest of the columns as features for MLP
y_mlp = dataset_mlp.iloc[:, 0]
X_mlp = dataset_mlp.iloc[:, 1:]

# Use the best-performing feature dimension from Task 2 (e.g., PCA with n_components=15) for MLP
pca_mlp = PCA(n_components=15)
X_pca_mlp = pca_mlp.fit_transform(X_mlp)

# Split the data into training and testing sets for MLP
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(X_pca_mlp, y_mlp, test_size=0.2, random_state=42)

# Define a reduced set of values for hidden_layer_sizes for MLP
hidden_layer_sizes_values_mlp = [(50,), (100,), (100, 50)]

# Define parameter search space for RandomizedSearchCV for MLP
param_random_mlp = {
    'hidden_layer_sizes': hidden_layer_sizes_values_mlp,
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# Initialize MLP classifier for MLP
mlp = MLPClassifier(random_state=42)

# Perform randomised search for hyperparameter tuning for MLP
random_search_mlp = RandomizedSearchCV(mlp, param_distributions=param_random_mlp, n_iter=50, cv=5, scoring='accuracy', random_state=42)
random_search_mlp.fit(X_train_mlp, y_train_mlp)

# Print the best hyperparameters for MLP
print(f"Best Hyperparameters for MLP: {random_search_mlp.best_params_}", file=open('output2.txt', 'a'))

# Train the model with the best hyperparameters on the entire training set for MLP
best_mlp = random_search_mlp.best_estimator_
best_mlp.fit(X_train_mlp, y_train_mlp)

# Evaluate on the test set for MLP
y_pred_mlp = best_mlp.predict(X_test_mlp)

# Calculate performance metrics for MLP
accuracy_mlp = accuracy_score(y_test_mlp, y_pred_mlp)
precision_mlp = precision_score(y_test_mlp, y_pred_mlp, pos_label='M')
recall_mlp = recall_score(y_test_mlp, y_pred_mlp, pos_label='M')
f1_mlp = f1_score(y_test_mlp, y_pred_mlp, pos_label='M')

accuracy_mlp = round(accuracy_mlp, 4)
precision_mlp = round(precision_mlp, 4)
recall_mlp = round(recall_mlp, 4)
f1_mlp = round(f1_mlp, 4)


# Print performance metrics for MLP
print("Test Set Performance Metrics for MLP:", file=open('output2.txt', 'a'))
print(f"Accuracy: {accuracy_mlp}", file=open('output2.txt', 'a'))
print(f"Precision: {precision_mlp}", file=open('output2.txt', 'a'))
print(f"Recall: {recall_mlp}", file=open('output2.txt', 'a'))
print(f"F1 Score: {f1_mlp}", file=open('output2.txt', 'a'))

# Print confusion matrix for MLP
conf_matrix_mlp = confusion_matrix(y_test_mlp, y_pred_mlp)
print(f"Confusion Matrix for MLP: {conf_matrix_mlp}", file=open('output2.txt', 'a'))
