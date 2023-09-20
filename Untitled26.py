#!/usr/bin/env python
# coding: utf-8

# # question 01
# Polynomial functions and kernel functions are both used in machine learning algorithms to transform data into a higher-dimensional feature space. This transformation is often done to make the data more amenable to classification or regression tasks, particularly when the data is not linearly separable in its original form.
# 
# Here's the relationship between polynomial functions and kernel functions:
# 
# Polynomial Functions:
# 
# A polynomial function is a mathematical function that involves powers of a variable raised to non-negative integer exponents.
# In machine learning, a polynomial feature transformation involves creating new features by taking all combinations of the original features raised to non-negative integer powers.
# For example, if you have a feature (x), a second-degree polynomial transformation would include features like (x^2), (x^3), and so on.
# Polynomial features can be used in linear models to allow them to capture non-linear relationships in the data.
# Kernel Functions:
# 
# A kernel function, in the context of machine learning, is a function that computes a dot product in some (possibly infinite-dimensional) feature space.
# It allows us to implicitly represent data in a higher-dimensional space without explicitly computing the coordinates of the data points in that space.
# Commonly used kernel functions include polynomial kernels, Gaussian (RBF) kernels, and more.
# A polynomial kernel is a specific type of kernel function that computes the dot product as if the data had been transformed using a polynomial feature transformation.
# Relationship:
# 
# The relationship between polynomial functions and polynomial kernels lies in the fact that a polynomial kernel is essentially performing a polynomial feature transformation, but in a more computationally efficient manner.
# 
# For example, if you use a second-degree polynomial kernel, it's equivalent to applying a second-degree polynomial feature transformation to the data. However, instead of explicitly calculating the transformed features, the kernel computes the dot product in the higher-dimensional space directly.
# 
# This is particularly useful when working with high-dimensional or even infinite-dimensional feature spaces, where explicitly computing the transformed features would be impractical or impossible.
# 
# In summary, while polynomial functions are a specific type of feature transformation, polynomial kernels are a way to achieve a similar effect, but in a more efficient and often more powerful manner, especially in the context of SVMs and other kernel-based methods.

# # question 02

# In[1]:


from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset (for demonstration purposes)
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize an SVM classifier with a polynomial kernel
svm_classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=42)

# Train the classifier on the training set
svm_classifier.fit(X_train, y_train)

# Predict labels for the testing set
y_pred = svm_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of SVM with Polynomial Kernel: {accuracy*100:.2f}%")


# # question 03
In Support Vector Regression (SVR), the parameter epsilon (ε) determines the width of the epsilon-insensitive tube around the regression line. This tube defines a margin within which errors are not penalized. Instances that fall outside this tube are considered errors and incur a penalty.

As you increase the value of epsilon:

1. **Wider Tube**:
   - A larger epsilon value increases the width of the epsilon-insensitive tube. This means that more data points are allowed to fall within the margin without incurring a penalty.

2. **More Support Vectors**:
   - With a wider tube, more data points can be accommodated within the margin. Consequently, more data points may become support vectors.

3. **Smoothing the Model**:
   - A larger epsilon allows for a more relaxed fit, which can lead to a smoother regression function. It's less sensitive to individual data points and may result in a model that generalizes better to unseen data.

4. **Reduced Model Complexity**:
   - In terms of model complexity, a larger epsilon tends to result in a simpler model because it allows for more data points to be considered as non-errors. This can be particularly useful when dealing with noisy or sparse data.

5. **Potentially Lower Accuracy**:
   - However, it's important to note that increasing epsilon too much might lead to underfitting, where the model becomes too simple and lacks the capacity to capture the underlying patterns in the data.

Overall, the choice of epsilon in SVR depends on the specific characteristics of the data and the problem at hand. It's often determined through cross-validation or other model selection techniques to find the value that provides the best trade-off between bias and variance, resulting in a model that generalizes well to unseen data.
# # question 04
Yes, the choice of various parameters in Support Vector Regression (SVR) can significantly impact the performance of the model. Let's discuss each parameter and how it influences SVR:

1. **Kernel Function**:
   - **Role**: The kernel function determines the type of mapping that is applied to transform the input data into a higher-dimensional feature space where the SVR algorithm can find a linear hyperplane.
   - **Examples**:
     - **Linear Kernel**: Suitable for linear relationships between the features and target variable.
     - **RBF (Radial Basis Function) Kernel**: More flexible and capable of capturing non-linear relationships. However, it requires tuning of the gamma parameter.
     - **Polynomial Kernel**: Useful for data with polynomial relationships.

2. **C Parameter**:
   - **Role**: The C parameter controls the trade-off between maximizing the margin (minimizing training error) and minimizing the classification error on the training set.
   - **Examples**:
     - **Small C**: Allows for a wider margin and more misclassifications. It might lead to a simpler model with higher bias and lower variance.
     - **Large C**: Penalizes misclassifications more heavily, resulting in a narrower margin and potentially a more complex model with lower bias but higher variance.

3. **Epsilon (ε) Parameter**:
   - **Role**: The epsilon parameter determines the width of the epsilon-insensitive tube around the regression line. It defines a region within which errors are not penalized.
   - **Examples**:
     - **Small ε**: Narrow tube, enforces a stricter fit to the data, which may lead to overfitting.
     - **Large ε**: Wider tube, allows for a more relaxed fit, which can lead to a smoother model with potentially better generalization.

4. **Gamma Parameter** (for RBF Kernel):
   - **Role**: The gamma parameter determines the influence of a single training example. A small gamma means a large influence, and a large gamma means a small influence.
   - **Examples**:
     - **Small Gamma**: Wider influence, smoother decision boundary. It may lead to underfitting.
     - **Large Gamma**: Narrow influence, more complex and potentially overfitting decision boundary.

**Examples**:

- **Scenario 1**: If you have prior knowledge that the relationship between features and target is highly non-linear, you might choose an RBF kernel with a relatively large gamma value.

- **Scenario 2**: If you have noisy data and want to reduce overfitting, you might increase the value of epsilon to allow for a wider epsilon-insensitive tube.

- **Scenario 3**: If you want to emphasize minimizing training errors and are willing to accept a more complex model, you might increase the value of the C parameter.

It's important to note that the optimal values for these parameters often need to be determined through techniques like cross-validation, grid search, or other model selection approaches, as they depend on the specific characteristics of the data and the problem at hand.
# # question 05 

# In[2]:


from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Optionally, you can print some information about the dataset
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(set(y))}")


# In[3]:


from sklearn.model_selection import train_test_split

# Split the dataset into a training set and a testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, you have the following variables available:
# X_train: Training features
# y_train: Corresponding labels for the training set
# X_test: Testing features
# y_test: Corresponding labels for the testing set

# Optionally, you can print the shapes of the sets to verify the split
print("Shapes of sets:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# In[4]:


from sklearn.preprocessing import StandardScaler

# Initialize a StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now, X_train_scaled and X_test_scaled are the scaled feature sets


# In[5]:


from sklearn.svm import SVC

# Initialize an instance of SVC
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier on the training data
svm_classifier.fit(X_train_scaled, y_train)


# In[6]:


# Predict labels for the testing data
y_pred = svm_classifier.predict(X_test_scaled)

# Now, y_pred contains the predicted labels for the testing data


# In[7]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# In[8]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their potential values
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto'],
}

# Initialize an SVC classifier
svm_classifier = SVC()

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Get the best estimator (classifier with the best parameters)
best_classifier = grid_search.best_estimator_

# Use the best classifier to predict labels for the testing data
y_pred_best = best_classifier.predict(X_test_scaled)

# Calculate the accuracy of the best classifier
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Best Parameters: {best_params}")
print(f"Accuracy with Best Parameters: {accuracy_best*100:.2f}%")


# In[9]:


# Initialize an SVC classifier with the best parameters
best_classifier = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'], random_state=42)

# Scale the entire dataset (if not already scaled)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the classifier on the entire dataset
best_classifier.fit(X_scaled, y)


# In[10]:


import joblib

# Define the file path where you want to save the classifier
file_path = 'svm_classifier_model.joblib'

# Save the trained classifier to the file
joblib.dump(best_classifier, file_path)

print(f"The trained classifier has been saved to '{file_path}' for future use.")


# In[ ]:




