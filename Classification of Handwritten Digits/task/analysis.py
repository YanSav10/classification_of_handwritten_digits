import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

x_data, y_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')[0]

n = x_data.shape[0]
m = np.prod(x_data.shape[1:])
x_reshaped = x_data.reshape(n, m)

x_train_partial, x_test_partial, y_train_partial, y_test_partial = train_test_split(x_data[:6000], y_data[:6000],
                                                                                    test_size=0.3, random_state=40)

x_train_partial_flat = x_train_partial.reshape(x_train_partial.shape[0], -1)
x_test_partial_flat = x_test_partial.reshape(x_test_partial.shape[0], -1)

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    score = accuracy_score(target_test, predictions)
    # print(f'Model: {model}\nAccuracy: {score:.3f}\n')

knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=40)
lr = LogisticRegression(random_state=40)
rf = RandomForestClassifier(random_state=40)

normalizer = Normalizer()

# Transform features
x_train_norm = normalizer.fit_transform(x_train_partial_flat)
x_test_norm = normalizer.transform(x_test_partial_flat)

# Fit, predict, and evaluate models with normalized data
fit_predict_eval(knn, x_train_norm, x_test_norm, y_train_partial, y_test_partial)
fit_predict_eval(dt, x_train_norm, x_test_norm, y_train_partial, y_test_partial)
fit_predict_eval(lr, x_train_norm, x_test_norm, y_train_partial, y_test_partial)
fit_predict_eval(rf, x_train_norm, x_test_norm, y_train_partial, y_test_partial)

# print("The answer to the 1st question: yes")

# Determine the two models with the best scores
models = [(knn, knn.score(x_test_norm, y_test_partial)),
          (dt, dt.score(x_test_norm, y_test_partial)),
          (lr, lr.score(x_test_norm, y_test_partial)),
          (rf, rf.score(x_test_norm, y_test_partial))]
models.sort(key=lambda x: x[1], reverse=True)

# print(f"The answer to the 2nd question: {models[0][0].__class__.__name__} - {models[0][1]:.3f}, "
#       f"{models[1][0].__class__.__name__} - {models[1][1]:.3f}")

x_train_final = x_train_norm
x_test_final = x_test_norm

# Initialize GridSearchCV for K-nearest Neighbors
knn_param_grid = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
knn_grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_param_grid, scoring='accuracy',
                               n_jobs=-1)
knn_grid_search.fit(x_train_final, y_train_partial)

# Print the best set of parameters for K-nearest Neighbors
print("K-nearest neighbours algorithm")
print("best estimator:", knn_grid_search.best_estimator_)
print("accuracy:", round(knn_grid_search.best_estimator_.score(x_test_final, y_test_partial), 3))

# Initialize GridSearchCV for Random Forest
rf_param_grid = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'], 'class_weight': ['balanced',
                                                                                                'balanced_subsample']}
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=rf_param_grid,
                              scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(x_train_final, y_train_partial)

# Print the best set of parameters for Random Forest
print("\nRandom forest algorithm")
print("best estimator:", rf_grid_search.best_estimator_)
print("accuracy:", round(rf_grid_search.best_estimator_.score(x_test_final, y_test_partial), 3))
