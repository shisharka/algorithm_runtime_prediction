import numpy as np
import pandas as pd
from sklearn import model_selection

### Constants

BROKEN_FEATURE_VALUES = [-512, -1024]
RAND = 3              # fixed random state
LOG10_TRESHOLD = 5e-3 # for log10 transformation of Y; Matlab constraint for min value of log argument
STD_TRESHOLD = 1e-6   # for constant columns removal
LINEAR_SIZE = 20      # for linear feature selection
DELTA = 1e-2          # default regularization parameter


### Helper functions

# functions for calculating mean and st. dev. vectors for feature matrix X; ignore broken features by default
def mean(X, ignore_broken_features=True):
  num_features = X.shape[1]
  mean = np.zeros(num_features)
  for i in range(0, num_features):
    if ignore_broken_features:
      mean[i] = np.mean([x for x in X[:, i] if x not in BROKEN_FEATURE_VALUES])
    else:
      mean[i] = np.mean(X[:, i])
  return mean


def std(X, ignore_broken_features=True):
  num_features = X.shape[1]
  std = np.zeros(num_features)
  for i in range(0, num_features):
    if ignore_broken_features:
      std[i] = np.std([x for x in X[:, i] if x not in BROKEN_FEATURE_VALUES])
    else:
      std[i] = np.std(X[:, i])
  return std


# function for removing constant columns (features) from feature matrix X
def remove_const_cols(X):
    std_vec = std(X)
    remove_indices = []
    for i in range(0, std_vec.shape[0]):
        if std_vec[i] < STD_TRESHOLD:
            remove_indices.append(i)
    return np.delete(X, remove_indices, axis=1)


# function for standardizing feature matrix X (mean 0, std 1); ignore broken features by default
# (in which case we're setting broken feature values to 0)
def standardize(X, mean_vec, std_vec, ignore_broken_features=True):
  X_scaled = np.zeros(X.shape)
  for i in range(0, X_scaled.shape[0]):
    for j in range(0, X_scaled.shape[1]):
      should_scale = (ignore_broken_features and X[i, j] not in BROKEN_FEATURE_VALUES) or ~ignore_broken_features
      if should_scale:
        X_scaled[i, j] = (X[i, j] - mean_vec[j]) / std_vec[j]
      else:
        X_scaled[i, j] = 0
  return X_scaled


# function for adding feature to lin. reg. model
# A = Xt * X + delta * I
def model_add_feature(X, Y, new_feature, A, A_inv, XtY, delta):
  tempX = X.T.dot(new_feature)
  tempX = tempX.reshape((tempX.shape[0], 1))
  A = np.hstack((A, tempX))
  A = np.vstack((A, np.append(tempX.T, new_feature.T.dot(new_feature))))
  A[-1, -1] += delta
  
  ### update the inverse stored in A_inv and train model ###
  n = A.shape[0] - 1
  
  # augment the matrix
  A_inv = np.hstack((A_inv, np.zeros((n, 1))))
  
  A_inv = np.vstack((A_inv, np.zeros((1, n + 1))))
  A_inv[n, n] = 1
  
  a_bar = A[n, 0:n].reshape((1, n))
  a_dot = A[n, n]
  
  # first rank one update
  u1 = np.vstack((np.zeros((n, 1)), 1))
  v1 = np.vstack((a_bar.T, 0))
  
  A_inv = A_inv - (A_inv.dot(u1).dot(v1.T).dot(A_inv)) / (1 + v1.T.dot(A_inv).dot(u1))
  
  # second rank one update
  u2 = np.vstack((a_bar.T, a_dot - 1))
  v2 = u1
  A_inv = A_inv - (A_inv.dot(u2).dot(v2.T).dot(A_inv)) / (1 + v2.T.dot(A_inv).dot(u2))[0, 0]

  # build the model
  XtY = np.vstack((XtY, new_feature.T.dot(Y)))
  return A_inv.dot(XtY)


# function for greedy forward feature selection
def forward_feature_selection(X_train, X_val, Y_train, Y_val, num_iterations):
  num_train_instances = X_train.shape[0]
  num_test_instances = X_val.shape[0]
  num_features = X_train.shape[1]
  
  num_iterations = min(num_iterations, num_features)
  
  X = np.ones((num_train_instances, 1))
  A = X.T.dot(X) + DELTA # a number
  A_inv = 1 / A
  XtY = X.T.dot(Y_train)
  
  X_test = np.zeros((num_test_instances, num_iterations + 2))
  X_test[:, 0] = np.ones(num_test_instances) # first column - ones
  
  features = np.array([])
  subs = np.array([])
  
  compute_training_rmse = False
  
  # for every set size
  for i in range(0, num_iterations):
    min_rmse = np.inf
    best_subset = 0
    # for each feature not in the set
    for j in range(0, num_features):
      if j in features:
        continue
      
      # construct the model with this feature added
      model = model_add_feature(X, Y_train, X_train[:, j], A, A_inv, XtY, DELTA)
      
      # record the RMSE on validation data for this feature
      X_test[:, i] = X_val[:, j]
      preds = X_test.dot(np.vstack((model, np.zeros((num_iterations - i, 1)))))
      rmse = np.sqrt(np.mean((preds - Y_val)**2))
      
      if compute_training_rmse:
        preds = np.hstack((X, X_train[:, j].reshape((num_train_instances, 1)))).dot(model)
        training_rmse = np.sqrt(np.mean((preds - Y_train)**2))
        print('Training RMSE:', training_rmse)
        
      if rmse < min_rmse:
        min_rmse = rmse
        best_subset = j

    if compute_training_rmse:
        # delimiter; easier to read
        print('--------------------------------')
    
    features = np.append(features, best_subset)
    X = np.hstack((X, X_train[:, best_subset].reshape((num_train_instances, 1))))
    
    A = X.T.dot(X) + DELTA * np.eye(X.shape[1])
    A_inv = np.linalg.pinv(A)
    XtY = X.T.dot(Y_train)
    XtY = XtY.reshape((XtY.shape[0], 1))
    model = A_inv.dot(XtY)
    model = model.reshape((model.shape[0], 1))
    preds = X_test.dot(np.vstack((model, np.zeros((num_iterations - i, 1)))))
    rmse = np.sqrt(np.mean(preds - Y_val)**2)
    subs = np.append(subs, { 'features': features, 'rmse': rmse })
              
  return subs 


### Reading the data

# X = pd.read_csv('data/INDU-HAND-RAND-feat.csv')
# X = X.drop('INSTANCE_ID', axis=1).get_values()
# print('dim(X) =', X.shape)

# Y = pd.read_csv('data/INDU-HAND-RAND-minisat.csv', names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME'].get_values()
# print('dim(Y) =', Y.shape)
# print('Y[0:10] =', Y[0:10])


### Transformation of the response variable y, e.g. log10:

# Y[Y < LOG10_TRESHOLD] = LOG10_TRESHOLD
# Y = np.log10(Y)
# print('After log10 transformation:')
# print('Y[0:10] =', Y[0:10])


### Removing constant columns, disregarding special values (-512, -1024):

# X_no_constant = remove_const_cols(X)
# print('After removing constant columns:')
# print('dim(X_no_constant) =', X_no_constant.shape)


### Train - test split:

# num_features = X_no_constant.shape[1]
# X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_no_constant, Y, train_size=0.5, random_state=RAND)
# print('dim(X_train) =', X_train.shape)
# print('dim(X_val) =', X_val.shape)


### Standardize the training data (mean 0, std 1), handling broken features.
### If an entry is one of the predefined problematic values (-512 or -1024),
### then standardization will not count those entries, and the entries will be set to zero.

# std_vec = std(X_no_constant) 
# mean_vec = mean(X_no_constant)
# ???? Shouldn't std and mean be calculated on train set?
# X_train_scaled = standardize(X_train, std_vec, mean_vec)
# X_val_scaled = standardize(X_val, std_vec, mean_vec)
# print("X_train_scaled[0:5, 0:5] =\n", X_train_scaled[0:5, 0:5])
# print("X_val_scaled[0:5, 0:5] =\n", X_val_scaled[0:5, 0:5])


### Linear forward selection to select the most informative features, using standardized data:

# subs = forward_feature_selection(X_train_scaled, X_val_scaled, Y_train, Y_val, LINEAR_SIZE)
# rmses = []
# for s in subs:
#   rmses.append(s['rmse'])
# best_index = np.argmin(rmses)
# selected_features = subs[best_index]['features']
# print(selected_features)
# print(rmses)


### Quadratic expansion of selected features:
### TODO