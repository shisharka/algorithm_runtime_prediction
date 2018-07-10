import numpy
# from forward_feature_selection import *
# from feature_selection_ffs import *
# from feature_selection_pca import *

# Constants
BROKEN_FEATURE_VALUES = [-512]
RAND = 1234           # fixed random state
LOG10_TRESHOLD = 5e-3 # Due to the resolution of CPU timer, runtimes below 0.01 seconds are measured as 0 seconds. 
                      # To make yi = log(ri) well defined in these cases,
                      # we count them as 0.005 (which, in log space, has the same
                      # distance from 0.01 as the next bigger value measurable with our CPU timer, 0.02).
STD_TRESHOLD = 1e-6   # for constant columns removal


# functions for calculating mean and st. dev. vectors for feature matrix X;
# ignore broken features by default
def mean(X, ignore_broken_features=True):
  num_features = X.shape[1]
  mean = numpy.zeros(num_features)
  for i in range(0, num_features):
    if ignore_broken_features:
      mean[i] = numpy.mean([x for x in X[:, i] if x not in BROKEN_FEATURE_VALUES])
    else:
      mean[i] = numpy.mean(X[:, i])
  return mean

def std(X, ignore_broken_features=True):
  num_features = X.shape[1]
  std = numpy.zeros(num_features)
  for i in range(0, num_features):
    if ignore_broken_features:
      std[i] = numpy.std([x for x in X[:, i] if x not in BROKEN_FEATURE_VALUES])
    else:
      std[i] = numpy.std(X[:, i])
  return std


# function for removing constant columns (features) from feature matrix X
def remove_const_cols(X):
  std_vec = std(X)
  remove_indices = numpy.array([])
  for i in range(0, std_vec.shape[0]):
    if std_vec[i] < STD_TRESHOLD:
      remove_indices = numpy.append(remove_indices, i)
  # print('Removed', remove_indices.shape[0], 'columns:', remove_indices)
  return numpy.delete(X, remove_indices, axis=1), remove_indices


# function for subtracting mean from feature matrix X; ignore broken features by default
# (in which case we're setting broken feature values to 0)
def center(X, mean_vec, ignore_broken_features=True):
  X_centered = numpy.zeros(X.shape)
  for i in range(0, X_centered.shape[0]):
    for j in range(0, X_centered.shape[1]):
      should_center = (ignore_broken_features and (X[i, j] not in BROKEN_FEATURE_VALUES)) or ~ignore_broken_features
      if should_center:
        X_centered[i, j] = X[i, j] - mean_vec[j]
      else:
        X_centered[i, j] = 0
  return X_centered


# function for standardizing feature matrix X (mean 0, std 1); ignore broken features by default
# (in which case we're setting broken feature values to 0)
def standardize(X, mean_vec, std_vec, ignore_broken_features=True):
  X_scaled = numpy.zeros(X.shape)
  for i in range(0, X_scaled.shape[0]):
    for j in range(0, X_scaled.shape[1]):
      should_scale = (ignore_broken_features and (X[i, j] not in BROKEN_FEATURE_VALUES)) or ~ignore_broken_features
      if should_scale:
        X_scaled[i, j] = (X[i, j] - mean_vec[j]) / std_vec[j]
      else:
        X_scaled[i, j] = 0
  return X_scaled


# function for quadratic expansion of feature matrix
def calculate_interactions(X, ignore_broken_features=True):
  num_features = X.shape[1]
  num_int_features = num_features + num_features * (num_features - 1) // 2
  X_int = numpy.zeros((X.shape[0], num_int_features)) 
  for i in range(0, X.shape[0]):
    for j in range(0, num_features):
      k_from = 0
      for l in range(0, j):
        k_from += num_features - l
      k_to = k_from + num_features - j
      for k in range(k_from, k_to):
        if ignore_broken_features and X[i, j] in BROKEN_FEATURE_VALUES:
          X_int[i, k] = X[i, j]
        elif ignore_broken_features and X[i, j + k - k_from] in BROKEN_FEATURE_VALUES:
          X_int[i, k] = X[i, j + k - k_from]
        else:
          X_int[i, k] = X[i, j] * X[i, j + k - k_from]
  return numpy.hstack((X, X_int))


# function for log10 transformation of response variable
def log10_transform(Y, use_treshold=True):
  if use_treshold:
    Y[Y < LOG10_TRESHOLD] = LOG10_TRESHOLD
  return numpy.log10(Y)


# function for setting broken feature values to mean value for that feature
def handle_broken_features(X, mean_vec):
  num_features = X.shape[1]
  num_samples = X.shape[0]

  X_new = numpy.zeros(X.shape)

  for i in range(0,  num_samples):
    for j in range(0, num_features):
      if X[i, j] in BROKEN_FEATURE_VALUES:
        X_new[i, j] = mean_vec[j]
      else:
        X_new[i, j] = X[i, j]

  return X_new
