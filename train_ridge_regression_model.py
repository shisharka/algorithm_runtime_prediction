from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
from data_preprocessing import mean, std, remove_const_cols, standardize

RAND = 3

def read_data(set_name):
  X = pd.read_csv('data/' + set_name + '-feat.csv')
  X = X.drop('INSTANCE_ID', axis=1).get_values()
  # print('dim(X) =', X.shape)

  Y = pd.read_csv('data/' + set_name + '-minisat.csv', names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME'].get_values()
  # print('dim(Y) =', Y.shape)
  # print('Y[0:10] =', Y[0:10])

  return X, Y


def determine_reg_param(X_train, Y_train, set_name):
  alphas = 10**np.linspace(-5, 5, 200)*0.5
  errors = np.array([])
  
  for a in alphas:
    reg = linear_model.Ridge(alpha=a)
    scores = model_selection.cross_val_score(reg, X_train, Y_train, cv=10, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    errors = np.append(errors, scores.mean())

  # ploting error w.r.t regularization parameter alpha
  # plt.plot(alphas, errors)
  # plt.xscale('log')
  # plt.show()

  min_index = errors.argmin()
  alpha = alphas[min_index]
  # print('Regularization parameter for', set_name, ':', alpha)
  return alpha

def train_ridge_regression_model(X, Y, set_name):

  # removing const columns
  # X = remove_const_cols(X)

  # train - test split
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=RAND, test_size=0.33)
  
  # standardizing data
  # std_vec = std(X_train) 
  # mean_vec = mean(X_train)
  # X_train = standardize(X_train, std_vec, mean_vec)
  # X_test = standardize(X_test, std_vec, mean_vec)
  
  alpha = determine_reg_param(X_train, Y_train, set_name)
  reg = linear_model.Ridge(alpha=alpha)
  reg.fit(X_train, Y_train)
  Y_predicted = reg.predict(X_test)
  mse = metrics.mean_squared_error(Y_test, Y_predicted)
  print(set_name, 'test RMSE:', np.sqrt(mse))

# nested cross-validation

def nested_cv(X, Y, set_name):
  skf = model_selection.StratifiedKFold(n_splits=10)
  scores = np.array([])

  for train_index, test_index in skf.split(X, Y):
    alpha = determine_reg_param(X[train_index], Y[train_index], set_name)
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X[train_index], Y[train_index])
    Y_predicted = reg.predict(X[test_index])
    mse = metrics.mean_squared_error(Y[test_index], Y_predicted)
    scores = np.append(scores, np.sqrt(mse))

  print(set_name, 'test RMSE:', scores.mean())
