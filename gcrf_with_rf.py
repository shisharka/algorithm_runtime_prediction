import code # for debugging, to set breakpoint use code.interact(local=dict(globals(), **locals()))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from GCRF import GCRF
from data_preprocessing import log10_transform

datasets = [
'INDU-IBM-SWV',
'RAND_SAT']

RAND = 1234
DELTA = 1e-5
SVD = False
LEARN = 'TNC'

# Function for calculating similarity matrix for Y (columns in Y are solver representations);
# delta is similarity metaparameter;
# similarity is calculated using the following formula: exp(-delta * norm(si - sj) ** 2) (scaled to [0, 1]),
# where si, sj are vector representations for solvers i, j;
# if svd is True, representations of solvers are calculated using svd on matrix Y
def S(Y, delta, svd=False):
  num_solvers = Y.shape[1]
  S = np.zeros((num_solvers, num_solvers))
  if svd == True:
    U, _sigma, _V = np.linalg.svd(Y.T)
  min = max = 0
  for k in range(0, num_solvers):
    for l in range(0, num_solvers):
      if k == l: continue
      if svd == True:
        S[k, l] = np.exp(- delta * np.linalg.norm(U[k, :] - U[l, :]) ** 2)
      else:
        S[k, l] = np.exp(- delta * np.linalg.norm(Y[:, k] - Y[:, l]) ** 2)
      S[l, k] = S[k, l]
      if S[k, l] > max: max = S[k, l]
      # if S[k, l] < min: min = S[k, l]
  # scaling to [0, 1]
  if max > 0: S = (S - min) / (max - min)
  return S

def determine_similarity_metaparam(X, Y):
  num_solvers = Y.shape[1]
  deltas = 10**np.linspace(-6, 0, 7)
  mean_square_errors = np.array([])
  
  gcrf = GCRF()
  rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)
  kf = KFold(n_splits=5, shuffle=True, random_state=RAND)

  for d in deltas:
    i = 0
    for (train_index, test_index) in kf.split(X, Y):
      i += 1; print(i)
      X_train = X[train_index]
      Y_train = Y[train_index]
      X_test = X[test_index]
      Y_test = Y[test_index]

      num_train_instances = X_train.shape[0]
      num_test_instances = X_test.shape[0]

      R_train = np.zeros((num_train_instances, num_solvers))
      Se_train = np.zeros((num_train_instances, 1, num_solvers, num_solvers))
      R_test = np.zeros((num_test_instances, num_solvers))
      Se_test = np.zeros((num_test_instances, 1, num_solvers, num_solvers))

      ### Setting R_train/R_test values
      for s in range(0, num_solvers):
        rf.fit(X_train, Y_train[:, s])
        R_train[:, s] = rf.predict(X_train)
        R_test[:, s] = rf.predict(X_test)

      ### Setting Se_train/Se_test values
      Se_train[:, 0, :, :] = S(Y_train, d, SVD)
      Se_test[:, 0, :, :] = S(Y_test, d, SVD)

      gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn=LEARN)
      predictions = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)

    n = Y_test.shape[0] * num_solvers
    mean_square_errors = \
      np.append(mean_square_errors, mean_squared_error(Y_test.reshape(n), predictions.reshape(n)))

  # ploting mse w.r.t metaparameter delta
  plt.plot(deltas, mean_square_errors)
  plt.xscale('log')
  plt.show()

  min_index = mean_square_errors.argmin()
  delta = deltas[min_index]
  print('Similarity metaparameter:', delta)
  return delta


for dataset in datasets:
  X = pd.read_csv('data/gcrf/' + dataset + '-feat.csv').drop('INSTANCE_ID', axis=1).get_values()
  Y = pd.read_csv('data/gcrf/' + dataset + '-results.csv').drop('INSTANCE_ID', axis=1).get_values()
  Y = log10_transform(Y)

  num_solvers = Y.shape[1]

  # structured model
  gcrf = GCRF()
  # unstructured models
  rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)

  gcrf_predictions = np.zeros(Y.shape)
  rf_predictions = np.zeros(Y.shape)

  kf = KFold(n_splits=5, shuffle=True, random_state=RAND)

  i = 1
  for (outer_train_index, outer_test_index) in kf.split(X, Y):
    print('Outer k-fold', i); i += 1

    X_train = X[outer_train_index]
    Y_train = Y[outer_train_index]
    X_test = X[outer_test_index]
    Y_test = Y[outer_test_index]

    num_train_instances = X_train.shape[0]
    num_test_instances = X_test.shape[0]

    R_train = np.zeros((num_train_instances, num_solvers))
    R_test = np.zeros((num_test_instances, num_solvers))

    Se_train = np.zeros((num_train_instances, 1, num_solvers, num_solvers))
    Se_test = np.zeros((num_test_instances, 1, num_solvers, num_solvers))

    delta = determine_similarity_metaparam(X_train, Y_train)
    # delta = DELTA

    j = 1
    for (inner_train_index, inner_test_index) in kf.split(X_train, Y_train):
      print('Inner k-fold', j); j += 1

      X_train_train = X_train[inner_train_index]
      Y_train_train = Y_train[inner_train_index]
      X_train_test = X_train[inner_test_index]
      Y_train_test = Y_train[inner_test_index]

      ### Setting R_train values
      for s in range(0, num_solvers):
        rf.fit(X_train_train, Y_train_train[:, s])
        R_train[inner_test_index, s] = rf.predict(X_train_test)

      ### Setting Se_train values
      for index in inner_test_index: Se_train[index, 0, :, :] = S(Y_train_train, delta, SVD)

    gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn=LEARN)
    # code.interact(local=dict(globals(), **locals()))
    print('Trained params')
    print('----------------')
    print('ALFA:', gcrf.alfa[0])
    print('BETA:', gcrf.beta[0])
    print('----------------')

    ### Setting R_test values
    for s in range(0, num_solvers):
      rf.fit(X_train, Y_train[:, s])
      R_test[:, s] = rf.predict(X_test)

    ### Setting Se_test values
    for index in range(0, num_test_instances): Se_test[index, 0, :, :] = S(Y_train, delta, SVD)

    gcrf_predictions[outer_test_index, :] = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
    rf_predictions[outer_test_index, :] = R_test

  np.save('predictions/' + dataset + '_rf_0.npy', rf_predictions)
  np.save('predictions/' + dataset + '_gcrf_0.npy', gcrf_predictions)
  a = np.load('predictions/' + dataset + '_rf_0.npy')
  b = np.load('predictions/' + dataset + '_gcrf_0.npy')
  print('!!!!!!!!!!', np.array_equal(rf_predictions, a))
  print('!!!!!!!!!!', np.array_equal(gcrf_predictions, b))

  print("\n")
  print('##########################', dataset, '##########################')
  n = Y.shape[0] * num_solvers
  print('GCRF for all solvers:')
  print('RMSE:', np.sqrt(mean_squared_error(Y.reshape(n), gcrf_predictions.reshape(n))))
  print('R2 score:', r2_score(Y.reshape(n), gcrf_predictions.reshape(n)))
  print("\n")
  print('GCRF 1st solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 0], gcrf_predictions[:, 0])))
  print('R2 score:', r2_score(Y[:, 0], gcrf_predictions[:, 0]))
  print("\n")  
  print('GCRF 2nd solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 1], gcrf_predictions[:, 1])))
  print('R2 score:', r2_score(Y[:, 1], gcrf_predictions[:, 1]))
  print("\n")
  print('GCRF 3rd solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 2], gcrf_predictions[:, 2])))
  print('R2 score:', r2_score(Y[:, 2], gcrf_predictions[:, 2]))

  print("\n\n")
  print('RF 1st solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 0], rf_predictions[:, 0])))
  print('R2 score:', r2_score(Y[:, 0], rf_predictions[:, 0]))
  print("\n")
  print('RF 2nd solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 1], rf_predictions[:, 1])))
  print('R2 score:', r2_score(Y[:, 1], rf_predictions[:, 1]))
  print("\n")
  print('RF 3rd solver:')
  print('RMSE:', np.sqrt(mean_squared_error(Y[:, 2], rf_predictions[:, 2])))
  print('R2 score:', r2_score(Y[:, 2], rf_predictions[:, 2]))
