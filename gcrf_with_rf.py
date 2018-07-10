import code
import numpy
from numpy import linalg
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from GCRF import GCRF
from data_preprocessing import mean, handle_broken_features, log10_transform
from sklearn.metrics import mean_squared_error, r2_score

RAND = 1234

# X = pd.read_csv('data/gcrf/RAND_SAT-feat.csv').drop('INSTANCE_ID', axis=1).get_values()
# Y = pd.read_csv('data/gcrf/RAND_SAT-results.csv').drop('INSTANCE_ID', axis=1).get_values()
X = pd.read_csv('data/gcrf/INDU-IBM-SWV-feat.csv').drop('INSTANCE_ID', axis=1).get_values()
Y = pd.read_csv('data/gcrf/INDU-IBM-SWV-results.csv').drop('INSTANCE_ID', axis=1).get_values()
Y = log10_transform(Y)

num_solvers = Y.shape[1]

# structured model
gcrf = GCRF()
# unstructured models
rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)

gcrf_predictions = numpy.zeros(Y.shape)
rf_predictions = numpy.zeros(Y.shape)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=RAND)

i = 1
for (train_index, test_index) in kf.split(X, Y):
  print('Outter k-fold', i)
  i += 1

  X_train = X[train_index]
  Y_train = Y[train_index]
  X_test = X[test_index]
  Y_test = Y[test_index]

  num_train_instances = X_train.shape[0]
  num_test_instances = X_test.shape[0]

  R_train = numpy.zeros((num_train_instances, num_solvers))
  R_test = numpy.zeros((num_test_instances, num_solvers))

  Se_train = numpy.zeros((num_train_instances, 1, num_solvers, num_solvers))
  Se_test = numpy.zeros((num_test_instances, 1, num_solvers, num_solvers))

  inner_kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=RAND)

  j = 1
  for (train, test) in kf.split(X_train, Y_train):
    print('Inner k-fold', j)
    j += 1

    X_train_train = X_train[train]
    Y_train_train = Y_train[train]
    X_train_test = X_train[test]
    Y_train_test = Y_train[test]

    # Setting R_train values
    # ### setting broken features to mean ###
    # mean_vec = mean(X_train_train)
    # X_train_train = handle_broken_features(X_train_train, mean_vec)
    # X_train_test = handle_broken_features(X_train_test, mean_vec)

    for s in range(0, num_solvers):
      rf.fit(X_train_train, Y_train_train[:, s])
      R_train[test, s] = rf.predict(X_train_test)


    # Setting Se_train values
    S = numpy.zeros((num_solvers, num_solvers))
    U, sigma, V = linalg.svd(Y_train_train.T)
    min = numpy.inf
    max = 0
    for k in range(0, num_solvers):
      for l in range(0, num_solvers):
        if k == l: continue
        S[k, l] = numpy.exp(- linalg.norm(U[k, :] - Y_train_train[l, :]) ** 2)
        S[l, k] = S[k, l]
        if S[k, l] > max: max = S[k, l]
        if S[k, l] < min: min = S[k, l]

    # scaling to [0, 1]
    if max - min > 0: S = (S - min) / (max - min)

    for index in test: Se_train[index, 0, :, :] = S

  gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn='TNC')
  # code.interact(local=dict(globals(), **locals()))
  print('Trained params')
  print('----------------')
  print('ALFA:', gcrf.alfa[0])
  print('BETA:', gcrf.beta[0])
  print('----------------')

  # Setting R_test values
  # ### setting broken features to mean ###
  # mean_vec = mean(X_train)
  # X_train = handle_broken_features(X_train, mean_vec)
  # X_test = handle_broken_features(X_test, mean_vec)

  for s in range(0, num_solvers):
    rf.fit(X_train, Y_train[:, s])
    R_test[:, s] = rf.predict(X_test)


  # Setting Se_test values
  S = numpy.zeros((num_solvers, num_solvers))
  U, sigma, V = linalg.svd(Y_train.T)
  min = numpy.inf
  max = 0
  for k in range(0, num_solvers):
    for l in range(0, num_solvers):
      if k == l: continue
      S[k, l] = numpy.exp(- linalg.norm(U[k, :] - U[l, :]) ** 2)
      S[l, k] = S[k, l]
      if S[k, l] > max: max = S[k, l]
      if S[k, l] < min: min = S[k, l]

  # scaling to [0, 1]
  if max - min > 0: S = (S - min) / (max - min)

  for index in range(0, num_test_instances): Se_test[index, 0, :, :] = S

  gcrf_predictions[test_index, :] = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
  rf_predictions[test_index, :] = R_test

n = Y.shape[0] * num_solvers
print('GCRF for all solvers:')
print('RMSE:', numpy.sqrt(mean_squared_error(Y.reshape(n), gcrf_predictions.reshape(n))))
print('R2 score:', r2_score(Y.reshape(n), gcrf_predictions.reshape(n)))

print('1st solver:')
print('RMSE:', numpy.sqrt(mean_squared_error(Y[:, 0], rf_predictions[:, 0])))
print('R2 score:', r2_score(Y[:, 0], rf_predictions[:, 0]))

print('2nd solver:')
print('RMSE:', numpy.sqrt(mean_squared_error(Y[:, 1], rf_predictions[:, 1])))
print('R2 score:', r2_score(Y[:, 1], rf_predictions[:, 1]))

print('3rd solver:')
print('RMSE:', numpy.sqrt(mean_squared_error(Y[:, 2], rf_predictions[:, 2])))
print('R2 score:', r2_score(Y[:, 2], rf_predictions[:, 2]))
