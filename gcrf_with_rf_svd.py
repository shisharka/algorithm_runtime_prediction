import code # for debugging, to set breakpoint use code.interact(local=dict(globals(), **locals()))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
from GCRF import GCRF
from data_preprocessing import log10_transform
from gcrf_similarity_matrix import similarity_matrix

# datasets = [
# 'INDU-IBM-SWV',
# 'RAND_SAT'
# ]

datasets = [
  'SATALL12S',
  'SATHAND12S',
  'SATINDU12S',
  'SATRAND12S'
]

RAND = 1234
DELTA = 1
SVD = True
LEARN = 'L-BFGS-B'
REDUCE_DIM_BY = 1


def determine_similarity_metaparam(X, Y):
  num_solvers = Y.shape[1]
  deltas = np.arange(1, 10) * DELTA
  mean_square_errors = np.array([])
  
  gcrf = GCRF()
  rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)
  kf = KFold(n_splits=5, shuffle=True, random_state=RAND)

  for d in deltas:
    scores = np.array([])
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
      Se_train[:, 0, :, :] = similarity_matrix(Y_train, d, SVD, REDUCE_DIM_BY)
      Se_test[:, 0, :, :] = similarity_matrix(Y_test, d, SVD, REDUCE_DIM_BY)

      gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn=LEARN)
      predictions = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
      n = Y_test.shape[0] * num_solvers
      scores = \
        np.append(scores, mean_squared_error(Y_test.reshape(n), predictions.reshape(n)))

    mean_square_errors = np.append(mean_square_errors, scores.mean())

  # # ploting mse w.r.t metaparameter delta
  # plt.plot(deltas, mean_square_errors)
  # plt.xscale('log')
  # plt.show()

  min_index = mean_square_errors.argmin()
  delta = deltas[min_index]
  print('Similarity metaparameter:', delta)
  return delta


def determine_dim_reduction(X, Y):
  num_solvers = Y.shape[1]
  dim_reductions = np.arange(1, num_solvers)
  mean_square_errors = np.array([])
  
  gcrf = GCRF()
  rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)
  kf = KFold(n_splits=5, shuffle=True, random_state=RAND)

  for d in dim_reductions:
    scores = np.array([])
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
      Se_train[:, 0, :, :] = similarity_matrix(Y_train, DELTA, SVD, d)
      Se_test[:, 0, :, :] = similarity_matrix(Y_test, DELTA, SVD, d)

      gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn=LEARN)
      predictions = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
      n = Y_test.shape[0] * num_solvers
      scores = \
        np.append(scores, mean_squared_error(Y_test.reshape(n), predictions.reshape(n)))

    mean_square_errors = np.append(mean_square_errors, scores.mean())

  min_index = mean_square_errors.argmin()
  dim_reduction = dim_reductions[min_index]
  print('Dimension reduction:', dim_reduction)
  return int(dim_reduction)


def determine_metaparams(X, Y):
  num_solvers = Y.shape[1]
  deltas = deltas = np.linspace(0.1, 2, 20)
  dim_reductions = np.arange(1, num_solvers)
  optimal_dim_reduction_mses = np.array([])
  optimal_dim_reductions = np.array([])

  gcrf = GCRF()
  rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_split=5, random_state=RAND)
  kf = KFold(n_splits=5, shuffle=True, random_state=RAND)

  for d in deltas:
    mean_square_errors = np.array([])
    for r in dim_reductions:
      cv_scores = np.array([])
      for (train_index, test_index) in kf.split(X, Y):
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
        Se_train[:, 0, :, :] = similarity_matrix(Y_train, d, SVD, r)
        Se_test[:, 0, :, :] = similarity_matrix(Y_test, d, SVD, r)

        gcrf.fit(R_train.reshape(num_train_instances * num_solvers, 1), Se_train, Y_train, learn=LEARN)
        predictions = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
        n = Y_test.shape[0] * num_solvers
        cv_scores = \
          np.append(cv_scores, mean_squared_error(Y_test.reshape(n), predictions.reshape(n)))

      mean_square_errors = np.append(mean_square_errors, cv_scores.mean())

    min_index = mean_square_errors.argmin()
    optimal_dim_reduction_mses = np.append(optimal_dim_reduction_mses, mean_square_errors.min())
    optimal_dim_reductions = np.append(optimal_dim_reductions, dim_reductions[min_index])

  min_index = optimal_dim_reduction_mses.argmin()
  delta = deltas[min_index]
  dim_reduction = optimal_dim_reductions[min_index]
  print('Similarity metaparam:', delta, 'dim reduction metaparam:', dim_reduction)
  return delta, int(dim_reduction)


for dataset in datasets:
  # X = pd.read_csv('data/gcrf/' + dataset + '-feat.csv').drop('INSTANCE_ID', axis=1).get_values()
  # Y = pd.read_csv('data/gcrf/' + dataset + '-results.csv').drop('INSTANCE_ID', axis=1).get_values()
  data = pd.read_csv('SATzilla2012_data/' + dataset + '.csv')
  features = data.drop(data.iloc[:, 0:156], axis=1)
  solver_times = data.filter(regex='_Time$', axis=1)
  X = features.get_values()
  Y = solver_times.get_values()
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

    # delta = DELTA
    # reduce_dim_by = REDUCE_DIM_BY
    # delta = determine_similarity_metaparam(X_train, Y_train)
    # reduce_dim_by = determine_dim_reduction(X_train, Y_train)
    delta, reduce_dim_by = determine_metaparams(X_train, Y_train)

    j = 1
    for (inner_train_index, inner_test_index) in kf.split(X_train, Y_train):
      print('  Inner k-fold', j); j += 1

      X_train_train = X_train[inner_train_index]
      Y_train_train = Y_train[inner_train_index]
      X_train_test = X_train[inner_test_index]
      Y_train_test = Y_train[inner_test_index]

      ### Setting R_train values
      for s in range(0, num_solvers):
        rf.fit(X_train_train, Y_train_train[:, s])
        R_train[inner_test_index, s] = rf.predict(X_train_test)

      ### Setting Se_train values
      for index in inner_test_index: Se_train[index, 0, :, :] = similarity_matrix(Y_train_train, delta, SVD, reduce_dim_by)

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
    for index in range(0, num_test_instances): Se_test[index, 0, :, :] = similarity_matrix(Y_train, delta, SVD, reduce_dim_by)

    gcrf_predictions[outer_test_index, :] = gcrf.predict(R_test.reshape(num_test_instances * num_solvers, 1), Se_test)
    rf_predictions[outer_test_index, :] = R_test

  np.save('gcrf_predictions/' + dataset + '_rf_svd_tweaking_metaparams.npy', rf_predictions)
  np.save('gcrf_predictions/' + dataset + '_gcrf_svd_tweaking_metaparams.npy', gcrf_predictions)
