from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import *


def determine_metaparams(X_train, Y_train):
  min_splits = numpy.arange(2, 11) # min samples split
  split_ratios = numpy.linspace(0.1, 1, 10) # max features
  optimal_split_ratio_mses = numpy.array([])
  optimal_split_ratios = numpy.array([])

  for n in min_splits:
    mean_square_errors = numpy.array([])
    for r in split_ratios:
      rf = RandomForestRegressor(n_estimators=10, max_features=r, min_samples_split=n, random_state=RAND)
      cv_scores = model_selection.cross_val_score(rf, X_train, Y_train, cv=10, scoring='neg_mean_squared_error')
      mean_square_errors = numpy.append(mean_square_errors, -cv_scores.mean())

    min_index = mean_square_errors.argmin()
    optimal_split_ratio_mses = mean_square_errors.min()
    optimal_split_ratios = numpy.append(optimal_split_ratios, split_ratios[min_index])
  
  min_index = optimal_split_ratio_mses.argmin()
  min_split = min_splits[min_index]
  split_ratio = optimal_split_ratios[min_index]

  print('Min samples split:', min_split, 'max features:', split_ratio)
  return min_split, split_ratio


def validate(X, Y, dataset):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RAND)
  # fold_rmses = numpy.array([])

  predictions = numpy.zeros(Y.shape)

  i = 0
  for train_index, test_index in kf.split(X, Y):
    i = i + 1
    print('K-fold', i)

    X_train = X[train_index]
    X_test = X[test_index]
    Y_train = Y[train_index]
    Y_test = Y[test_index]

    min_split, split_ratio = determine_metaparams(X_train, Y_train)

    ### setting missing features to mean ###
    mean_vec = mean(X_train)
    X_train = handle_broken_features(X_train, mean_vec)
    X_test = handle_broken_features(X_test, mean_vec)

    rf = RandomForestRegressor(n_estimators=200, max_features=split_ratio, min_samples_split=min_split, random_state=RAND)
    rf.fit(X_train, Y_train)
    Y_predicted = rf.predict(X_test)
    predictions[test_index] = Y_predicted

  numpy.save('rf_predictions/' + dataset + '_rf_with_tuning.npy', predictions)

  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), metrics.r2_score(Y, predictions)
