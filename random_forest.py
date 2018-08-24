from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import *

# cross-validation
def validate(X, Y):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  # n_estimators - the number of trees in the RandomForestRegressor
  # max_features - the number of features to consider when looking for the best split (percentage)
  # min_samples_split - minimum number of samples required to split an internal node
  rf = RandomForestRegressor(n_estimators=2000, max_features=0.5, min_samples_split=5, random_state=RAND)

  predictions = numpy.zeros(Y.shape)

  kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RAND)
  # fold_rmses = numpy.array([])

  i = 0
  for train_index, test_index in kf.split(X, Y):
    i = i + 1
    print('K-fold', i)

    X_train = X[train_index]
    X_test = X[test_index]
    Y_train = Y[train_index]
    Y_test = Y[test_index]

    ### setting missing features to mean ###
    mean_vec = mean(X_train)
    X_train = handle_broken_features(X_train, mean_vec)
    X_test = handle_broken_features(X_test, mean_vec)

    rf.fit(X_train, Y_train)
    Y_predicted = rf.predict(X_test)
    predictions[test_index] = Y_predicted

    # mse = metrics.mean_squared_error(Y_test, Y_predicted)
    # fold_rmses = numpy.append(fold_rmses, numpy.sqrt(mse))

  # return fold_rmses.mean()
  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), numpy.sqrt(metrics.r2_score(Y, predictions))
