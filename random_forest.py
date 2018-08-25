from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import *

def validate(X, Y, dataset, attempt = 0):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  # n_estimators - the number of trees in the RandomForestRegressor
  # max_features - the number of features to consider when looking for the best split (percentage)
  # min_samples_split - minimum number of samples required to split an internal node
  n_estimators = 10
  if attempt == 1:
    n_estimators = 20
  elif attempt == 2:
    n_estimators = 30
  elif attempt == 3:
    n_estimators = 50
  elif attempt == 4:
    n_estimators = 100
  elif attempt == 5:
    n_estimators = 200
  elif attempt == 6:
    n_estimators = 1000
  elif attempt == 7:
    n_estimators = 2000
  rf = RandomForestRegressor(n_estimators=n_estimators, max_features=0.5, min_samples_split=5, random_state=RAND)

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

  numpy.save('rf_predictions/' + dataset + '_rf_' + str(attempt) + '.npy', predictions)

  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), metrics.r2_score(Y, predictions)
