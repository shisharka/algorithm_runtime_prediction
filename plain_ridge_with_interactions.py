from data_preprocessing import *
from regularization_metaparam import *

def validate(X, Y, dataset):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  ### calculating interactions ###
  X = calculate_interactions(X)

  kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RAND)

  predictions = numpy.zeros(Y.shape)

  i = 0
  for train_index, test_index in kf.split(X, Y):
    i = i + 1
    print('K-fold', i)

    X_train = X[train_index]
    X_test = X[test_index]
    Y_train = Y[train_index]
    Y_test = Y[test_index]

    ### removing const columns ###
    X_train, remove_indices = remove_const_cols(X_train)
    X_test = numpy.delete(X_test, remove_indices, axis=1)

    ### standardizing ###
    std_vec = std(X_train)
    mean_vec = mean(X_train)
    X_train = standardize(X_train, mean_vec, std_vec)
    X_test = standardize(X_test, mean_vec, std_vec)

    alpha = determine_regularization_metaparam(X_train, Y_train)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    predictions[test_index] = Y_predicted 

  numpy.save('ridge_predictions/' + dataset + '_plain_ridge_with_interactions.npy', predictions)

  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), metrics.r2_score(Y, predictions)
