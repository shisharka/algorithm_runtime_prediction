from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from data_preprocessing import *
from regularization_metaparam import *

NUM_L_FEATURES = 30 # fixed number of features for the first selection

def validate(X, Y, dataset, attempt = 0):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

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

    ### standardizing for feature selection ###
    std_vec = std(X_train)
    mean_vec = mean(X_train)
    X_train_new = standardize(X_train, mean_vec, std_vec)

    ### feature selection ###
    # print('For first feature selection:')
    print('Before feature selection:')
    alpha = determine_regularization_metaparam(X_train_new, Y_train)
    print('---------------------------------------')
    selection = RFE(linear_model.Ridge(alpha=alpha), NUM_L_FEATURES, step=1).fit(X_train_new, Y_train)
    if attempt == 1:
      selection = RFECV(linear_model.Ridge(alpha=alpha), step=1, cv=5).fit(X_train_new, Y_train)
    X_train = selection.transform(X_train)
    print('Selected', X_train.shape[1], 'features in the first phase')
    X_test = selection.transform(X_test)

    ### calculate interactions ###
    X_train = calculate_interactions(X_train)
    X_test = calculate_interactions(X_test)

    ### removing const columns from feature matrix with interactions ###
    X_train, remove_indices = remove_const_cols(X_train)
    X_test = numpy.delete(X_test, remove_indices, axis=1)
    
    ### standardizing ###
    std_vec = std(X_train)
    mean_vec = mean(X_train)
    X_train = standardize(X_train, mean_vec, std_vec)
    X_test = standardize(X_test, mean_vec, std_vec)

    ### second feature selection ###
    # print('For second feature selection:')
    # alpha = determine_regularization_metaparam(X_train, Y_train)
    # print('---------------------------------------')
    selection = RFECV(linear_model.Ridge(alpha=alpha), step=1, cv=5).fit(X_train, Y_train)
    X_train = selection.transform(X_train)
    print('Selected', X_train.shape[1], 'features in the second phase')
    X_test = selection.transform(X_test)

    alpha = determine_regularization_metaparam(X_train, Y_train)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    predictions[test_index] = Y_predicted

  numpy.save('ridge_predictions/' + dataset + '_ridge_with_rfe_cv_' + str(attempt) + '.npy', predictions)

  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), metrics.r2_score(Y, predictions)
