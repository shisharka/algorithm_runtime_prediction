from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from data_preprocessing import *
from regularization_metaparam import *

NUM_L_FEATURES = 30
NUM_Q_FEATURES = 60


# cross-validation
def validate(X, Y):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=RAND)
  fold_rmses = numpy.array([])

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
    # alpha = DEFAULT_REG_PARAM
    print('Reg metaparam for first feature selection:')
    alpha = determine_regularization_metaparam(X_train_new, Y_train)
    print('---------------------------------------')
    selection = RFECV(linear_model.Ridge(alpha=alpha), step=1, cv=5).fit(X_train_new, Y_train)
    # selection = RFE(linear_model.Ridge(alpha=alpha), NUM_L_FEATURES, step=1).fit(X_train_new, Y_train)
    X_train = selection.transform(X_train)
    print('Selected', X_train.shape[1], 'features')
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
    # print('Reg metaparam for second feature selection:')
    # alpha = determine_regularization_metaparam(X_train, Y_train)
    # print('---------------------------------------')
    selection = RFECV(linear_model.Ridge(alpha=alpha), step=1, cv=5).fit(X_train, Y_train)
    # selection = RFE(linear_model.Ridge(alpha=alpha), NUM_Q_FEATURES, step=1).fit(X_train, Y_train)
    X_train = selection.transform(X_train)
    print('Selected', X_train.shape[1], 'features in second phase')
    X_test = selection.transform(X_test)

    alpha = determine_regularization_metaparam(X_train, Y_train)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, Y_predicted)
    fold_rmses = numpy.append(fold_rmses, numpy.sqrt(mse))

  return fold_rmses.mean()
