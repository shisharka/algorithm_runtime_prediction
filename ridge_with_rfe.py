from sklearn.feature_selection import RFE
from data_preprocessing import *
from regularization_metaparam import *

NUM_L_FEATURES     = 30 # fixed num of features for first selection
NUM_Q_FEATURES     = 75 # fixed num of features for second selection
MAX_NUM_L_FEATURES = 30 # max num of features for determine_num_feat_for_selection (first selection)
MAX_NUM_Q_FEATURES = 60 # max num of features for determine_num_feat_for_selection (second selection)


def determine_num_feat_for_selection(X_train, Y_train, max_num_features, alpha):
  nums = numpy.arange(5, max_num_features + 1, 5)
  scores = []

  kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=RAND)
  r = linear_model.Ridge(alpha=alpha)

  for n in nums:
    fold_mses = []
    for tr_index, val_index in kf.split(X_train, Y_train):
      X_tr = X_train[tr_index]
      X_val = X_train[val_index]
      Y_tr = Y_train[tr_index]
      Y_val = Y_train[val_index]

      selection = RFE(r, n, step=1).fit(X_tr, Y_tr)
      X_tr = selection.transform(X_tr)
      X_val = selection.transform(X_val)

      r.fit(X_tr, Y_tr)
      Y_pred = r.predict(X_val)
      mse = metrics.mean_squared_error(Y_val, Y_pred)
      fold_mses = numpy.append(fold_mses, mse)

    scores = numpy.append(scores, fold_mses.mean())

  min_score_index = numpy.argmin(scores)
  print('Selecting', nums[min_score_index], 'features')
  return nums[min_score_index]


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
    n = NUM_L_FEATURES
    print('Reg metaparam for first feature selection:')
    alpha = determine_regularization_metaparam(X_train_new, Y_train)
    print('---------------------------------------')
    # n = determine_num_feat_for_selection(X_train_new, Y_train, MAX_NUM_L_FEATURES, alpha)
    selection = RFE(linear_model.Ridge(alpha=alpha), n, step=1).fit(X_train_new, Y_train)
    X_train = selection.transform(X_train)
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
    n = NUM_Q_FEATURES
    # print('Reg metaparam for second feature selection:')
    # alpha = determine_regularization_metaparam(X_train, Y_train)
    # print('---------------------------------------')
    # n = determine_num_feat_for_selection(X_train, Y_train, MAX_NUM_Q_FEATURES, alpha)
    selection = RFE(linear_model.Ridge(alpha=alpha), n, step=1).fit(X_train, Y_train)
    X_train = selection.transform(X_train)
    X_test = selection.transform(X_test)

    alpha = determine_regularization_metaparam(X_train, Y_train)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, Y_predicted)
    fold_rmses = numpy.append(fold_rmses, numpy.sqrt(mse))

  return fold_rmses.mean()
