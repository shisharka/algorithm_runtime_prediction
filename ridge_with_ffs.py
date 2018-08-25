from forward_feature_selection import *


def determine_metaparams(X_train, Y_train):
  alphas = 10**numpy.linspace(-6, 2, 9)
  nums = numpy.array([1, 2, 4, 8, 16, 32, 64])
  num_scores = numpy.array([])
  num_alphas = numpy.array([])
  
  for n in nums:
    alpha_scores = numpy.array([])
    set_of_feature_indices, best_index = get_important_features(X_train, Y_train, n)
    selected_feature_indices = numpy.array(set_of_feature_indices[best_index]).astype(int)
    # print('Selected', selected_feature_indices.shape[0], 'features in grid search')
    X_train_new = X_train[:, selected_feature_indices]
    for a in alphas:
      ridge = linear_model.Ridge(alpha=a)
      scores = model_selection.cross_val_score(ridge, X_train_new, Y_train, cv=5, scoring='neg_mean_squared_error')
      alpha_scores = numpy.append(alpha_scores, -scores.mean())
    min_index = alpha_scores.argmin()
    alpha = alphas[min_index]
    num_scores = numpy.append(num_scores, alpha_scores.min())
    num_alphas = numpy.append(num_alphas, alpha)

  # ploting error w.r.t number of pca components
  # plt.plot(nums, num_scores)
  # plt.show()

  min_index = num_scores.argmin()
  num = nums[min_index]
  alpha = num_alphas[min_index]
  print('Max number of quadratic features:', num, 'reg param:', alpha)
  return num, alpha


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
    set_of_feature_indices, best_index = get_important_features(X_train_new, Y_train, MAX_L_FEATURES)

    selected_feature_indices = numpy.array(set_of_feature_indices[best_index]).astype(int)
    print('Selected', selected_feature_indices.shape[0], 'features')
    X_train = X_train[:, selected_feature_indices]
    X_test = X_test[:, selected_feature_indices]

    ### calculating interactions ###
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
    n = MAX_Q_FEATURES
    alpha = DEFAULT_REG_PARAM
    if attempt == 1:
      alpha = determine_regularization_metaparam(X_train, Y_train)
    elif attempt == 2:
      n, alpha = determine_metaparams(X_train, Y_train)
    set_of_feature_indices, best_index = get_important_features(X_train, Y_train, n)

    selected_feature_indices = numpy.array(set_of_feature_indices[best_index]).astype(int)
    print('Selected', selected_feature_indices.shape[0], 'features in second phase')
    X_train = X_train[:, selected_feature_indices]
    X_test = X_test[:, selected_feature_indices]

    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    predictions[test_index] = Y_predicted

  numpy.save('ridge_predictions/' + dataset + '_ridge_with_ffs_' + str(attempt) + '.npy', predictions)

  return numpy.sqrt(metrics.mean_squared_error(Y, predictions)), metrics.r2_score(Y, predictions)
