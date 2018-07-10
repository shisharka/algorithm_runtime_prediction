from data_preprocessing import *
from regularization_metaparam import *
from feature_selection_pca import *

MAX_PCA_COMPONENTS = 40


def determine_metaparams(X_train, Y_train):
  alphas = 10**numpy.linspace(-6, 2, 9)
  nums = numpy.arange(5, MAX_PCA_COMPONENTS + 5, 5)
  num_scores = numpy.array([])
  num_alphas = numpy.array([])
  
  for n in nums:
    alpha_scores = numpy.array([])
    pca = PCA(n_components = n)
    pca.fit(X_train)
    X_train_new = pca.transform(X_train)
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
  print('Number of pca components:', num, 'reg param:', alpha)
  return num, alpha


# cross-validation
def validate(X, Y):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  ### calculating interactions ###
  X = calculate_interactions(X)

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

    ### subtracting mean ###
    mean_vec = mean(X_train)
    X_train = center(X_train, mean_vec)
    X_test = center(X_test, mean_vec)

    ### feature selection using PCA ###
    # with required explained variance ratio
    X_train, X_test = pca_feature_selection(X_train, X_test, num_components=1-1e-6)
    # # with tuning num_components metaparam
    # num_components, alpha = determine_metaparams(X_train, Y_train)
    # X_train, X_test = pca_feature_selection(X_train, X_test, num_components=num_components)

    # ### removing const columns ###
    # X_train, remove_indices = remove_const_cols(X_train)
    # X_test = numpy.delete(X_test, remove_indices, axis=1)

    # ### standardizing ###
    # mean_vec = numpy.zeros(X_train.shape[1])
    # # mean_vec = mean(X_train)
    # std_vec = std(X_train)
    # X_train = standardize(X_train, mean_vec, std_vec)
    # X_test = standardize(X_test, mean_vec, std_vec)

    alpha = determine_regularization_metaparam(X_train, Y_train)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    Y_predicted = ridge.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, Y_predicted)
    fold_rmses = numpy.append(fold_rmses, numpy.sqrt(mse))

  return fold_rmses.mean()
