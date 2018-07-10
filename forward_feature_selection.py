import numpy
from data_preprocessing import *
from regularization_metaparam import *

MAX_L_FEATURES = 30 # for first phase forward feature selection
MAX_Q_FEATURES = 20 # for second phase forward feature selection


# function for greedy forward feature selection
def forward_feature_selection(X, Y, num_iterations):
  num_samples = X.shape[0]
  num_features = X.shape[1]
  
  num_iterations = min(num_iterations, num_features)

  kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=RAND)

  X_new = numpy.array([])

  selected_feature_indices = numpy.array([])
  best_feature_index = 0
  subs = numpy.array([])

  for i in range(0, num_iterations):
    min_rmse = numpy.inf
    for j in range(0, num_features):
      X_temp = X_new
      
      # print('Trying feature', j)
      if j in selected_feature_indices:
        # print('Feature', j, 'already selected')
        continue

      if X_temp.shape[0] == 0:
        # adding first feature
        X_temp = X[:, j].reshape((num_samples, 1))
      else:
        X_temp = numpy.hstack((X_temp, X[:, j].reshape((num_samples, 1))))

      fold_rmses = numpy.array([])
      for train_index, test_index in kf.split(X_temp, Y):
        X_train = X_temp[train_index]
        X_test = X_temp[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        ridge = linear_model.Ridge(alpha=DEFAULT_REG_PARAM)
        ridge.fit(X_train, Y_train)
        Y_predicted = ridge.predict(X_test)
        mse = metrics.mean_squared_error(Y_test, Y_predicted)
        fold_rmses = numpy.append(fold_rmses, numpy.sqrt(mse))

      if fold_rmses.mean() < min_rmse:
        min_rmse = fold_rmses.mean()
        best_feature_index = j

    # adding the best feature to X_new
    if X_new.shape[0] == 0:
      X_new = X[:, best_feature_index].reshape((num_samples, 1))
    else:
      X_new = numpy.hstack((X_new, X[:, best_feature_index].reshape((num_samples, 1))))
    selected_feature_indices = numpy.append(selected_feature_indices, best_feature_index)
    subs = numpy.append(subs, { 'features': selected_feature_indices, 'rmse': min_rmse })

  return subs
  

def get_important_features(X_train, Y_train, num_iterations):
  print('Getting important features...')
  subs = forward_feature_selection(X_train, Y_train, num_iterations)
  set_of_feature_indices = []
  rmses = numpy.array([])
  for i in range(0, subs.shape[0]):
    set_of_feature_indices.append(subs[i]['features'])
    rmses = numpy.append(rmses, subs[i]['rmse'])

  return set_of_feature_indices, numpy.argmin(rmses)
