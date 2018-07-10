import numpy
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt


DEFAULT_REG_PARAM = 1e-2 # default regularization param used for forward feature selection


def determine_regularization_metaparam(X_train, Y_train):
  alphas = 10**numpy.linspace(-6, 2, 9)
  mean_square_errors = numpy.array([])
  
  for a in alphas:
    ridge = linear_model.Ridge(alpha=a)
    scores = model_selection.cross_val_score(ridge, X_train, Y_train, cv=10, scoring='neg_mean_squared_error')
    mean_square_errors = numpy.append(mean_square_errors, -scores.mean())

  # ploting mse w.r.t regularization parameter alpha
  # plt.plot(alphas, mean_square_errors)
  # plt.xscale('log')
  # plt.show()

  min_index = mean_square_errors.argmin()
  alpha = alphas[min_index]
  print('Regularization parameter:', alpha)
  return alpha
