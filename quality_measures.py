import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import log10_transform

datasets = [
  'SATALL12S',
  # 'SATHAND12S',
  # 'SATINDU12S',
  # 'SATRAND12S'
]

for dataset in datasets:
  print('#######################', dataset, '#######################')
  data = pd.read_csv('SATzilla2012_data/' + dataset + '.csv')
  solver_times = data.filter(regex='_Time$', axis=1)
  Y = log10_transform(solver_times.get_values())

  gcrf_predictions = np.load('satzilla_predictions/' + dataset + '_gcrf_svd_tweaking_metaparams.npy')
  rf_predictions = np.load('satzilla_predictions/' + dataset + '_rf_svd_tweaking_metaparams.npy')
  # gcrf_predictions = np.load('predictions/' + dataset + '_gcrf_tweaking_sim_metaparam.npy')
  # rf_predictions = np.load('predictions/' + dataset + '_rf_tweaking_sim_metaparam.npy')
  # gcrf_predictions = np.load('gcrf_svd_predictions/' + dataset + '_gcrf_svd_tweaking_dim_reduction.npy')
  # rf_predictions = np.load('gcrf_svd_predictions/' + dataset + '_rf_svd_tweaking_dim_reduction.npy')

  num_solvers = Y.shape[1]

  for i in range(0, num_solvers):
    print('Solver', i + 1)
    print('GCRF RMSE:', round(np.sqrt(mean_squared_error(Y[:, i], gcrf_predictions[:, i])), 6), '|',
          'GCRF R2:', round(r2_score(Y[:, i], gcrf_predictions[:, i]), 6))
    print('RF RMSE:  ', round(np.sqrt(mean_squared_error(Y[:, i], rf_predictions[:, i])), 6), '|',
          'RF R2:  ', round(r2_score(Y[:, i], rf_predictions[:, i]), 6))
    print("\n")
