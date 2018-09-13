import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import log10_transform

solvers = [
  'ebglucose',
  'ebminisat',
  'glucose2',
  'glueminisat',
  'lingeling',
  'lrglshr',
  'minisatpsm',
  'mphaseSAT64',
  'precosat',
  'qutersat',
  'rcl',
  'restartsat',
  'cryptominisat2011',
  'spear-sw',
  'spear-hw',
  'eagleup',
  'sparrow',
  'marchrw',
  'mphaseSATm',
  'satime11',
  'tnm',
  'mxc09',
  'gnoveltyp2',
  'sattime',
  'sattimep',
  'clasp2',
  'clasp1',
  'picosat',
  'mphaseSAT',
  'sapperlot',
  'sol'
] 

datasets = [
  'SATALL12S',
  'SATHAND12S',
  'SATINDU12S',
  'SATRAND12S'
]

# possible model values: tweaking_similarity_metaparam, svd_tweaking_metaparams
model = 'tweaking_similarity_metaparam'
if len(sys.argv) > 1:
  model = sys.argv[1]

print('Evaluating', model, 'model:')
print("\n\n")

for dataset in datasets:
  print('#######################', dataset, '#######################')
  data = pd.read_csv('SATzilla2012_data/' + dataset + '.csv')
  solver_times = data.filter(regex='_Time$', axis=1).get_values()
  Y = log10_transform(solver_times)


  gcrf_predictions = np.load('gcrf_predictions/' + dataset + '_gcrf_' + model + '.npy')
  rf_predictions = np.load('gcrf_predictions/' + dataset + '_rf_' + model + '.npy')
  # gcrf_predictions = np.load('predictions/' + dataset + '_gcrf_tweaking_sim_metaparam.npy')
  # rf_predictions = np.load('predictions/' + dataset + '_rf_tweaking_sim_metaparam.npy')
  # gcrf_predictions = np.load('gcrf_svd_predictions/' + dataset + '_gcrf_svd_tweaking_dim_reduction.npy')
  # rf_predictions = np.load('gcrf_svd_predictions/' + dataset + '_rf_svd_tweaking_dim_reduction.npy')

  num_solvers = Y.shape[1]

  for i in range(0, num_solvers):
    print('Solver', solvers[i])
    print('GCRF RMSE:', round(np.sqrt(mean_squared_error(Y[:, i], gcrf_predictions[:, i])), 6), '|',
          'GCRF R2:', round(r2_score(Y[:, i], gcrf_predictions[:, i]), 6))
    print('RF RMSE:  ', round(np.sqrt(mean_squared_error(Y[:, i], rf_predictions[:, i])), 6), '|',
          'RF R2:  ', round(r2_score(Y[:, i], rf_predictions[:, i]), 6))
    print("\n")
  print('General evaluation:')
  n = Y.shape[0] * num_solvers
  print(np.sqrt(mean_squared_error(Y.reshape(n), gcrf_predictions.reshape(n))))
  print(r2_score(Y.reshape(n), gcrf_predictions.reshape(n)))
