import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import log10_transform

datasets = [
  'INDU-HAND-RAND-minisat',
  'SAT_Competition_RACE_HAND-minisat',
  'SAT_Competition_RACE_RAND-minisat',
  'SAT_Competition_RACE_INDU-minisat',
  'IBM-SWV-minisat',
  'IBM-ALL-minisat',
  'SWV-minisat',
  'SAT_Competition_RACE_INDU-cryptominisat',
  'IBM-SWV-cryptominisat',
  'IBM-ALL-cryptominisat',
  'SWV-cryptominisat',
  'SAT_Competition_RACE_INDU-spear',
  'IBM-SWV-spear',
  'IBM-ALL-spear',
  'SWV-spear',
  'SAT_Competition_RACE_RAND_SAT-tnm',
  'SAT_Competition_RACE_RAND_SAT-saps'
]

# possible model values:
# plain_ridge, plain_ridge_with_interactions,
# ridge_with_ffs_0, ridge_with_ffs_1, ridge_with_ffs_2,
# ridge_with_pca_0, ridge_with_pca_1, ridge_with_pca_2,
# ridge_with_rfe_0, ridge_with_rfe_1, ridge_with_rfe_2, ridge_with_rfe_3, ridge_with_rfe_4,
# ridge_with_rfe_cv_0, ridge_with_rfe_cv_1
model = 'plain_ridge'
if len(sys.argv) > 1:
  model = sys.argv[1]

print('Evaluating', model, 'model:')
print("\n\n")

for dataset in datasets:
  print('#######################', dataset, '#######################')

  solver_times = pd.read_csv('data/' + dataset + '.csv', names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME'].get_values()
  Y = log10_transform(solver_times)

  predictions = np.load('ridge_predictions/' + dataset + '_' + model + '.npy')

  print('RMSE:', round(np.sqrt(mean_squared_error(Y, predictions)), 6), '|',
        'R2:', round(r2_score(Y, predictions), 6))
  print("\n")
