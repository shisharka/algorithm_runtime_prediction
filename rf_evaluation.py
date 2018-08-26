import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import log10_transform

datasets = [
  'INDU-HAND-RAND-minisat',
  # 'SAT_Competition_RACE_HAND-minisat',
  # 'SAT_Competition_RACE_RAND-minisat',
  # 'SAT_Competition_RACE_INDU-minisat',
  # 'IBM-SWV-minisat',
  # 'IBM-ALL-minisat',
  # 'SWV-minisat',
  # 'SAT_Competition_RACE_INDU-cryptominisat',
  # 'IBM-SWV-cryptominisat',
  # 'IBM-ALL-cryptominisat',
  # 'SWV-cryptominisat',
  # 'SAT_Competition_RACE_INDU-spear',
  # 'IBM-SWV-spear',
  # 'IBM-ALL-spear',
  # 'SWV-spear',
  # 'SAT_Competition_RACE_RAND_SAT-tnm',
  # 'SAT_Competition_RACE_RAND_SAT-saps'
]

# possible model values: rf_0, rf_1, rf_2, rf_3, rf_4, rf_5, rf_6, rf_7, rf_with_tuning
model = 'rf_0'
if len(sys.argv) > 1:
  model = sys.argv[1]

print('Evaluating', model, 'model:')
print("\n\n")

for dataset in datasets:
  print('#######################', dataset, '#######################')

  solver_times = pd.read_csv('data/' + dataset + '.csv', names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME'].get_values()
  Y = log10_transform(solver_times)

  predictions = np.load('rf_predictions/' + dataset + '_' + model + '.npy')

  print('RMSE:', round(np.sqrt(mean_squared_error(Y, predictions)), 6), '|',
        'R2:', round(r2_score(Y, predictions), 6))
  print("\n")
