import pandas as pd

def read_data(set_name, solver_name):
  X = pd.read_csv('data/' + set_name + '-feat.csv')
  X = X.drop('INSTANCE_ID', axis=1).get_values()
  # print('dim(X) =', X.shape)

  Y = pd.read_csv('data/' + set_name + '-' + solver_name + '.csv',
                  names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME'].get_values()
  # print('dim(Y) =', Y.shape)
  # print('Y[0:10] =', Y[0:10])

  return X, Y
