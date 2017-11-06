from train_ridge_regression_model import *

SET_NAMES = ['IBM-ALL',
             'IBM-SWV-INDU',
             'IBM-SWV',
             'INDU-HAND-RAND',
             'QCP-ALL',
             'QCP-SAT',
             'SAT_Competition_RACE_HAND',
             'SAT_Competition_RACE_INDU',
             'SAT_Competition_RACE_RAND',
             'SWGCP-ALL',
             'SWV']

for set_name in SET_NAMES:
  X, Y = read_data(set_name)
  # train_ridge_regression_model(X, Y, set_name)
  nested_cv(X, Y, set_name)