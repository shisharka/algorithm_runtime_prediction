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
  # Output:
  # IBM-ALL test RMSE: 426865.232115
  # IBM-SWV-INDU test RMSE: 1075.28825007
  # IBM-SWV test RMSE: 641.238659529
  # INDU-HAND-RAND test RMSE: 1379.59287058
  # QCP-ALL test RMSE: 67.3003649458
  # QCP-SAT test RMSE: 287.371432033
  # SAT_Competition_RACE_HAND test RMSE: 1252.8796923
  # SAT_Competition_RACE_INDU test RMSE: 1251.72677966
  # SAT_Competition_RACE_RAND test RMSE: 882.406349045
  # SWGCP-ALL test RMSE: 255.217111066
  # SWV test RMSE: 0.360709881881

  nested_cv(X, Y, set_name)
  # Output:
  # IBM-ALL test RMSE: 74189.0667692
  # IBM-SWV-INDU test RMSE: 1445.34487284
  # IBM-SWV test RMSE: 470.289016472
  # INDU-HAND-RAND test RMSE: 1378.66325249
  # QCP-ALL test RMSE: 62.3454788267
  # QCP-SAT test RMSE: 91.850178902
  # SAT_Competition_RACE_HAND test RMSE: 1560.06349696
  # SAT_Competition_RACE_INDU test RMSE: 1884.27447752
  # SAT_Competition_RACE_RAND test RMSE: 1184.59263688
  # SWGCP-ALL test RMSE: 130.80471732
  # SWV test RMSE: 0.215628428283