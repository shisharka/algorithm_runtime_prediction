from read_data import *
# import plain_ridge as rr_plain
import plain_ridge_with_interactions as rr_plain2
# import ridge_with_ffs as rr_ffs
# import ridge_with_pca as rr_pca
# import ridge_with_rfe as rr_rfe
# import ridge_with_rfe_cv as rr_rfe_cv
# import random_forest as rf
# import random_forest_tuning_metaparams as rf2

SOLVER_SET_NAMES = {
   'minisat': ['INDU-HAND-RAND',
               'SAT_Competition_RACE_HAND',
               'SAT_Competition_RACE_RAND',
               'SAT_Competition_RACE_INDU',
               'IBM-SWV',
               'IBM-ALL',
               'SWV'],

  'cryptominisat': ['SAT_Competition_RACE_INDU',
                    'IBM-SWV',
                    'IBM-ALL',
                    'SWV'],

  'spear': ['SAT_Competition_RACE_INDU',
            'IBM-SWV',
            'IBM-ALL',
            'SWV'],

  'tnm': ['SAT_Competition_RACE_RAND_SAT'],

  'saps': ['SAT_Competition_RACE_RAND_SAT']
}

for solver_name, set_names in SOLVER_SET_NAMES.items():
  for set_name in set_names:
    X, Y = read_data(set_name, solver_name)

    # print('Ridge regression model...')
    # rmse = rr_plain.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse)

    print('Ridge regression model with interactions...')
    rmse, r2 = rr_plain2.validate(X, Y)
    print(solver_name, set_name, 'RMSE:', rmse, 'R2:', r2)

    # print('Ridge regression model with forward feature selection...')
    # rmse, r2 = rr_ffs.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse, 'R2:', r2)

    # print('Ridge regression model with PCA feature selection...')
    # rmse = rr_pca.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse)

    # print('Ridge regression model with recursive feature elimination...')
    # rmse, r2 = rr_rfe.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse, 'R2:', r2)

    # print('Ridge regression model with recursive feature elimination (number of features determined with cv)...')
    # rmse = rr_rfe_cv.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse)

    # print('Random forest model...')
    # rmse = rf.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse)

    # print('Random forest model with metaparam tuning...')
    # rmse = rf2.validate(X, Y)
    # print(solver_name, set_name, 'RMSE:', rmse)

    print("\n")

  print("\n\n")
