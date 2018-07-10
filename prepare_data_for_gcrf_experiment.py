import pandas as pd

# # merging data for minisat, cryptominisat and spear
# X_indu = pd.read_csv('data/SAT_Competition_RACE_INDU-feat.csv')
# X_ibm_swv = pd.read_csv('data/IBM-SWV-feat.csv')
# X = pd.concat((X_indu, X_ibm_swv), axis=0, ignore_index=True)
# X.to_csv('data/gcrf/INDU-IBM-SWV-feat.csv', index=None)

# Y_minisat_indu = pd.read_csv('data/SAT_Competition_RACE_INDU-minisat.csv',
#                              names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_minisat_ibm_swv = pd.read_csv('data/IBM-SWV-minisat.csv',
#                                 names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_minisat = pd.concat((Y_minisat_indu, Y_minisat_ibm_swv), axis=0, ignore_index=True)
# Y_minisat.to_csv('data/gcrf/INDU-IBM-SWV-minisat.csv', header=None, index=None)

# Y_cryptominisat_indu = pd.read_csv('data/SAT_Competition_RACE_INDU-cryptominisat.csv',
#                                    names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_cryptominisat_ibm_swv = pd.read_csv('data/IBM-SWV-cryptominisat.csv',
#                                       names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_cryptominisat = pd.concat((Y_cryptominisat_indu, Y_cryptominisat_ibm_swv), axis=0, ignore_index=True)
# Y_cryptominisat.to_csv('data/gcrf/INDU-IBM-SWV-cryptominisat.csv', header=None, index=None)

# Y_spear_indu = pd.read_csv('data/SAT_Competition_RACE_INDU-spear.csv',
#                            names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_spear_ibm_swv = pd.read_csv('data/IBM-SWV-spear.csv',
#                               names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_spear = pd.concat((Y_spear_indu, Y_spear_ibm_swv), axis=0, ignore_index=True)
# Y_spear.to_csv('data/gcrf/INDU-IBM-SWV-spear.csv', header=None, index=None)

# # merging data for minisat, tnm and saps
# X = pd.read_csv('data/SAT_Competition_RACE_RAND_SAT-feat.csv')
# X.to_csv('data/gcrf/RAND_SAT-feat.csv', index=None)

# rand_sat_instance_ids = X['INSTANCE_ID']
# Y_minisat_randsat = pd.read_csv('data/SAT_Competition_RACE_RAND-minisat.csv',
#                                 names=['INSTANCE_ID', 'SOLVER_TIME']) \
#                       .set_index('INSTANCE_ID') \
#                       .filter(items=rand_sat_instance_ids, axis=0) \
#                       .reset_index()
# Y_minisat_randsat.to_csv('data/gcrf/RAND_SAT-minisat.csv', header=None, index=None)

# Y_tnm_randsat = pd.read_csv('data/SAT_Competition_RACE_RAND_SAT-tnm.csv',
#                             names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_tnm_randsat.to_csv('data/gcrf/RAND_SAT-tnm.csv', header=None, index=None)

# Y_saps_randsat = pd.read_csv('data/SAT_Competition_RACE_RAND_SAT-saps.csv',
#                              names=['INSTANCE_ID', 'SOLVER_TIME'])
# Y_saps_randsat.to_csv('data/gcrf/RAND_SAT-saps.csv', header=None, index=None)

# creating table with results for INDU-IBM-SWV
Y_minisat = pd.read_csv('data/gcrf/INDU-IBM-SWV-minisat.csv',
                         names=['INSTANCE_ID', 'SOLVER_TIME'])
Y_cryptominisat = pd.read_csv('data/gcrf/INDU-IBM-SWV-cryptominisat.csv',
                              names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME']
Y_spear = pd.read_csv('data/gcrf/INDU-IBM-SWV-spear.csv',
                      names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME']

Y = pd.concat((Y_minisat, Y_cryptominisat, Y_spear), axis=1, ignore_index=True)
Y.to_csv('data/gcrf/INDU-IBM-SWV-results.csv', header=['INSTANCE_ID', 'MINISAT', 'CRYPTOMINISAT', 'SPEAR'], index=None)

# creating table with results for RAND_SAT
Y_minisat = pd.read_csv('data/gcrf/RAND_SAT-minisat.csv',
                         names=['INSTANCE_ID', 'SOLVER_TIME'])
Y_tnm = pd.read_csv('data/gcrf/RAND_SAT-tnm.csv',
                    names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME']
Y_saps = pd.read_csv('data/gcrf/RAND_SAT-saps.csv',
                     names=['INSTANCE_ID', 'SOLVER_TIME'])['SOLVER_TIME']

Y = pd.concat((Y_minisat, Y_tnm, Y_saps), axis=1, ignore_index=True)
Y.to_csv('data/gcrf/RAND_SAT-results.csv', header=['INSTANCE_ID', 'MINISAT', 'TNM', 'SAPS'], index=None)