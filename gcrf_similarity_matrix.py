import numpy as np

# Function for calculating similarity matrix for Y (columns in Y are solver representations);
# delta is similarity metaparameter;
# similarity is calculated using the following formula: exp(-delta * norm(si - sj) ** 2) (scaled to [0, 1]),
# where si, sj are vector representations for solvers i, j;
# if svd is True, representations of solvers are calculated using svd on matrix Y,
# while dimensionality is reduced by reduce_dim_by
def similarity_matrix(Y, delta, svd=False, reduce_dim_by=1):
  num_solvers = Y.shape[1]
  S = np.zeros((num_solvers, num_solvers))
  if svd == True:
    U, _sigma, _V = np.linalg.svd(Y.T)
    U = U[:, :-reduce_dim_by]
  min = max = 0
  for k in range(0, num_solvers):
    for l in range(0, num_solvers):
      if k == l: continue
      if svd == True:
        S[k, l] = np.exp(- delta * np.linalg.norm(U[k, :] - U[l, :]) ** 2)
      else:
        S[k, l] = np.exp(- delta * np.linalg.norm(Y[:, k] - Y[:, l]) ** 2)
      S[l, k] = S[k, l]
      if S[k, l] > max: max = S[k, l]
  # scaling to [0, 1]
  if max > 0: S = (S - min) / (max - min)
  return S
