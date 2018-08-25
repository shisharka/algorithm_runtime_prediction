from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# feature selection based on PCA
def pca_feature_selection(X_train, X_test, num_components):
  # select the number of components such that the amount of
  # variance that needs to be explained is greater than the percentage specified by n_components;
  # in this case 0 < num_components < 1
  if num_components > 0 and num_components < 1:
    pca = PCA(n_components = num_components, svd_solver = 'full')
  else:
    pca = PCA(n_components = num_components)

  pca.fit(X_train)

  # print('Explained variance:', pca.explained_variance_)
  # plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o')
  # plt.show()
  print('Number of components:', pca.n_components_)
  return pca.transform(X_train), pca.transform(X_test)
