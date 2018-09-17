# for decision tree demo only
import graphviz 
from sklearn import tree
from data_preprocessing import log10_transform

# cross-validation
def validate(X, Y, feature_names):
  ### log10 transformation of response variable ###
  Y = log10_transform(Y)

  dt = tree.DecisionTreeRegressor()

  dt = dt.fit(X, Y)

  dot_data = tree.export_graphviz(dt, out_file=None, 
                                      feature_names=feature_names,  
                                      # class_names=iris.target_names,  
                                      filled=True, rounded=True,  
                                      special_characters=True)
  graph = graphviz.Source(dot_data)  
  graph.view()
