from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

#steps = [ ("preprocessing", StandardScaler()),
#              ("classifier", MLPRegressor()),]
#pipe = Pipeline(steps)

pipe = make_pipeline(StandardScaler(), MLPRegressor())
print(pipe)

