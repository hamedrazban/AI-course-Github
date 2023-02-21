from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor

import joblib
import numpy as np

S = pd.read_csv("./DP_force_data.csv")

data_list = S.iloc[0:-1, 1:7]
Fx_label_list = S.iloc[1:, 7] #Fx
Fy_label_list = S.iloc[1:, 8] #Fy
Fz_label_list = S.iloc[1:, 9] #yaw torque
test_size = 0.1

#dataset and model for Fx
X1, y1 = shuffle(data_list.values, Fx_label_list, random_state=0)
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size)
#print(x1_train.shape)
#print(x1_test.shape)

mlp1 = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(10, 10),
                               activation='tanh',
                               max_iter=1000),
        transformer=StandardScaler()
    )
)

mlp1.fit(x1_train, y1_train.ravel())
print('mlp1.socre: ', mlp1.score(x1_test, y1_test))

#print('mlp1: ', mlp1.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))

#dataset and model for Fx
X2, y2 = shuffle(data_list.values, Fy_label_list, random_state=0)
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size)
#print(x2_train.shape)
#print(x2_test.shape)

mlp2 = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(10, 10),
                               activation='tanh',
                               max_iter=1000),
        transformer=StandardScaler()
    )
)

mlp2.fit(x2_train, y2_train.ravel())
print('mlp2.score: ', mlp2.score(x2_test, y2_test))

#print('mlp2: ', mlp2.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))


#dataset and model for Yaw torque
X3, y3 = shuffle(data_list.values, Fz_label_list, random_state=0)
x3_train, x3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=test_size)
#print(x3_train.shape)
#print(x3_test.shape)

mlp3 = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(15, 12),
                               activation='tanh',
                               max_iter=1000),
        transformer=StandardScaler()
    )
)

mlp3.fit(x3_train, y3_train.ravel())
print('mlp3.score: ', mlp3.score(x3_test, y3_test))

#print('mlp3: ', mlp3.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))

# save model
joblib.dump(mlp1, './saved model/mlp1.pkl')
joblib.dump(mlp2, './saved model/mlp2.pkl')
joblib.dump(mlp3, './saved model/mlp3.pkl')