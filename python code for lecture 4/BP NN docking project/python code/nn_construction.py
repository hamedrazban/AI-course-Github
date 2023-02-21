from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor

import joblib


S = pd.read_csv("../data file/docking_data.csv") #['x', 'y', 'psi', 'u', 'v', 'r', 'e_pos_x','e_pos_y','e_psi''cmd_rpm','act_rpm','cmd_rudder','act_rudder']

rpm_data_list = S.iloc[0:-1, [4,5,6,7,8,9]] # u,v,r,e_pos_x, e_pos_y, e_psi
#rpm_data_list = S.iloc[0:-1, 1:7] # x,y,psi,u,v,r
rpm_label_list = S.iloc[1:, 10] #act_rpm

rudder_data_list = S.iloc[0:-1, [4,5,6,7,8,9]] # u,v,r, e_pos_x, e_pos_y, e_psi
#rudder_data_list = S.iloc[0:-1, 1:7] # u,v,r, e_pos_x, e_pos_y, e_psi
rudder_label_list = S.iloc[1:, 12] #act_rudder

test_size = 0.1


#dataset and model for rudder
X2, y2 = shuffle(rudder_data_list.values, rudder_label_list, random_state=0)

x2_train, x2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size)
print(x2_train.shape)
print(x2_test.shape)

mlp2 = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000),
)
#mlp2 = make_pipeline(
#    StandardScaler(),
#    TransformedTargetRegressor(
#        regressor=MLPRegressor(hidden_layer_sizes=(10,10),
#                               #activation='tanh',
#                               max_iter=1000),
#        transformer=StandardScaler()
#    )
#)

mlp2.fit(x2_train, y2_train.ravel())
print('mlp2.socre: ', mlp2.score(x2_test, y2_test))

#dataset and model for rpm
X1, y1 = shuffle(rpm_data_list.values, rpm_label_list, random_state=0)

x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size)
print(x1_train.shape)
print(x1_test.shape)

mlp1 = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000),
)


mlp1.fit(x1_train, y1_train.ravel())
print('mlp1.socre: ', mlp1.score(x1_test, y1_test))


# save model
joblib.dump(mlp1, '../saved_model/mlp1.pkl')
joblib.dump(mlp2, '../saved_model/mlp2.pkl')
