import numpy as np
import pandas as pd
from docking import docking_PID

test_num = 500
success_num = 0

r = 1 #m
rho = np.random.random(test_num)*(2*np.pi)
psi = (np.random.random(test_num)-0.5)*(2*np.pi)

df_empty = np.zeros((0, 13)) # space for the attributes ['x', 'y', 'psi', 'u', 'v', 'r','e_x', 'e_y', 'e_psi','cmd_rpm','act_rpm','cmd_rudder','act_rudder']

for i in range(test_num):
    x_d = np.array([0, 0])

    init_x = 500 + r * np.cos(rho[i])
    init_y = 400 + r * np.sin(rho[i])
    init_psi = psi[i] #in rad

    R = np.array([[np.cos(init_psi), -np.sin(init_psi)],
                  [np.sin(init_psi), np.cos(init_psi)]])
    e_pos = np.linalg.inv(R).dot(np.array([[init_x],[init_y]])-x_d.reshape(-1,1))
    e_psi = init_psi - np.arctan2(x_d[1] - init_y, x_d[0] - init_x) #in rad
    if e_psi > np.pi:
        e_psi -= 2 * np.pi
    elif e_psi <= -np.pi:
        e_psi += 2 * np.pi

    x0 = [init_x, init_y, init_psi, 0, 0, 0, e_pos[0,0], e_pos[1,0], e_psi, 0, 0, 0, 0]

    print(i, 'th sample x0: ', x0)

    dp = docking_PID(X0=np.array(x0),
                 x_d=x_d,
                 h=1,
                 t_f=2000,
                 pid_rpm=[10, 0, 5],
                 pid_rudder=[.2, 0.0, 5],
                 e_thresh=10) #small e_thresh will result in rudder change at the last moment

    success, restemp = dp.run()
    print('restemp.shape ', restemp.shape)
    if success:
        success_num += 1
        df_empty = np.concatenate((df_empty, restemp))

print('total success: ', success_num)
data = pd.DataFrame(df_empty, columns=['x', 'y', 'psi', 'u', 'v', 'r', 'e_x', 'e_y', 'e_psi', 'cmd_rpm','act_rpm','cmd_rudder','act_rudder'])
result_file = "../data file/docking_data.csv"

data.to_csv(result_file)

