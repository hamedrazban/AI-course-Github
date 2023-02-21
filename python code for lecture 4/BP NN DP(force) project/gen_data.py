import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DP_force import DP_force_PID

test_num = 300
success_num = 0

DP_range_outer = 20 #m
DP_range_inner = 10 #m

df_empty = np.zeros((0, 9)) # space for the attributes ['x', 'y', 'psi', 'u', 'v', 'r','Fx','Fy','yaw_torque']

r = np.random.random(test_num)*(DP_range_outer-DP_range_inner) + DP_range_inner
rho = np.random.random(test_num)*(2*np.pi)
psi = (np.random.random(test_num)-0.5)*(2*np.pi)

for i in range(test_num):
    init_x = r[i]*np.cos(rho[i])
    init_y = r[i] * np.sin(rho[i])
    init_psi = psi[i] #in rad

    x0 = [init_x, init_y, init_psi, 0, 0, 0]

    print(i, 'th sample x0: ', x0)

    dp = DP_force_PID(X0=np.array(x0),
                      x_d=np.array([0,0,0]),
                      h=1,
                      t_f=2000,
                      Kp=np.diag([2e3, 2e3, 1e6]),
                      Ki=np.diag([0, 0, 0]),
                      Kd=np.diag([1e3, 1e3, 5e5]),
                      e_thresh=np.array([0.5,0.5,np.deg2rad(3)]))

    success, restemp = dp.run()
    print(i, 'th PID success: ', success)
    print('restemp.shape ',restemp.shape)
    if success:
        success_num += 1
        df_empty = np.concatenate((df_empty, restemp))

print('total success: ', success_num)
data = pd.DataFrame(df_empty, columns=['x', 'y', 'psi', 'u', 'v', 'r', 'Fx','Fy','Yaw_torque'])
result_file = "./DP_force_data.csv"

data.to_csv(result_file)


# #plot
# ship_length = 81
# ship_width = 23
# ship_shape = np.array([[ship_length / 2, 0],
#                        [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), ship_width / 2],
#                        [-ship_length / 2, ship_width / 2],
#                        [-ship_length / 2, -ship_width / 2],
#                        [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), -ship_width / 2],
#                        [ship_length / 2, 0]]).T
# plt.figure(1)
# plt.plot(data_eta[0,:],data_eta[1,:],'b-')
# for i in np.arange(0, len(t), 50):
#     ship_cur_pos = data_eta[0:2, i].reshape(-1, 1)
#     ship_heading = data_eta[2, i]
#     ship_R = np.array([[np.cos(ship_heading), - np.sin(ship_heading)],
#                        [np.sin(ship_heading), np.cos(ship_heading)]
#                        ])
#     ship_orientation = np.tile(ship_cur_pos, (1, 6)) + np.dot(ship_R, ship_shape)
#     plt.plot(ship_orientation[0], ship_orientation[1], 'g-')
#
# plt.title('Trajectory')
# plt.xlabel('x')
# plt.ylabel('y')
#
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(t,np.rad2deg(data_eta[2,:]),'b-')
# axs[0].set_title('Heading')
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('Heading (deg)')
#
# axs[1].plot(t,data_nu[0,:],'b-', label='surge')
# axs[1].plot(t,data_nu[1,:],'r-', label='sway')
# axs[1].legend()
# axs[1].set_title('Velocity')
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Velocity (m/s)')
#
# plt.show()