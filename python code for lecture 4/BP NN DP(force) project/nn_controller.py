import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DP_force import DP_force_PID


DP_range_outer = 20 #m
DP_range_inner = 10 #m

# load NN models
mlp1 = joblib.load('./saved model/mlp1.pkl')
mlp2 = joblib.load('./saved model/mlp2.pkl')
mlp3 = joblib.load('./saved model/mlp3.pkl')

r = np.random.random(1)*(DP_range_outer-DP_range_outer) + DP_range_inner
rho = np.random.random(1)*(2*np.pi)
psi = (np.random.random(1)-0.5)*(2*np.pi)

init_x = r[0]*np.cos(rho)
init_y = r[0] * np.sin(rho)
init_psi = psi[0] #in rad

x0 = [init_x, init_y, init_psi, 0, 0, 0]

print('x0: ', x0)

dp = DP_force_PID(X0=x0,
                  x_d=np.array([0,0,0]),
                  h=1,
                  t_f=2000,
                  Kp=np.diag([0, 0, 0]),
                  Ki=np.diag([0, 0, 0]),
                  Kd=np.diag([0, 0, 0]),
                  e_thresh=np.array([0.6,0.6,np.deg2rad(5)])) # we set a wider error threshold for better tolerance of solutions

success, NN_data = dp.run_NN_control(mlp1, mlp2, mlp3)
print('success: ', success)
print('NN_data.shape ',NN_data.shape)
print('Final state: ')
print(' x = ', NN_data[-1,0])
print(' y = ', NN_data[-1,1])
print(' psi = ', NN_data[-1,2])
print(' u = ', NN_data[-1,3])
print(' v = ', NN_data[-1,4])
print(' r = ', NN_data[-1,5])

#plot
ship_length = 31#81
ship_width = 9.9#23
ship_shape = np.array([[ship_length / 2, 0],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), ship_width / 2],
                       [-ship_length / 2, ship_width / 2],
                       [-ship_length / 2, -ship_width / 2],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), -ship_width / 2],
                       [ship_length / 2, 0]]).T
fig = plt.figure(1)
NN_data = NN_data.T
plt.plot(NN_data[0,:],NN_data[1,:],'b-',label='Trajectory')


#plot the desired pos and heading
ship_final_pos = np.array([0,0]).reshape(-1, 1)
ship_final_heading = 0
ship_R = np.array([[np.cos(ship_final_heading), - np.sin(ship_final_heading)],
                   [np.sin(ship_final_heading), np.cos(ship_final_heading)]
                   ])
ship_orientation = np.tile(ship_final_pos, (1, 6)) + np.dot(ship_R, ship_shape)
plt.plot(ship_orientation[0], ship_orientation[1], 'r--', label='Desired state')

row, col = NN_data.shape
for i in np.arange(0, col, 40):
    ship_cur_pos = NN_data[0:2, i].reshape(-1, 1)
    ship_heading = NN_data[2, i]
    ship_R = np.array([[np.cos(ship_heading), - np.sin(ship_heading)],
                       [np.sin(ship_heading), np.cos(ship_heading)]
                       ])
    ship_orientation = np.tile(ship_cur_pos, (1, 6)) + np.dot(ship_R, ship_shape)
    line_opacity = i/col
    plt.plot(ship_orientation[0], ship_orientation[1], 'g-', alpha=line_opacity)

plt.plot(ship_orientation[0], ship_orientation[1], 'g-', label='Actual state')

ax = plt.gca()
ax.set_aspect('equal')

plt.legend()
plt.title('Trajectory')
plt.xlabel('x')
plt.ylabel('y')

fig2 = plt.figure(figsize=(14,4))
ax1 = fig2.add_subplot(131)
ax2 = fig2.add_subplot(132)
ax3 = fig2.add_subplot(133)

h = dp.get_time_step()
t = np.arange(col)*h

ax1.plot(t,NN_data[3,:],'b-')
ax1.set_title('Surge speed')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)')

ax2.plot(t,NN_data[4,:],'b-')
ax2.set_title('Sway speed')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Speed (m/s)')

ax3.plot(t,np.rad2deg(NN_data[5,:]),'b-')
ax3.set_title('Heading chage rate')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Yaw rate (deg/s)')


fig3 = plt.figure(figsize=(14,4))
ax1 = fig3.add_subplot(131)
ax2 = fig3.add_subplot(132)
ax3 = fig3.add_subplot(133)

ax1.plot(t[1:],NN_data[6,1:],'b-')
ax1.set_title('Surge force')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Fx (N)')

ax2.plot(t[1:],NN_data[7,1:],'b-')
ax2.set_title('Sway force')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Fy (N)')

ax3.plot(t[1:],np.rad2deg(NN_data[8,1:]),'b-')
ax3.set_title('Yaw torque')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Fz (Nm)')


plt.show()


