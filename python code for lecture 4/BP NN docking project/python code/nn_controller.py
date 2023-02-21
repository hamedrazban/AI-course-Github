import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docking import docking_PID



# load NN models
mlp1 = joblib.load('../saved_model/mlp1.pkl')
mlp2 = joblib.load('../saved_model/mlp2.pkl')

r = 1 #m
rho = np.random.random()*(2*np.pi)
psi = (np.random.random()-0.5)*(2*np.pi)

x_d = np.array([0, 0])

init_x = 500 + r * np.cos(rho)
init_y = 400 + r * np.sin(rho)
init_psi = psi #in rad
R = np.array([[np.cos(init_psi), -np.sin(init_psi)],
              [np.sin(init_psi), np.cos(init_psi)]])
e_pos = np.linalg.inv(R).dot(np.array([[init_x], [init_y]]))
print(e_pos.shape)
e_psi = init_psi - np.arctan2(x_d[1] - init_y, x_d[0] - init_x)  # in rad

x0 = [init_x, init_y, init_psi, 0, 0, 0, e_pos[0,0], e_pos[1,0], e_psi, 0, 0, 0, 0]

print('x0: ', x0)

e_thresh = 20 #distance error threshold
docking = docking_PID(X0=np.array(x0),
                 x_d=x_d,
                 h=1,
                 t_f=2000,
                 pid_rpm=[0, 0, 0],
                 pid_rudder=[0, 0, 0],
                 e_thresh=e_thresh)

success, NN_data = docking.run_NN_control(mlp1, mlp2)

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
ship_length = 81
ship_width = 23
ship_shape = np.array([[ship_length / 2, 0],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), ship_width / 2],
                       [-ship_length / 2, ship_width / 2],
                       [-ship_length / 2, -ship_width / 2],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), -ship_width / 2],
                       [ship_length / 2, 0]]).T
fig = plt.figure(1)
NN_data = NN_data.T
plt.plot(NN_data[0,:],NN_data[1,:],'b-',label='Trajectory')


#plot the acceptance area
half_fill_num = 20
theta = np.linspace(0,2*np.pi,2*half_fill_num)
x = x_d[0] + e_thresh * np.cos(theta)
y = x_d[1] + e_thresh * np.sin(theta)
#plt.plot(x, y, 'r--', label='Acceptance area')
plt.fill_between(x[:half_fill_num],y[:half_fill_num], y[half_fill_num:],facecolor='red', alpha=0.9)

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

h = docking.get_time_step()
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
ax1 = fig3.add_subplot(121)
ax2 = fig3.add_subplot(122)

ax1.plot(t[1:],NN_data[10,1:],'b-')
ax1.set_title('Actual RPM')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('RPM')

ax2.plot(t[1:],NN_data[12,1:],'b-')
ax2.set_title('Actual rudder angle')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (deg)')

plt.show()


