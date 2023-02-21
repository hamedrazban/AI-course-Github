import numpy as np
import matplotlib.pyplot as plt

M = np.array([[1.1e7, 0, 0],
                      [0, 1.1e7, 8.4e6],
                      [0, 8.4e6, 5.8e9]])  # Mass matrix

D = np.array([[3.0e5, 0, 0],
              [0, 5.5e5, 6.4e5],
              [0, 6.4e5, 1.2e8]])  # Damping matrix

step = 1
t = np.arange(0, 1000, step)

data_eta = np.zeros((3,len(t))) #ship position and heading in world frame
data_nu = np.zeros((3,len(t)))  #ship velocity in body frame

M_inv = np.linalg.inv(M)

# initialization for PID control
Kp = np.diag([2e3, 2e3, 1e6])
Ki = np.diag([0, 0, 0])
Kd = np.diag([1e3, 1e3, 5e5])

e_previous = np.array([0,0,0]).reshape(-1,1)
e_int = np.array([0,0,0]).reshape(-1,1)

x0 = np.array([0, 0, 0]).reshape(-1, 1)
xd = np.array([200, 150, np.deg2rad(45)]).reshape(-1, 1)

eta = x0
nu = np.array([0,0,0]).reshape(-1,1)

for i in np.arange(1, len(t)):
    psi = eta[-1,0]
    R = np.array([[np.cos(psi), -np.sin(psi),    0],
                  [np.sin(psi),  np.cos(psi),    0],
                  [0,            0,              1]], dtype=np.float32)

    #PID control
    e = np.dot(np.linalg.inv(R), (eta - xd))
    e_int = e_int + e
    tau = -np.dot(Kp, e) - np.dot(Ki, e_int)* step - np.dot(Kd, (e - e_previous))/step
    #print(tau)
    e_previous = e

    acc = np.dot(M_inv,(tau - np.dot(D, nu)))
    nu = nu + acc * step
    spd = np.dot(R, nu) #spd in world frame

    eta = eta + spd * step

    if eta[-1] >= 2*np.pi:
        eta[-1] -= 2*np.pi
    elif eta[-1] <= -2*np.pi:
        eta[-1] += 2 * np.pi

    data_nu[:, i] = nu.reshape(-1)
    data_eta[:, i] = eta.reshape(-1)


#plot
ship_length = 81
ship_width = 23
ship_shape = np.array([[ship_length / 2, 0],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), ship_width / 2],
                       [-ship_length / 2, ship_width / 2],
                       [-ship_length / 2, -ship_width / 2],
                       [ship_length / 2 - ship_width / 2 * np.tan(45 * np.pi / 180), -ship_width / 2],
                       [ship_length / 2, 0]]).T
plt.figure(1)
plt.plot(data_eta[0,:],data_eta[1,:],'b-')
for i in np.arange(0, len(t), 50):
    ship_cur_pos = data_eta[0:2, i].reshape(-1, 1)
    ship_heading = data_eta[2, i]
    ship_R = np.array([[np.cos(ship_heading), - np.sin(ship_heading)],
                       [np.sin(ship_heading), np.cos(ship_heading)]
                       ])
    ship_orientation = np.tile(ship_cur_pos, (1, 6)) + np.dot(ship_R, ship_shape)
    plt.plot(ship_orientation[0], ship_orientation[1], 'g-')

plt.title('Trajectory')
plt.xlabel('x')
plt.ylabel('y')

fig, axs = plt.subplots(1, 2)
axs[0].plot(t,np.rad2deg(data_eta[2,:]),'b-')
axs[0].set_title('Heading')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Heading (deg)')

axs[1].plot(t,data_nu[0,:],'b-', label='surge')
axs[1].plot(t,data_nu[1,:],'r-', label='sway')
axs[1].legend()
axs[1].set_title('Velocity')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Velocity (m/s)')

plt.show()