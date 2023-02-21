import numpy as np
import matplotlib.pyplot as plt

tau = np.array([1e6, 0, 1e6]).reshape(-1, 1)

M = np.array([[1.1e7, 0, 0],
                      [0, 1.1e7, 8.4e6],
                      [0, 8.4e6, 5.8e9]])  # Mass matrix

D = np.array([[3.0e5, 0, 0],
              [0, 5.5e5, 6.4e5],
              [0, 6.4e5, 1.2e8]])  # Damping matrix

step = .1
t = np.arange(0, 2000, step)

data_eta = np.zeros((3,len(t))) #ship position and heading in world frame
data_nu = np.zeros((3,len(t)))  #ship velocity in body frame

eta = data_eta[:,0].reshape(-1,1)
nu = data_nu[:,0].reshape(-1,1)

M_inv = np.linalg.inv(M)

for i in np.arange(1, len(t)):
    psi = eta[-1,0]
    R = np.array([[np.cos(psi), -np.sin(psi),    0],
                  [np.sin(psi),  np.cos(psi),    0],
                  [0,            0,              1]], dtype=np.float32)

    acc = np.dot(M_inv,(tau - np.dot(D,nu)))
    nu = nu + acc * step
    spd = np.dot(R, nu) #spd in world frame

    eta = eta + spd * step

    if eta[-1] >= 2*np.pi:
        eta[-1] -= 2*np.pi
    elif eta[-1] <= -2*np.pi:
        eta[-1] += 2 * np.pi

    data_nu[:, i] = nu.reshape(-1)
    data_eta[:, i] = eta.reshape(-1)

fig, axs = plt.subplots(1, 3)
axs[0].plot(data_eta[0,:],data_eta[1,:],'b-')
axs[0].set_title('Position')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].plot(t,np.rad2deg(data_eta[2,:]),'b-')
axs[1].set_title('Heading')
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('heading (deg)')

axs[2].plot(t,data_nu[0,:],'b-', label='surge')
axs[2].plot(t,data_nu[1,:],'r-', label='sway')
axs[2].legend()
axs[2].set_title('Velocity')
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('velocity (m/s)')

plt.show()