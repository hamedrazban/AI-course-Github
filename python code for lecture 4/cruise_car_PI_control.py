import matplotlib.pyplot as plt
import numpy as np

#            ________
#           |        |
# f / _____ |   m    | ______\ F
#   \       | _______|       /

def car_speed_ode(t, v, m, F, f):
    # m is the mass, F is the traction force, f is the friction
    return (F - f) / m

m = 1500
f = 1000
desire_speed = 10

dt = 0.2
T = np.arange(0,20,dt)
speed = np.zeros(T.shape)
U = np.zeros(T.shape)

Kp = 5000
Ki = 800

e_int = 0

for i in range(0, len(T)-1):
    # PI controller
    e = speed[i] - desire_speed
    e_int = e_int + e
    U[i] = -Kp * e - Ki * e_int * dt

    # 4th-RK integration
    K1 = car_speed_ode(T[i], speed[i], m, U[i], f)
    K2 = car_speed_ode(T[i] + 0.5 * dt, speed[i] + 0.5 * dt * K1, m, U[i], f)
    K3 = car_speed_ode(T[i] + 0.5 * dt, speed[i] + 0.5 * dt * K2, m, U[i], f)
    K4 = car_speed_ode(T[i] + dt, speed[i] + dt * K3, m, U[i], f)

    speed[i+1] = speed[i] + dt * (K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)


# plot
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax1.plot(T, speed, 'r', label='speed (m/s)')
ax1.plot(T, 10*np.ones(T.shape), 'r--')
ax1.legend()
ax1.set_xlabel('Time (s)')
ax2 = fig.add_subplot(122)
ax2.plot(T[1:], U[:-1], 'b', label='control of traction force (N)')
ax2.legend()
ax2.set_xlabel('Time (s)')
plt.show()


