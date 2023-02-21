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
K = np.array([1000, 5000, 12000])

dt = 0.2
T = np.arange(0,10,dt)

speed = np.zeros((len(K), len(T)))

for i in range(0, len(K)):
    for j in range(0, len(T)-1):
        # P controller
        e = speed[i, j] - desire_speed
        #F = -K[i] * e + f
        F = -K[i] * e # If we do not know the fricition, there will be steady - state offset

        # 4th-RK integration
        K1 = car_speed_ode(T[j], speed[i, j], m, F, f)
        K2 = car_speed_ode(T[j] + 0.5 * dt, speed[i, j] + 0.5 * dt * K1, m, F, f)
        K3 = car_speed_ode(T[j] + 0.5 * dt, speed[i, j] + 0.5 * dt * K2, m, F, f)
        K4 = car_speed_ode(T[j] + dt, speed[i, j] + dt * K3, m, F, f)

        speed[i,j+1] = speed[i, j] + dt * (K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)


# plot
h1, = plt.plot(T, speed[0,:], 'r-', label='K=1000')
h2, = plt.plot(T, speed[1,:], 'g-', label='K=5000')
h3, = plt.plot(T, speed[2,:], 'b-', label='K=10000')
plt.legend(handles=[h1, h2, h3])
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.show()


