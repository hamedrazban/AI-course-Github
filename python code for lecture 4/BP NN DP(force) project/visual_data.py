import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./DP_force_data.csv')

print(len(data))


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)

idx_start = 0
idx_end = 0


for i in range(0, len(data)):
    #since u=0 and v=0 for inital condition, we can used it to find a new start of test
    if np.linalg.norm(data.iloc[i, 4:6]) == 0 \
            or i==len(data)-1:
        print(i, 'th pos=',data.iloc[i, :6])
        idx_end = i
        ax1.plot(data.iloc[idx_start:idx_end, 1], data.iloc[idx_start:idx_end, 2]) #trajectory
        ax2.plot(np.arange(idx_end-idx_start), np.rad2deg(data.iloc[idx_start:idx_end, 3])) #heading
        ax3.plot(np.arange(idx_end-idx_start), data.iloc[idx_start:idx_end, 7],label='Fx') #Fx
        ax4.plot(np.arange(idx_end - idx_start), data.iloc[idx_start:idx_end, 8],label='Fy')  # Fy
        ax5.plot(np.arange(idx_end-idx_start),data.iloc[idx_start:idx_end, 9]) # yaw_torque
        idx_start = idx_end

ax1.set_title('Trajectory')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

ax2.set_title('Heading')
ax2.set_xlabel('time (s)')
ax2.set_ylabel(r"$\psi$ (deg)")

ax3.set_title('Fx')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Forces (N)')

ax4.set_title('Fy')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('Forces (N)')

ax5.set_title('Yaw torque')
ax5.set_xlabel('time (s)')
ax5.set_ylabel('Torque (Nm)')
# plt.savefig('C:/data_visual', bbox_inches='tight')
plt.show()

