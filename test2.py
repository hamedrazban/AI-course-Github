import random
import numpy as np
import matplotlib.pyplot as plt
for i in range(0,100):
    a=np.array([i,i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9])
    b=np.array([i,i,i,i,i,i,i,i,i,i])
    plt.plot(b,a)
plt.show()