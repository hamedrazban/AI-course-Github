import numpy as np
import torch
torch.rand()
x=np.ones((3,4))
for i in range(3):
    for j in range(4):
        x[i,j]=i+j
print (x)
print(x[1,0:])
print(x[1,-1])
