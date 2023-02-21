import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def thrust_forcce ( w ) : #equation (1)
    if w >= 0 :
        T0 = kp * w ** 2
    elif w < 0 :
        T0 = kn * w * abs ( w )
    return T0
kp = 1.47 * 10 ** 5
kn = 1.65 * 10 ** 5
w = np.linspace ( -132 / 60 , 132 / 60 , 100 )
T0 = np.zeros ( len ( w ) )
for i in range ( 0 , len ( w ) ) : #for loop to calculate trust force
    T0 [ i ] = thrust_forcce ( w [ i ] )
fig = plt.figure (figsize=(10,10))
ax_thrust = fig.add_subplot ( 111 )
ax_thrust.plot ( w , T0 , label="Propeller trust force")
ax_thrust.set_xlabel("w: Shaft speed (Hz)")
ax_thrust.set_ylabel("T0: Propeller trust force (N)")
ax_thrust.legend()
plt.show ()
# from the curve we have delta as the ruder angel and f_d for drag force coefficient  f_l for lift force coefficient
delta = np.array ( [ -45 , -40 , -30 , -20 , -10 , 0 , 10 , 20 , 30 , 40 , 45 ] )
f_d = np.array ( [ 0.21 , 0.31 , 0.54 , 0.78 , 0.93 , 1 , 0.93 , 0.78 , 0.54 , 0.31 , 0.21 ] ) / 100
f_l = np.array ( [ -0.45 , -0.46 , -0.44 , -0.35 , -0.20 , 0 , 0.20 , 0.35 , 0.44 , 0.46 , 0.45 ] ) / 100
fD_poly = np.polyfit ( delta , f_d , 5 )
fL_poly = np.polyfit ( delta , f_l , 5 )
print("polynomial for FD are:",np.poly1d(fD_poly))
print("polynomial for FL are:",np.poly1d(fL_poly))
delta_plot = np.linspace ( -45 , 45 , 100 )
fD = np.polyval ( fD_poly , delta_plot )
fL = np.polyval ( fL_poly , delta_plot )
fig2=plt.figure(figsize=(10,10))
ax_LiftDrag=fig2.add_subplot(111)
ax_LiftDrag.plot ( delta_plot , fD , "b-" , label="(1-Drag force coefficient)" )
ax_LiftDrag.plot ( delta , f_d , "b*" )
ax_LiftDrag.plot ( delta_plot , fL , "r-" , label="Lift force coefficient" )
ax_LiftDrag.plot ( delta , f_l , "r*" )
ax_LiftDrag.legend ()
ax_LiftDrag.grid ()
ax_LiftDrag.set_xlabel("Rudder angel (degree)")
ax_LiftDrag.set_ylabel("Force (%T0)")
plt.show ()


def get_force_torque ( w , Delta ) : # Delta in degree
    T0 = thrust_forcce ( w )
    fD = np.polyval ( fD_poly , Delta )
    fL = np.polyval ( fL_poly , Delta )
    FBX = 2 * (T0 * fD)
    FBY=2 * T0 * fL
    TZB=2 * T0 * fL * 41.5
    return FBX , FBY , TZB


# -------------------simulation of ship maneuvering-------------------------
def rhs(X_old): #phi=X_old[2] is in radian and Delta is in degree
    y1= X_old [ 3 ] * np.cos ( X_old [ 2 ] * np.pi / 180 ) - X_old [ 4 ] \
        * np.sin ( X_old [ 2 ] * np.pi / 180 )
    y2= X_old [ 3 ] * np.sin ( X_old [ 2 ] * np.pi / 180 ) + X_old [ 4 ] *\
        np.cos ( X_old [ 2 ] * np.pi / 180 )
    y3= X_old [ 5 ]
    y4= -X_old [ 3 ] * 0.02727272727273 + 9.09090909090909 * 10 ** (-8) * FXB \
        + 0.0181818181818182 * np.cos ( X_old [ 2 ] * np.pi / 180 )
    y5= -X_old [ 4 ] * 0.0135671049361174 - X_old [ 5 ] * 0.0424293701623614 +\
        9.10097436914556 * 10 ** (-8) * FYB - 1.31807215001418 * 10 ** (-10) * \
        TZB - 0.0182019487382911 * np.sin ( X_old [ 2 ] * np.pi / 180 )
    y6=  -X_old [ 4 ] * 9.06959169890713 * 10 ** (-5) - X_old [ 5 ] * 0.0206282057397649\
         - 1.31807215001418 * 10 ** (-10) * FYB + 1.72604686311381 * 10 ** (-10) * TZB + \
         2.63614430002837 * 10 ** (-5) * np.sin ( X_old [ 2 ] * np.pi / 180 )
    return np.array([y1,y2,y3,y4,y5,y6])
def Runge_Kutta (fun,dt):
    k1 = dt * fun(X_old)
    k2 = dt * fun(X_old + 0.5 * k1)
    k3 = dt * fun(X_old + 0.5 * k2)
    k4 = dt * fun(X_old + k3)
    return X_old + (1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4
#############defining variables and constants#############
tEnd=1000
dt=0.05
t=0
w=100/60
X_old_0=np.array([0,0,0,0,0,0]) #initial condition
X_old=X_old_0
X_new=np.zeros(6,float)
data=[]# for storing the results in loop
time=[]
data.append(X_old_0)
time.append(0)
##############soluton by loop############
while t<tEnd:
    Delta=30*np.sin(0.06*t)
    FXB,FYB,TZB=get_force_torque(w,Delta)
    X_new=Runge_Kutta(rhs,dt)
    time.append(t)
    data.append(X_new)
    X_old=X_new
    t+=dt
data=np.array(data)
time=np.array(time)
#############plot the results###############

fig3=plt.figure(figsize=(8,9))
ax_maneuver=fig3.add_subplot(122)
ax_maneuver.plot(data[:,1],data[:,0],"r-",label="ship maneuvering")
ax_maneuver.set_xlabel("Y (m)")
ax_maneuver.set_ylabel("X (m)")
ax_surgeSpeed=fig3.add_subplot(321)
ax_surgeSpeed.plot(time,data[:,3])
ax_surgeSpeed.set_title('surge speed')
ax_surgeSpeed.set_xlabel('time')
ax_surgeSpeed.set_ylabel('m/s')
ax_swaySpeed=fig3.add_subplot(323)
ax_swaySpeed.plot(time,data[:,4])
ax_swaySpeed.set_xlabel('time')
ax_swaySpeed.set_ylabel('m/s')
ax_swaySpeed.set_title('sway speed')
ax_headingRate=fig3.add_subplot(325)
ax_headingRate.plot(time,data[:,5])
ax_headingRate.set_title('heading rate')
ax_headingRate.set_xlabel('time')
ax_headingRate.set_ylabel('radian')

plt.tight_layout()
ax_maneuver.legend()
plt.show()

# -------------------begin PID control-------------------------
desire_phi = 30  # degree, because Delta is in degree

Kp = 30
Ki = 0
Kd = 7000
# with Delta limit
# Kp = 30
# Ki = -0.0001
# Kd = 7000

# without Delta limit
# Kp = 2.16
# Ki = -0.0001
# Kd = -0.0098
dt = 0.1
w=100/60
T = np.arange(0,2000,dt)
X_state = np.zeros((6, len(T)))
X_state[:,0] = np.array([0, 0, 0, 0, 0, 0])    # initial the condition
U = np.zeros(T.shape)

e_previous = 0
e_int = 0

for i in range(0, len(T)-1):
    # PID controller
    e = X_state[2, i]*180/np.pi - desire_phi
    e_int = e_int + e
    U[i] = -Kp * e - Ki * e_int * dt - Kd * (e - e_previous) / dt
    if U[i]>45:
        U[i]=45
    if U[i]<-45:
        U[i]=-45
    e_previous = e
    FXB , FYB , TZB = get_force_torque ( w , U[i] )

    # 4th-RK integration
    K1 = rhs(X_state[:,i])
    K2 = rhs(X_state[:,i] + 0.5 * dt * K1)
    K3 = rhs(X_state[:,i] + 0.5 * dt * K2)
    K4 = rhs(X_state[:,i] + dt * K3)

    X_state[:,i+1] = X_state[:,i] + dt * (K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)
print("Error:",e)
fig4=plt.figure(figsize=(8,3))
ax_PID=fig4.add_subplot(111)
ax_PID.plot(T,U,"b--",label="Delta")
ax_PID.plot(T,X_state[2,:]*180/np.pi,"r-",label="phi")
ax_PID.plot(T,30*np.ones(T.shape),"g--",linewidth=0.5,label="desired phi")
ax_PID.plot(T,45*np.ones(T.shape),"k--",linewidth=0.5, label="Ruder max angel")
ax_PID.set_xlabel("time (S)")
ax_PID.set_ylabel("Degree")
ax_PID.legend()
fig5=plt.figure(figsize=(3,8))
ax_PIDxy=fig5.add_subplot(111)
ax_PIDxy.plot(X_state[1,:],X_state[0,:])
ax_PIDxy.set_xlabel("Y axis (m)")
ax_PIDxy.set_ylabel("X axis (m)")
plt.tight_layout()

plt.show()