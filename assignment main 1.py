import random

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor



# kp = 1.47* 10 ** 5
# kn = 1.65* 10 ** 5

class pid_control():

    def __init__(self,RPM,Ttime):
        self.RPM=RPM
        self.Ttime=Ttime
        # self.kp=kp
        # self.kn=kn
        # self.Delta=Delta

    # def thrust_forcce (self) : #equation (1)
    #     if self.w >= 0 :
    #         T0 = self.kp * self.w ** 2
    #     elif self.w < 0 :
    #         T0 = self.kn * self.w * abs ( self.w )
    #     return T0

    # w = np.linspace ( -132 / 60 , 132 / 60 , 100 )
    # T0 = np.zeros ( len ( w ) )
    # for i in range ( 0 , len ( w ) ) : #for loop to calculate trust force
    #     T0 [ i ] = thrust_forcce ( w [ i ] )
    # fig = plt.figure (figsize=(10,10))
    # ax_thrust = fig.add_subplot ( 111 )
    # ax_thrust.plot ( w , T0 , label="Propeller trust force")
    # ax_thrust.set_xlabel("w: Shaft speed (Hz)")
    # ax_thrust.set_ylabel("T0: Propeller trust force (N)")
    # ax_thrust.legend()
    # plt.show ()
    # # from the curve we have delta as the ruder angel and f_d for drag force coefficient  f_l for lift force coefficient
    # delta = np.array ( [ -45 , -40 , -30 , -20 , -10 , 0 , 10 , 20 , 30 , 40 , 45 ] )
    # f_d = np.array ( [ 0.21 , 0.31 , 0.54 , 0.78 , 0.93 , 1 , 0.93 , 0.78 , 0.54 , 0.31 , 0.21 ] ) / 100
    # f_l = np.array ( [ -0.45 , -0.46 , -0.44 , -0.35 , -0.20 , 0 , 0.20 , 0.35 , 0.44 , 0.46 , 0.45 ] ) / 100
    # self.fD_poly = np.polyfit ( delta , f_d , 5 )
    # fL_poly = np.polyfit ( delta , f_l , 5 )

    # print("polynomial for FD are:",np.poly1d(fD_poly))
    # print("polynomial for FL are:",np.poly1d(fL_poly))
    # delta_plot = np.linspace ( -45 , 45 , 100 )
    # fD = np.polyval ( fD_poly , delta_plot )
    # fL = np.polyval ( fL_poly , delta_plot )
    # fig2=plt.figure(figsize=(10,10))
    # ax_LiftDrag=fig2.add_subplot(111)
    # ax_LiftDrag.plot ( delta_plot , fD , "b-" , label="(1-Drag force coefficient)" )
    # ax_LiftDrag.plot ( delta , f_d , "b*" )
    # ax_LiftDrag.plot ( delta_plot , fL , "r-" , label="Lift force coefficient" )
    # ax_LiftDrag.plot ( delta , f_l , "r*" )
    # ax_LiftDrag.legend ()
    # ax_LiftDrag.grid ()
    # ax_LiftDrag.set_xlabel("Rudder angel (degree)")
    # ax_LiftDrag.set_ylabel("Force (%T0)")
    # plt.show ()


    def get_force_torque (self,Delta) : # Delta in degree
        delta = np.array([-45, -40, -30, -20, -10, 0, 10, 20, 30, 40, 45])
        f_d = np.array([0.21, 0.31, 0.54, 0.78, 0.93, 1, 0.93, 0.78, 0.54, 0.31, 0.21]) / 100
        f_l = np.array([0.48, 0.49, 0.46, 0.36, 0.20, 0, -0.20, -0.36, -0.46, -0.49, -0.48]) / 100
        rpm=np.array([-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,])
        nominal_force=np.array([-303081,-282027,-253778,-215396,-176700,-133455,-102758,-73990,-51829,-33492,-19360,-13605,-2938,0,3007,15374,24149,41713,59350,83202,110691,142079,174066,208059,242513,276954,314476])
        nominal_force_poly=np.polyfit(rpm,nominal_force,5)
        fD_poly = np.polyfit(delta, f_d, 5)
        fL_poly = np.polyfit(delta, f_l, 5)


        # T0 = self.thrust_forcce ()
        fD = np.polyval ( fD_poly , Delta )
        fL = np.polyval ( fL_poly , Delta )
        NominalForce=np.polyval(nominal_force_poly,self.RPM)


        FXB = 2 * NominalForce * fD
        FYB=2 * NominalForce * fL
        TZB= FYB * 41.5
        return FXB , FYB , TZB


    # -------------------simulation of ship maneuvering-------------------------
    def rhs(self,X_old,FXB,FYB,TZB): #phi=X_old[2] is in radian and Delta is in degree
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
    # def Runge_Kutta (self,fun,dt):
    #     k1 = dt * fun(X_old)
    #     k2 = dt * fun(X_old + 0.5 * k1)
    #     k3 = dt * fun(X_old + 0.5 * k2)
    #     k4 = dt * fun(X_old + k3)
    #     return X_old + (1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4


    # -------------------begin PID control-------------------------
    def pid (self):
        desire_phi = 45  # degree, because Delta is in degree

        Kp = 10
        Ki = 0
        Kd = 0

        dt = 0.1
        # w=100/60
        phi_0=random.random()*180
        T = np.arange(0,self.Ttime,dt)
        X_state = np.zeros((6, len(T)))
        X_state[:,0] = np.array([0, 0, 51, 0, 0, 0])    # initial the condition

        U = np.zeros(T.shape)
        U_rate = np.zeros(T.shape)
        forces=np.zeros((3,len(T)))
        forces[:,0]=np.array([0,0,0])

        e_previous = 0
        e_int = 0

        for i in range(0, len(T)-1):
            # PID controller
            e = X_state[2, i] - desire_phi
            if abs(e)< 0.1:
                break

            e_int = e_int + e
            U[i] = -Kp * e - Ki * e_int * dt - Kd * (e - e_previous) / dt
            if U[i]>45:
                U[i]=45
            if U[i]<-45:
                U[i]=-45
            if i>0:
                if (U[i]-U[i-1])/dt > 3.7:
                    U[i]=3.7*dt+U[i-1]
                U_rate[i]=(U[i]-U[i-1])/dt
            e_previous = e
            FXB , FYB , TZB = self.get_force_torque (U[i] )
            forces[:,i]=np.array([FXB,FYB,TZB])

            # 4th-RK integration
            K1 = self.rhs(X_state[:,i],FXB,FYB,TZB)
            K2 = self.rhs(X_state[:,i] + 0.5 * dt * K1,FXB,FYB,TZB)
            K3 = self.rhs(X_state[:,i] + 0.5 * dt * K2,FXB,FYB,TZB)
            K4 = self.rhs(X_state[:,i] + dt * K3,FXB,FYB,TZB)

            # ensure phi within [-pi, pi]
            # if X_state[2, i] > np.pi:
            #     X_state[2, i] -= np.pi*2
            # elif X_state[2, i] <= -np.pi:
            #     X_state[2, i] += np.pi*2

            X_state[:,i+1] = X_state[:,i] + dt * (K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)
        return e,T,X_state,U,U_rate,forces
    def plot_pid(self):
        e,T,X_state,U,U_rate,forces=self.pid()# or we can define this variables with self in constructor
        print('error:',e)
        fig4=plt.figure(figsize=(7,7))
        ax_PID=fig4.add_subplot(321)
        ax_PID.plot(T,U,"b--",label="Delta")
        ax_PID.plot(T,X_state[2,:],"r-",label="phi")

        ax_PID.plot(T,45*np.ones(T.shape),"g--",linewidth=0.5,label="desired phi")
        ax_PID.plot(T,45*np.ones(T.shape),"k--",linewidth=0.5, label="Ruder max angel")
        ax_PID.set_xlabel("time (S)")
        ax_PID.set_ylabel("Degree")
        ax_PID.legend()


        # fig5=plt.figure(figsize=(2,5))
        ax_PIDxy=fig4.add_subplot(322)
        ax_PIDxy.plot(X_state[1,:],X_state[0,:])
        ax_PIDxy.set_xlabel("Y axis (m)")
        ax_PIDxy.set_ylabel("X axis (m)")
        ax_PID.legend()

        ax_forces=fig4.add_subplot(323)
        ax_forces.plot(T,forces[0,:],'r-',label='FBX')
        ax_forces.plot(T,forces[1,:],'b-',label='FBY')
        ax_forces.plot(T,forces[2,:],'g-',label='TZB')
        ax_forces.legend()


        ax_velocities=fig4.add_subplot(324)
        ax_velocities.plot(T,X_state[3,:],'r-',label='v surge')
        ax_velocities.plot(T,X_state[4,:],'b-',label='v sway')
        ax_velocities.legend()


        ax_algles=fig4.add_subplot(325)
        ax_algles.plot(T,U_rate,'r-',label='Rudder changing rate')
        ax_algles.plot(T,X_state[5,:],'b-',label='heading changing rate')
        ax_algles.legend()
        plt.tight_layout()


        plt.show()
        # plt.savefig('pid-fig.jpg')


##############################data generation######################################
s=pid_control(80,2000)
s.plot_pid()
sucess_num=0
data=np.zeros((1,9))
RPM=80
Ttime=500
test_num=500
# for i in range (test_num):
#     pid_data=pid_control(RPM,Ttime)
#     e,T,X_state,U,U_rate,forces=pid_data.pid()
#     print("error:",e)
#     if abs(e)<=5:
#         sucess_num+=1
#         a=np.concatenate((X_state.T,forces.T),axis=1)
#
#         data=np.vstack((data,a))
#
#         print("number of succeed controls",sucess_num)
#
# data = pd.DataFrame(data, columns=['x', 'y', 'phi', 'v_surge', 'v_sway', 'phi_dot', 'FBX','FBY','TBZ'])
# result_file = "./pid_data.csv"
# data.to_csv(result_file)
#


if False:
    S = pd.read_csv ( "./pid_data.csv" )

    data_list = S.iloc [ 0 :-1 , 1 :7 ]
    Fx_label_list = S.iloc [ 1 : , 7 ]  # Fx
    Fy_label_list = S.iloc [ 1 : , 8 ]  # Fy
    Fz_label_list = S.iloc [ 1 : , 9 ]  # yaw torque
    test_size = 0.1

    # dataset and model for Fx
    X1 , y1 = shuffle ( data_list.values , Fx_label_list , random_state=0 )
    x1_train , x1_test , y1_train , y1_test = train_test_split ( X1 , y1 , test_size=test_size )
    # print(x1_train.shape)
    # print(x1_test.shape)

    mlp1 = make_pipeline (
        StandardScaler () ,
        TransformedTargetRegressor (
            regressor=MLPRegressor ( hidden_layer_sizes=(10 , 10) ,
                                     activation='tanh' ,
                                     max_iter=1000 ) ,
            transformer=StandardScaler ()
        )
    )

    mlp1.fit ( x1_train , y1_train.ravel () )
    print ( 'mlp1.socre: ' , mlp1.score ( x1_test , y1_test ) )

    # print('mlp1: ', mlp1.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))

    # dataset and model for Fx
    X2 , y2 = shuffle ( data_list.values , Fy_label_list , random_state=0 )
    x2_train , x2_test , y2_train , y2_test = train_test_split ( X2 , y2 , test_size=test_size )
    # print(x2_train.shape)
    # print(x2_test.shape)

    mlp2 = make_pipeline (
        StandardScaler () ,
        TransformedTargetRegressor (
            regressor=MLPRegressor ( hidden_layer_sizes=(10 , 10) ,
                                     activation='tanh' ,
                                     max_iter=1000 ) ,
            transformer=StandardScaler ()
        )
    )

    mlp2.fit ( x2_train , y2_train.ravel () )
    print ( 'mlp2.score: ' , mlp2.score ( x2_test , y2_test ) )

    # print('mlp2: ', mlp2.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))

    # dataset and model for Yaw torque
    X3 , y3 = shuffle ( data_list.values , Fz_label_list , random_state=0 )
    x3_train , x3_test , y3_train , y3_test = train_test_split ( X3 , y3 , test_size=test_size )
    # print(x3_train.shape)
    # print(x3_test.shape)

    mlp3 = make_pipeline (
        StandardScaler () ,
        TransformedTargetRegressor (
            regressor=MLPRegressor ( hidden_layer_sizes=(15 , 12) ,
                                     activation='tanh' ,
                                     max_iter=1000 ) ,
            transformer=StandardScaler ()
        )
    )

    mlp3.fit ( x3_train , y3_train.ravel () )
    print ( 'mlp3.score: ' , mlp3.score ( x3_test , y3_test ) )

    # print('mlp3: ', mlp3.predict(np.array([-9.627573332,-2.014017937,-1.644032329,-0.00715073,0.015809356,0.003983173]).reshape(1,-1)))

    # save model
    joblib.dump ( mlp1 , './saved model/mlp1.pkl' )
    joblib.dump ( mlp2 , './saved model/mlp2.pkl' )
    joblib.dump ( mlp3 , './saved model/mlp3.pkl' )