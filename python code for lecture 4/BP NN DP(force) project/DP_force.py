import numpy as np
import matplotlib.pyplot as plt

class DP_force_PID():
    def __init__(self,X0,x_d,h,t_f,Kp,Ki,Kd,e_thresh):
        '''
               Dynamical positioning the ship to a desired position and orientation by applying force & torque on the ship
               X0 = 6 by 1 array, initial state for ship model [x,y,psi,u,v,r]'
               psi in rad
               x_d = 3 by 1 array, desired position [x_d, y_d, psi_d]',
               t_f = final simulation time / maximum time
               h = sampling time
               Kp,Ki,Kd = P,I,D parameters,  each of them is a 3 by 3 matrix
               e_thresh = 1 by 3 vector, threshold used to evaluate position error(m) and heading error (rad)
               '''

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.e_thresh = e_thresh.reshape(-1,1)


        self.tf = t_f
        self.h = h
        self.x_d = x_d

        self.M = np.array([[1.1e7, 0, 0],
                          [0, 1.1e7, 8.4e6],
                          [0, 8.4e6, 5.8e9]])  # Mass matrix
        self.M_inv = np.linalg.inv(self.M)

        self.D = np.array([[3.0e5, 0, 0],
                          [0, 5.5e5, 6.4e5],
                          [0, 6.4e5, 1.2e8]])  # Damping matrix


        self.T = np.arange(0, self.tf, self.h)
        self.N = len(self.T)  # number of samples

        self.X = np.zeros((9, self.N))  # x,y,psi,u,v,r,tau
        self.X[:6, 0] = X0

    def run(self):
        '''
        run for one instance of DP_force_PID
        '''
        success = 0 #flag for indicating successful PID control to x_d
        # initialization for PID control
        e_previous = np.array([0, 0, 0]).reshape(-1, 1)
        e_int = np.array([0, 0, 0]).reshape(-1, 1)

        for i in range(self.N - 1):
            eta = self.X[:3, i]
            psi = eta[-1]
            R = np.array([[np.cos(psi), -np.sin(psi), 0],
                          [np.sin(psi), np.cos(psi), 0],
                          [0, 0, 1]], dtype=np.float32)

            # PID control
            e = np.dot(np.linalg.inv(R), (eta - self.x_d)).reshape(-1,1)

            e_int = e_int + e
            tau = -np.dot(self.Kp, e) - np.dot(self.Ki, e_int) * self.h - np.dot(self.Kd, (e - e_previous)) / self.h
            # print(tau)
            e_previous = e

            k1 = self.ship_model(self.T[i], self.X[:6, i], tau)
            k2 = self.ship_model(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, tau)
            k3 = self.ship_model(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, tau)
            k4 = self.ship_model(self.T[i] + self.h, self.X[:6, i] + self.h * k3, tau)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            self.X[6:, i + 1] = tau.reshape(-1)

            # ensure psi within [-pi, pi]
            if self.X[2, i + 1] > np.pi:
                self.X[2, i + 1] -= 2 * np.pi
            elif self.X[2, i + 1] <= -np.pi:
                self.X[2, i + 1] += 2 * np.pi

            if (np.absolute(e) <= self.e_thresh).all():
                #print('i=', i, 'e=',e, '(e <= self.e_thresh).all()  ',(e <= self.e_thresh).all())
                success = 1
                break
        res = self.X[:, :i + 2].reshape(9, -1) #since index i+1 has values updated
        return (success, res.T)

    #perform NN controller for DP
    def run_NN_control(self, mlp1, mlp2, mlp3):
        '''
        run for one instance of NN controller for DP operation
        :param mlp1, MLP NN model for Fx prediction
        :param mlp2, MLP NN model for Fy prediction
        :param mlp3, MLP NN model for Fz prediction
        '''
        success = 0 #flag for indicating successful PID control to x_d
        # MLP use x,y,psi,u,v,r to predict Fx, Fy, and Fz


        for i in range(self.N - 1):
            eta = self.X[:3, i]
            psi = eta[-1]
            R = np.array([[np.cos(psi), -np.sin(psi), 0],
                          [np.sin(psi), np.cos(psi), 0],
                          [0, 0, 1]], dtype=np.float32)
            e = np.dot(np.linalg.inv(R), (eta - self.x_d)).reshape(-1, 1)

            Fx = mlp1.predict(self.X[:6, i].reshape(1,-1))
            Fy = mlp2.predict(self.X[:6, i].reshape(1, -1))
            Fz = mlp3.predict(self.X[:6, i].reshape(1, -1))
            tau = np.array([Fx, Fy, Fz]).reshape(-1,1)

            k1 = self.ship_model(self.T[i], self.X[:6, i], tau)
            k2 = self.ship_model(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, tau)
            k3 = self.ship_model(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, tau)
            k4 = self.ship_model(self.T[i] + self.h, self.X[:6, i] + self.h * k3, tau)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            self.X[6:, i + 1] = tau.reshape(-1)

            # ensure psi within [-pi, pi]
            if self.X[2, i + 1] > np.pi:
                self.X[2, i + 1] -= 2 * np.pi
            elif self.X[2, i + 1] <= -np.pi:
                self.X[2, i + 1] += 2 * np.pi


            if (np.absolute(e) <= self.e_thresh).all():
                #print('i=', i, 'e=',e, '(e <= self.e_thresh).all()  ',(e <= self.e_thresh).all())
                success = 1
                break
        res = self.X[:, :i + 2].reshape(9, -1) #since index i+1 has values updated
        return (success, res.T)

    # ship dynamic model
    def ship_model(self, t, xship, tau):
        '''
        :param t: current time
        :param xship: x,y,psi,u,v,r
        :param tau: force & torque generated by PID controller
        :return: time derivative of xship
        '''


        xship = np.array(xship).reshape(-1, 1)

        F = tau.reshape(-1,1)  # return 3 by 1 array
        #print('F shape:', F.shape)

        tmp_psi = xship[2]
        R = np.array([[np.cos(tmp_psi), -np.sin(tmp_psi), 0],
                      [np.sin(tmp_psi), np.cos(tmp_psi), 0],
                      [0, 0, 1]], dtype=float)

        r1 = np.hstack((np.zeros((3, 3)), R))
        r2 = np.hstack((np.zeros((3, 3)), -np.dot(self.M_inv, self.D)))

        A = np.vstack((r1, r2))
        B = np.vstack((np.zeros((3, 1)), np.dot(self.M_inv, F)))

        xplus = (np.dot(A, xship) + B).T

        return xplus

    def get_time_step(self):
        return self.h