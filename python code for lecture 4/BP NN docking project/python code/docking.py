import numpy as np
import pandas as pd

class docking_PID():
    def __init__(self, X0,x_d,h,t_f,pid_rpm,pid_rudder,e_thresh):
        '''
        Dynamical positioning the ship to a desired position and orientation
        X0 = 13 by 1 array, initial state for ship model [x,y,psi, u,v,r, e_pos_x, e_pos_y, e_psi, cmd_rpm,act_rpm, cmd_rudder_angle,act_rudder_angle]'
        psi in rad
        x_d = 2 by 1 array, desired position [x_d, y_d]',
        t_f = final simulation time
        h = sampling time
        pid_rpm = 3 by 1 array, PID parameter (kp, ki and kd) for cmd_rpm
        pid_rudder = 3 by 1 array, PID parameter(kp, ki and kd) for cmd_rudder
        e_thresh = threshold used to evaluate position error(m)
        '''

        self.pid_rpm = pid_rpm
        self.pid_rudder = pid_rudder

        self.tf = t_f
        self.h = h
        self.x_d = x_d
        self.e_thresh = e_thresh

        self.T = np.arange(0, self.tf, self.h)
        self.N = len(self.T) # number of samples

        self.X = np.zeros((13,self.N)) #0:3->eta, 3:6->nv, 6:9 e_pos_x, e_pos_y and e_psi in body frame, 9->cmd_rpm, 10->act_rpm, 11->cmd_rudder, 12->act_rudder
        self.X[:,0] = X0

        self.M = np.array([[1.1e7, 0, 0],
                          [0, 1.1e7, 8.4e6],
                          [0, 8.4e6, 5.8e9]])  # Mass matrix
        self.M_inv = np.linalg.inv(self.M)
        self.D = np.array([[3.0e5, 0, 0],
                          [0, 5.5e5, 6.4e5],
                          [0, 6.4e5, 1.2e8]])  # Damping matrix

        # propulsion coefficient
        Dt = pd.read_csv('../data file/rpm.txt', names=['rpm', 'f_coef'])
        self.rpm_ploy = np.polyfit(Dt.rpm, Dt.f_coef, deg=5)

        Dr = pd.read_csv('../data file/rudder.txt',
                         names=['angle_lift', 'lift_coef', 'angle_prop', 'prop_coef'])
        self.l_ploy = np.polyfit(Dr.angle_lift, Dr.lift_coef, deg=5)

        self.p_ploy = np.polyfit(Dr.angle_prop, Dr.prop_coef, deg=5)


    def run(self):
        '''
        run for one instance of docking to pos x_d using initial state
        '''
        success = 0 #flag for indicating successful docking to x_d

        # initialization for PID control
        e_rpm_previous = 0
        e_rpm_int = 0

        e_rudder_previous = 0
        e_rudder_int = 0

        for i in range(self.N-1):
            eta = self.X[:3, i]
            nu = self.X[3:6, i]

            psi = eta[-1] # psi is the yaw angle(in radian)


            ## RPM PID control
            #piece-wise surge speed
            e_pos_dist = np.linalg.norm(eta[:-1] - self.x_d.reshape(-1))
            #print('e_pos ', e_pos)
            #surge_desire = min(6.0, 6*np.exp(0.06 * e_pos_dist - 6))  # m/s

            if e_pos_dist>=200:
                surge_desire = 6
            else:
                surge_desire = 6*e_pos_dist/200 # m/s

            #if surge_desire!=6.0:
            #    print('aaa i=', i)
            e_surge = nu[0] - surge_desire
            e_rpm_int = e_rpm_int + e_surge
            cmd_rpm = -self.pid_rpm[0] * e_surge - self.pid_rpm[1] * e_rpm_int * self.h - self.pid_rpm[2] * (e_surge - e_rpm_previous) / self.h # in RPM
            #cmd_rpm = max(10,cmd_rpm)
            e_rpm_previous = e_surge
            #cmd_rpm = 80
            #print('cmd_rpm',cmd_rpm)

            ## rudder PID control
            psi_desire = np.arctan2(self.x_d[1] - eta[1], self.x_d[0] - eta[0]) #in rad

            e_psi = psi - psi_desire
            if e_psi > np.pi:
                e_psi -= 2 * np.pi
            elif e_psi <= -np.pi:
                e_psi += 2 * np.pi


            #if self.X[12, i]<0: #if actual RPM is in reverse mode, then rudder does not take effect on propulsion
            #    cmd_rudder = 0
            #else:
            e_rudder_int = e_rudder_int + e_psi
            cmd_rudder = np.rad2deg(-self.pid_rudder[0] * e_psi - self.pid_rudder[1] * e_rudder_int * self.h\
                     - self.pid_rudder[2] * (e_psi - e_rudder_previous) / self.h) # in degree
            e_rudder_previous = e_psi
            #print('cmd_rudder',cmd_rudder)

            #obtain the actual rpm and ruuder angle that will apply to the ship model
            act = self.get_actual(
                cmd_rpm=cmd_rpm, cmd_rudder=cmd_rudder,
                act_rpm=self.X[10, i],act_rudder_angle=self.X[12,i],
                delt_t=self.h
            )

            k1 = self.ship(self.T[i], self.X[:6, i], act)
            k2 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, act)
            k3 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, act)
            k4 = self.ship(self.T[i] + self.h, self.X[:6, i] + self.h * k3, act)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            #ensure psi within [-pi, pi]
            #if self.X[2, i + 1] > np.pi:
            #    self.X[2, i + 1] -= 2*np.pi
            #elif self.X[2, i + 1] <= -np.pi:
            #    self.X[2, i + 1] += 2*np.pi

            #calc e_pos and e_psi for index i+1
            e_pos_future = self.X[0:2, i+1] - self.x_d
            R = np.array([[np.cos(self.X[3, i+1]), -np.sin(self.X[3, i+1])],
                         [np.sin(self.X[3, i+1]), np.cos(self.X[3, i+1])]])
            e_pos_future_in_b = np.linalg.inv(R).dot(e_pos_future.reshape(-1,1))
            self.X[6:8, i + 1] = e_pos_future_in_b.reshape(-1)

            e_psi_future = self.X[2, i + 1] - np.arctan2(self.x_d[1] - self.X[1, i + 1], self.x_d[0] - self.X[0, i + 1])
            if e_psi_future > np.pi:
                e_psi_future -= 2*np.pi
            elif e_psi_future <= -np.pi:
                e_psi_future += 2*np.pi
            self.X[8, i + 1] = e_psi_future
            #fill in the calculated cmd and act rpm and rudder angle
            self.X[9, i + 1] = cmd_rpm
            self.X[10, i + 1] = act[0]
            self.X[11, i + 1] = cmd_rudder
            self.X[12, i + 1] = act[1]

            if e_pos_dist < self.e_thresh:
                success = 1
                break
        res = self.X[:,:i+2].reshape(13,-1) #since we update values to X[i+1]
        return (success, res.T)

    def get_force(self, rpm, rudder):
        '''
        :param rpm: current rpm
        :param rudder: current rudder angle in degree
        :return: force torque in surge, sway and yaw direction
        '''
        if np.abs(rpm) > 132 or np.abs(rudder) > 45:
            print('input thruster rpm or rudder angle out of range')

        arm = -41.5
        rudder_deg = rudder #note the polynomial use rudder in degree
        nominalf = np.polyval(self.rpm_ploy, rpm) * 1e6
        propulsionf = np.polyval(self.p_ploy, rudder_deg) * nominalf
        liftf = np.polyval(self.l_ploy, rudder_deg) * nominalf
        if rpm > 0:
            ft = np.array([2 * propulsionf, 2 * liftf, 2 * arm * liftf]).reshape(-1, 1)
        elif rpm == 0:
            ft = np.array([0, 0, 0]).reshape(-1, 1)
        else:
            ft = np.array([2 * nominalf, 0, 0]).reshape(-1, 1)
        return ft

    def get_actual(self, cmd_rudder, cmd_rpm, act_rudder_angle, act_rpm, delt_t):
        '''
        :param cmd_rudder: command rudder angle in degree
        :param cmd_rpm: command rmp in RPM
        :param act_rudder_angle, act_rpm: current rudder angle and RPM
        :return: actual rudder anger in degree and actual RPM
        '''

        #print('act_rpm type ', type(act_rpm))

        rudder_max = 45 #in degree
        rudder_dot_max = 3.7

        rpm_max = 132 # in RPM
        rpm_dot_max = 13

        if act_rudder_angle < cmd_rudder:
            if act_rudder_angle + rudder_dot_max * delt_t > cmd_rudder:
                act_rudder_angle_ = cmd_rudder
            else:
                act_rudder_angle_ = act_rudder_angle + rudder_dot_max * delt_t
        else:
            if act_rudder_angle - rudder_dot_max * delt_t < cmd_rudder:
                act_rudder_angle_ = cmd_rudder
            else:
                act_rudder_angle_ = act_rudder_angle - rudder_dot_max * delt_t

        if act_rudder_angle_ > rudder_max:
            act_rudder_angle_ = rudder_max
        elif act_rudder_angle_ < -rudder_max:
            act_rudder_angle_ = -rudder_max

        if act_rpm < cmd_rpm:
            if act_rpm + rpm_dot_max * delt_t > cmd_rpm:
                act_rpm_ = cmd_rpm
            else:
                act_rpm_ = act_rpm + rpm_dot_max * delt_t
        else:
            if act_rpm - rpm_dot_max * delt_t < cmd_rpm:
                act_rpm_ = cmd_rpm
            else:
                act_rpm_ = act_rpm - rpm_dot_max * delt_t

        if act_rpm_ > rpm_max:
            act_rpm_ = rpm_max
        elif act_rpm_ < -rpm_max:
            act_rpm_ = -rpm_max

        return np.array([float(act_rpm_), float(act_rudder_angle_)])

    # ship dynamic model
    def ship(self, t, xship, act):
        '''
        :param t: current time
        :param xship: x,y,psi, u,v,r, i.e., [eta nv]'
        :param act: include act rpm and act rudder angle in degree
        :return: time derivative of xship
        '''


        xship = np.array(xship).reshape(-1, 1)

        F = self.get_force(act[0], act[1]) #return 3 by 1 array
        #F = self.get_force(150, 0)
        #print('F:', F)

        tmp_psi = xship[2,0]
        R = np.array([[np.cos(tmp_psi), -np.sin(tmp_psi), 0],
                      [np.sin(tmp_psi), np.cos(tmp_psi), 0],
                      [0, 0, 1]], dtype=float)

        r1 = np.hstack((np.zeros((3, 3)), R))
        r2 = np.hstack((np.zeros((3, 3)), -np.dot(self.M_inv, self.D)))

        A = np.vstack((r1, r2))
        B = np.vstack((np.zeros((3, 1)), np.dot(self.M_inv, F)))

        xplus = (np.dot(A, xship) + B).T

        return xplus

    # perform NN controller for docking
    def run_NN_control(self, mlp1, mlp2):
        '''
            run for one instance of NN controller for docking operation
            :param mlp1, MLP NN model for rpm prediction
            :param mlp2, MLP NN model for rudder angle prediction
            '''
        success = 0  # flag for indicating successful control to x_d

        # MLP use u,v,r,e_pos_x,e_pos_y,e_psi, to predict rpm and ruuder angle
        ##################i have to fix the rest today.
        success = 0 #flag for indicating successful docking to x_d

        for i in range(self.N-1):
            eta = self.X[:3, i]

            cmd_rpm = mlp1.predict(self.X[3:9, i].reshape(1, -1)) #u,v,r,e_pos_x, e_pos_y, e_psi
            #print('cmd_rpm ', cmd_rpm)
            #rudder_input = np.array([self.X[3, i],self.X[5, i],self.X[6, i],self.X[7, i],self.X[8, i]])
            rudder_input = np.array([self.X[3:9, i]])
            cmd_rudder = mlp2.predict(rudder_input.reshape(1, -1))
            #print('cmd_rudder ', cmd_rudder)

            #obtain the actual rpm and ruuder angle that will apply to the ship model
            act = self.get_actual(
                cmd_rpm=cmd_rpm, cmd_rudder=cmd_rudder,
                act_rudder_angle=self.X[12,i],act_rpm=self.X[10,i],
                delt_t=self.h
            )
            #print('act= ',act)
            k1 = self.ship(self.T[i], self.X[:6, i], act)
            k2 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k1, act)
            k3 = self.ship(self.T[i] + 0.5 * self.h, self.X[:6, i] + 0.5 * self.h * k2, act)
            k4 = self.ship(self.T[i] + self.h, self.X[:6, i] + self.h * k3, act)
            self.X[:6, i + 1] = self.X[:6, i] + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            #ensure psi within [-pi, pi]
            #if self.X[2, i + 1] > np.pi:
            #    self.X[2, i + 1] -= 2*np.pi
            #elif self.X[2, i + 1] <= -np.pi:
            #    self.X[2, i + 1] += 2*np.pi

            #calc e_pos and e_psi for index i+1
            e_pos_future = self.X[0:2, i+1] - self.x_d
            R = np.array([[np.cos(self.X[3, i+1]), -np.sin(self.X[3, i+1])],
                         [np.sin(self.X[3, i+1]), np.cos(self.X[3, i+1])]])
            e_pos_future_in_b = np.linalg.inv(R).dot(e_pos_future.reshape(-1,1))
            self.X[6:8, i + 1] = e_pos_future_in_b.reshape(-1)

            e_psi_future = self.X[2, i + 1] - np.arctan2(self.x_d[1] - self.X[1, i + 1], self.x_d[0] - self.X[0, i + 1])
            if e_psi_future > np.pi:
                e_psi_future -= 2*np.pi
            elif e_psi_future <= -np.pi:
                e_psi_future += 2*np.pi
            self.X[8, i + 1] = e_psi_future
            #fill in the calculated cmd and act rpm and rudder angle
            self.X[9, i + 1] = cmd_rpm
            self.X[10, i + 1] = act[0]
            self.X[11, i + 1] = cmd_rudder
            self.X[12, i + 1] = act[1]

            e_pos_dist = np.linalg.norm(eta[:-1] - self.x_d.reshape(-1))
            if e_pos_dist < self.e_thresh:
                success = 1
                break
        res = self.X[:,:i+2].reshape(13,-1) #since we update values to X[i+1]
        return (success, res.T)

    def get_time_step(self):
        return  self.h