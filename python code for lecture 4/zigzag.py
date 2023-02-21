import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class zigzag(object):
    def __init__(self, x0, zigzag_rudder, zigzag_heading):
        self.x0 = x0
        self.zigzag_rudder = zigzag_rudder
        self.zigzag_heading = zigzag_heading

        Dt = pd.read_csv('./rpm.txt', names=['rpm', 'f_coef'])
        self.rpm_ploy = np.polyfit(Dt.rpm, Dt.f_coef, deg=5)

        Dr = pd.read_csv('./rudder.txt',
                         names=['angle_lift', 'lift_coef', 'angle_prop', 'prop_coef'])
        self.l_ploy = np.polyfit(Dr.angle_lift, Dr.lift_coef, deg=5)

        self.p_ploy = np.polyfit(Dr.angle_prop, Dr.prop_coef, deg=5)

    #get the command force applied on ship
    def get_force(self, rpm, rudder):

        if np.abs(rpm) > 150 or np.abs(rudder) > 45:
            print('input thruster rpm or rudder angle out of range')

        arm = -41.5
        nominalf = np.polyval(self.rpm_ploy, rpm) * 1e5
        propulsionf = np.polyval(self.p_ploy, rudder) * nominalf
        liftf = np.polyval(self.l_ploy, rudder) * nominalf
        if rpm > 0:
            ft = np.array([2 * propulsionf, 2 * liftf, 2 * arm * liftf]).reshape(-1, 1)
        elif rpm == 0:
            ft = np.array([0, 0, 0]).reshape(-1, 1)
        else:
            ft = np.array([2 * nominalf, 0, 0]).reshape(-1, 1)
        return ft

    #get the actual control command on ship
    def get_actual(self, cmd_rudder, cmd_rpm, act_rudder_angle, act_rpm, delt_t):
        rudder_max = 45
        rudder_dot_max = 3.7

        rpm_max = 132
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

        return np.array([act_rpm_, act_rudder_angle_])

    # ship dynamic model
    def ship(self, t, xship, act):
        M = np.array([[1.1e7, 0, 0],
                      [0, 1.1e7, 8.4e6],
                      [0, 8.4e6, 5.8e9]])  # Mass matrix

        D = np.array([[3.0e5, 0, 0],
                      [0, 5.5e5, 6.4e5],
                      [0, 6.4e5, 1.2e8]])  # Damping matrix

        xship = np.array(xship).reshape(-1, 1) #x, y, yaw, x_dot, y_dot, yaw_rate

        F = self.get_force(act[0], act[1])

        R = np.array([[np.cos(xship[2,0]), -np.sin(xship[2,0]), 0],
                      [np.sin(xship[2,0]), np.cos(xship[2,0]), 0],
                      [0, 0, 1]], dtype=float)

        r1 = np.hstack((np.zeros((3, 3)), R))
        r2 = np.hstack((np.zeros((3, 3)), -np.dot(np.linalg.inv(M), D)))

        A = np.vstack((r1, r2))
        B = np.vstack((np.zeros((3, 1)), np.dot(np.linalg.inv(M), F)))

        xplus = (np.dot(A, xship) + B).T

        return xplus

    def solve(self, t_span = 800):
        h = 1
        X0 = self.x0 #x, y, yaw, x_dot, y_dot, yaw_rate

        T = np.arange(0, t_span, h)
        X = np.zeros((6, len(T)))  #x,y, psi, x_dot, y_dot, psi_dot
        cmd_rudder_value = np.zeros((1, len(T)))
        act_rudder_value = np.zeros((1, len(T)))
        act_rpm_value = np.zeros((1, len(T)))

        X[:, 0] = X0
        cmd_rud = 0
        for i in range(len(T) - 1):
            psi = np.rad2deg(X[2, i])

            # 20/20 zigzag control
            if T[i] >= 100 and T[i]<=100 +h:
                cmd_rud = self.zigzag_rudder
            elif cmd_rud != 0:
                if psi > self.zigzag_heading:
                    cmd_rud = -self.zigzag_rudder
                elif psi < -self.zigzag_heading:
                    cmd_rud = self.zigzag_rudder

            cmd_rudder_value[0, i + 1] = cmd_rud

            act = self.get_actual(
                cmd_rpm=80, cmd_rudder=cmd_rud, act_rudder_angle=act_rudder_value[0, i], act_rpm=act_rpm_value[0, i],
                delt_t=h
            )

            k1 = self.ship(T[i], X[:, i], act)
            k2 = self.ship(T[i] + 0.5 * h, X[:, i] + 0.5 * h * k1, act)
            k3 = self.ship(T[i] + 0.5 * h, X[:, i] + 0.5 * h * k2, act)
            k4 = self.ship(T[i] + h, X[:, i] + h * k3, act)
            X[:, i + 1] = X[:, i] + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)

            act_rpm_value[0, i + 1] = act[0]
            act_rudder_value[0, i + 1] = act[1]

        out = np.vstack((T, X, cmd_rudder_value, act_rudder_value))

        return out.T

if __name__ == "__main__":
    z = zigzag(x0=[0,0,0,0,0,0], zigzag_rudder=20, zigzag_heading=20)
    res = z.solve(t_span=1000)

    plt.plot(res[:,1], res[:,2], 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()

    plt.figure(2)
    plt.plot(res[:, 0],np.rad2deg(res[:, 3]), 'b', label='Heading(deg)')
    plt.plot(res[:, 0],res[:, 7], 'r', label='Command rudder(deg)')
    plt.plot(res[:, 0],res[:, 8], 'g', label='Act rudder(deg)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle')
    plt.grid()
    plt.legend(bbox_to_anchor =(1.05, 1.1),ncol=3)
    plt.show()