import numpy as np
import math


def homogeneous_trans_matrix(a, alpha, d, theta):  # (a alpha d theta)
    # D-H Homogeneous Transformation Matrix
    return np.array([
        [
            math.cos(theta),
            -math.sin(theta)*math.cos(alpha),
            math.sin(theta)*math.sin(alpha),
            a*math.cos(theta)
        ],
        [
            math.sin(theta),
            math.cos(theta)*math.cos(alpha),
            -math.cos(theta)*math.sin(alpha),
            a*math.sin(theta)
        ],
        [0.0, math.sin(alpha), math.cos(alpha), d],
        [0.0, 0.0, 0.0, 1.0]
    ])

def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0][0] - current_pos[0][0]
    y_diff = goal_pos[1][0] - current_pos[1][0]
    z_diff = goal_pos[2][0] - current_pos[2][0]
    return np.math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

if __name__ == "__main__":
    q = [0, 1.02, 0, 1.21, 0, 1.84]
    point = np.matrix([[.0], [.0], [.79]])

    Kp = np.random.rand(3, 3)
    Kp = np.dot(Kp, Kp.transpose())
    Ko = np.random.rand(3, 3)
    Ko = np.dot(Ko, Ko.transpose())

    while True:
        q1, q2, q3, q4, q5, q6 = q

        H1 = homogeneous_trans_matrix(0.0, math.pi/2, 0.0, q1)
        H2 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.0, q2)
        H3 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.4, q3)
        H4 = homogeneous_trans_matrix(0.0, math.pi/2, 0.0, q4)
        H5 = homogeneous_trans_matrix(0.0, math.pi/2, 0.39, q5)
        H6 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.0, q6)
        # H7 = homogeneous_trans_matrix(0.0, 0.0, 0.0, q7)

        T2 = np.linalg.multi_dot([H1, H2])
        T3 = np.linalg.multi_dot([H1, H2, H3])
        T4 = np.linalg.multi_dot([H1, H2, H3, H4])
        T5 = np.linalg.multi_dot([H1, H2, H3, H4, H5])
        T6 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6])
        # T7 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7])  # same as T6

        # end-effector displacement from base frame
        P = T6[0:3, 3].reshape(-1, 1)
        # P7 = T7[0:3, 3].reshape(-1, 1)  # same as T7

        z0 = np.array([0, 0, 1]).reshape(-1, 1)
        z1 = H1[0:3, 2].reshape(-1, 1)
        z2 = T2[0:3, 2].reshape(-1, 1)
        z3 = T3[0:3, 2].reshape(-1, 1)
        z4 = T4[0:3, 2].reshape(-1, 1)
        z5 = T5[0:3, 2].reshape(-1, 1)

        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = H1[0:3, 3].reshape(-1, 1)
        p2 = T2[0:3, 3].reshape(-1, 1)
        p3 = T3[0:3, 3].reshape(-1, 1)
        p4 = T4[0:3, 3].reshape(-1, 1)
        p5 = T5[0:3, 3].reshape(-1, 1)
        p5 = T5[0:3, 3].reshape(-1, 1)

        up = np.concatenate((np.cross(z0, P - p0, axis=0), np.cross(z1, P - p1, axis=0),
                            np.cross(z2, P - p2, axis=0), np.cross(z3, P - p3, axis=0),
                            np.cross(z4, P - p4, axis=0), np.cross(z5, P - p5, axis=0)), axis=1)
        down = np.concatenate((z0, z1, z2, z3, z4, z5), axis=1)
        J = np.concatenate((up, down))
        Ji = np.linalg.pinv(J)

        Rd = np.matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        Re = T6[0:3, 0:3]
        p = Rd.dot(Re.T)
        tau = math.acos((p[0, 0] + p[1, 1] + p[2, 2] - 1) / 2)  # (2.27)
        r = (1 / 2 * math.sin(tau)) * \
            np.matrix([[p[2, 1] - p[1, 2]], [p[0, 2] - p[2, 0]],
                    [p[1, 0] - p[0, 1]]])  # (2.28)
        eo = r * math.sin(tau)
        # print(Rd, '\n\n', Re, '\n\n', p, '\n\n', eo)
        # print(np.square(r)) should sum up to 1 ! 

        nd = Rd[0:3, 0]
        ne = Re[0:3, 0].reshape(1, 3)
        sd = Rd[0:3, 1]
        se = Re[0:3, 1].reshape(1, 3)
        ad = Rd[0:3, 2]
        ae = Re[0:3, 2].reshape(1, 3)
        L = -0.5 * (nd.dot(ne) + sd.dot(se) + ad.dot(ae))
        Lt = L.T
        Li = np.linalg.inv(L)

        pd = np.array([0.57, -0.15, 0.1]).reshape(3,1)
        wd = np.array([0.07, 0.05, -0.966]).reshape(3,1)

        
        if not np.all(np.linalg.eigvals(Kp) > 0) or not np.all(np.linalg.eigvals(Ko) > 0):
            print('not positive definite')
            exit()

        ep = P - pd

        dis = distance_to_goal(P, pd)

        if dis < 0.1:
            print(dis)
            print(ep)
            print(eo)
            print(q)
            break

        pos = pd + Kp.dot(ep)
        rot = Li.dot(Lt.dot(wd) + Ko.dot(eo))
        qv = Ji.dot(np.concatenate((pos,rot)))

        # xd − k(q) is reduced within a given threshold

        delta = .05
        q = np.array(qv.T)[0]
        # q = q * np.array(qv.T)[0] * delta
        



'''

pd = np.matrix([[point[0]], [point[1]], [point[2]]])
    Rd = R

    while True:
        q1, q2, q3, q4, q5, q6, q7 = q

        H1 = homogeneous_trans_matrix(0.0, math.pi/2, 0.0, q1)
        H2 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.0, q2)
        H3 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.4, q3)
        H4 = homogeneous_trans_matrix(0.0, math.pi/2, 0.0, q4)
        H5 = homogeneous_trans_matrix(0.0, math.pi/2, 0.39, q5)
        H6 = homogeneous_trans_matrix(0.0, -math.pi/2, 0.0, q6)
        # H7 = homogeneous_trans_matrix(0.0, 0.0, 0.0, q7)

        T2 = np.linalg.multi_dot([H1, H2])
        T3 = np.linalg.multi_dot([H1, H2, H3])
        T4 = np.linalg.multi_dot([H1, H2, H3, H4])
        T5 = np.linalg.multi_dot([H1, H2, H3, H4, H5])
        T6 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6])
        # T7 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7])  # same as T6

        # end-effector displacement from base frame
        P = T6[0:3, 3].reshape(-1, 1)
        # P7 = T7[0:3, 3].reshape(-1, 1)  # same as T7

        z0 = np.array([0, 0, 1]).reshape(-1, 1)
        z1 = H1[0:3, 2].reshape(-1, 1)
        z2 = T2[0:3, 2].reshape(-1, 1)
        z3 = T3[0:3, 2].reshape(-1, 1)
        z4 = T4[0:3, 2].reshape(-1, 1)
        z5 = T5[0:3, 2].reshape(-1, 1)

        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = H1[0:3, 3].reshape(-1, 1)
        p2 = T2[0:3, 3].reshape(-1, 1)
        p3 = T3[0:3, 3].reshape(-1, 1)
        p4 = T4[0:3, 3].reshape(-1, 1)
        p5 = T5[0:3, 3].reshape(-1, 1)
        p5 = T5[0:3, 3].reshape(-1, 1)

        up = np.concatenate((np.cross(z0, P - p0, axis=0), np.cross(z1, P - p1, axis=0),
                             np.cross(z2, P - p2, axis=0), np.cross(z3,
                                                                    P - p3, axis=0),
                             np.cross(z4, P - p4, axis=0), np.cross(z5, P - p5, axis=0)), axis=1)
        down = np.concatenate((z0, z1, z2, z3, z4, z5), axis=1)
        J = np.concatenate((up, down))
        Ji = np.linalg.pinv(J)

        Re = T6[0:3, 0:3]
        p = Rd.dot(Re.T)
        tau = math.acos((p[0, 0] + p[1, 1] + p[2, 2] - 1) / 2)  # (2.27)
        r = (1 / 2 * math.sin(tau)) * \
            np.matrix([[p[2, 1] - p[1, 2]], [p[0, 2] - p[2, 0]],
                       [p[1, 0] - p[0, 1]]])  # (2.28)
        eo = r * math.sin(tau)
        # print(Rd, '\n\n', Re, '\n\n', p, '\n\n', eo)
        # print(np.square(r)) should sum up to 1 !

        nd = Rd[0:3, 0]
        ne = Re[0:3, 0].reshape(1, 3)
        sd = Rd[0:3, 1]
        se = Re[0:3, 1].reshape(1, 3)
        ad = Rd[0:3, 2]
        ae = Re[0:3, 2].reshape(1, 3)
        L = -0.5 * (nd.dot(ne) + sd.dot(se) + ad.dot(ae))
        Lt = L.T
        Li = np.linalg.inv(L)

        # pd = np.array([0.57, -0.15, 0.1]).reshape(3,1)
        wd = np.array([0.07, 0.05, -0.966]).reshape(3, 1)

        Kp = np.random.rand(3, 3)
        Kp = np.dot(Kp, Kp.transpose())
        Ko = np.random.rand(3, 3)
        Ko = np.dot(Ko, Ko.transpose())

        if not np.all(np.linalg.eigvals(Kp) > 0) or not np.all(np.linalg.eigvals(Ko) > 0):
            print('not positive definite')
            exit()

        ep = P - pd

        dis = distance_to_goal(P, pd)

        if dis < 0.1:
            break

        pos = pd + Kp.dot(ep)
        rot = Li.dot(Lt.dot(wd) + Ko.dot(eo))
        qv = Ji.dot(np.concatenate((pos, rot)))

        # xd − k(q) is reduced within a given threshold

        delta = 0.1
        q = q + np.append(np.array(qv.T)[0], .0) * delta


'''