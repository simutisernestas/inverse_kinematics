#! /usr/bin/env python3

"""
    # {Ernestas}
    # {Simutis}
"""

import numpy as np
import math


def scara_IK(point):
    x = point[0] - 0.07
    y = point[1]
    z = point[2]
    a1, a2 = (0.3, 0.35)
    square_sum = x**2 + y**2
    c2 = (square_sum - a1**2 - a2**2)/(2*a1*a2)
    s2 = math.sqrt(1-c2**2)
    theta_2 = np.arctan2(s2, c2)
    s1 = ((a1+a2*c2)*y - a2*s2*x)/square_sum
    c1 = ((a1+a2*c2)*x + a2*s2*y)/square_sum
    theta_1 = np.arctan2(s1, c1)
    return [theta_1, theta_2, z]


def homogeneous_trans_matrix(a, alpha, d, theta):  
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


def kuka_IK(point, R, joint_positions):
    q = joint_positions
    pd = np.matrix([[point[0]], [point[1]], [point[2]]])
    Rd = np.matrix(R)

    # Kp = np.random.rand(3, 3)
    # Kp = np.dot(Kp, Kp.transpose())
    # Ko = np.random.rand(3, 3)
    # Ko = np.dot(Ko, Ko.transpose())

    # if not np.all(np.linalg.eigvals(Kp) > 0) or not np.all(np.linalg.eigvals(Ko) > 0):
    #     print('not positive definite')
    #     exit()

    iq = [math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, -math.pi/2, 0.0]
    while True:
        q1, q2, q3, q4, q5, q6, q7 = q 

        H1 = homogeneous_trans_matrix(0.0, iq[0], 0.311, q1)
        H2 = homogeneous_trans_matrix(0.0, iq[1], 0.0, q2)
        H3 = homogeneous_trans_matrix(0.0, iq[2], 0.4, q3)
        H4 = homogeneous_trans_matrix(0.0, iq[3], 0.0, q4)
        H5 = homogeneous_trans_matrix(0.0, iq[4], 0.39, q5)
        H6 = homogeneous_trans_matrix(0.0, iq[5], 0.0, q6)
        H7 = homogeneous_trans_matrix(0.0, iq[6], 0.078, q7)

        T2 = np.linalg.multi_dot([H1, H2])
        T3 = np.linalg.multi_dot([H1, H2, H3])
        T4 = np.linalg.multi_dot([H1, H2, H3, H4])
        T5 = np.linalg.multi_dot([H1, H2, H3, H4, H5])
        T6 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6])
        T7 = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7])

        # end-effector displacement from base frame
        P = T7[0:3, 3].reshape(-1, 1)

        # contribution to the angular velocity
        z0 = np.array([0, 0, 1]).reshape(-1, 1)
        z1 = H1[0:3, 2].reshape(-1, 1)
        z2 = T2[0:3, 2].reshape(-1, 1)
        z3 = T3[0:3, 2].reshape(-1, 1)
        z4 = T4[0:3, 2].reshape(-1, 1)
        z5 = T5[0:3, 2].reshape(-1, 1)
        z6 = T6[0:3, 2].reshape(-1, 1)

        # contribution to the linear velocity
        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = H1[0:3, 3].reshape(-1, 1)
        p2 = T2[0:3, 3].reshape(-1, 1)
        p3 = T3[0:3, 3].reshape(-1, 1)
        p4 = T4[0:3, 3].reshape(-1, 1)
        p5 = T5[0:3, 3].reshape(-1, 1)
        p6 = T6[0:3, 3].reshape(-1, 1)

        J = np.concatenate(
            (
                np.concatenate((
                    np.cross(z0, P - p0, axis=0), np.cross(z1, P - p1, axis=0),
                    np.cross(z2, P - p2, axis=0), np.cross(z3, P - p3, axis=0),
                    np.cross(z4, P - p4, axis=0), np.cross(z5, P - p5, axis=0),
                    np.cross(z6, P - p6, axis=0),
                ), axis=1),
                np.concatenate((z0, z1, z2, z3, z4, z5, z6), axis=1)
            )
        )
        Ji = np.linalg.pinv(J)

        # angle error
        Re = T7[0:3, 0:3]
        nd = Rd[0:3, 0]
        ne = Re[0:3, 0].reshape(1, 3)
        sd = Rd[0:3, 1]
        se = Re[0:3, 1].reshape(1, 3)
        ad = Rd[0:3, 2]
        ae = Re[0:3, 2].reshape(1, 3)
        # p = Rd.dot(Re.T)
        # tau = math.acos((p[0, 0] + p[1, 1] + p[2, 2] - 1) / 2)  # (2.27)
        # r = (1 / 2 * math.sin(tau)) * \
        #     np.matrix([[p[2, 1] - p[1, 2]], [p[0, 2] - p[2, 0]],
        #                [p[1, 0] - p[0, 1]]])  # (2.28)
        # eo = r * math.sin(tau)

        eo = np.matrix(.5 * (
            np.cross(ne, nd.reshape(1, 3)) +
            np.cross(se, sd.reshape(1, 3)) +
            np.cross(ae, ad.reshape(1, 3)))
        ).T
        # angle_err = np.sum(np.absolute(eo))
        # not abs

        # check for absolute values
        ep = P - pd
        # distance error
        # dis_err = distance_to_goal(P, pd)
        E = np.concatenate((ep, eo))
        norm = np.linalg.norm(E)
        # err_sum = np.sum(E)
        # print('Errsum')

        # me = min(angle_err, me)
        # qv = Ji.dot(np.concatenate((P, wd)))
        qv = Ji.dot(E)
        q -=  np.array(qv.T)[0]
        # q = q - np.array(qv.T)[0] * delta

        if norm < 0.01:
            # print(q)
            break

        # err_sum2 = np.linalg.norm(np.concatenate((ep, eo)))

        # if err_sum < 0.6:
        #     print(q)
        #     break

        # dis = distance_to_goal(P, pd)
        # if dis < 0.1:
        #     print(q)
        #     break

        # L = -0.5 * (nd.dot(ne) + sd.dot(se) + ad.dot(ae))
        # Lt = L.T
        # Li = np.linalg.inv(L)
        # pos = pd + Kp.dot(ep)
        # rot = Li.dot(Lt.dot(wd) + Ko.dot(eo))
        # qv = Ji.dot(np.concatenate((pos, rot)))

        # qv = Ji.dot(np.concatenate((P, W)))
        # delta = 0.1
        # q = q + np.array(qv.T)[0] * delta
        # q = q + np.append(np.array(qv.T)[0], .0) * delta
        # q = np.array(qv.T)[0]
        # print(q)

    return q
