import numpy as np
import math


def transformation_matrix(a, b, c, d):  # (a alpha d theta)
    # D-H Homogeneous Transformation Matrix
    return np.array([
        [math.cos(d), -math.sin(d)*math.cos(b),
         math.sin(d)*math.sin(b), a*math.cos(d)],
        [math.sin(d), math.cos(d)*math.cos(b), -
         math.cos(d)*math.sin(b), a*math.sin(d)],
        [0.0, math.sin(b), math.cos(b), c],
        [0.0, 0.0, 0.0, 1.0]
    ])


if __name__ == "__main__":
    q1, q2, q3, q4, q5, q6, q7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    A1 = transformation_matrix(0.0, math.pi/2, 0.0, q1)
    A2 = transformation_matrix(0.0, -math.pi/2, 0.0, q2)
    A3 = transformation_matrix(0.0, -math.pi/2, 0.0, q3)
    A4 = transformation_matrix(0.0, math.pi/2, 0.3, q4)
    A5 = transformation_matrix(0.0, math.pi/2, 0.0, q5)
    A6 = transformation_matrix(0.0, -math.pi/2, 0.0, q6)
    A7 = transformation_matrix(0.0, 0.0, 0.0, q7)

    T2 = np.linalg.multi_dot([A1, A2])
    T3 = np.linalg.multi_dot([A1, A2, A3])
    T4 = np.linalg.multi_dot([A1, A2, A3, A4])
    T5 = np.linalg.multi_dot([A1, A2, A3, A4, A5])
    T6 = np.linalg.multi_dot([A1, A2, A3, A4, A5, A6])

    z0 = np.array([0, 0, 1]).reshape(-1, 1)
    z1 = A1[0:3, 2].reshape(-1, 1)
    z2 = T2[0:3, 2].reshape(-1, 1)
    z3 = T3[0:3, 2].reshape(-1, 1)
    z4 = T4[0:3, 2].reshape(-1, 1)
    z5 = T5[0:3, 2].reshape(-1, 1)

    p0 = np.array([0, 0, 0]).reshape(-1, 1)
    p1 = A1[0:3, 3].reshape(-1, 1)
    p2 = T2[0:3, 3].reshape(-1, 1)
    p3 = T3[0:3, 3].reshape(-1, 1)
    p4 = T4[0:3, 3].reshape(-1, 1)
    p5 = T5[0:3, 3].reshape(-1, 1)

    P = T6[0:3, 3].reshape(-1, 1)

    J = np.array([
        [np.cross(z0, P - p0, axis=0), np.cross(z1, P - p1, axis=0), np.cross(z2, P - p2, axis=0),
         np.cross(z3, P - p3, axis=0), np.cross(z4, P - p4, axis=0), np.cross(z5, P - p5, axis=0)],
        [z0, z1, z2, z3, z4, z5]
    ])

    # JACOBIAN !
    print(J)