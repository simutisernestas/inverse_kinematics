#! /usr/bin/env python3

"""
    # {student full name}
    # {student email}
"""

import numpy as np

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]

    a1, a2 = [0.4670,0.4005]

    # Second joint
    c2 = (pow(x, 2) + pow(y, 2) - pow(a1, 2) - pow(a2, 2)) / (2*a1*a2)

    if (c2 >= -1) and (c2 <= 1):
        s2 = np.sqrt(1 - pow(c2, 2))
    else:
        s2 = 0

    q[1] = np.arctan2(s2, c2)

    # First joint
    s1 = ((a1 + a2*c2)*y - a2*s2*x) / (pow(x, 2) + pow(y, 2))

    c1 = ((a1 + a2*c2)*x + a2*s2*y) / (pow(x, 2) + pow(y, 2))

    q[0] = np.arctan2(s1, c1)

    # Third joint
    q[2] = z

    return q


def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions  # it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    return q


if __name__ == '__main__':

    point = [1,5,10]

    ans = scara_IK(point)

    print(ans)
    