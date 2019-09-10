#! /usr/bin/env python3

"""
    # {student full name}
    # {student email}
"""

import numpy as np
import math
from numpy import *


def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]
    a1, a2 = (0.3, 0.35)
    c2 = (pow(x, 2) + pow(y, 2) - pow(a1, 2) - pow(a2, 2)) / (2*a1*a2)
    s2 = np.sqrt(1 - pow(c2, 2)) if (c2 >= -1) and (c2 <= 1) else 0
    q[1] = math.degrees(np.arctan2(s2, c2))
    s1 = ((a1 + a2*c2)*y - a2*s2*x) / (pow(x, 2) + pow(y, 2))
    c1 = ((a1 + a2*c2)*x + a2*s2*y) / (pow(x, 2) + pow(y, 2))
    q[0] = math.degrees(np.arctan2(s1, c1))
    q[2] = -z
    return q


def scara_IK_2(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]
    a1, a2 = (0.3, 0.35)
    r = math.sqrt(x**2 + y**2)
    fi2 = math.degrees(math.atan(1.18))
    fi1 = math.degrees(math.acos((a2**2 - a1**2 - r**2) / (-2*a1*r)))
    q[0] = fi2 - fi1
    fi3 = math.degrees(math.acos((r**2 - a1**2 - a2**2) / (-2*a2*a1)))
    q[1] = 180 - fi3
    q[2] = -z
    return q


def scara_IK_3(point):
    x = point[0]
    y = point[1]
    z = point[2]
    a1, a2 = (0.3, 0.35)
    delta = x**2 + y**2
    c2 = (delta - a1**2 - a2**2)/(2*a1*a2)
    s2 = sqrt(1-c2**2)
    theta_2 = arctan2(s2, c2)
    s1 = ((a1+a2*c2)*y - a2*s2*x)/delta
    c1 = ((a1+a2*c2)*x + a2*s2*y)/delta
    theta_1 = arctan2(s1, c1)
    return [rad2deg(theta_1), rad2deg(theta_2), -z]


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
    point = [0.57, -0.15, 0.1]
    ans = scara_IK(point)
    print(ans)
    ans = scara_IK_2(point)
    print(ans)
    ans = scara_IK_3(point)
    print(ans)
