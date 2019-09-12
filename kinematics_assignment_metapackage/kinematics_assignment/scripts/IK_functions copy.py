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


def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions  # it must contain 7 elements

    """
    Fill in your IK solution here and return the seven joint values in q
    """

    return q