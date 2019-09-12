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


def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = joint_positions  # it must contain 7 elements
    return q