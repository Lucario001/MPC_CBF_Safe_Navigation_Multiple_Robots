# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:16:30 2023

@author: Bijo Sebastian
"""
pioneer_wheel_radius = 0.0975  #[m] Radius of the wheels on each pioneer
pioneer_track_width = 0.4      #[m] Distance betweent he two wheels of pioneer

pioneer_max_W = 0.3            #[rad/seconds] Maximum angular velocity (omega) of pioneer
pioneer_max_V = 0.7            #[m/seconds] Maximum linear velocity (V) of pioneer

goal_threshold = 0.4           #[m] The threshold distance at whihc robot is declared to be at goal

"""
Created on Fri Jul 19 15:04:08 2024

@author: kanni
"""

No_of_robots = 8               # No of robots used in the simulation

length_of_robot = 0.5          # in meters
l = length_of_robot / 4        # Look ahead distance in m assuming the lidar is placed in the center of the robot and it's diameter is half the length of the robot

us = 2                         # Control input size
sv = 2                         # State variable size

lidar_radius = 6.0             # [m] the approximate range of lidar sensor set by me

T = 0.001                      # [s] Sample time in seconds

d = 1.0

alpha = 0.1