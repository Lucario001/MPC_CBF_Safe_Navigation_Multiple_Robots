# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:02:52 2024

@author: kanni
"""

import robot_params as rp
import numpy as np
import time
import sim_interface
import Model_Predictive_Controller
from casadi import *

# Function that tells which agents are in the range of each agent's lidar

def get_obstacles(current_locations):
    store_obstacles = [[] for _ in range(rp.No_of_robots)]
    for i in range(rp.No_of_robots):
        for j in range(rp.No_of_robots):
            if i != j:
                if np.sqrt((current_locations[i][0] - current_locations[j][0]) ** 2 + (current_locations[i][1] - current_locations[j][1]) ** 2) <= rp.lidar_radius:
                    store_obstacles[i].append([current_locations[j][0], current_locations[j][1]])
    return store_obstacles

def transition(sample_time, x_casadi, u_casadi, cur_casadi):
    xnew_casadi = MX.zeros(x_casadi.shape[0], 1)
    xnew_casadi[0, 0] = x_casadi[0, 0] + (u_casadi[0, 0] * np.cos(cur_casadi[2]) - rp.l * u_casadi[1, 0] * np.sin(cur_casadi[2])) * sample_time
    xnew_casadi[1, 0] = x_casadi[1, 0] + (u_casadi[0, 0] * np.sin(cur_casadi[2]) + rp.l * u_casadi[1, 0] * np.cos(cur_casadi[2])) * sample_time
        
    return xnew_casadi

def main():

    store_current_locations = np.array([[0.5, 0.5, 0.0], [0.5, 2.5, 0.0], [0.5, 4.5, 0.0], [0.5, 6.5, 0.0], [4.5, 0.5, np.pi], [4.5, 2.5, np.pi], [4.5, 4.5, np.pi], [4.5, 6.5, np.pi]])
    store_goal_locations = np.array([[4.5, 6.5], [4.5, 2.5], [4.5, 4.5], [4.5, 0.5], [0.5, 2.5], [0.5, 6.5], [0.5, 0.5], [0.5, 4.5]])
    
    #store_current_locations = np.array([[0.5, 0.5, 0.0], [0.5, 4.5, 0.0], [4.5, 0.5, np.pi], [4.5, 6.5, np.pi]])
    #store_goal_locations = np.array([[4.5, 6.5], [4.5, 0.5], [0.5, 2.5], [0.5, 4.5]])
    
    MPC_Controllers = [None for _ in range(rp.No_of_robots)]
    for i in range(rp.No_of_robots):
        MPC_Controllers[i] = Model_Predictive_Controller.Model_Predictive_Controller(
            prediction_horizon_steps = 6, 
            control_horizon_steps = 4, 
            state_dim = 2,
            control_input_dim = 2,
            sample_time = rp.T,
            terminal_weight = 1.0,
            intermediate_weight = 1e-6,
            control_weight = 1e-6,
            state_transition_function = transition,
            x_goal = store_goal_locations[i]
        )

    list_robots = [None for _ in range(rp.No_of_robots)]
    if sim_interface.sim_init():
        for i in range(rp.No_of_robots):
            list_robots[i] = sim_interface.Pioneer(i + 1)
            list_robots[i].localize_robot()
            store_current_locations[i] = list_robots[i].current_state
            list_robots[i].goal_state = store_goal_locations[i]
    else:
        print ('Failed connecting to remote API server')

    reached_goal = True                                                # Variable to store the status of the robots
    for i in range(rp.No_of_robots):
        reached_goal = reached_goal and list_robots[i].robot_at_goal()
        
    store_u = [[0.0, 0.0] for _ in range(rp.No_of_robots)]
    
    while not reached_goal:

        obstacles = get_obstacles(store_current_locations)
        
        if (sim_interface.start_simulation()):
            for i in range(rp.No_of_robots):
                store_u[i] = MPC_Controllers[i].solve(store_current_locations[i], store_u[i], obstacles[i])
                u = store_u[i]
                list_robots[i].run_MPC_Controller(u[0], u[1], rp.T)
        else:
            print ('Failed to start simulation')
        
        reached_goal = True
        for i in range(rp.No_of_robots):
            list_robots[i].localize_robot()
            store_current_locations[i] = list_robots[i].current_state
            reached_goal = reached_goal and list_robots[i].robot_at_goal()
            #MPC_Controllers[i].intermediate_weight = 10 ** (-np.sqrt((list_robots[i].current_state[0] - list_robots[i].goal_state[0]) ** 2 + (list_robots[i].current_state[1] - list_robots[i].goal_state[1]) ** 2))
            print(list_robots[i].robot_at_goal())
        print(store_u)
    
    #shutdown
    sim_interface.sim_shutdown()
    time.sleep(2.0)
    return

if __name__ == '__main__':
    main()
    print('Program ended')