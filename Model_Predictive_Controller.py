# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:00:14 2024

@author: kanni
"""

from casadi import *
import numpy as np
import robot_params as rp

class Model_Predictive_Controller:
    def __init__(self, prediction_horizon_steps, control_horizon_steps, state_dim, control_input_dim, sample_time, terminal_weight, intermediate_weight, control_weight, state_transition_function, x_goal):
        
        self.Np = prediction_horizon_steps
        self.Nc = control_horizon_steps
        
        self.x_goal = x_goal                                                    # [goalx, goaly] 
        
        self.state_dim = state_dim                                              # It is 2 [sx, sy]
        self.control_input_dim = control_input_dim                              # It is 2 [V, W]
        
        self.sample_time = sample_time
        
        self.terminal_weight = terminal_weight
        self.intermediate_weight = intermediate_weight
        self.control_weight = control_weight
        
        self.transition_function = state_transition_function
        
    def solve(self, x_init, prev_control, obstacle_locations):
        
        self.x_init = x_init                                                    # [x, y, angle]
    
        self.prev_control = prev_control
        
        self.obstacle_locations = obstacle_locations
        
        opti = Opti()
        # For storing the predicted control inputs
        u_casadi = opti.variable(self.control_input_dim, self.Np)
        # The above is a matrix where the control inputs at one time instance are represented in one column and there are Np columns
        
        u0_casadi = opti.parameter(self.control_input_dim, 1)
        
        for i in range(self.control_input_dim):
            opti.set_value(u0_casadi[i, 0], self.prev_control[i])
    
        
        for i in range(self.Nc, self.Np):
            u_casadi[:, i] = u_casadi[:, self.Nc - 1]
        
        # For storing the predicted states
        x_casadi = opti.variable(self.state_dim, self.Np)
        
        # For storing the initial condition
        x0_casadi = opti.parameter(self.state_dim, 1)
        
        # For storing the goal condition
        xgoal_casadi = opti.parameter(self.state_dim, 1)    
        
        current_casadi = MX.zeros(len(self.x_init), self.Np)
        for i in range(len(self.x_init)):
            current_casadi[i, 0] = self.x_init[i]
    
        
        opti.set_value(x0_casadi[0], self.x_init[0] + rp.l * np.cos(self.x_init[2]))
        opti.set_value(x0_casadi[1], self.x_init[1] + rp.l * np.sin(self.x_init[2]))
        opti.set_value(xgoal_casadi[0], self.x_goal[0])
        opti.set_value(xgoal_casadi[1], self.x_goal[1])
        
        transformation_matrix = MX.ones(2, 2)
        transformation_matrix[0, 0] = np.cos(current_casadi[2, 0])
        transformation_matrix[0, 1] = - rp.l * np.sin(current_casadi[2, 0])
        transformation_matrix[1, 0] = np.sin(current_casadi[2, 0])
        transformation_matrix[1, 1] = rp.l * np.cos(current_casadi[2, 0])
            
        x_casadi[:, 0] = self.transition_function(self.sample_time, x0_casadi, u_casadi[:, 0], current_casadi[:, 0])
        
        
        for i in range(1, self.Np):
            current_casadi[0, i] = current_casadi[0, i - 1] + u_casadi[0, i - 1] * np.cos(current_casadi[2, i - 1]) * self.sample_time
            current_casadi[1, i] = current_casadi[1, i - 1] + u_casadi[0, i - 1] * np.sin(current_casadi[2, i - 1]) * self.sample_time
            current_casadi[2, i] = current_casadi[2, i - 1] + u_casadi[1, i - 1] * self.sample_time
            x_casadi[:, i] = self.transition_function(self.sample_time, x_casadi[:, i - 1], u_casadi[:, i], current_casadi[:, i])
            
        cost = 0 
        for i in range(0, self.Np):
            if i < self.Np - 1:
                cost += self.intermediate_weight * (x_casadi[:, i] - xgoal_casadi[:, 0]).T @ (x_casadi[:, i] - xgoal_casadi[:, 0])
            else:
                cost += self.terminal_weight * (x_casadi[:, i] - xgoal_casadi[:, 0]).T @ (x_casadi[:, i] - xgoal_casadi[:, 0])
            cost += self.control_weight * (u_casadi[:, i].T @ u_casadi[:, i])
        
        opti.minimize(cost)
        
        for i in range(0, self.Np):
            opti.set_initial(u_casadi[:, i], self.prev_control)
        
        obstacle_matrix = MX.zeros(len(self.obstacle_locations), 2)
        cbf_vector = MX.zeros(len(self.obstacle_locations), 1)
        flag = 0
        for i in range(len(self.obstacle_locations)):
            flag = 1
            obstacle_matrix[i, 0] = 2 * (x0_casadi[0, 0] - self.obstacle_locations[i][0])
            obstacle_matrix[i, 1] = 2 * (x0_casadi[1, 0] - self.obstacle_locations[i][1])
            cbf_vector[i, 0] = - rp.alpha * ((x0_casadi[0, 0] - self.obstacle_locations[i][0]) ** 2 + (x0_casadi[1, 0] - self.obstacle_locations[i][1]) ** 2 - rp.d ** 2)
            
        boundary_matrix = MX.zeros(2, 2)
        cbf_boundary = MX.zeros(2, 1)
        boundary_matrix[0, 0] = 9.75 + 0.25 - 2 * x0_casadi[0, 0]
        boundary_matrix[1, 1] = 9.75 + 0.25 - 2 * x0_casadi[1, 0]
        cbf_boundary[0, 0] = - rp.alpha * ((9.75 + 0.25) * x0_casadi[0, 0] - x0_casadi[0, 0] ** 2 - 0.25 * 9.75)
        cbf_boundary[1, 0] = - rp.alpha * ((9.75 + 0.25) * x0_casadi[1, 0] - x0_casadi[1, 0] ** 2 - 0.25 * 9.75)
        
        
        u_limits = MX.zeros(2, 1)
        l_limits = MX.zeros(2, 1)
        u_limits[0, 0] = 0.7
        l_limits[0, 0] = - 0.7
        u_limits[1, 0] = 0.3
        l_limits[1, 0] = - 0.3
        
        for i in range(self.Np):
            if flag:
                opti.subject_to(obstacle_matrix @ transformation_matrix @ u_casadi[:, i] >= cbf_vector)
            opti.subject_to(boundary_matrix @ transformation_matrix @ u_casadi[:, i] >= cbf_boundary)
            opti.subject_to(u_casadi[:, i] <= u_limits[:, 0])
            opti.subject_to(u_casadi[:, i] >= l_limits[:, 0])

            transformation_matrix[0, 0] = np.cos(current_casadi[2, i])
            transformation_matrix[0, 1] = - rp.l * np.sin(current_casadi[2, i])
            transformation_matrix[1, 0] = np.sin(current_casadi[2, i])
            transformation_matrix[1, 1] = rp.l * np.cos(current_casadi[2, i])
            
            flag = 0
            for j in range(len(self.obstacle_locations)):
                flag = 1
                obstacle_matrix[j, 0] = 2 * (x_casadi[0, i] - self.obstacle_locations[j][0])
                obstacle_matrix[j, 1] = 2 * (x_casadi[1, i] - self.obstacle_locations[j][1])
                cbf_vector[j, 0] = - rp.alpha * ((x_casadi[0, i] - self.obstacle_locations[j][0]) ** 2 + (x_casadi[1, i] - self.obstacle_locations[j][1]) ** 2 - rp.d ** 2)
            
            boundary_matrix[0, 0] = 9.75 + 0.25 - 2 * x_casadi[0, i]
            boundary_matrix[1, 1] = 9.75 + 0.25 - 2 * x_casadi[1, i]
            cbf_boundary[0, 0] = - rp.alpha * ((9.75 + 0.25) * x_casadi[0, i] - x_casadi[0, i] ** 2 - 0.25 * 9.75)
            cbf_boundary[1, 0] = - rp.alpha * ((9.75 + 0.25) * x_casadi[1, i] - x_casadi[1, i] ** 2 - 0.25 * 9.75)
        
        solver_opts = {'ipopt' : {'print_level' : 0, 'linear_solver' : 'mumps'}}
        opti.solver('ipopt', solver_opts)
        
        sol = opti.solve()
        
        return [sol.value(u_casadi)[0, 0], sol.value(u_casadi)[1, 0]]
        
            
            
            
        
            
            
        
        
        
        
        
        
            