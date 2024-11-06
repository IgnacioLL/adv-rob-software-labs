#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import time
from bezier import Bezier
from tools import setcubeplacement
from config import CUBE_PLACEMENT_TARGET
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 15_000. # proportional gain (P of PD)
Kv = 3_000   # derivative gain (D of PD)

v_Kp = 0
v_Kv = 0


q_previous_error = None
v_previous_error = None

def controllaw(sim, robot, trajs, tcurrent, cube):

    global q_previous_error, v_previous_error
    
    q, vq = sim.getpybulletstate()

    q_target = trajs[0](tcurrent)
    v_target = trajs[1](tcurrent)
    
    # Initialize previous errors to 0.
    if q_previous_error is None and v_previous_error is None:
        q_previous_error = np.zeros(q_target.shape)
        v_previous_error = np.zeros(v_target.shape)

    q_errors = (q_target - q)
    q_errors_d = (q_errors - q_previous_error)
    
    v_errors = (v_target - vq)
    v_errors_d = (v_errors - v_previous_error)

    torques = []
    for q_error, e_d, v_error, v_error_d in zip(q_errors, q_errors_d, v_errors, v_errors_d):

        u = Kp*q_error + Kv*e_d + v_Kp*v_error + v_Kv*v_error_d
        torques.append(u)

    sim.step(torques)

    v_previous_error = v_errors.copy()
    q_previous_error = q_errors.copy()

    

 

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    # path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    import pickle as pkl
    path = pkl.load(open("path.pkl", "rb"))

    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    def maketraj(path_points,T): #TODO compute a real trajectory !
        q_of_t = Bezier(path_points,t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    path.insert(0, q0)
    path.append(qe)
    total_time=10
    trajs = maketraj(path, total_time)  

    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    