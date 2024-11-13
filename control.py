#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import time
from bezier import Bezier
from tools import setcubeplacement, getcubeplacement
from config import CUBE_PLACEMENT_TARGET
    
# in my solution these gains were good enough for all joints but you might want to tune this.
KP = 8000 # proportional gain (P of PD)
KV = 10   # derivative gain (D of PD)

BEZIER_REDUNDANCY = 2

def controllaw(sim, robot, trajs, tcurrent, cube, Kp=15_000, Kv=1000, q_previous_error=None, v_previous_error=None):
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

        u = Kp*q_error + Kv*e_d 
        torques.append(u)

    sim.step(torques)

    return q_errors.copy(), v_errors.copy()

def maketraj(path_points,T):
    q_of_t = Bezier(path_points,t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    return q_of_t, vq_of_t, vvq_of_t

    
if __name__ == "__main__":
        
    from tools import setupwithpybullet, rununtil
    from config import DT, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath
    import pickle as pkl
    import uuid
    import os
    import random

    robot, sim, cube = setupwithpybullet()
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)

    if successinit and successend:
        sim.setqsim(q0)

        extra_args = {'n_samples': 250, 'n_nodes_to_add':5}

        length = 0
        tries = 0
        while length < 3 and tries < 3:
            path, _ = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, control=True, **extra_args)
            length = len(path)
            tries +=1

        # Create redundancy in BEZIER
        new_path = [p for p in path for _ in range(BEZIER_REDUNDANCY)]

        total_time=4
        trajs = maketraj(new_path, total_time)  

        tcur = 0.
        q_errors, v_errors = None, None
        while tcur < total_time:
            q_errors, v_errors = rununtil(controllaw, DT, sim, robot, trajs, tcur, cube, KP, KV, q_errors, v_errors)
            tcur += DT
    else:
        print("Without successfull grasp in start or end cube position")
                

        
    
    