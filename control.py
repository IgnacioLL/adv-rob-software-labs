#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import time
from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 100.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    #TODO 

    q_target = trajs[0](tcurrent)
    v_target = trajs[1](tcurrent)

    previous_error_q = np.zeros(q_target.shape)

    errors = (q_target - q)

    torques = []
    for error, error_old in zip(errors, previous_error_q):
        e_d = (error - error_old)/DT
        u = Kp*error + Kv*e_d
        torques.append(u)

    sim.step(torques)

    previous_error_q = errors.copy()

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
    # pkl.dump(path, open("path.pkl", "wb"))
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
    total_time=4
    trajs = maketraj(path, total_time)  

    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    