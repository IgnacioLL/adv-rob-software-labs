#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON,  CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, PULL_CUBE

from tools import setcubeplacement, jointlimitscost, distanceToObstacle

from scipy.optimize import fmin_bfgs,fmin_slsqp
from numpy.linalg import norm,inv,pinv,svd,eig
import time

def cost_v1(q, robot, target_left, target_right, lh_frameid, rh_frameid):
    #eff = endeffector(q)
    pin.framesForwardKinematics(robot.model,robot.data,q)
    
    eff_lh = robot.data.oMf[lh_frameid]
    eff_rh = robot.data.oMf[rh_frameid]
    
    cost_lh = norm(eff_lh.np - target_left.np)**2
    cost_rh = norm(eff_rh.np - target_right.np)**2

    cost_d = cost_lh + cost_rh

    q_epsilon = (robot.model.upperPositionLimit - robot.model.lowerPositionLimit) / 100
    cost_joint_limits = jointlimitscost(robot, q - q_epsilon) + jointlimitscost(robot, q + q_epsilon)
    
    return cost_d + cost_joint_limits

def callback(q):
    #updatevisuals(viz, robot, cube, q)
    #time.sleep(.5)
    pass

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None, tol=0.0001, control=False):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    target_left = getcubeplacement(cube, LEFT_HOOK)
    target_right = getcubeplacement(cube, RIGHT_HOOK)

    lh_frameid = robot.model.getFrameId('LARM_EFF')
    rh_frameid = robot.model.getFrameId('RARM_EFF')

    
    qopt_bfgs = fmin_bfgs(
        cost_v1, 
        qcurrent, 
        callback=callback, 
        args=(robot, target_left, target_right, lh_frameid,rh_frameid),
        full_output=True,
        disp=False
    )

    tolerable_error = False
    if qopt_bfgs[1] < tol and not collision(robot, qopt_bfgs[0]) and not jointlimitsviolated(robot, qopt_bfgs[0]):
        tolerable_error = True
        if control:
            grip_adjustment = (target_right.translation - target_left.translation)*PULL_CUBE
            target_right.translation = target_right.translation - grip_adjustment
            target_left.translation = target_left.translation + grip_adjustment

            qopt_bfgs = fmin_bfgs(
                cost_v1, 
                qcurrent, 
                callback=callback, 
                args=(robot, target_left, target_right, lh_frameid,rh_frameid),
                full_output=True,
                disp=False
            )
        
    return qopt_bfgs[0], tolerable_error
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    updatevisuals(viz, robot, cube, q0)

    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    updatevisuals(viz, robot, cube, qe)

        
