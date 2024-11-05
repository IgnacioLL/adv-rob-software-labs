#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time

import heapq
from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

def generate_random_cube():
    
    random_Cube = pin.SE3(pin.utils.rotate('z', 0.),np.array([0.33, 0.0, 0.93]))
    # For Random Rotation
    #random_Cube = random_Cube.Random()
    #random_Cube.translation = np.array([0.33, 0.0, 0.93])
    
    rd_delta_translations = np.random.random(3)
    rd_delta_translations[0] = (rd_delta_translations[0] - 0.5)
    rd_delta_translations[1] = (rd_delta_translations[1] - 0.5)
    rd_delta_translations[2] = (rd_delta_translations[2])

    random_Cube.translation = random_Cube.translation + rd_delta_translations
    
    return random_Cube

def euclid_dist(cube_1, cube_2) -> float:
    return np.sqrt(np.sum((cube_2.translation - cube_1.translation)**2))

class Node:
    def __init__(self, cube_coordinates, q):
        self.cube = cube_coordinates
        self.q = q
        self.neighbours = {}
        
class Node_Graph:
    
    def __init__(self, root_node):
        self.root_node = root_node
        self.nodes = []
        
        self.START_NODE_ID = self.add_node(self.root_node)
        self.END_NODE_ID = 1
        
    def add_node(self, new_node):
        self.nodes.append(new_node)
        
        return len(self.nodes) - 1
        
    def add_edge(self, node1_id, node2_id):
        
        start_node = self.nodes[node1_id]
        end_node = self.nodes[node2_id]
                
        dist = euclid_dist(start_node.cube, end_node.cube) 
        self.nodes[node1_id].neighbours[node2_id] = dist
        
        return True
        
    def find_closest_neigbour(self, random_cube):
        
        min_distance = np.inf
        for i, node in enumerate(self.nodes):
            if i == self.END_NODE_ID:
                continue
            
            distance = euclid_dist(node.cube, random_cube)
            if  distance < min_distance:
                min_distance = distance
                nearest_node_id = i
                
        return nearest_node_id

def is_edge_valid(q, cube_root, cube_leaf, n_steps) -> bool:

    quat_root = pin.Quaternion(cube_root.rotation)
    quat_leaf = pin.Quaternion(cube_leaf.rotation)

    total_translation = cube_leaf.translation - cube_root.translation
    delta_translation = total_translation / n_steps

    cube_interim = cube_root.copy()
    q_interim = q.copy()
    for i in range(n_steps):
        quat_delta = quat_root.slerp(i/n_steps, quat_leaf)
        cube_interim.rotation = quat_delta.matrix()
        
        cube_interim.translation += delta_translation
        
        qt, success = computeqgrasppose(robot, q_interim, cube, cube_interim, viz=None)
        
        #setcubeplacement(robot, cube, cube_interim)
        #updatevisuals(viz, robot, cube, q=qt)
 
        if success:
            q_interim = qt
            pass
        else:
            return False
        
    return True

def create_path(N, q0, c0, qe, ce, n_steps_interpol=20):
    #setcubeplacement(robot, cube, c0)
    #updatevisuals(viz, robot, cube, q=q0)
    
    start_node = Node(c0, q0)
    g = Node_Graph(start_node)
    end_node = Node(ce, qe)
    end_node_id = g.add_node(end_node)

    prev_q = q0.copy()
    available_path = False
    for _ in range(N):
        # maybe make a better randomizer how this is done 
        sampled_cube = generate_random_cube()
        
        qt, success_grasp = computeqgrasppose(robot, q0, cube, sampled_cube, viz=None)
        if success_grasp:
            closest_node_id = g.find_closest_neigbour(sampled_cube)
            closest_node = g.nodes[closest_node_id]

            valid_edge = is_edge_valid(closest_node.q, closest_node.cube, sampled_cube, n_steps=n_steps_interpol)
            
            if valid_edge:
                #print('Valid Path')
                new_node = Node(sampled_cube, qt)
                
                new_node_id = g.add_node(new_node)
                g.add_edge(closest_node_id, new_node_id)
                
                valid_finish = is_edge_valid(qt, sampled_cube, ce, n_steps=n_steps_interpol)
                if valid_finish:
                    g.add_edge(new_node_id, end_node_id)
                    available_path = True
        
    return g, available_path

def shortest_path(g):
    TARGET_NODE_ID = 1
    
    num_nodes = len(g.nodes)
    distances = {i: np.inf for i in range(num_nodes)}
    distances[g.START_NODE_ID] = 0
    previous_nodes = {i: None for i in range(num_nodes)}

    priority_queue = [(0, g.START_NODE_ID)]  # (distance, node_id)
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node_id = heapq.heappop(priority_queue)
        #print(current_node_id, round(current_distance, 2))

        if current_node_id == TARGET_NODE_ID:
            
            # Trace back the path from target to start node
            path = []
            while current_node_id is not None:
                path.append(current_node_id)
                current_node_id = previous_nodes[current_node_id]
            return path[::-1], distances[TARGET_NODE_ID]

        if current_distance > distances[current_node_id]:
            continue

        for neighbour_id, neighbour_distance in g.nodes[current_node_id].neighbours.items():
            #print('\t', neighbour_id, round(neighbour_distance, 2))
            new_distance = current_distance + neighbour_distance
            if new_distance < distances[neighbour_id]:
                distances[neighbour_id] = new_distance
                previous_nodes[neighbour_id] = current_node_id
                heapq.heappush(priority_queue, (new_distance, neighbour_id))

    return None, np.inf

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    graph, success = create_path(
        N=500,
        q0=qinit,
        c0=cubeplacementq0,
        qe=qgoal,
        ce=cubeplacementqgoal,
        n_steps_interpol=15
    )

    if success:
        path, length = shortest_path(graph)
    else:
        return np.array([])
    
    path_qs = []
    for id in path:
        path_qs.append(graph.nodes[id].q)

    return path_qs

def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
