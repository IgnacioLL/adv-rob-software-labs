#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: Ignacio Lloret and Hypolite 
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
from tqdm import tqdm
import math


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


def create_spaced_index(length, k):
    length = length - 1
    
    step_size = length / k
    
    cumulate = 0
    indexes = []
    for i in range(k):
        cumulate += step_size
        if math.ceil(cumulate) < length:
            indexes.append(math.ceil(cumulate))
    
    indexes.append(length)
    indexes = list(set(indexes))
    indexes.sort()
    return indexes
        

def compute_interpolation(robot, cube, q, cube_root, cube_leaf, n_steps, control) -> bool:

    quat_root = pin.Quaternion(cube_root.rotation)
    quat_leaf = pin.Quaternion(cube_leaf.rotation)

    total_translation = cube_leaf.translation - cube_root.translation
    delta_translation = total_translation / n_steps
    
    cube_interim = cube_root.copy()
    q_interim = q.copy()
    
    qs_interpolated = [(cube_interim.copy(), q_interim.copy())]
    for i in range(1, n_steps+1):
        quat_delta = quat_root.slerp(i/n_steps, quat_leaf)
        cube_interim.rotation = quat_delta.matrix()
        cube_interim.translation += delta_translation
        
        qt, success = computeqgrasppose(robot, q_interim, cube, cube_interim, viz=None, control=control)
 
        if success:
            qs_interpolated.append((cube_interim.copy(), qt.copy()))
            q_interim = qt
        else:
            return np.array([]), False

    return qs_interpolated, True

def add_interpolations(g, qs_interpolated, closest_node_id, n_interpolations=1, end_node_id=None):
    
    if not n_interpolations <= len(qs_interpolated) - 1:
        n_interpolations = len(qs_interpolated) - 1

    prev_node_id = closest_node_id
    
    idx_list = create_spaced_index(len(qs_interpolated), k=n_interpolations)
    
    for idx in idx_list:
        if end_node_id and idx == idx_list[-1]:
            g.add_edge(prev_node_id, end_node_id)
        else:
            new_node = Node(qs_interpolated[idx][0], qs_interpolated[idx][1])
            new_node_id = g.add_node(new_node)
            g.add_edge(prev_node_id, new_node_id)
            prev_node_id = new_node_id

    return prev_node_id 

def create_path(robot, cube, q0, c0, qe, ce, n_samples=500, n_steps_interpol=20, n_steps_graph_interpolations=10, control=False):
    #setcubeplacement(robot, cube, c0)
    #updatevisuals(viz, robot, cube, q=q0)
    
    assert n_steps_graph_interpolations <= n_steps_interpol, "The number of interpolations added to the graph must be lower than the number of interpolations computed"

    start_node = Node(c0, q0)
    g = Node_Graph(start_node)
    end_node = Node(ce, qe)
    end_node_id = g.add_node(end_node)

    prev_q = q0.copy()
    available_path = False
    print("Creating sample path: ")
    for _ in tqdm(range(n_samples), ascii=True, unit='samples'):
        # maybe make a better randomizer how this is done 
        sampled_cube = generate_random_cube()
        qt, success_grasp = computeqgrasppose(robot, q0, cube, sampled_cube, viz=None, control=control)
       
        if success_grasp:
            closest_node_id = g.find_closest_neigbour(sampled_cube)
            closest_node = g.nodes[closest_node_id]
            qs_interpolated, success_edge = compute_interpolation(robot, cube, closest_node.q, closest_node.cube, sampled_cube, 
                                                                  n_steps=n_steps_interpol, control=control)
        
            if success_edge:
                last_node_id = add_interpolations(g, qs_interpolated, closest_node_id, n_steps_graph_interpolations)
                qs_interpolated_end, success_end = compute_interpolation(robot, cube, qs_interpolated[-1][1], qs_interpolated[-1][0], ce, 
                                                                         n_steps=n_steps_interpol, control=control)
                
                if success_end:
                    add_interpolations(g, qs_interpolated_end, last_node_id, n_steps_graph_interpolations, end_node_id=end_node_id)
                    available_path = True
        
    return g, available_path

def shortest_path(g: Node_Graph):
    TARGET_NODE_ID = 1
    
    num_nodes = len(g.nodes)
    distances = {i: np.inf for i in range(num_nodes)}
    distances[g.START_NODE_ID] = 0
    previous_nodes = {i: None for i in range(num_nodes)}

    priority_queue = [(0, g.START_NODE_ID)]  # (distance, node_id)
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node_id = heapq.heappop(priority_queue)

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
            new_distance = current_distance + neighbour_distance
            if new_distance < distances[neighbour_id]:
                distances[neighbour_id] = new_distance
                previous_nodes[neighbour_id] = current_node_id
                heapq.heappush(priority_queue, (new_distance, neighbour_id))

    return None, np.inf

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, cube, qinit,qgoal,cubeplacementq0, cubeplacementqgoal, control=False, **kwargs):
    graph, success = create_path(
        robot,
        cube,
        q0=qinit,
        c0=cubeplacementq0,
        qe=qgoal,
        ce=cubeplacementqgoal,
        control=control,
        **kwargs
    )

    if not success:
        return np.array([]), 0
    
    path, length = shortest_path(graph)
    
    return [graph.nodes[id].q for id in path], length

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

    extra_args = {'n_samples': 300, 'n_steps_graph_interpolations': 5}
    path, _ = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, **extra_args)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
