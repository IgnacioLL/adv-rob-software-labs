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

from tools import getcubeplacement, setcubeplacement
from setup_meshcat import updatevisuals
import solution


def generate_random_cube(random_rotation=False):
    
    random_Cube = pin.SE3(pin.utils.rotate('z', 0.),np.array([0.0, 0.0, 0.0]))

    if random_rotation:
        random_Cube = random_Cube.Random()
        random_Cube.translation =np.array([0.0, 0.0, 0.0])
    
    rd_delta_translations = np.random.random(3)
    rd_delta_translations[0] = (rd_delta_translations[0] * 0.6) + 0.15
    rd_delta_translations[1] = (rd_delta_translations[1] - 0.5)
    rd_delta_translations[2] = (rd_delta_translations[2]*0.6) + 0.93

    random_Cube.translation = random_Cube.translation + rd_delta_translations
    
    return random_Cube

def euclid_dist(cube_1, cube_2) -> float:
    return np.sqrt(np.sum((cube_2.translation - cube_1.translation)**2))

def get_n_nodes_to_add(max_points, cube_pose_1, cube_pose_2):
    n_nodes_to_add = min(max_points, max(int(euclid_dist(cube_pose_1, cube_pose_2) * 15), 1))

    return n_nodes_to_add

class Node:
    def __init__(self, cube_coordinates, q):
        self.cube_pose = cube_coordinates
        self.q = q
        self.children = {}
        
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
                
        dist = euclid_dist(start_node.cube_pose, end_node.cube_pose) 
        self.nodes[node1_id].children[node2_id] = dist
        
        return True
        
    def find_closest_node(self, random_cube_pose):
        
        min_distance = np.inf
        for i, node in enumerate(self.nodes):
            if i == self.END_NODE_ID:
                continue
            
            distance = euclid_dist(node.cube_pose, random_cube_pose)
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
        

def compute_interpolation(robot, cube, q_start, cube_start_pose, cube_end_pose, n_steps, control) -> bool:

    # Get Quaternion of Rotations, to calculate delta step for rotation
    quat_start = pin.Quaternion(cube_start_pose.rotation)
    quat_end = pin.Quaternion(cube_end_pose.rotation)

    # Get Delta Translation Step
    total_translation = cube_end_pose.translation - cube_start_pose.translation
    delta_translation = total_translation / n_steps
        
    q_cube_interpolated = [(cube_start_pose.copy(), q_start.copy())]
    successful_interpolation = True
    for i in range(1, n_steps+1):
        # Set Cube New Pose
        cube_inter_pose = q_cube_interpolated[-1][0].copy()
        quat_delta = quat_start.slerp(i/n_steps, quat_end)
        cube_inter_pose.rotation = quat_delta.matrix()
        cube_inter_pose.translation += delta_translation
        
        qt, success = computeqgrasppose(robot, q_cube_interpolated[-1][1], cube, cube_inter_pose, viz=None, control=control)
        if success:
            q_cube_interpolated.append((cube_inter_pose.copy(), qt.copy()))
            #q_inter = qt
        else:
            successful_interpolation = False
            break

    return q_cube_interpolated, successful_interpolation

def add_interpolations(g, q_cube_interpolated, start_node_id, success_interpolation, n_nodes_to_add=3, end_node_id=None):
    if not n_nodes_to_add <= len(q_cube_interpolated) - 1:
        n_nodes_to_add = len(q_cube_interpolated) - 1
    
    # Get a list of all the indexex of the q, cube pairs we're adding to the graph
    if success_interpolation:
        idx_list = create_spaced_index(len(q_cube_interpolated), k=n_nodes_to_add)
    else:
        # If interpolation wasn't successful, only add the middle successful interpolated point
        idx_list = [len(q_cube_interpolated) // 2]

    prev_node_id = start_node_id
    for idx in idx_list:
        cube_pose_new_node, q_new_node = q_cube_interpolated[idx]

        if end_node_id and idx == idx_list[-1]:
            # Special condition of adding edge to already existing node, then end node
            g.add_edge(prev_node_id, end_node_id)
        else:
            new_node = Node(cube_pose_new_node, q_new_node)
            new_node_id = g.add_node(new_node)
            g.add_edge(prev_node_id, new_node_id)

            prev_node_id = new_node_id

    return prev_node_id 

def create_path(robot, cube, q0, c0, qe, ce, n_samples=500, n_steps_interpol=20, control=False):
    
    # Instantiate Graph
    start_node = Node(c0, q0)
    g = Node_Graph(start_node)
    end_node = Node(ce, qe)
    end_node_id = g.add_node(end_node)


    available_path = False
    for i in tqdm(range(n_samples), ascii=True, unit='samples'):
        # Sample Cube, and compute if robot can grasp it, from start configuration
        sampled_cube_pose = generate_random_cube()
        qt, success_grasp = computeqgrasppose(
            robot, 
            q0,
            cube, 
            cubetarget=sampled_cube_pose, 
            viz=None, 
            control=control
        )
       
        if success_grasp:
            # Find closest Node in neighbour, and interpolate to see if linear path is doable
            closest_node_id = g.find_closest_node(sampled_cube_pose)
            closest_node = g.nodes[closest_node_id]
            q_cube_interpolated, success_interpolation = compute_interpolation(
                robot, 
                cube, 
                q_start=closest_node.q,
                cube_start_pose=closest_node.cube_pose, 
                cube_end_pose=sampled_cube_pose, 
                n_steps=n_steps_interpol,
                control=control
            )
        
            if len(q_cube_interpolated) > 2:
                n_nodes_to_add = get_n_nodes_to_add(len(q_cube_interpolated)-1, closest_node.cube_pose, sampled_cube_pose)
                last_node_id = add_interpolations(
                    g, 
                    q_cube_interpolated, 
                    closest_node_id,
                    success_interpolation,
                    n_nodes_to_add
                )

                # Try to reach end position from the last new node added
                qs_interpolated_end, success_end = compute_interpolation(
                    robot, 
                    cube, 
                    q_start=q_cube_interpolated[-1][1],
                    cube_start_pose=q_cube_interpolated[-1][0], 
                    cube_end_pose=ce, 
                    n_steps=n_steps_interpol, 
                    control=control
                )
                    
                if success_end:
                    n_nodes_to_add = get_n_nodes_to_add(len(qs_interpolated_end)-1, sampled_cube_pose, ce)
                    add_interpolations(
                        g, 
                        qs_interpolated_end, 
                        last_node_id, 
                        success_interpolation,
                        n_nodes_to_add, 
                        end_node_id=end_node_id
                    )
                    #solution.plot_2d_points(graph=g)
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

        for children_id, children_distance in g.nodes[current_node_id].children.items():
            new_distance = current_distance + children_distance
            if new_distance < distances[children_id]:
                distances[children_id] = new_distance
                previous_nodes[children_id] = current_node_id
                heapq.heappush(priority_queue, (new_distance, children_id))

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

    extra_args = {'n_samples': 250}
    path, _ = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, **extra_args)

    setcubeplacement(robot, cube, CUBE_PLACEMENT_TARGET)

    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
