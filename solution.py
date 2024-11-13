#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:43:26 2023

@author: stonneau
"""

from os.path import dirname, join, abspath
import numpy as np
import pinocchio as pin #the pinocchio library
from pinocchio.utils import rotate

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import matplotlib.pyplot as plt
import uuid

#helpers 
#if needed, you can store the placement of the right hand in the left hand frame here
LMRREF = pin.SE3(pin.Quaternion(-0,0, 0, 1 ), np.array(np.array([0, 0, 0])))
RMLREF = LMRREF.inverse()


def plot_2d_points(graph):

    # Define the nodes with their coordinates and neighbors
    # Initialize a figure
    plt.figure(figsize=(16, 12))

    # start cube position
    xyz_start = CUBE_PLACEMENT.translation
    plt.plot(xyz_start[1], xyz_start[2], 'x', markersize=20)
    # end cube position
    xyz_end = CUBE_PLACEMENT_TARGET.translation
    plt.plot(xyz_end[1], xyz_end[2], 'x', markersize=20)
    # obstacle position
    xyz_end = [0.43,-0.1,0.94]
    plt.plot(xyz_end[1], xyz_end[2], 'x', markersize=50)


    # Plot the nodes
    for node_id, node in enumerate(graph.nodes):
        x, y, z = node.cube_pose.translation
        
        plt.plot(y, z, 'o', markersize=10)  # label=f'Node {node_id}'
        #plt.text(x + 0.1, y + 0.1, f'{node_id}', fontsize=12)  # Label the node

    # Plot the directed edges
    for node_id, node in enumerate(graph.nodes):
        x_start, y_start, z_start = node.cube_pose.translation
        
        for neighbour_id in node.children.keys():
            # if node_id in sp and neighbour_id in sp:
            #     viz_edges = {'ec': 'red', 'linewidth': 1.0}
            # else:
            #     viz_edges = {'ec': 'black', 'linewidth': 0.5}
            viz_edges = {'ec': 'black', 'linewidth': 0.5}

            x_end, y_end, z_end = graph.nodes[neighbour_id].cube_pose.translation
            
            # Draw an arrow for directed edge
            plt.arrow(
                y_start, 
                z_start, 
                y_end - y_start, 
                z_end - z_start, 
                length_includes_head=True,
                fc='gray', 
                linestyle='-', 
                **viz_edges
            ) # head_width=0.2,  

    # Display the plot
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Graph Visualization with Directed Edges")
    plt.legend()
    plt.grid(True)
    #plt.savefig('plots/2d_coordinates_' + str(len(graph.nodes)) + '_' + str(uuid.uuid4()) + '.png')
    plt.show()