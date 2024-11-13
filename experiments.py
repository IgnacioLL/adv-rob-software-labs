from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import optuna


from tools import setupwithmeshcat, setupwithpybullet, rununtil, getcubeplacement
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT
from inverse_geometry import computeqgrasppose
from path import computepath
from control import controllaw, maketraj

import itertools
import time

import argparse


def check_path_hyperparameters():
    ## Check path.py and how hyperparameters vary
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")


    n_samples = [100, 250, 500, 1000]
    graph_interpolations = [1, 2, 3, 5]
    try_extra_args = [
        {'n_samples': n, 'n_steps_graph_interpolations': graph_interpolations}
        for n, graph_interpolations in itertools.product(n_samples, graph_interpolations)
        ]


    n_samples = []
    graph_interpolations = []
    execution_times = []
    path_length = []
    euclidean_distances = []
    for extra_args in try_extra_args:
        start_time = time.time()
        path, length = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, **extra_args)
        execution_time = time.time() - start_time

        
        n_samples.append(extra_args['n_samples'])
        graph_interpolations.append(extra_args['n_steps_graph_interpolations'])
        execution_times.append(execution_time)
        path_length.append(len(path))
        euclidean_distances.append(length)


    columns_names = ['n_samples', 'graph_interpolations', 'execution_times', 'path_length', 'euclidean_distances']
    data = pd.DataFrame([n_samples, graph_interpolations, execution_times, path_length, euclidean_distances], index=columns_names).T

    data.to_csv("experiments/experiments.csv", index=False)

    sns.lineplot(data=data, y='execution_times', x='n_samples', hue='graph_interpolations')
    plt.savefig("experiments/execution_times.png")

    sns.lineplot(data=data, y='path_length', x='n_samples', hue='graph_interpolations')
    plt.savefig("experiments/path_length.png")

    sns.lineplot(data=data, y='euclidean_distances', x='n_samples', hue='graph_interpolations')
    plt.savefig("experiments/euclidean_distances.png")


import time

def optimize(trial):
    retries = 3  
    attempt = 0
    success = False
    total_error = float('inf') 

    while attempt < retries and not success:
        try:
            robot, sim, cube = setupwithpybullet()

            q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
            qe, successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)

            if not successinit or not successend:
                raise ValueError("Failed to compute grasps")

            sim.setqsim(q0)

            # extra_args = {'n_samples': 250, 'n_steps_graph_interpolations': 5}
            # length = 0
            # while length < 3:
            #     path, _ = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, control=True, **extra_args)
            #     length = len(path)

            import os
            import random
            import pickle as pkl

            paths = os.listdir("path/")
            rand = random.randint(0, len(paths)-1)
            path = pkl.load(open(f"path/{paths[rand]}", "rb"))

            # Suggest values for the hyperparameters we want to tune
            bezier_redundancy = trial.suggest_int('BEZIER_REDUNDANCY', 1, 3) 
            kp = trial.suggest_float('KP', 100, 10_000, log=True)  
            kv = trial.suggest_float('KV', 20, 1_000, log=True)  

            # Recreate the path with redundancy
            new_path = [p for p in path for _ in range(bezier_redundancy)]
            total_time = 4
            trajs = maketraj(new_path, total_time)

            # Initialize variables for simulation
            tcur = 0.0
            q_errors, v_errors = None, None
            total_error = 0.0

            # Run the simulation loop
            while tcur < total_time:
                q_errors, v_errors = rununtil(controllaw, DT, sim, robot, trajs, tcur, cube, kp, kv, q_errors, v_errors)
                # Calculate error metric (adjust based on the desired outcome)
                total_error += sum(abs(err) for err in q_errors)  # Example: sum of absolute errors
                print(q_errors)
                tcur += DT

            success = True  
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(1) 

    if not success:
        print("Optimization failed after several attempts.")
        return float('inf')  
    
    # Return the total error if successful
    return total_error


def check_control_hyperparameters():
    study = optuna.create_study(direction="minimize")
    study.optimize(optimize, n_trials=20)  # Adjust n_trials as needed
    study_results = study.trials_dataframe()
    study_results.to_csv("experiments/control_experiment.csv", index=False)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter checks.")
    parser.add_argument(
        "--path", 
        action="store_true", 
        default=False,
        help="Run check for path hyperparameters (default: False)"
    )
    parser.add_argument(
        "--control", 
        action="store_true", 
        default=False,
        help="Run check for control hyperparameters (default: False)"
    )

    args = parser.parse_args()

    if args.path:
        check_path_hyperparameters()
    if args.control:
        check_control_hyperparameters()