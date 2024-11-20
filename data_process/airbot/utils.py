import json
import os
import random
import logging
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from absl import app, flags, logging
from functools import partial
from multiprocessing import Pool

def squash(path): 
    im = Image.open(path)
    out = np.asarray(im).astype(np.uint8)
    return out

def split_and_categorize_trajectories(data_list):
    categorized_trajectories = defaultdict(list)
    current_traj = []
    current_traj_id = None
    current_category = None  # Track the current category

    for item in tqdm(data_list, desc='split into trajectories'):
        # Split the path by '/' and get relevant segments
        third_image_path_parts = item["third_image"].split('/')
        
        # Assume both paths are similar, so take the fourth-to-last and third-to-last segments
        category = third_image_path_parts[-4]  # e.g., "duck_in_blue_bowl"
        traj_id = third_image_path_parts[-3]   # e.g., "9"
        
        # Combine category and trajectory ID to form a unique trajectory identifier
        full_traj_id = f"{category}/{traj_id}"

        # Check if we are starting a new trajectory
        if full_traj_id != current_traj_id:
            # Append the completed trajectory to the previous category, if any
            if current_traj:
                categorized_trajectories[current_category].append(current_traj)
            # Start a new trajectory
            current_traj = [item]
            current_traj_id = full_traj_id
            current_category = category
        else:
            # Continue with the current trajectory
            current_traj.append(item)

    # Append the last trajectory if it exists
    if current_traj:
        categorized_trajectories[current_category].append(current_traj)

    return categorized_trajectories

def process_dc(task, trajectories, outpath_train, outpath_val, train_proportion):
    
    all_traj = trajectories[task]
    random.shuffle(all_traj)
    num_traj = len(all_traj)
    
    train_ep = 0
    val_ep = 0

    for itraj, tp in tqdm(enumerate(all_traj), desc=f'Processing {task} data'):
        
        episode = []
        
        for step in tp:
            episode.append({
                'image': squash(step['third_image']),
                # 'wrist_image': squash(step['wrist_image']),
                'state': np.array(step['state']).astype(np.float32),
                'action': np.array(step['action'][0]).astype(np.float32),
                'language_instruction': step['instruction'],
            })       
            

        if itraj < int(num_traj * train_proportion):
            file_name = f'{task}_episode_{str(train_ep)}.npy'
            with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, file_name), "wb") as f:
                np.save(f, episode)
            train_ep += 1
        else:
            file_name = f'{task}_episode_{str(val_ep)}.npy'
            with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, file_name), "wb") as f:
                np.save(f, episode)
            val_ep += 1
