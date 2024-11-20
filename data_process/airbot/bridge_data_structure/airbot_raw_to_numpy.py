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


def process_images(traj):  # processes images at a trajectory level
    
    images_out = defaultdict(list)
    tlen = len(traj)
    
    demo = traj[0]
    
    img_name = ['images0', 'images1', 'images2']
    
    if 'wrist_image' in demo:
        names = ['third_image', 'wrist_image']
    else:
        names = ['third_image']

    for i, name in enumerate(names):
        for t in range(tlen):
            images_out[img_name[i]].append(squash(traj[t][name]))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in images_out.keys():
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs

def process_actions(traj):
    action = [np.array(i['action']) for i in traj]
    return action[:-1]

def process_state(traj):
    prob = [np.array(i['state']) for i in traj]
    state = prob[:-1]
    next_state = prob[1:]
    return state, next_state

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

def process_dc(trajectories, task, train_ratio=0.9):
    all_traj = trajectories[task]
    random.shuffle(all_traj)
    num_traj = len(all_traj)
    all_dicts_train = list()
    all_dicts_test = list()
    for itraj, tp in tqdm(enumerate(all_traj), desc=f'Processing {task} data'):
        try:
            out = dict()

            obs, next_obs = process_images(tp)
            acts = process_actions(tp)
            state, next_state = process_state(tp)
            
            term = [0] * len(acts)
            term[-1] = 1
            lang = tp[0]['instruction']

            out["observations"] = obs
            out["observations"]["state"] = state
            out["next_observations"] = next_obs
            out["next_observations"]["state"] = next_state

            out["observations"] = [
                dict(zip(out["observations"], t))
                for t in zip(*out["observations"].values())
            ]
            out["next_observations"] = [
                dict(zip(out["next_observations"], t))
                for t in zip(*out["next_observations"].values())
            ]

            out["actions"] = acts
            out["terminals"] = term
            out["language"] = lang

            traj_len = len(out["observations"])
            assert len(out["next_observations"]) == traj_len
            assert len(out["actions"]) == traj_len
            assert len(out["terminals"]) == traj_len

            if itraj < int(num_traj * train_ratio):
                all_dicts_train.append(out)
            else:
                all_dicts_test.append(out)
                
        except FileNotFoundError as e:
            logging.error(e)
            continue
        except AssertionError as e:
            logging.error(e)
            continue

    return all_dicts_train, all_dicts_test
    
def make_numpy(task, trajectories, output_dir, train_proportion):
    outpath = tf.io.gfile.join(output_dir, "numpy")
    outpath =  tf.io.gfile.join(outpath, str(task))

    if os.path.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)
    
    train, val = process_dc(
        trajectories, task, train_ratio=train_proportion
    )
        
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, train)
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, val)

FLAGS = flags.FLAGS

flags.DEFINE_string("origin_data_json", 
                    '/home/dodo/zyn/octo/finetune_data/data_process/AIR-bot/AIR-toykitchen-v3.json', 
                    "Input path")
flags.DEFINE_string("output_path", '/home/dodo/zyn/octo/finetune_data/data/airbot-test', "Output path")

flags.DEFINE_bool("overwrite", True, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")


def main(argv):
    with open(FLAGS.origin_data_json, 'r') as file:
        trajs = json.load(file)
        
    trajectories = split_and_categorize_trajectories(trajs)
    
    tasks = trajectories.keys()
    worker_fn = partial(make_numpy, trajectories=trajectories, output_dir=FLAGS.output_path, train_proportion=FLAGS.train_proportion)
    with Pool(FLAGS.num_workers) as p:
        list(tqdm(p.imap(worker_fn, tasks), total=len(tasks)))
    # for task in tqdm(tasks):
    #     make_numpy(trajectories, task, FLAGS.output_path, train_proportion=FLAGS.train_proportion)
    

if __name__ == "__main__":
    app.run(main)