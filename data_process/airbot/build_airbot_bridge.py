import tensorflow_datasets as tfds
import os
from absl import app, flags, logging
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from utils import process_bridge_tf, split_and_categorize_trajectories
import tensorflow as tf

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

FLAGS = flags.FLAGS

flags.DEFINE_string("origin_data_json", 
                    '/data/rsp_data/newair_rel_eef_25_0120_rsp.json', 
                    "Input path")
flags.DEFINE_string("output_path", '/data/rsp_data', "Output path")

flags.DEFINE_bool("overwrite", True, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.99, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 16, "Number of threads to use")

def main(argv):
    '''
    Compare to OXE data, bridge data directly output tfrecord
    '''
    with open(FLAGS.origin_data_json, 'r') as file:
        trajs = json.load(file)
    trajectories = split_and_categorize_trajectories(trajs)
    
    outpath = FLAGS.output_path
    if os.path.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return
        
    outpath = tf.io.gfile.join(outpath, "tfrecord")
    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)
    
    tasks = trajectories.keys()

    for task in tqdm(tasks):
        process_bridge_tf(task, trajectories=trajectories, outpath_train=outpath_train, outpath_val=outpath_val, train_proportion=FLAGS.train_proportion)

    # worker_fn = partial(process_bridge_tf, trajectories=trajectories, outpath_train=outpath_train, outpath_val=outpath_val, train_proportion=FLAGS.train_proportion)
    # with Pool(FLAGS.num_workers) as p:
    #     list(tqdm(p.imap(worker_fn, tasks), total=len(tasks)))

if __name__ == "__main__":
    app.run(main)
