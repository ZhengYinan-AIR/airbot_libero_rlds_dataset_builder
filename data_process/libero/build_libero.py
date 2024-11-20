import tensorflow_datasets as tfds
from libero_dataset_dataset_builder import LiberoDataset
import os
from absl import app, flags, logging
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from utils import process_dc, split_and_categorize_trajectories
import tensorflow as tf

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

FLAGS = flags.FLAGS

flags.DEFINE_string("origin_data_json", 
                    './libero130-ac.json', 
                    "Input path")
flags.DEFINE_string("output_path", '/data2/rlds_finetune_data/libero', "Output path")

flags.DEFINE_bool("overwrite", True, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 16, "Number of threads to use")

def main(argv):
    '''
    STAGE 1: JSON TO NUMPY
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
        
    outpath = tf.io.gfile.join(outpath, "numpy")
    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)
    
    tasks = trajectories.keys()
    worker_fn = partial(process_dc, trajectories=trajectories, outpath_train=outpath_train, outpath_val=outpath_val, train_proportion=FLAGS.train_proportion)
    with Pool(FLAGS.num_workers) as p:
        list(tqdm(p.imap(worker_fn, tasks), total=len(tasks)))
    # debug
    # for task in tqdm(tasks):
    #     make_numpy(task, trajectories, FLAGS.output_path, train_proportion=FLAGS.train_proportion)

    '''
    STAGE 2: NUMPY TO TF
    '''
    builder = LiberoDataset(data_dir=FLAGS.output_path)
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=outpath))

if __name__ == "__main__":
    app.run(main)
