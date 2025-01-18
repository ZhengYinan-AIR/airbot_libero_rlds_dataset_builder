cd data_process/libero
sudo -E /home/amax/anaconda3/envs/openvla/bin/python build_libero_bridge.py \
--origin_data_json='/data/libero/libero_10-ac.json' \
--output_path='/data/libero/libero_10' \
--num_workers=2 \
--train_proportion=0.99 \


