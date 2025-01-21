export CUDA_VISIBLE_DEVICES=0,1,2,3


cd data_process/airbot
sudo -E /home/amax/anaconda3/envs/openvla/bin/python build_airbot_bridge.py \
--origin_data_json='/data/rsp_data/newair_rel_eef_25_0120_rsp.json' \
--output_path='/data/rsp_data/airbot_tf' \
--num_workers=2 \
--train_proportion=0.99 \


