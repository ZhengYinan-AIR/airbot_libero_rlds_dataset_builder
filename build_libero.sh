cd data_process/libero
sudo -E /home/dodo/miniconda3/envs/octo/bin/python build_libero.py \
--origin_data_json='/home/dodo/ljx/BearRobot/data/libero/libero130_no-op-ac.json' \
--output_path='/data2/rlds_finetune_data/libero' \
--train_proportion=0.99 \


