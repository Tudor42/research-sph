#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/dataset_db.sh >> db_dataset.out 2>&1 &

DATA_ROOT=./lagrangebench_data

##### 2D dataset
for seed in {18..21} 
do
    echo "Run with seed = $seed"
    python main_jax_sph.py config=cases/ft2d.yaml seed=$seed case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/
    python main_jax_sph.py config=cases/ft2d.yaml seed=$seed case.state0_path=$DATA_ROOT/relaxed/ft2d_2_0.02_$seed.h5 io.data_path=$DATA_ROOT/raw/FT2D_every250/
done

#python gen_dataset.py --src_dir=$DATA_ROOT/raw/FT2D_every250/ --dst_dir=$DATA_ROOT/datasets/FT2D_every250/ --split=2_1_1
