# train
python main.py ctdet --exp_id kitti_dla --dataset kitti --batch_size 16 --num_epochs 70 --lr_step 45,60 --gpus 0,1
# test
python test.py ctdet --exp_id kitti_dla --dataset kitti --resume --flip_test
