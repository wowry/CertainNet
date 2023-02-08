# train
python main.py ctdet --exp_id kitti_certainnet --dataset kitti --arch certainnet_34 --ablation 6 --batch_size 16 --num_epochs 80 --lr_step 45,60 --wh_weight 0.2 --gpus 0,1
# test
python test.py ctdet --exp_id kitti_certainnet --dataset kitti --arch certainnet_34 --ablation 6 --resume
