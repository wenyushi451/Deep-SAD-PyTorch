python3 main.py customize vgg ../log/DeepSAD/customize ../../../data/20200803/pure_white/prepared/train --ratio_known_normal 0.99 --ratio_known_outlier 0 --ratio_pollution 0.0 --lr 0.0001 --n_epochs 50 --lr_milestone 50 --batch_size 32 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 0 --ae_batch_size 32 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1
