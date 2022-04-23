#python3 start_offline.py --batch 64 --epoch 80 --dor 0.6 --decay 1e-7 --lr 9e-4 --k 30 --lift 30 --state_dim 4 --control_dim 1 --validate_k 1400 --lr_decay 0.99
python3 start_online.py --batch 20 --epoch 80 --dor 0.8 --decay 1e-8 --lr 2e-4 --k 30 --lift 30 --state_dim 4 --control_dim 1 --validate_k 1400 --lr_decay 0.99
