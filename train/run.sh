# ReFine
python refine_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python refine_train.py --dataset mnist --hid 50 --epoch 200 --ratio 0.2 --lr 1e-4
python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3
python refine_train.py --dataset vg --hid 250 --epoch 100 --ratio 0.2 --lr 1e-3

# PGExplainer
python pg_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python pg_train.py --dataset mnist --hid 50 --epoch 200 --ratio 0.2 --lr 1e-4
python pg_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3
python pg_train.py --dataset vg --hid 250 --epoch 100 --ratio 0.2 --lr 1e-3