python gl-ncplr_test.py -a resnet50_neighbor -d market1501 --logs logs/market1501/ncplr --resume checkpoints/market1501/ncplr

python gl-ncplr_test.py -a resnet50_neighbor -d market1501 --logs logs/market1501/gl-ncplr --resume checkpoints/market1501/gl-ncplr

python gl-ncplr_test.py -a resnet50_neighbor -d msmt17 --logs logs/msmt17/ncplr --resume checkpoints/msmt17/ncplr

python gl-ncplr_test.py -a resnet50_neighbor -d msmt17 --logs logs/msmt17/gl-ncplr --resume checkpoints/msmt17/gl-ncplr