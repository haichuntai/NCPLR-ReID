CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py --logs logs/market

CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d personx --eps 0.7 --rampup-value 0.8 --eps-neighbor-gap 0.1 --logs logs/personx

CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d dukemtmcreid --eps 0.7  --eps-neighbor 0.3 --logs logs/dukemtmcreid

CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d msmt17 --iters 400 --eps 0.7 --eps-neighbor 0.3 --logs logs/msmt17

CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d veri --iters 400 --eps 0.6 --height 224 --width 224 --eps-neighbor 0.3 --logs logs/veri