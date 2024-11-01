python gl-ncplr_train.py -b 256 -a resnet50_neighbor -d market1501 --momentum 0.1 --eps 0.5 --num-instances 16 --height 320 \
    --use-refinelabels --ce-option 1 --eps-neighbor 0.2 --alpha 0.1 --cmalpha 0.3 --topk-s 0 --topk-spart 0 \
    --use-part --use-mixdistneighbbor --eps-partneighbor 0.2 --lambda1 0.2 --lambda2 0.2 \
    --extra-option 2 --temp-KLlogits 5.0 --temp-KLloss 5.0 \
    --logs logs/market1501/gl-ncplr


