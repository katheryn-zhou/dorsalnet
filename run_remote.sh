#!/bin/bash
set -e

# Somehow, the path is not correctly set - I don't totally get it
PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

case $DATASET in
    "crcns-pvc1")
        dataset_num=0
        ;;
    "crcns-pvc4")
        dataset_num=1
        ;;
    "crcns-mt1")
        dataset_num=2
        ;;
    "crcns-mt2")
        dataset_num=3
        ;;
    "packlab-mst")
        dataset_num=4
        ;;
    *)
        echo "Unknown dataset"
        exit 0;
esac

datasets=(mt1_norm_neutralbg mt2 pvc1-repeats pvc4 mst_norm_neutralbg)
max_cells=(83 43 22 24 35)
dataset=${datasets[$dataset_num]}
max_cell=${max_cells[$dataset_num]}

pip install -r requirements.txt
aws s3 sync "s3://yourheadisthere/data_derived/$DATASET" "/data/data_derived/$DATASET"

# Not sure if actually necessary.
chown -R nobody:nogroup /data
chown -R nobody:nogroup /cache

size=8
ckpt_root=/data/checkpoints
data_root=/data/data_derived
cache_root=/cache
slowfast_root=../slowfast

models=(airsim_04 MotionNet)
dataset=mt2
max_cell=43

for model in "${models[@]}";
do
    echo "$dataset" "$model"
    for ((subset = 0; subset <= $max_cell; subset++))
    do
        echo "Fitting cell $subset"
        python train_convex.py \
            --exp_name mt_boosting_revision \
            --dataset "$dataset" \
            --features "$model" \
            --subset "$subset" \
            --batch_size 8 \
            --cache_root $cache_root \
            --ckpt_root $ckpt_root \
            --data_root $data_root \
            --slowfast_root $slowfast_root \
            --aggregator downsample \
            --aggregator_sz $size \
            --skip_existing \
            --subsample_layers \
            --autotune \
            --no_save \
            --save_predictions \
            --method boosting

        # Clear cache.
        rm -f $cache_root/*
    done
done