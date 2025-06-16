###################################################################
# File Name: eval.sh
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Wed Mar 31 15:51:59 2021
###################################################################
#!/bin/bash

EXP_NAME=$1
CFG_PATH=config/${EXP_NAME}.yaml
SMRY_ROOT=results/summary/${EXP_NAME}
mkdir -p $SMRY_ROOT
# CUDA_VISIBLE_DEVICES=$2 python -u test/test_sot_siamfc.py --record_inout --config $CFG_PATH | tee results/summary/${EXP_NAME}/sot_siamfc.log 2>&1 
#CUDA_VISIBLE_DEVICES=$2 python -u test/test_sot_cfnet.py --config $CFG_PATH | tee results/summary/${EXP_NAME}/sot_cfnet.log 2>&1
# CUDA_VISIBLE_DEVICES=$2 python -u test/test_vos.py --config $CFG_PATH | tee results/summary/${EXP_NAME}/vos.log 2>&1
# CUDA_VISIBLE_DEVICES=$2 python -u test/test_vos.py --config $CFG_PATH --QANNPath /home/kang_you/SpikeZIP_transformer/output/T-SNN_resnet50_imagenet_relu_QANN_QAT_act16_weightbit32/checkpoint-29.pth \
#     --level 16 --weight_quantization_bit 32 --time_step 32 --record_inout --encoding_type analog --log_dir /home/kang_you/UniTrack-main/output | tee results/summary/${EXP_NAME}/vos.log 2>&1
# CUDA_VISIBLE_DEVICES=$2 python -u test/test_mot.py --config $CFG_PATH | tee results/summary/${EXP_NAME}/mot.log 2>&1
CUDA_VISIBLE_DEVICES=$2 python -u test/test_mot.py --config $CFG_PATH --QANNPath /home/kang_you/SpikeZIP_transformer/output/T-SNN_resnet50_imagenet_relu_QANN_QAT_act16_weightbit32/checkpoint-90.pth \
    --level 16 --weight_quantization_bit 32 --time_step 16 --record_inout --encoding_type analog --log_dir /home/kang_you/UniTrack-main/output | tee results/summary/${EXP_NAME}/mot-T=16.log 2>&1
#CUDA_VISIBLE_DEVICES=$2 python -u test/test_mots.py --config $CFG_PATH | tee  results/summary/${EXP_NAME}/mots.log 2>&1
#CUDA_VISIBLE_DEVICES=$2 python -u test/test_posetrack.py --config $CFG_PATH | tee results/summary/${EXP_NAME}/posetrack.log 2>&1
