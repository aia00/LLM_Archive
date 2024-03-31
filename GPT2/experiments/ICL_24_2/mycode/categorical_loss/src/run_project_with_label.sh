#!/bin/bash
source activate ykwang_ICL || conda activate ykwang_ICL
conda deactivate
source activate ykwang_ICL || conda activate ykwang_ICL
cd /home/guanting/yikai/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/src
export CUDA_VISIBLE_DEVICES="0,1"
python train.py --config conf/multiple_task_with_label.yaml