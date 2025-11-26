CUDA_VISIBLE_DEVICES=1 python tools/train.py rebuttal/Ablation_AAP/2.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/distill/mmdet/IRSTD-reti50-18/mse+hcl.py

CUDA_VISIBLE_DEVICES=2 python tools/model_converters/convert_kd_ckpt_to_student.py work_dirs/nuaa-sirst101-50-mykd/epoch_600.pth --out-path checkpoint/pr


python tools/visualizations/vis_tsne.py rebuttal/scale8/mse+hcl.py \
    --checkpoint rebuttal/scale8/epoch_400.pth --vis-stage neck \
    --cfg-options model.neck.out_indices=[0,1,2,3] \
    --max-num-class 1 \
    --max-num-samples 100 \
    --perplexity 30 \
    --n-iter 1000 \
    --device cuda:0