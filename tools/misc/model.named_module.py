from mmdet.apis import init_detector

# 初始化模型
config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
model = init_detector(config, checkpoint)

# 打印所有模块名称和类型
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")