import torch
from torchvision import models, transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from torch import nn

class DetrWithSeparateBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.backbone(x)
        return self.detection_head(x)

def detrResnet101():
    # 加载DETR模型和预训练权重
    model_name = "facebook/detr-resnet-101"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    return model, processor


def detrResnet50():
    # 加载DETR模型和预训练权重
    model_name = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    return model, processor


def detrResnet50Pipe():
    model_name = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    pipemodel = DetrWithSeparateBackbone(model.model)
    return pipemodel, processor


if __name__ == "__main__":
    model, processor = detrResnet50()
    # 确保模型在评估模式
    model.eval()

    # 加载本地图片
    img_path = '/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/elm_1st_car_truck_train_CL/val/car/ILSVRC2012_val_00000049.JPEG'  # 替换为图片的路径
    image = Image.open(img_path)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 预处理图像
    input_image = transform(image).unsqueeze(0)

    # 将图片输入到模型中进行推理
    with torch.no_grad():
        outputs = model(input_image)

    # 获取推理结果
    target_sizes = torch.tensor([image.size[::-1]])  # [height, width]
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # 可视化检测结果
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 绘制检测框
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        xmin, ymin, xmax, ymax = box.tolist()
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=2, edgecolor='r', facecolor='none'))
        ax.text(xmin, ymin, f"{model.config.id2label[label.item()]}: {score:.2f}",
                color='r', fontsize=12, weight='bold')

    plt.savefig("test_detr.jpg")
