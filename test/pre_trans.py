# 预处理验证集合数据，防止每次都花好久现做预处理

import torch
from torchvision import transforms, datasets

def pre_trans():
    transform = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    taskname = 'elm_1st_car_truck_train'

    val_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{taskname}_CL/val/')

    # 预处理数据集并保存为文件
    preprocessed_data = []

    for data in val_dataset:
        inputs, labels = data
        preprocessed_inputs = transform(inputs)
        preprocessed_data.append((preprocessed_inputs, labels))

    # 将预处理后的数据保存为文件
    torch.save(preprocessed_data, f'test/val_{taskname}.pth')

if __name__ == "__main__":
    pre_trans()