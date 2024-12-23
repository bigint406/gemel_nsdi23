from models.model_architectures import *
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


taskname = "elm_1st_car_truck_train"

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 64
val_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{taskname}_CL/val/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

torch.cuda.set_device(2)
model = resnet101(3)
model.load_state_dict(torch.load("/mnt/data/zs/samba/gemel_nsdi23/test/models/elm_1st_car_truck_train/resnet50_model_epoch_17.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'预训练的ResNet-50在Imagenet-1k验证集上的准确率为: {accuracy:.4f}%')