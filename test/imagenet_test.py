from models.model_architectures import resnet50
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

classes_to_merge = {
    'fish': ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041'],
    'cat': ['n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052'],
    'car': ['n02814533', 'n02930766', 'n03100240', 'n03930630'],
    'truck': ['n03417042', 'n03796401', 'n04461696', 'n04467665'],
    'train': ['n02917067', 'n03272562', 'n03393912', 'n03895866']
}

merging_classes_mapping = {}

for k,v in classes_to_merge.items():
    for i in v:
        merging_classes_mapping[i] = k

# model = resnet50(2)


# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载Imagenet-1k验证集
original_dataset = datasets.ImageFolder('/mnt/data/zs/samba/datasets/imagenet-1k/val', transform=transform)

# 将需要合并的类别合并成一个新类别
merged_class_samples = []
for idx, (img_path, label) in enumerate(original_dataset.samples):
    class_name = original_dataset.classes[label]
    if class_name in classes_to_merge:
        new_label = len(original_dataset.classes)  # 创建一个新类别
    else:
        new_label = label
    merged_class_samples.append((img_path, new_label))

# 定义新的数据集
merged_dataset = torch.utils.data.Subset(original_dataset, indices=range(len(merged_class_samples)))
merged_dataset.samples = merged_class_samples



val_loader = DataLoader(original_dataset, batch_size=64, shuffle=False)

# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=True)
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