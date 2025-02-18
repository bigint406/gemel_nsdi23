from models.model_architectures import *
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

taskname = "elm_1st_car_truck_train"
modelname = "vgg_16"
if not os.path.exists(f'test/models/{modelname}/{taskname}'):
    os.makedirs(f'test/models/{modelname}/{taskname}')
if not os.path.exists(f'test/img/{modelname}'):
    os.makedirs(f'test/img/{modelname}')
if not os.path.exists(f'test/logs/{modelname}'):
    os.makedirs(f'test/logs/{modelname}')

# 设置随机种子
seed = 42
torch.cuda.set_device(1)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{taskname}_CL/train/', transform=transform)
val_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{taskname}_CL/val/', transform=transform)

# 定义 DataLoader
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的ResNet-50模型
model = vgg16(3)
# model.load_state_dict(torch.load("/mnt/data/zs/samba/gemel_nsdi23/test/models/main_2nd_cat_fish/resnet50_model_epoch_9.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
# for param in model.fc.parameters():
    param.requires_grad = True
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

losses = []
accuracies = []

# 训练模型
num_epochs = 5
for epoch in range(0, num_epochs):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    # 在验证集上测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {100 * accuracy:.2f}%')

    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)

    # if (epoch+1) % 10 == 0:
    torch.save(model.state_dict(), f'test/models/{modelname}/{taskname}/epoch_{epoch}.pth')

# 绘制损失和准确率曲线
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(1, num_epochs+1), losses, color=color, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(range(1, num_epochs+1), accuracies, color=color, label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training Loss and Validation Accuracy')
plt.savefig(f'test/img/{modelname}/{taskname}.png')

with open(f"test/logs/{modelname}/{taskname}.log", 'w') as f:
    f.write(str(losses))
    f.write("\n")
    f.write(str(accuracies))