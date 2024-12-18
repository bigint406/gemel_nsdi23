from models.model_architectures import resnet50
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子
seed = 42
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
original_dataset = datasets.ImageFolder('/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/main_2nd_cat_fish_CL/val/', transform=transform)
# 切分数据集为训练集和验证集
train_size = 0.8
train_indices, val_indices = train_test_split(list(range(len(original_dataset))), train_size=train_size, random_state=seed)

train_dataset = torch.utils.data.Subset(original_dataset, train_indices)
val_dataset = torch.utils.data.Subset(original_dataset, val_indices)

# 定义 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的ResNet-50模型
model = resnet50(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
accuracies = []

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
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
        # print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {100 * accuracy:.2f}%')

    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'test/models/resnet50_model_epoch_{epoch}.pth')

# 绘制损失和准确率曲线
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
plt.savefig('test/img/training_plot.png')

with open("test/logs/retrain.log", 'w') as f:
    f.write(str(losses))
    f.write("\n")
    f.write(str(accuracies))