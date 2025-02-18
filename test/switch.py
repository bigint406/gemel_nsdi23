import torch
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import nvtx
import json
from DeepSpeed.test import resnet101_pal
from models.model_architectures import *
# 设置随机种子
seed = 42
torch.cuda.set_device(1)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

with open('test/config.json', 'r') as f:
    data = json.load(f)

batch_size = 256

tasknames = ['elm_1st_car_truck_train', 'main_2nd_cat_fish']
model_dict = {}

cpu = torch.device('cpu')
gpu = torch.device('cuda')

# Task 1
model_dict[tasknames[0]] = {}
model_dict[tasknames[0]]['unmerged_acc'] = 0.90

# Initialize model structure and load weights
model_elm_1st = resnet101_pal(3, torch.load(data['models_path']['resnet101']['pytorch'])) # 3 classes
# model_elm_1st = resnet101(3) # 3 classes
# model_elm_1st = vgg16(3) # 3 classes
# model_elm_1st.load_state_dict(torch.load(data['models_path']['resnet101']['pytorch']))
# model_elm_1st.load_state_dict(torch.load('/mnt/data/zs/samba/gemel_nsdi23/test/models/vgg_16/elm_1st_car_truck_train/epoch_4.pth'))
model_dict[tasknames[0]]['model'] = model_elm_1st
model_dict[tasknames[0]]['name'] = tasknames[0]
# train_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{tasknames[0]}_CL/train/', transform=transform)
val_dataset = torch.load('test/val_elm_1st_car_truck_train.pth')
# model_dict[tasknames[0]]['train_loader'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model_dict[tasknames[0]]['val_loader'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model_dict[tasknames[0]]['iter'] = iter(model_dict[tasknames[0]]['val_loader'])

# # Task 2
# model_dict[tasknames[1]] = {}
# model_dict[tasknames[1]]['unmerged_acc'] = 0.90

# # Initialize model structure and load weights
# model_main_2nd = resnet50(2) # 2 classes
# model_main_2nd.load_state_dict(torch.load('/mnt/data/zs/samba/gemel_nsdi23/test/models/main_2nd_cat_fish_resnet50/resnet50_model_epoch_4.pth'))
# model_dict[tasknames[1]]['model'] = model_main_2nd
# model_dict[tasknames[1]]['name'] = tasknames[1]
# # train_dataset = datasets.ImageFolder(f'/mnt/data/zs/samba/gemel_nsdi23/dataset_formation/datasets/{tasknames[1]}_CL/train/', transform=transform)
# val_dataset = torch.load('test/val_main_2nd_cat_fish.pth')
# # model_dict[tasknames[1]]['train_loader'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# model_dict[tasknames[1]]['val_loader'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# model_dict[tasknames[1]]['iter'] = iter(model_dict[tasknames[1]]['val_loader'])

num_iter = 3
warmup_iter = 2
num_iter += warmup_iter

# t = time.time()
# for i in range(num_iter):
#     if i == warmup_iter-1:# 预热
#         start = time.time()
#     for task in model_dict.values():
#         model = task['model']

#         nvtx.push_range(f"model {task['name']} loading")
#         model.to(gpu)
#         model.eval()
#         nvtx.pop_range()

#         with torch.no_grad():
#             correct = 0
#             total = 0

#             nvtx.push_range(f"model {task['name']} data loading")
#             try:
#                 images, labels = next(task['iter'])
#             except StopIteration:
#                 task['iter'] = iter(task['val_loader'])
#                 images, labels = next(task['iter'])
#             nvtx.pop_range()

#             nvtx.push_range(f"model {task['name']} data copy")
#             images, labels = images.to(gpu), labels.to(gpu)
#             nvtx.pop_range()

#             nvtx.push_range(f"model {task['name']} inference")
#             outputs = model(images)
#             nvtx.pop_range()

#             nvtx.push_range(f"model {task['name']} validate")
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             accuracy = correct / total
#             nvtx.pop_range()

#             dt = time.time()
#             print(f'Iteration {i+1}/{num_iter}, Validation Accuracy: {100 * accuracy:.2f}%, time: {(dt - t):.2f}')
#             t = dt
#         model.to(cpu)

# print(f'Total time: {(time.time() - start):.2f}')

for task in model_dict.values():
    task['iter'] = iter(task['val_loader'])
    task['model'].to(gpu)

t = time.time()
for i in range(num_iter):
    if i == warmup_iter-1:# 预热
        start = time.time()
    for task in model_dict.values():
        model = task['model']
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            nvtx.push_range(f"{task['name']} data loading")
            try:
                images, labels = next(task['iter'])
            except StopIteration:
                task['iter'] = iter(task['val_loader'])
                images, labels = next(task['iter'])
            nvtx.pop_range()

            nvtx.push_range(f"{task['name']} data copy")
            images, labels = images.to(gpu), labels.to(gpu)
            nvtx.pop_range()

            nvtx.push_range(f"{task['name']} inference")
            outputs = model(images)
            nvtx.pop_range()

            nvtx.push_range(f"{task['name']} validate")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            nvtx.pop_range()

            dt = time.time()
            # print(f'Iteration {i+1}/{num_iter}, Validation Accuracy: {100 * accuracy:.2f}%, time: {(dt - t):.2f}')
            t = dt

print(f'Total time: {(time.time() - start):.2f}')