# 测试在一张卡上跑多个模型的效率
import sys
sys.path.append('.')
import time
import torch
from torch.utils.data import DataLoader
from models.model_architectures import *
import torch.multiprocessing as mp
import json

def run_model(task, deviceID):
    seed = 42
    torch.cuda.set_device(deviceID)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    batch_size = 2
    num_iter = 1000
    warmup_iter = 10
    num_iter += warmup_iter
    
    device = torch.device(f'cuda:{deviceID}')

    val_dataset = torch.load(task['val_dataset'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    it = iter(val_loader)

    model = task['model']
    task_name = task['name']
    model.to(device)
    model.eval()
    
    for i in range(num_iter):
        if i == warmup_iter - 1:
            start = time.time()

        with torch.no_grad():
            correct = 0
            total = 0

            try:
                images, labels = next(it)
            except StopIteration:
                it = iter(val_loader)
                images, labels = next(it)

            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = correct / total

            if i % 20 == 0:
                print(f'Model: {task_name}, Iter {i}/{num_iter}, Validation Accuracy: {100 * accuracy:.2f}%')
    
    dt = time.time() - start
    print(f'Model finish: {task_name}, total time: {dt:.3g}')

# 创建两个线程分别运行两个模型

if __name__ == '__main__':
    device = 1
    with open('test/config_cluster2gpu.json', 'r') as f:
        data = json.load(f)

    tasknames = ['elm_1st_car_truck_train', 'main_2nd_cat_fish']
    model_dict = {}

    torch.cuda.set_device(device)
    
    model_dict[tasknames[0]] = {}
    model_elm_1st = resnet101(3) # 3 classes
    model_elm_1st.load_state_dict(torch.load(data['models_path']['resnet101']['pytorch'], weights_only=True, map_location=torch.device(f'cuda:{device}')))
    model_dict[tasknames[0]]['model'] = model_elm_1st
    model_dict[tasknames[0]]['name'] = tasknames[0]
    model_dict[tasknames[0]]['val_dataset'] = data['data_path']['val_car_truck_train']

    model_dict[tasknames[1]] = {}
    model_main_2nd = resnet50(2) # 2 classes
    model_main_2nd.load_state_dict(torch.load(data['models_path']['resnet50']['pytorch'], weights_only=True, map_location=torch.device(f'cuda:{device}')))
    model_dict[tasknames[1]]['model'] = model_main_2nd
    model_dict[tasknames[1]]['name'] = tasknames[1]
    model_dict[tasknames[1]]['val_dataset'] = data['data_path']['val_cat_fish']

    mp.set_start_method('spawn')

    processes = []
    for task in model_dict.values():
        p = mp.Process(target=run_model, args=(task,device), daemon=True)
        processes.append(p)

    # 启动进程
    for p in processes:
        p.start()
        
    # 等待两个进程结束
    for p in processes:
        p.join()