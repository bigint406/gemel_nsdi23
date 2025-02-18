import deepspeed
import time
from models.model_architectures import *
import nvtx
import json
from torch.utils.data import DataLoader

def resnet101_pal(n, param):
    config = {
        "tensor_parallel": {"tp_size": 2},
    }
    model = resnet101(n)
    model.load_state_dict(param)
    engine = deepspeed.init_inference(model=model, config=config)
    return engine.model

if __name__ == '__main__':
    with open('test/config_cluster2gpu.json', 'r') as f:
        data = json.load(f)

    num_iter = 3
    warmup_iter = 2
    batch_size = 32
    num_iter += warmup_iter
    model = resnet101_pal(3, torch.load(data['models_path']['resnet101']['pytorch']))
    model.eval()
    t = time.time()
    val_dataset = torch.load(data['data_path']['val_car_truck_train'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    it = iter(val_loader)
    
    for i in range(num_iter):
        if i == warmup_iter-1:# 预热
            start = time.time()
        with torch.no_grad():
            correct = 0
            total = 0
            
            nvtx.push_range(f"data loading")
            try:
                images, labels = next(it)
            except StopIteration:
                it = iter(val_loader)
                images, labels = next(it)
            nvtx.pop_range()

            # nvtx.push_range(f"{task['name']} data copy")
            # images, labels = images.to(gpu), labels.to(gpu)
            # nvtx.pop_range()

            nvtx.push_range(f"inference")
            outputs = model(images)
            nvtx.pop_range()

            # nvtx.push_range(f"{task['name']} validate")
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # accuracy = correct / total
            # nvtx.pop_range()

            dt = time.time()
            # print(f'Iteration {i+1}/{num_iter}, Validation Accuracy: {100 * accuracy:.2f}%, time: {(dt - t):.2f}')
            t = dt

    print(f'Total time: {(time.time() - start):.2f}')
