# 不同bs下执行一次推理的耗时
import sys
sys.path.append('.')
import torch
import time
import json
from torch.utils.data import DataLoader
import nvtx

from models.model_architectures import *
from DETR.resnet import detrResnet50, detrResnet101
# 设置随机种子

def main():
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    deviceID = 0
    cpu = torch.device('cpu')
    gpu = torch.device(f'cuda:{deviceID}')
    
    with open('test/config.json', 'r') as f:
        data = json.load(f)
    max_batch_size = 256
    # 加载预处理后的验证集数据
    val_dataset = torch.load(data['data_path']['val_car_truck_train'])

    model, processor = detrResnet50()
    # model = resnet101(3)
    # model.load_state_dict(torch.load(data['models_path']['resnet101']['pytorch'], weights_only=True, map_location=gpu))
    model.to(gpu)
    model.eval()

    batch_size = 16
    num_iter = 50
    warmup_iter = num_iter//20
    num_iter += warmup_iter

    data = {}
    
    while batch_size <= max_batch_size:
        nvtx.push_range(str(batch_size))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        it = iter(val_loader)
        for i in range(num_iter):

            with torch.no_grad():
                try:
                    images, labels = next(it)
                except StopIteration:
                    it = iter(val_loader)
                    images, labels = next(it)
                images, labels = images.to(gpu), labels.to(gpu)

                start = time.time()
                outputs = model(images, return_dict=False)

                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                # accuracy = correct / total
                dt = time.time() - start

            # print(f'BS {len(labels)}/{batch_size}, Iteration {i+1}/{num_iter}, accuracy {accuracy:.3g}, time: {dt:.3g}')

            if i >= warmup_iter:
                bs = len(labels)
                if data.get(bs) is None:
                    data[bs] = {'sum': dt, 'num': 1}
                else:
                    data[bs]['sum'] += dt
                    data[bs]['num'] += 1

        batch_size *= 2
        nvtx.pop_range()

    print("batchsize,time")
    for k, v in data.items():
        print(f"{k},{v['sum']/v['num']:.3g}")


if __name__ == "__main__":
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()
    main()
