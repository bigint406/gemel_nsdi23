# 不同bs下执行一次推理的耗时
from line_profiler import LineProfiler
import torch
import time
from torch.utils.data import DataLoader

from models.model_architectures import resnet101
# 设置随机种子
seed = 42
torch.cuda.set_device(1)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

cpu = torch.device('cpu')
gpu = torch.device('cuda')

def main():
    max_batch_size = 600
    taskname = 'elm_1st_car_truck_train'
    # 加载预处理后的验证集数据
    val_dataset = torch.load(f'test/val_{taskname}.pth')

    model = resnet101(3)
    model.load_state_dict(torch.load('/mnt/data/zs/samba/gemel_nsdi23/test/models/elm_1st_car_truck_train/resnet50_model_epoch_18.pth'))
    model.to(gpu)
    model.eval()

    batch_size = 1
    num_iter = 500
    warmup_iter = num_iter//20
    num_iter += warmup_iter

    data = {}
    
    while batch_size <= max_batch_size:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        it = iter(val_loader)
        for i in range(num_iter):

            with torch.no_grad():
                correct = 0
                total = 0
                try:
                    images, labels = next(it)
                except StopIteration:
                    it = iter(val_loader)
                    images, labels = next(it)
                images, labels = images.to(gpu), labels.to(gpu)
                start = time.time()
                outputs = model(images)
                dt = time.time() - start
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                # accuracy = correct / total

            print(f'BS {len(labels)}/{batch_size}, Iteration {i+1}/{num_iter}, time: {dt:.3g}')

            if i >= warmup_iter:
                bs = len(labels)
                if data.get(bs) is None:
                    data[bs] = {'sum': dt, 'num': 1}
                else:
                    data[bs]['sum'] += dt
                    data[bs]['num'] += 1

        batch_size *= 2

    print("batchsize,time")
    for k, v in data.items():
        print(f"{k},{v['sum']/v['num']:.3g}")


if __name__ == "__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
