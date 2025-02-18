import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import asyncio
import time

import uvicorn
from models.model_architectures import *
from torch.utils.data import DataLoader
import torch
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='/mnt/data/zs/samba/gemel_nsdi23/test/merge_batch_test/server.log', level=logging.INFO)
logger.info('Started')

# 请求数据模型
class RequestData(BaseModel):
    input_data: str

# 批量处理参数
batch_size = 16  # 每个批次的请求数量
batch_lock = asyncio.Lock()  # 用于同步批处理
batch_requests = []  # 收集请求数据
batch_responses = {}  # 存储每个请求的响应
batch_event = asyncio.Event()  # 用于通知批量已达数量
batch_finish = asyncio.Event()  # 用于通知批量处理完成

cpu = torch.device('cpu')
gpu = torch.device('cuda')

# 后台任务：批量处理模型推理
async def run_batch_processor():
    global batch_requests, batch_responses
    # 设置随机种子
    seed = 42
    torch.cuda.set_device(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    with open('test/config.json', 'r') as f:
        data = json.load(f)
    model = resnet101(3)
    model.load_state_dict(torch.load(data['models_path']['resnet101']['pytorch']))
    model.to(gpu)
    val_dataset = torch.load('test/val_elm_1st_car_truck_train.pth')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    i = iter(val_loader)
    # 等待批次请求达到指定大小
    logger.info("waiting batch")

    while True:
        try:
            await asyncio.wait_for(batch_event.wait(), 0.2)
            logger.info("batch ready")
        except asyncio.TimeoutError:
            pass

        async with batch_lock:
            # 提取当前批次的请求
            if not batch_requests or len(batch_requests) == 0:
                continue
            elif len(batch_requests) < batch_size:
                logger.info("batch timeout")

            current_batch = batch_requests[:batch_size]
            batch_requests = batch_requests[batch_size:]

            # 如果队列不足一个batch，重置事件
            if not batch_requests or len(batch_requests) < batch_size:
                batch_event.clear()

        # 提取数据并调用模型
        try:
            images, labels = next(i)
        except StopIteration:
            i = iter(val_loader)
            images, labels = next(i)
        images, labels = images.to(gpu), labels.to(gpu)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # inputs = [request.input_data for request in batch_data]
        # results = fake_model(inputs)

        # 将结果分发回每个请求
        for request_id, _ in current_batch:
            batch_responses[request_id] = True

        logger.info("batch finish")
        batch_finish.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("initing")
    asyncio.create_task(run_batch_processor())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: RequestData):
    global batch_requests, batch_responses

    # 当前请求的唯一ID
    request_id = id(request)

    # logger.info(f"request get {request.input_data}")

    # 将请求加入批次队列
    async with batch_lock:
        batch_requests.append((request_id, request))
        batch_responses[request_id] = None  # 占位，等待结果

        # 如果达到批量大小，触发批处理
        
        if len(batch_requests) >= batch_size:
            # logger.info("ready to model")
            batch_event.set()  # 通知处理线程

    # 等待批处理完成
    while batch_responses[request_id] is None:
        await batch_finish.wait()
        batch_finish.clear()

    # 返回该请求的结果
    return batch_responses.pop(request_id)

if __name__ == '__main__':
    uvicorn.run(app_dir="/mnt/data/zs/samba/gemel_nsdi23/test/merge_batch_test", app="server:app", host="0.0.0.0", port=30123, reload=False)