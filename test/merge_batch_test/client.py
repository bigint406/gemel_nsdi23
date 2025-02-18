import numpy as np
import threading
import time
import requests
import matplotlib.pyplot as plt

BS = 16 # 一次性发多少请求，用于gamma分布
REQ_PER_SEC = 100 # 每秒多少请求
latencies = []  # 存储请求的延迟
url = 'http://localhost:30123/predict'  # 请替换为实际的服务端URL

# 请求处理函数
def process_request():
    start_time = time.time()
    response = requests.post(url, json={"input_data":"123"})
    end_time = time.time()
    latencies.append(end_time - start_time)

# 主函数
if __name__ == '__main__':
    # distribution = 'gamma'
    distribution = 'poisson'
    num_req = BS*50 # 总共的请求数量

    st = []

    while num_req > 0:
        req_num = 1
        if distribution == "poisson": # 请求为泊松过程，时间间隔为指数分布
            interarrival_time = np.random.exponential(1/REQ_PER_SEC)
            time.sleep(interarrival_time)
        elif distribution == "gamma": # 时间间隔为gamma分布
            req_num = min(BS, num_req)
            interarrival_time = np.random.gamma(req_num, 1/REQ_PER_SEC)
            time.sleep(interarrival_time)
        
        # 提交请求事件
        for _ in range (req_num):
            sub_thread = threading.Thread(target=process_request)
            st.append(sub_thread)
            sub_thread.start()

        num_req -= req_num

    for t in st:
        t.join()
    # 统计请求延迟
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)

    print(f"Avg Latency: {avg_latency} seconds")
    print(f"Max Latency: {max_latency} seconds")
    print(f"Min Latency: {min_latency} seconds")

    with open("/mnt/data/zs/samba/gemel_nsdi23/test/merge_batch_test/latency.txt","w") as f:
        f.write(str(latencies))

    # 计算CDF
    sorted_latencies = np.sort(latencies)
    cumulative_prob = np.linspace(0, 1, len(sorted_latencies))

    # 绘制CDF
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_latencies, cumulative_prob, label='CDF')
    plt.xlabel('Latency')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF) of Latencies')
    plt.grid(True)
    plt.legend()
    plt.savefig('test/merge_batch_test/cdf_plot.png')  # 保存为图片文件
    plt.close()

    # 绘制直方图
    plt.figure(figsize=(12, 6))
    plt.hist(latencies, bins=30, density=True, alpha=0.75, color='b', edgecolor='black')
    plt.xlabel('Latency')
    plt.ylabel('Density')
    plt.title('Histogram of Latencies')
    plt.grid(True)
    plt.savefig('test/merge_batch_test/histogram_plot.png')  # 保存为图片文件
    plt.close()
