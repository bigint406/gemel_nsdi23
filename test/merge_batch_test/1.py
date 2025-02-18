import numpy as np
import matplotlib.pyplot as plt

# 示例数据，用实际数据替换
latencies = np.random.normal(loc=0, scale=1, size=1000)

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
plt.savefig('cdf_plot.png')  # 保存为图片文件
plt.close()

# 绘制直方图
plt.figure(figsize=(12, 6))
plt.hist(latencies, bins=30, density=True, alpha=0.75, color='b', edgecolor='black')
plt.xlabel('Latency')
plt.ylabel('Density')
plt.title('Histogram of Latencies')
plt.grid(True)
plt.savefig('histogram_plot.png')  # 保存为图片文件
plt.close()