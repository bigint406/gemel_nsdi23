import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# 示例数据：一维数据点
data = np.array([1, 2, 3, 4, 8, 9, 10])

# 使用 'ward' 方法计算层次聚类
Z = linkage(data.reshape(-1, 1), method='ward')

# 设置最大距离（阈值）
max_distance = 2

# 根据最大距离构建聚类，返回每个点的簇编号
clusters = fcluster(Z, t=max_distance, criterion='distance')

# 打印聚类结果
print("聚类结果:", clusters)

# 可视化聚类结果
plt.scatter(data, np.zeros_like(data), c=clusters, cmap='viridis')
plt.title("Hierarchical Clustering with Distance Constraint (1D Data)")
plt.xlabel('Data Points')
plt.yticks([])  # 一维数据，y轴没有实际意义

# plt.savefig("cluster.jpg")