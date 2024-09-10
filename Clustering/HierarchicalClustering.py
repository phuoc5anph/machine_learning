import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Thiết lập số điểm dữ liệu và số cụm
n_points = 100
n_clusters = 3

# Tạo ngẫu nhiên các tâm cụm
centers = np.random.rand(n_clusters, 3) * 10

# Tạo dữ liệu dựa trên các tâm cụm
points = np.zeros((n_points, 3))
labels = np.zeros(n_points, dtype=int)


for i in range(n_points):
    # Chọn ngẫu nhiên một tâm cụm
    cluster_idx = np.random.choice(range(n_clusters))

    # Tạo điểm dữ liệu quanh tâm cụm
    point = centers[cluster_idx] + np.random.randn(3)

    points[i] = point
    labels[i] = cluster_idx

# Trực quan hóa dữ liệu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm dữ liệu theo từng cụm
for cluster_idx in range(n_clusters):
    cluster_points = points[labels == cluster_idx]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_idx + 1}')

# Đặt tên cho các trục
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Hiển thị biểu đồ
plt.legend()
plt.show()
class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = []
        self.labels = None

    def fit(self, X):
        # Khởi tạo mỗi điểm dữ liệu là một cụm đơn lẻ ban đầu
        self.clusters = [[x] for x in X]
        while len(self.clusters) > self.n_clusters:
            # Tìm hai cụm gần nhau nhất để gộp
            min_dist = float('inf')
            merge_indices = None
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = self.distance(self.clusters[i], self.clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_indices = (i, j)

            # Gộp hai cụm gần nhau nhất thành một cụm mới
            i, j = merge_indices
            merged_cluster = self.clusters[i] + self.clusters[j]
            del self.clusters[j]
            del self.clusters[i]
            self.clusters.append(merged_cluster)

        # Gán nhãn cho các điểm dữ liệu
        self.labels = np.zeros(len(X), dtype=int)
        for i, cluster in enumerate(self.clusters):
            for sample_index in cluster:
                self.labels[sample_index.astype(int)] = i

    def distance(self, cluster1, cluster2):
        # Tính toán khoảng cách giữa hai cụm
        # Có thể sử dụng khoảng cách Euclidean, Manhattan, cosine, ...
        # Trong ví dụ này, sử dụng khoảng cách Euclidean
        centroid1 = np.mean(cluster1, axis=0)
        centroid2 = np.mean(cluster2, axis=0)
        return np.linalg.norm(centroid1 - centroid2)

model = HierarchicalClustering(n_clusters = 3)
model.fit(points)

# Trực quan hóa dữ liệu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Vẽ các điểm dữ liệu theo từng cụm
for cluster_idx in range(model.n_clusters):
    cluster_points = model.clusters[cluster_idx]
    print(cluster_points)
    X = np.array([arr[0] for arr in cluster_points])
    Y = np.array([arr[1] for arr in cluster_points])
    Z = np.array([arr[2] for arr in cluster_points])
    ax.scatter(X, Y, Z, label=f'Cluster {model.labels[cluster_idx] + 1}')

# Đặt tên cho các trục
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Hiển thị biểu đồ
plt.legend()
plt.show()
