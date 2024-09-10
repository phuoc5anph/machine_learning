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

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.zeros(n_samples, dtype=int)  # Khởi tạo nhãn ban đầu

        cluster_id = 1
        for i in range(n_samples):
            if self.labels[i] != 0:  # Đã được gán nhãn trước đó
                continue

            neighbors = self._get_neighbors(X, i)
            if len(neighbors) < self.min_samples:  # Không đủ điểm lân cận
                self.labels[i] = -1  # Gán nhãn là điểm nhiễu (noise point)
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _get_neighbors(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        self.labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.labels[neighbor] == -1:  # Điểm nhiễu, gán vào cụm
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:  # Chưa được gán nhãn
                self.labels[neighbor] = cluster_id
                new_neighbors = self._get_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            j += 1


model = DBSCAN(2, 3)
model.fit(points)
# In kết quả
print("Labels:", model.labels)

# Tạo danh sách màu sắc cho các nhãn
colors = ['r', 'g', 'b']

# Tạo subplot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vẽ biểu đồ 3D
for i, label in enumerate(np.unique(model.labels)):
    print(np.unique(model.labels))
    # Lấy chỉ mục của các điểm thuộc cùng một nhãn
    indices = np.where(model.labels == label)[0]
    # Lấy tập điểm thuộc cùng một nhãn
    cluster_points = points[indices]
    # Vẽ các điểm thuộc cùng một nhãn với màu tương ứng
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label}')

# Đặt nhãn cho các trục
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Đặt tiêu đề và chú thích cho biểu đồ
ax.set_title('Clustering Result')
ax.legend()

# Hiển thị biểu đồ 3D
plt.show()






