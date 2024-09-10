import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def pairwise_euclidean_distance(self, X):
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])  # tính khoảng cách Euclidean
        return distance_matrix

    def calculate_perplexity(self, distances, perplexity=30.0, tol=1e-5):
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            beta = 1.0
            betamin = -np.inf
            betamax = np.inf
            Di = distances[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n_samples)))]
            H, thisP = 0.0, 0.0
            while np.abs(H - perplexity) > tol:
                sumP = np.sum(np.exp(-Di * beta))
                thisP = np.exp(-Di * beta) / sumP
                H = np.log(sumP) + beta * np.sum(Di * thisP)
                if H > perplexity:
                    betamin = beta
                    if betamax == np.inf or betamax == -np.inf:
                        beta *= 2.0
                    else:
                        beta = (beta + betamax) / 2.0
                else:
                    betamax = beta
                    if betamin == np.inf or betamin == -np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + betamin) / 2.0
            P[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n_samples)))] = thisP
        return P

    def calculate_Q(self, Y):
        n_samples = Y.shape[0]
        Q = np.zeros((n_samples, n_samples))
        distances = self.pairwise_euclidean_distance(Y)
        inv_distances = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv_distances, 0.0)
        sum_inv_distances = np.sum(inv_distances)
        Q = inv_distances / sum_inv_distances
        return Q

    def fit_transform(self, X):
        n_samples, n_features = X.shape

        # Khởi tạo Y (dữ liệu đầu ra) với giá trị ngẫu nhiên
        Y = np.random.randn(n_samples, self.n_components)

        # Huấn luyện t-SNE
        for i in range(self.n_iter):
            distances = self.pairwise_euclidean_distance(Y)
            P = self.calculate_perplexity(distances, self.perplexity)
            Q = self.calculate_Q(Y)
            grad = np.zeros((n_samples, self.n_components))
            for j in range(n_samples):
                grad[j] = 4.0 * np.dot((P[j] - Q[j]) * distances[j], Y[j] - Y)
            Y -= self.learning_rate * grad

        return Y


df = pd.read_csv("Iris.csv")
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
# Example usage
X = X.astype(float)
Y = df["Species"].to_numpy()
tsne = TSNE(n_components=3, perplexity=30, learning_rate=0.1, n_iter=30)

new_X = tsne.fit_transform(X)
for label in set(Y):
    X_class = new_X[Y == label]
    plt.scatter(X_class[:, 0], X_class[:, 1], label=label)

plt.legend()
plt.savefig('SNE.png')
