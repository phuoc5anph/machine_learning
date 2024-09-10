import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PCA:
    def __init__(self, n_dimention: int):
        self.n_dimention = n_dimention

    def fit_transform(self, X):
        mean = np.mean(X, axis=0)
        X = X - mean
        cov = X.T.dot(X) / X.shape[0]
        #tri rieng, vector rieng
        eigen_values, eigen_vectors, = np.linalg.eig(cov)
        select_index = np.argsort(eigen_values)[::-1][:self.n_dimention]
        U = eigen_vectors[:, select_index]
        print(cov)
        print(eigen_values)
        print(eigen_vectors)
        print(cov.dot(eigen_vectors.T[0]))
        print(eigen_values[0]*eigen_vectors.T[0])
        X_new = X.dot(U)
        return X_new


if __name__ == "__main__":
    df = pd.read_csv("Iris.csv")
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
    Y = df["Species"].to_numpy()

    pca = PCA(n_dimention=2)
    new_X = pca.fit_transform(X)

    for label in set(Y):
        X_class = new_X[Y == label]
        plt.scatter(X_class[:, 0], X_class[:, 1], label=label)

    plt.legend()
    plt.savefig('PCA.png')

