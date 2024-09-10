from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('dataset.csv')
df_x = df.iloc[:, 0:9].astype(float)
df_y = df.iloc[:, 9].astype(float)
# dữ liệu được random theo cách 365->dữ liệu sẽ được chia theo cùng một cách mỗi khi bạn chạy
X, X_test, Y, Y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=365)

import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(n_samples, n_features)

        # Add a column of ones for the intercept term
        X = np.column_stack((np.ones(n_samples), X))
        print(X.shape)
        print(X)
        # Initialize coefficients
        self.coef_ = np.zeros(n_features + 1)
        print(self.coef_)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            prev_coef = np.copy(self.coef_)
            print(prev_coef)

            for j in range(n_features + 1):
                X_j = X[:, j]
                X_not_j = np.delete(X, j, axis=1)
                #print(X.shape)
                y_pred = self.predict(X_not_j)

                c_j = 2 * np.dot(X_j, y - y_pred) / n_samples
                d_j = np.sum(X_j ** 2) / n_samples

                # Soft thresholding function
                if c_j < -self.alpha:
                    self.coef_[j] = (c_j + self.alpha) / d_j
                elif c_j > self.alpha:
                    self.coef_[j] = (c_j - self.alpha) / d_j
                else:
                    self.coef_[j] = 0

            # Update intercept
            self.intercept_ = self.coef_[0]

            # Check convergence
            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break

    def predict(self, X):
        n_samples, n_features = X.shape

        # Add a column of ones for the intercept term
        X = np.column_stack((np.ones(n_samples), X))

        return np.dot(X, self.coef_) + self.intercept_


model_2 = LassoRegression(0.00001, 20, 0.2)
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test.values)
print(y_pred2)
print("% Sai lệch trung bình: ",(sum(abs((y_pred2/Y_test-1)))/len(Y_test)))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()