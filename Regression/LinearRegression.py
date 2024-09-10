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

# Tính toán các tham số w
class LinearRegression_2:
    def __init__(self, learning_rate=0.00001, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Khởi tạo trọng số và bias ban đầu
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            # Tính giá trị dự đoán
            y_pred = np.dot(X, self.weights) + self.bias

            # Tính gradient của hàm mất mát
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Cập nhật trọng số và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

model_2 = LinearRegression_2()
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test.values)
print(y_pred2)
print("% Sai lệch trung bình: ",(sum(abs((y_pred2/Y_test-1)))/len(Y_test)))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()