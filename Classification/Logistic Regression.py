from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import make_classification
import numpy as np

# Tạo bộ dữ liệu với 1000 mẫu, 4 đặc trưng và 2 lớp
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_classes=2)


# dữ liệu được random theo cách 365->dữ liệu sẽ được chia theo cùng một cách mỗi khi bạn chạy
X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.1, random_state=36)




class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Khởi tạo weights và bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            # Tính giá trị dự đoán
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

            # Tính gradient
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Cập nhật weights và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Tính giá trị dự đoán
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

        # Chuyển đổi giá trị dự đoán thành nhãn dự đoán
        y_pred_class = np.where(y_pred >= 0.5, 1, 0)

        return y_pred_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

model_2 = LogisticRegression(3)
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test)

sumtrue = 0;
for i in range(len(y_pred2)):
    if y_pred2[i] == Y_test[i]:
        sumtrue += 1

print("% dự đoán đúng: ",sumtrue/len(y_pred2))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()