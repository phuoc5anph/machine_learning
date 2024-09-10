from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import make_classification
import numpy as np

# Tạo bộ dữ liệu với 1000 mẫu, 4 đặc trưng và 2 lớp
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_classes=2)
y[y == 0] = -1

# dữ liệu được random theo cách 365->dữ liệu sẽ được chia theo cùng một cách mỗi khi bạn chạy
X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.1, random_state=36)
class SupportVectorMachine:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_samples, self.num_features = X.shape

        # Khởi tạo các tham số
        self.weights = np.zeros(self.num_features)
        self.bias = 0

        # Huấn luyện SVM
        for _ in range(self.num_iterations):
            for i in range(self.num_samples):
                if self.y[i] * self.predict(self.X[i]) < 1:
                    self.update_parameters(i)

        print(self.weights)
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

    def update_parameters(self, i):
        self.weights += self.learning_rate * ((self.y[i] * self.X[i]) - (2 * (1 / self.num_iterations) * self.weights))
        self.bias += self.learning_rate * (self.y[i])


model = SupportVectorMachine()
model.fit(X, Y)

y_pred2 = model.predict(X_test)

sumtrue = 0;
for i in range(len(y_pred2)):
    if y_pred2[i] == Y_test[i]:
        sumtrue += 1

print("% dự đoán đúng: ",sumtrue/len(y_pred2))

sb.regplot(x=model.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()