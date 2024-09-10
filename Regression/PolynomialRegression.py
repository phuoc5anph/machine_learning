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

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coef_ = None

    def compute_features(self, X):
        features = []
        for i in range(1, self.degree + 1):
            features.append(X ** i)
        return np.column_stack(features)

    def fit(self, X, y):
        features = self.compute_features(X)

        # Tính toán các hệ số bằng công thức của hồi quy tuyến tính
        self.coef_ = np.linalg.inv(features.T.dot(features)).dot(features.T).dot(y)

    def predict(self, X):
        features = self.compute_features(X)
        return np.dot(features, self.coef_)


model_2 = PolynomialRegression(1)
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test.values)
print(y_pred2)
print("% Sai lệch trung bình: ",(sum(abs((y_pred2/Y_test-1)))/len(Y_test)))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()