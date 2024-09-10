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

class RidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha  # Tham số alpha (điều chuẩn)

    def fit(self, X, y):
        # Thêm cột 1 vào ma trận X để tính toán bias
        #X = np.column_stack((np.ones(len(X)), X))

        n_features = X.shape[1]
        eye = np.eye(n_features)  # Ma trận đơn vị

        # Tính toán các hệ số bằng công thức của Ridge Regression
        self.coef_ = np.linalg.inv(X.T.dot(X) + self.alpha * eye).dot(X.T).dot(y)

    def predict(self, X):
        # Thêm cột 1 vào ma trận X để tính toán bias
        #X = np.column_stack((np.ones(len(X)), X))

        # Dự đoán giá trị
        y_pred = X.dot(self.coef_)
        return y_pred


model_2 = RidgeRegression(8)
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test.values)
print(y_pred2)
print("% Sai lệch trung bình: ",(sum(abs((y_pred2/Y_test-1)))/len(Y_test)))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()