from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv('dataset.csv')

# Xem 5 dòng đầu của dataset
print(df.head())

# Xem thông tin tổng quan về dataset
print(df.info())

# Thống kê mô tả về dữ liệu
print(df.describe())

# Vẽ biểu đồ histogram cho DienTich
plt.hist(df['DienTich'], bins=10, edgecolor='k')
plt.xlabel('DienTich')
plt.ylabel('Số lượng')
plt.title('Biểu đồ Histogram DienTich')
plt.show()

# Vẽ biểu đồ histogram cho GiayTO
plt.hist(df['GiayTo'], bins=10, edgecolor='k')
plt.xlabel('GiayTo')
plt.ylabel('Số lượng')
plt.title('Biểu đồ Histogram GiayTo')
plt.show()

# Biểu đồ scatter plot giữa DienTich và USD
plt.scatter(df['DienTich'], df['USD'], c='blue', label='DienTich vs. USD')
plt.xlabel('DienTich')
plt.ylabel('USD')
plt.legend()
plt.title('Biểu đồ DienTich vs. USD')
plt.show()

# Biểu đồ scatter plot giữa DienTich và USD
plt.scatter(df['Phongngu'], df['USD'], c='blue', label='DienTich vs. USD')
plt.xlabel('DienTich')
plt.ylabel('USD')
plt.legend()
plt.title('Biểu đồ Phongngu vs. USD')
plt.show()


# Biểu đồ scatter plot giữa DienTich và USD
plt.scatter(df['SoTang'], df['USD'], c='blue', label='DienTich vs. USD')
plt.xlabel('DienTich')
plt.ylabel('USD')
plt.legend()
plt.title('Biểu đồ Phongngu vs. USD')
plt.show()



# Xóa các hàng có giá trị trống
df = df.dropna()


df_x = df.iloc[:, 0:9].astype(float)/1000
df_y = df.iloc[:, 9].astype(float)

# dữ liệu được random theo cách 365->dữ liệu sẽ được chia theo cùng một cách mỗi khi bạn chạy
X, X_test, Y, Y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=365)


class LinearRegression3:
    def __init__(self):
        self.coef_ = None  # Để lưu trữ hệ số beta1
        self.intercept_ = None  # Để lưu trữ hệ số beta0

    def fit(self, X, y):
        # Thêm cột 1 vào ma trận X để tính beta0 (hệ số chặn)
        X_with_intercept = np.column_stack((np.ones(len(X)), X))

        # Sử dụng công thức ma trận để tính hệ số beta
        beta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

        # Hệ số beta0 (intercept) là phần tử đầu tiên của vectơ beta
        self.intercept_ = beta[0]
        print(self.intercept_)
        # Hệ số beta1 (hệ số của biến độc lập) là phần tử thứ hai trở đi của vectơ beta
        self.coef_ = beta[1:]
        print(self.coef_)

    def predict(self, X):
        if self.coef_ is not None and self.intercept_ is not None:
            # Dự đoán dựa trên hệ số beta0 và beta1
            y_pred = self.intercept_ + X.dot(self.coef_)
            return y_pred
        else:
            raise ValueError("Chưa có hệ số hồi quy. Hãy gọi phương thức fit() trước.")



model_2 = LinearRegression3()
model_2.fit(X, Y)
y_pred2 = model_2.predict(X_test.values)
print(y_pred2)
print("% Sai lệch trung bình: ",(sum(abs((y_pred2/Y_test-1)))/len(Y_test)))

sb.regplot(x=model_2.predict(X_test), y = Y_test)
plt.xlabel("Giá dự đoán")
plt.ylabel('Giá thực tế')
plt.show()
#65.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 162500.0
print(64021 + 1.31 * 65.0 + 145 * 2.0 + 3.6 * 1 + 1.05 * 1 + 2.44 * 1 - 1.18 * 0 - 8.96 * 0 - 1.66 * 0)
