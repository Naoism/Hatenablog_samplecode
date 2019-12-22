import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats

plt.style.use('ggplot')


size = 400  # データ数を400とする
r = 0.8  # xとzの間の相関係数の設定値
z = np.random.randn(size)  # zは標準正規分布から生成
z_lie = np.random.randn(size)  # xを生成するための仮データ
x = r * z + (1 - r ** 2) ** 0.5 * z_lie  # zとの相関係数が0.8となるxを生成

r_out = np.corrcoef(x, z)[1, 0]  # 生成したx, zの相関係数

y = 1.5 * x + 1.1 * z + np.random.randn(400)  # yの生成

# 散布図の図示
plt.scatter(x, y, s = 8)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


x_lm = x.reshape(-1, 1)  # xを二次元化
y_lm = y.reshape(-1, 1)  # yを二次元化

# 線形回帰
lm = linear_model.LinearRegression()
lm.fit(x_lm, y_lm)

print(lm.coef_)  # 回帰係数の出力


xyz = np.concatenate([x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)])  # 3変数を同じアレイに格納

xyz_0 = np.zeros(3).reshape(3,1)  # 3行1列のアレイを用意
xyz_1 = np.zeros(3).reshape(3,1)
xyz_2 = np.zeros(3).reshape(3,1)
xyz_3 = np.zeros(3).reshape(3,1)

for i in range(size):
    if xyz[2,i] <= np.percentile(z, 25):  # 25パーセンタイル以下
        xyz_0 = np.append(xyz_0, xyz[:,i].reshape(-1,1), axis=1)
    elif np.percentile(z, 25) < xyz[2,i] <= np.percentile(z, 50):  # 25パーセンタイル以上50パーセンタイル以下
        xyz_1 = np.append(xyz_1, xyz[:,i].reshape(-1,1), axis=1)
    elif np.percentile(z, 50) < xyz[2,i] <= np.percentile(z, 75):  # 50パーセンタイル以上75パーセンタイル以下
        xyz_2 = np.append(xyz_2, xyz[:,i].reshape(-1,1), axis=1)
    else:  # 75パーセンタイル以上
        xyz_3 = np.append(xyz_3, xyz[:,i].reshape(-1,1), axis=1)

xyz_0 = np.delete(xyz_0, 0, 1)  # 1列目のすべて0の要素をドロップ
xyz_1 = np.delete(xyz_1, 0, 1)
xyz_2 = np.delete(xyz_2, 0, 1)
xyz_3 = np.delete(xyz_3, 0, 1)

# 散布図作成
plt.scatter(x_0, y_0, s = 8, label = "i = 0")
plt.scatter(x_1, y_1, s = 8, label = "i = 1")
plt.scatter(x_2, y_2, s = 8, label = "i = 2")
plt.scatter(x_3, y_3, s = 8, label = "i = 3")
plt.title("X - Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()


# データ抽出
x_0 = xyz_0[0, :].reshape(-1, 1)
y_0 = xyz_0[1, :].reshape(-1, 1)
x_1 = xyz_1[0, :].reshape(-1, 1)
y_1 = xyz_1[1, :].reshape(-1, 1)
x_2 = xyz_2[0, :].reshape(-1, 1)
y_2 = xyz_2[1, :].reshape(-1, 1)
x_3 = xyz_3[0, :].reshape(-1, 1)
y_3 = xyz_3[1, :].reshape(-1, 1)

# 線形回帰
lm_0 = linear_model.LinearRegression()
lm_0.fit(x_0, y_0)
lm_1 = linear_model.LinearRegression()
lm_1.fit(x_1, y_1)
lm_2 = linear_model.LinearRegression()
lm_2.fit(x_2, y_2)
lm_3 = linear_model.LinearRegression()
lm_3.fit(x_3, y_3)

# 回帰係数計算
b_0 = lm_0.coef_[0, 0]
b_1 = lm_1.coef_[0, 0]
b_2 = lm_2.coef_[0, 0]
b_3 = lm_3.coef_[0, 0]

# 作図
fig = plt.figure(figsize = (12,8))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)

ax0.scatter(x_0, y_0, s = 8)
ax0.plot(x_0, lm_0.predict(x_0), color = 'red', label = "b = {}".format(round(b_0,2))) 
ax0.set_title("i = 0")
ax0.set_xlabel("X_0")
ax0.set_ylabel("Y")
ax0.legend(loc="upper left")

ax1.scatter(x_1, y_1, s = 8)
ax1.plot(x_1, lm_1.predict(x_1), color = 'red', label = "b = {}".format(round(b_1,2))) 
ax1.set_title("i = 1")
ax1.set_xlabel("X_1")
ax1.set_ylabel("Y")
ax1.legend(loc="upper left")

ax2.scatter(x_2, y_2, s = 8)
ax2.plot(x_2, lm_2.predict(x_2), color = 'red', label = "b = {}".format(round(b_2,2))) 
ax2.set_title("i = 2")
ax2.set_xlabel("X_2")
ax2.set_ylabel("Y")
ax2.legend(loc="upper left")

ax3.scatter(x_3, y_3, s = 8)
ax3.plot(x_3, lm_3.predict(x_3), color = 'red', label = "b = {}".format(round(b_3,2))) 
ax3.set_title("i = 3")
ax3.set_xlabel("X_3")
ax3.set_ylabel("Y")
ax3.legend(loc="upper left")

plt.tight_layout()  # 図の体裁を整えるため


# 標準誤差を入手
_, _, _, _, std_0 = stats.linregress(xyz_0[0, :], xyz_0[1, :])
_, _, _, _, std_1 = stats.linregress(xyz_1[0, :], xyz_1[1, :])
_, _, _, _, std_2 = stats.linregress(xyz_2[0, :], xyz_2[1, :])
_, _, _, _, std_3 = stats.linregress(xyz_3[0, :], xyz_3[1, :])

# 分散を計算
r_var_0 = 1 / (std_0**2)
r_var_1 = 1 / (std_1**2)
r_var_2 = 1 / (std_2**2)
r_var_3 = 1 / (std_3**2)

# 分散の重み付け平均を計算
mean_ = ((r_var_0 * b_0) + (r_var_0 * b_1) + (r_var_0 * b_2) + (r_var_0 * b_3)) \
        / (r_var_0 + r_var_1 + r_var_2 + r_var_3)

print(mean_)
