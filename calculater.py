import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

# 生成模拟数据
V = np.array([1, 2, 3, 4, 5])  # 电压
B = 2.0  # 磁场常数
R = V / B  # 半径
u_V = 0.05 * V  # 假设 5% 误差
u_R = 0.05 * R

# 计算 X, Y
X1 = V / B
Y1 = R

X2 = 1 / V
Y2 = 1 / (B * R)

# 线性拟合
model = LinearModel()
params1 = model.guess(Y1, x=X1)
fit1 = model.fit(Y1, params1, x=X1)

params2 = model.guess(Y2, x=X2)
fit2 = model.fit(Y2, params2, x=X2)

# 提取斜率
slope1 = fit1.params['slope'].value
slope2 = fit2.params['slope'].value

# 画图
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X1, Y1, label="V/B vs R", color='blue')
plt.plot(X1, fit1.best_fit, linestyle="--", color="black")
plt.xlabel("X = V/B")
plt.ylabel("Y = R")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(X2, Y2, label="1/V vs 1/BR", color='red')
plt.plot(X2, fit2.best_fit, linestyle="--", color="black")
plt.xlabel("X = 1/V")
plt.ylabel("Y = 1/BR")
plt.legend()

plt.show()

print(f"Slope from V/B = R: {slope1:.4f}")
print(f"Slope from 1/V = 1/BR: {slope2:.4f}")
print(f"Product of slopes (should be 1): {slope1 * slope2:.4f}")
