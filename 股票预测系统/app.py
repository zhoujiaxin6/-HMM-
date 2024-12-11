import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd  # 用于数据处理，特别是读取和操作csv文件
from sklearn.model_selection import KFold

from hmm import HMM  # 引入自定义的HMM模型

# 设置中文字体，避免出现方框
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

# 读取数据
data = pd.read_csv('StockData/000001.SZ.csv')  # 从CSV文件中读取该股票的历史数据

# 获取股票的收盘价数据
Origin_data = data["close"]  # 收盘价列
data_size = 1200  # 设置使用的历史数据的长度
stock_data = data["close"][:data_size]  # 截取前1200条数据用于训练
x = data.index[:data_size]  # 使用日期索引作为x轴数据

# 使用指数加权移动平均进行平滑处理
span = 30  # 设置加权因子，值越小越平滑
filter_data = stock_data.ewm(span=span, adjust=False).mean()

# 计算误差和相对误差
error = np.sum(np.abs(stock_data - filter_data)) / len(stock_data)  # 计算绝对误差
error_rate = np.sum(np.abs(stock_data - filter_data) / np.abs(stock_data)) / len(stock_data)  # 计算相对误差

# 输出误差结果
print("原始股价与平滑后的股价之间的误差")
print("误差为：{:.6f}".format(error))  # 打印绝对误差
print("相对误差率为：{:.2f}%".format(error_rate * 100))  # 打印相对误差率

# 绘制原始数据和滤波后的数据
plt.figure(figsize=(8, 5), dpi=200)  # 设置图表大小和分辨率
plt.plot(x, list(stock_data), label="原始数据")  # 绘制原始股价数据
plt.plot(x, filter_data, label="滤波后的数据", color='red')  # 绘制滤波后的股价数据
plt.title('平安银行股票数据')
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 准备数据用于HMM训练
X = np.array(stock_data).reshape(-1, 1)  # 将股价数据转为二维数组，作为HMM的输入

# 配置HMM模型的参数
hiden_state_num = 16 # 隐藏状态的个数
X_dim = 1  # 输入数据的维度，股价数据为一维
epoch = 40  # 训练迭代次数
if_kmeans = True  # 是否使用KMeans算法初始化高斯分布均值


# 定义k折交叉验证函数
def cross_validation(X, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]

        # 创建并训练HMM模型
        model = HMM(hiden_state_num, X_dim, epoch, if_kmeans)
        model.train(X_train)

        # 进行预测
        pred = model.predict(X_test, len(X_test))

        # 计算MSE
        mse = np.mean((X_test - pred) ** 2)
        mse_scores.append(mse)

    return mse_scores

# 执行k折交叉验证
mse_scores = cross_validation(X, n_splits=5)

# 计算平均MSE和标准差
mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)

print(f"平均MSE: {mse_mean:.4f}, 标准差: {mse_std:.4f}")

# 绘制MSE的直方图
plt.figure(figsize=(8, 5))
plt.hist(mse_scores, bins=5, color='green', alpha=0.7)
plt.title('MSE直方图')
plt.xlabel('MSE')
plt.ylabel('频率')
plt.grid(True)
plt.show()

# 创建并训练HMM模型
model = HMM(hiden_state_num, X_dim, epoch, if_kmeans)  # 初始化HMM模型
model.train(X)  # 训练HMM模型

# 绘制训练过程中的对数似然值变化
plt.figure(figsize=(8, 5), dpi=200)  # 设置图表大小和分辨率
plt.rcParams["figure.dpi"] = 150  # 设置图表的分辨率
plt.figure(figsize=(5, 3))  # 设置图表大小
plt.plot(model.L)  # 绘制对数似然值的变化
plt.xlabel("迭代次数")  # x轴标签
plt.title("对数似然值")  # 图表标题

# 设置预测的样本数
pred_num = 1000  # 预测1000个数据点
now_x = np.array(stock_data).reshape(-1, 1)[1:pred_num + 1]  # 选择训练数据的一部分进行预测
pred = model.predict(now_x, pred_num)  # 使用HMM模型进行预测

# 计算预测误差和相对误差
error = now_x - pred.reshape(-1, 1)  # 计算预测误差

# 绘制原始数据、预测数据和误差的对比图
plt.rcParams["figure.dpi"] = 120  # 设置图表的分辨率
plt.figure(figsize=(7, 7))  # 设置图表大小
plt.subplot(3, 1, 1)  # 子图1：绘制原始数据
plt.plot(list(data.index)[1:pred_num + 1], list(now_x), label="原始数据")
plt.ylabel("收盘价")
plt.title("原始值")  # 原始数据标题

plt.subplot(3, 1, 2)  # 子图2：绘制预测数据
plt.plot(list(data.index)[1:pred_num + 1], pred, "r", label="预测数据")
plt.ylabel("收盘价")
plt.title("预测值")  # 预测数据标题

plt.subplot(3, 1, 3)  # 子图3：绘制误差
plt.plot(list(data.index)[1:pred_num + 1], error, "gray", label="误差")
plt.ylabel("误差值")
plt.title("误差值")  # 误差标题
plt.tight_layout()  # 调整布局
plt.show()  # 显示图表

# 重新进行预测并计算误差
pred_num = 1000  # 设置新的预测样本数
now_x = np.array(stock_data).reshape(-1, 1)[1:pred_num + 1]  # 选择数据进行预测
pred = model.predict(now_x, pred_num)  # 预测

# 绘制原始数据与预测数据对比图
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(list(data.index)[1:pred_num + 1], list(now_x), label="原始数据")
plt.plot(list(data.index)[1:pred_num + 1], pred, "r", label="预测数据")
plt.xticks(rotation=45)  # 旋转x轴标签
plt.title("HMM静态预测拟合效果")  # 标题
plt.ylabel("收盘价")
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 进行更多时间步的预测
pred_num = 1005
observe_start = 1000
now_x = np.array(Origin_data).reshape(-1, 1)[1:pred_num + 1]  # 选择数据进行预测
pred = model.predict(now_x, pred_num)  # 预测
now_x = now_x[observe_start - 1:pred_num - 1]  # 获取预测的真实数据
pred = pred[observe_start:pred_num]  # 获取预测结果

# 计算误差并输出结果
error = now_x - pred.reshape(-1, 1)  # 计算误差
error_rate = np.sum(np.abs(error) / now_x) / len(error)  # 计算相对误差
print("HMM模型进行多步预测后的误差，与实际股价之间的误差")
print("平均误差为：", np.sum(np.abs(error)) / len(error))  # 打印平均误差
print("相对误差率为：{:.2f}%".format(error_rate * 100))  # 打印相对误差率

# 预测更多时间步的股票数据
end = 1230  # 设置预测的结束时间
b = model.predict_more(X, end)  # 进行更多的时间步预测

# 绘制股票的变化情况
plt.rcParams["figure.dpi"] = 120  # 设置图表分辨率
plt.figure(figsize=(7, 7))  # 设置图表大小
plt.subplot(2, 1, 1)  # 绘制原始数据与预测数据对比图
plt.plot(list(data["close"][data_size:end]), label="原始数据")  # 绘制原始股价数据
plt.plot(b[data_size:end], "r", label="预测数据")  # 绘制预测的股价数据
plt.xticks(rotation=90)  # 旋转x轴标签
plt.ylabel("收盘价")
plt.ylim(0, 25)  # 设置y轴范围
plt.legend()  # 显示图例
plt.title("股票变化情况")  # 标题
plt.show()  # 显示图表
