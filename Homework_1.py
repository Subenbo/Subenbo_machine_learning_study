# -*- coding: utf-8 -*-
"""
北京市空气质量数据分析（基于线性回归）
- 读取本地 Excel 文件
- 目标变量：AQI（空气质量指数）
- 特征变量：PM2.5, PM10, SO2, CO, NO2, O3 对其进行分析 通过分析最终选用PM2.5, PM10两组相关性较高
"""

# 1. 导入库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

# 设置字体（使用微软雅黑，支持中文和上标）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局选项
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)        # 自动检测宽度
pd.set_option('display.max_colwidth', 50)   # 列内容最多显示50字符
pd.set_option('display.expand_frame_repr', False)  # 禁用多行表示


# 2. 加载空气质量数据（Excel文件）
file_path = r'D:\Users\WangBo\PycharmProjects\PythonProject1\.venv\北京市空气质量数据.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 将日期列转换为datetime类型（若尚未转换），便于后续可能的时间特征提取，但不作为特征
if '日期' in df.columns:
    df['日期'] = pd.to_datetime(df['日期'])

# 定义目标变量和特征列
target_col = 'AQI'          # 目标变量：空气质量指数
feature_cols = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']  # 数值特征

# 提取特征矩阵和目标向量
X_raw = df[feature_cols].copy()
y_raw = df[target_col].copy()

# 处理缺失值（用均值填充，也可根据实际选择删除或插值）
print("缺失值统计：")
print(X_raw.isnull().sum())
X_raw.fillna(X_raw.mean(), inplace=True)
y_raw.fillna(y_raw.mean(), inplace=True)

# 3.1 数据基本信息查看
print("\n=== 数据集前5行 ===")
print(X_raw.head())
print("\n=== 目标变量前5行 ===")
print(y_raw.head())

print("\n=== 数据类型与缺失值 ===")
print(X_raw.info())
print("\n=== 数值特征统计描述 ===")
print(X_raw.describe())
print("\n=== 目标变量统计描述 ===")
print(y_raw.describe())

# 3.2 目标变量分布分析
plt.figure(figsize=(10, 6))
plt.hist(y_raw, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
y_raw.plot(kind='kde', color='red', linewidth=2)
plt.title('空气质量指数(AQI)分布', fontsize=14)
plt.xlabel('AQI', fontsize=12)
plt.ylabel('密度', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 3.3 特征与目标变量的相关性分析
# 计算所有特征与目标变量的相关系数
correlations = X_raw.corrwith(y_raw).sort_values(ascending=False)
print("\n=== 特征与目标变量（AQI）的相关系数 ===")
print(correlations)

# 可视化相关性（条形图）
plt.figure(figsize=(12, 6))
correlations.plot(kind='bar', color=['green' if x > 0 else 'red' for x in correlations])
plt.title('特征与AQI的相关性', fontsize=14)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('特征', fontsize=12)
plt.ylabel('相关系数', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.show()

# 3.4 强相关特征与目标变量的散点图
# 选择与目标变量相关性最强的2个正相关和2个负相关特征
top_positive = correlations.index[:2]   # 前2个正相关特征
top_negative = correlations.index[-2:]  # 最后2个负相关特征
selected_features = list(top_positive) + list(top_negative)

plt.figure(figsize=(15, 10))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    plt.scatter(X_raw[feature], y_raw, alpha=0.6, color='purple', edgecolor='white', s=30)
    # 添加趋势线（线性拟合）
    z = np.polyfit(X_raw[feature], y_raw, 1)
    p = np.poly1d(z)
    plt.plot(X_raw[feature], p(X_raw[feature]), "r--")
    plt.title(f'{feature} vs AQI (Corr: {correlations[feature]:.2f})', fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('AQI', fontsize=10)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 3.5 特征间的相关性热力图
plt.figure(figsize=(14, 10))
corr_matrix = X_raw.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('特征间相关性矩阵', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# 方法1：基于相关系数手动筛选（选择绝对值>0.8的特征）
corr_threshold = 0.8
selected_features = correlations[abs(correlations) > corr_threshold].index.tolist()
print(f"\n基于相关系数筛选的特征（|corr| > {corr_threshold}）：{selected_features}")

# 方法2：计算VIF剔除多重共线性特征（可选）
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["特征"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)

# 初始VIF计算
if len(selected_features) > 0:
    vif_df = calculate_vif(X_raw[selected_features])
    print("\n初始VIF值：")
    print(vif_df)

    # 剔除VIF>10的特征（通常认为VIF>10存在强共线性）
    while vif_df["VIF"].max() > 10:
        drop_feature = vif_df.iloc[0]["特征"]
        selected_features.remove(drop_feature)
        vif_df = calculate_vif(X_raw[selected_features])
        print(f"\n剔除特征 {drop_feature} 后VIF值：")
        print(vif_df)

    # 更新特征矩阵
    X_selected = X_raw[selected_features]
    print(f"\n最终筛选后的特征：{selected_features}")
else:
    print("警告：没有特征与目标变量相关性绝对值 > 0.8，请降低阈值或检查数据")
    X_selected = X_raw  # 使用全部特征

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
print("\n标准化后的特征前5行：")
print(X_scaled_df.head())
print("\n标准化后特征的均值和标准差：")
print(pd.DataFrame({
    "均值": X_scaled_df.mean().round(4),
    "标准差": X_scaled_df.std().round(4)
}))

# 划分训练集和测试集（测试集占20%）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw.values, test_size=0.2, random_state=42
)
print(f"\n训练集特征形状：{X_train.shape}，测试集特征形状：{X_test.shape}")
print(f"训练集目标形状：{y_train.shape}，测试集目标形状：{y_test.shape}")

# 初始化线性回归模型
lr_model = LinearRegression(fit_intercept=True)

# 使用训练集训练模型
lr_model.fit(X_train, y_train)

# 输出模型参数
print("\n=== 线性回归模型参数 ===")
print(f"截距项（w0）：{lr_model.intercept_:.4f}")
print("特征系数（w1...wn）：")
for feature, coef in zip(selected_features, lr_model.coef_):
    print(f"  {feature}: {coef:.4f}")

# 对训练集和测试集进行预测
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# 查看部分预测结果
print("\n=== 预测结果示例（前5条） ===")
comparison = pd.DataFrame({
    "真实值": y_test,
    "预测值": y_test_pred.round(2),
    "误差": (y_test_pred - y_test).round(2)
})
comparison.to_csv("线性回归预测结果.csv", index=False)
print(comparison.head())

# 评估函数
def evaluate(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {dataset_name} 评估指标 ===")
    print(f"均方误差（MSE）：{mse:.4f}")
    print(f"均方根误差（RMSE）：{rmse:.4f}")
    print(f"决定系数（R²）：{r2:.4f}")
    return mse, rmse, r2

# 评估训练集和测试集
train_mse, train_rmse, train_r2 = evaluate(y_train, y_train_pred, "训练集")
test_mse, test_rmse, test_r2 = evaluate(y_test, y_test_pred, "测试集")

# 分析过拟合情况
if abs(train_r2 - test_r2) < 0.1:
    print("\n模型未出现明显过拟合（训练集与测试集R²差异较小）")
else:
    print("\n警告：模型可能存在过拟合（训练集与测试集R²差异较大）")

# 7.1 真实值 vs 预测值散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue', edgecolor='white', label='测试集')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
plt.title(f'真实AQI vs 预测AQI (R²={test_r2:.4f})', fontsize=14)
plt.xlabel('真实AQI', fontsize=12)
plt.ylabel('预测AQI', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 7.2 预测误差分布直方图
plt.figure(figsize=(10, 6))
errors = y_test - y_test_pred
plt.hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
plt.title(f'预测误差分布 (RMSE={test_rmse:.4f})', fontsize=14)
plt.xlabel('误差', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 7.3 特征重要性条形图（基于系数绝对值）
if len(selected_features) > 0:
    plt.figure(figsize=(12, 6))
    feature_importance = np.abs(lr_model.coef_)
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.bar([selected_features[i] for i in sorted_idx],
            [feature_importance[i] for i in sorted_idx],
            color=['green' if lr_model.coef_[i] > 0 else 'red' for i in sorted_idx])
    plt.title('特征重要性（系数绝对值）', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('系数绝对值（重要性）', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    # 添加正负标记
    for pos, i in enumerate(sorted_idx):
        plt.text(pos, feature_importance[i] + 0.1,
                 '+' if lr_model.coef_[i] > 0 else '-',
                 ha='center', fontweight='bold')
    plt.show()

# 预测数据
new_data = pd.DataFrame([[14, 43]], columns=['PM2.5', 'PM10'])
new_scaled = scaler.transform(new_data)
pred = lr_model.predict(new_scaled)
print(f"预测 AQI: {pred[0]:.1f}")