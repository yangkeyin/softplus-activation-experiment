import numpy as np
from sklearn.model_selection import train_test_split
import torch



def generate_data(data_range, num_points, test_size, target_function, device):
    """
    生成并分割目标函数数据集
    Args:
        data_range (list): 数据范围，例如 [-2*pi, 2*pi]
        num_points (int): 数据点数量
        test_size (float): 测试集比例
    Returns:
        X_train_t, y_train_t, X_test_t, y_test_t: 训练集和测试集的 PyTorch Tensors
    """
    X = np.linspace(data_range[0], data_range[1], num_points).reshape(-1, 1)
    y = target_function(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 转换为PyTorch Tensors并移动到设备
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

# 插值方法
def generate_inter_data(data_range, num_train_points, num_test_points, target_function, device):
    """
    对训练集进行插值，生成更多数据点
    Args:
        X_train (np.array): 原始训练集输入
        y_train (np.array): 原始训练集输出
        X_test (np.array): 测试集输入
        num_points (int): 插值后数据点数量
    Returns:
        X_train_interp, y_train_interp, X_test: 插值后的训练集和测试集输入
    """
    # 对训练集进行插值
    # --- 创建测试集 (密集的线性网格) ---
    X_test = np.linspace(data_range[0], data_range[1], num_test_points).reshape(-1, 1)
    y_test = target_function(X_test)

    # --- 创建训练集 (稀疏的随机采样) ---
    # 使用固定的数据种子(DATA_SEED)来确保每次运行N=10时，拿到的都是同一批随机点
    np.random.seed(42)
    X_train = np.random.uniform(data_range[0], data_range[1], num_train_points).reshape(-1, 1)
    y_train = target_function(X_train)
    
    # 转换为PyTorch Tensors并移动到设备
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

def generate_outer_data(data_range, num_train_points, num_test_points, target_function, device):
    """
    对训练集进行插值，生成更多数据点
    Args:
        X_train (np.array): 原始训练集输入
        y_train (np.array): 原始训练集输出
        X_test (np.array): 测试集输入
        num_points (int): 插值后数据点数量
    Returns:
        X_train_interp, y_train_interp, X_test: 插值后的训练集和测试集输入
    """
    # 对训练集进行插值
    # --- 创建测试集 (密集的线性网格) ---
    # 从[0, data_range[1]]中随机采样
    test_range = [0, data_range[1]]
    X_test = np.linspace(test_range[0], test_range[1], num_test_points).reshape(-1, 1)
    y_test = target_function(X_test)

    # --- 创建训练集 (稀疏的随机采样) ---
    # 从[-data_range[1], 0]中随机采样
    train_range = [data_range[0], 0]
    X_train = np.linspace(train_range[0], train_range[1], num_train_points).reshape(-1, 1)
    y_train = target_function(X_train)
    
    # 转换为PyTorch Tensors并移动到设备
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

def generate_2D_data(data_range, num_train_points, num_test_points, target_function, device):
    """
    生成并分割目标函数数据集
    Args:
        data_range (list): 数据范围，例如 [-2*pi, 2*pi]
        num_points (int): 数据点数量
        test_size (float): 测试集比例
    Returns:
        X_train_t, y_train_t, X_test_t, y_test_t: 训练集和测试集的 PyTorch Tensors
    """
    # --- 计算训练集 ---
    # 定义每个维度上的点数
    points_per_dim_train = int(np.sqrt(num_train_points))
    # 使用 meshgrid 创建一个网格
    x_range_train = np.linspace(data_range[0][0], data_range[0][1], points_per_dim_train)
    y_range_train = np.linspace(data_range[1][0], data_range[1][1], points_per_dim_train)
    xx_train, yy_train = np.meshgrid(x_range_train, y_range_train)
    # 将网格展平以创建 (N, 2) 的输入数据
    X_train = np.vstack([xx_train.ravel(), yy_train.ravel()]).T
    # 计算目标值
    y_train = target_function(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)

    # --- 计算测试集 ---
    points_per_dim_test = int(np.sqrt(num_test_points))
    
    # 使用 meshgrid 创建一个网格
    np.random.seed(42)
    x_range_test = np.random.uniform(data_range[0][0], data_range[0][1], points_per_dim_test)
    y_range_test = np.random.uniform(data_range[1][0], data_range[1][1], points_per_dim_test)
    xx_test, yy_test = np.meshgrid(x_range_test, y_range_test)
    
    # 将网格展平以创建 (N, 2) 的输入数据
    X_test = np.vstack([xx_test.ravel(), yy_test.ravel()]).T
    
    # 计算目标值
    y_test = target_function(X_test[:, 0], X_test[:, 1]).reshape(-1, 1)

    # 转换为PyTorch Tensors并移动到设备
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

