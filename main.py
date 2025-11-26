import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class LassoComparison:
    def __init__(self, n_samples=500, n_features=1000, sparsity=0.1, lambda_val=0.1, device='cuda'):
        self.n_samples = n_samples # 样本数量
        self.n_features = n_features # 特征数量
        self.lambda_val = lambda_val # 稀疏程度，整个矩阵非0元素占比
        self.device = device

        # 生成测试数据作为benchmark
        self.A = torch.randn(n_samples, n_features, device=device) # 从正态分布随机抽样，矩阵形状(n_samples,n_features)
        self.x_true = torch.zeros(n_features, device=device) # 真实解，形状(n_features)
        nonzero_idx = torch.randperm(n_features)[:int(sparsity * n_features)] # 非0元素位置：选取随机排列后前sparsity比例个位置
        self.x_true[nonzero_idx] = torch.randn(len(nonzero_idx), device=device)

        self.b = self.A @ self.x_true + 0.1 * torch.randn(n_samples, device=device) # Ax + epsilon = b，epsilon的标准差0.1

        self.x_init = torch.zeros(n_features, device=device) # 初始化

    def obj(self, x):
        """
        计算目标函数值
        :return: 目标函数值
        """
        term1 = 0.5 * torch.norm(self.A @ x - self.b) ** 2 # loss项
        term2 = self.lambda_val * torch.norm(x, 1) # 正则项
        return term1 + term2

    """
    不同方法-->
    """

    def ordinary_gd(self, max_iter=1000, lr=0.01):
        """
        普通梯度下降法：这里是次梯度下降，正则项在0点不可导
        :param max_iter: 最大迭代步数
        :param lr: 学习率
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        objs = []

        for i in range(max_iter):
            # loss项梯度
            loss_gd = self.A.T @ (self.A @ x - self.b)

            # 正则项的次梯度
            l1_subgd = self.lambda_val * torch.sign(x) # 可导点
            # 不可导点x=0，次梯度可以取[-lambda, lambda]之间的任意值
            l1_subgd[x == 0] = 0 # 取0

            # 总梯度
            gradient = loss_gd + l1_subgd

            x = x - lr * gradient

            obj = self.obj(x)
            objs.append(obj.item())

        return objs


    def soft_threshold(self, x, threshold):
        """
        软阈值算子，用于L1正则化的proximal算子
        :return: x的l1范数proximal近端算子，张量，形状同x
        """
        prox_x = torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
        return prox_x

    def proximal_gd(self, max_iter=1000, lr=0.01):
        """
        proximal下降法
        :param max_iter:最大迭代步数
        :param lr: 学习率
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        objs = []

        for i in range(max_iter):
            # 对光滑loss二次项求梯度，然后下降
            gradient = self.A.T @ (self.A @ x - self.b)
            x_temp = x - lr * gradient

            # l1正则项计算proximal算子
            x = self.soft_threshold(x_temp, lr * self.lambda_val) # 阈值是 学习率x正则化系数

            obj = self.obj(x)
            objs.append(obj.item())

        return objs


    def smoothed_gd(self, max_iter=1000, lr=0.01, epsilon=1e-4):
        """
        光滑化梯度下降法：使用huber光滑化近似l1正则项
        :param max_iter: 最大迭代步数
        :param lr: 学习率
        :param epsilon: 光滑化参数
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        objs = []

        for i in range(max_iter):
            # loss项梯度
            loss_gd = self.A.T @ (self.A @ x - self.b)

            # 正则项光滑化的梯度
            # huber近似：abs(x)约等于sqrt(x^2 + epsilon)
            l1_smooth_gd = self.lambda_val * x / torch.sqrt(x ** 2 + epsilon)

            # 总梯度
            gradient = loss_gd + l1_smooth_gd

            x = x - lr * gradient

            obj = self.obj(x)
            objs.append(obj.item())

        return objs

    def admm(self, max_iter=1000, rho=1.0):
        """
        交替方向乘子法
        :param max_iter: 最大迭代步数
        :param rho: 惩罚参数
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        z = self.x_init.clone()  # 辅助变量，用于admm分割变量
        u = torch.zeros_like(x)  # 对偶变量，拉格朗日乘子

        objs = []

        # 预计算A^TA + rho I的逆矩阵
        ATA = self.A.T @ self.A
        I = torch.eye(self.n_features, device=self.device)
        M = ATA + rho * I

        L = torch.linalg.cholesky(M) # 使用Cholesky分解求逆
        M_inv = torch.cholesky_inverse(L)

        # # 如果Cholesky失败，使用伪逆
        # M_inv = torch.linalg.pinv(M)


        for i in range(max_iter):
            # x更新子问题
            x = M_inv @ (self.A.T @ self.b + rho * (z - u))

            # z更新子问题
            z = self.soft_threshold(x + u, self.lambda_val / rho)

            # 对偶变量更新
            u = u + x - z

            obj = self.obj(x)
            objs.append(obj.item())

        return objs

    def coordinate_descent(self, blocks, max_iter=1000):
        """
        块坐标下降法：每次更新一个特征块（块内按推导的单变量规则更新）
        :param max_iter: 最大迭代步数
        :param blocks:特征块分组列表，每个元素是特征索引的子列表，如[[0,1], [2,3]]
                        索引为非负整数，不超出特征数量，所有索引无重复且覆盖所有特征
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        objs = []

        # 预计算
        ATA = (self.A.T @ self.A).to(self.device)
        ATb = (self.A.T @ self.b).to(self.device)
        norm_a_sq = torch.norm(self.A, p=2, dim=0) ** 2  # 每个特征列a_j的二范数平方
        norm_a_sq = norm_a_sq.to(self.device)

        n_blocks = len(blocks)  # 块的数量（类初始化时传入的特征分组）

        for i in range(max_iter):
            # 随机打乱块的更新顺序
            block_order = torch.randperm(n_blocks, device=self.device)

            for block_idx in block_order:
                current_block = blocks[block_idx]  # 当前块的特征索引（比如[0,1]）

                # 遍历块内每个特征，按规则更新
                for j in current_block:
                    a_j_T_cj = ATb[j] - (ATA[j] @ x - norm_a_sq[j] * x[j])

                    mu = self.lambda_val  # 正则系数mu
                    if a_j_T_cj > mu:
                        x[j] = (a_j_T_cj - mu) / norm_a_sq[j]
                    elif a_j_T_cj < -mu:
                        x[j] = (a_j_T_cj + mu) / norm_a_sq[j]
                    else:
                        x[j] = 0.0

            # 每完成一轮块更新，记录目标函数值
            obj = self.obj(x)
            objs.append(obj.item())

        return objs



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("初始化Lasso问题比较器...")

    # 实例化
    lasso = LassoComparison(
        n_samples=200,  # 样本数量
        n_features=500,  # 特征维度
        sparsity=0.1,  # 稀疏度
        lambda_val=0.1,  # 正则化系数
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"使用设备: {lasso.device}")
    print(f"问题规模: {lasso.n_samples} × {lasso.n_features}")
    print(f"稀疏度: {lasso.lambda_val}")

    max_iter = 500
    lr = 0.01
    epsilon = 1e-4
    rho = 1.0
    blocks = [[i] for i in range(lasso.n_features)]

    lasso.ordinary_gd(max_iter=max_iter, lr=lr)
    lasso.smoothed_gd(max_iter=max_iter, lr=lr, epsilon=epsilon)
    lasso.proximal_gd(max_iter=max_iter, lr=lr)
    lasso.admm(max_iter=max_iter, rho=rho)
    lasso.coordinate_descent(max_iter=max_iter, blocks=blocks)



