import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



class LassoComparison:
    def __init__(self, A, b, lambda_val=0.1, device='cuda'):
        self.A = A
        self.b = b
        self.n_samples, self.n_features = A.shape
        self.lambda_val = lambda_val # 稀疏程度，整个矩阵非0元素占比
        self.device = device
        #self.x_init = torch.zeros(self.n_features, device=device) # 初始化
        # 小随机初始化：正态分布(0, 0.01)，替换原全零
        self.x_init = torch.randn(self.n_features, device=device) * 0.01

        self.L = torch.linalg.norm(A, ord=2) ** 2  # Lipschitz常数，以下这些方法都使用自适应学习率


    def obj(self, x):
        """计算目标函数值"""
        if x.isnan().any():
            return torch.tensor(float('inf'), device=self.device)

        term1 = 0.5 * torch.norm(self.A @ x - self.b) ** 2
        term2 = self.lambda_val * torch.norm(x, 1)
        obj_val = term1 + term2

        # 防止数值爆炸，强制置为1e10防止崩溃
        if obj_val > 1e10:
            return torch.tensor(1e10, device=self.device)

        return obj_val

    """
    不同方法-->
    """

    def ordinary_gd(self, max_iter=1000, lr=None):
        """
        普通梯度下降法：这里是次梯度下降，正则项在0点不可导
        :param max_iter: 最大迭代步数
        :param lr: 自适应学习率
        :return: 每一步的obj，列表形式存储
        """
        if lr is None:
            lr = 1.0 / (self.L + 10)  # 基于Lipschitz的自适应学习率

        x = self.x_init.clone()
        objs = []

        # lasso问题不连续，因此这里是次梯度下降
        for i in range(max_iter):
            loss_gd = self.A.T @ (self.A @ x - self.b)
            l1_subgd = self.lambda_val * torch.sign(x)
            l1_subgd[x == 0] = 0

            gradient = loss_gd + l1_subgd

            # 梯度裁剪防止梯度爆炸导致目标函数急剧上升，若范数过大则将梯度缩放至范数为1e3
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e3:
                gradient = gradient / grad_norm * 1e3

            x = x - lr * gradient

            # 投影到合理范围：截断，防止数值太大
            x = torch.clamp(x, -1e3, 1e3)

            obj = self.obj(x)
            objs.append(obj.item())

            # 如果目标函数值过大或出现NaN，说明算法发散，用1e10标记发散并提前停止
            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))  #
                break

        return objs


    def soft_threshold(self, x, threshold):
        """
        软阈值算子，用于L1正则化的proximal算子
        :return: x的l1范数proximal近端算子，张量，形状同x
        """
        prox_x = torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
        return prox_x

    def proximal_gd(self, max_iter=1000, lr=None):
        """
        proximal下降法
        :param max_iter:最大迭代步数
        :param lr: 学习率
        :return: 每一步的obj，列表形式存储
        """
        if lr is None:
            lr = 1.0 / (self.L + 10) * 2

        x = self.x_init.clone()
        objs = []

        for i in range(max_iter):
            gradient = self.A.T @ (self.A @ x - self.b)

            # 梯度裁剪
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e3:
                gradient = gradient / grad_norm * 1e3

            x_temp = x - lr * gradient
            x = self.soft_threshold(x_temp, lr * self.lambda_val)

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

        return objs


    def smoothed_gd(self, max_iter=1000, lr=None, epsilon=1e-4):
        """
        光滑化梯度下降法：使用huber光滑化近似l1正则项
        :param max_iter: 最大迭代步数
        :param lr: 学习率
        :param epsilon: 光滑化参数
        :return: 每一步的obj，列表形式存储
        """
        if lr is None:
            lr = 1.0 / (self.L + 10)

        x = self.x_init.clone()
        objs = []

        for i in range(max_iter):
            loss_gd = self.A.T @ (self.A @ x - self.b)
            l1_smooth_gd = self.lambda_val * x / torch.sqrt(x ** 2 + epsilon)

            gradient = loss_gd + l1_smooth_gd

            # 梯度裁剪
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e3:
                gradient = gradient / grad_norm * 1e3

            x = x - lr * gradient
            x = torch.clamp(x, -1e3, 1e3)

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

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

        L = torch.linalg.cholesky(M)  # 使用Cholesky分解求逆
        M_inv = torch.cholesky_inverse(L)

        for i in range(max_iter):
            # x更新子问题
            x = M_inv @ (self.A.T @ self.b + rho * (z - u))

            # z更新子问题
            z = self.soft_threshold(x + u, self.lambda_val / rho)

            # 对偶变量更新
            u = u + x - z

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

        return objs

    def coordinate_descent(self, max_iter=1000):
        """
        块坐标下降法：每次更新一个特征块（块内按推导的单变量规则更新）
        :param max_iter: 最大迭代步数
        :return: 每一步的obj，列表形式存储
        """
        x = self.x_init.clone()
        objs = []

        # 预计算
        ATA = (self.A.T @ self.A).to(self.device)
        ATb = (self.A.T @ self.b).to(self.device)
        norm_a_sq = torch.norm(self.A, p=2, dim=0) ** 2  # 每个特征列a_j的二范数平方

        for i in range(max_iter):
            # 随机打乱特征顺序
            feature_order = torch.randperm(self.n_features, device=self.device)

            # 按随机顺序更新每个特征
            for j in feature_order:
                # 计算梯度分量
                a_j_T_cj = ATb[j] - (ATA[j] @ x - norm_a_sq[j] * x[j])
                mu = self.lambda_val  # 软阈值更新

                if norm_a_sq[j] > 1e-10:  # 避免除以0
                    if a_j_T_cj > mu:
                        x[j] = (a_j_T_cj - mu) / norm_a_sq[j]
                    elif a_j_T_cj < -mu:
                        x[j] = (a_j_T_cj + mu) / norm_a_sq[j]
                    else:
                        x[j] = 0.0

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

        return objs

    def fista(self, max_iter=1000, lr=None):
        """标准FISTA：加速的proximal GD"""
        if lr is None:
            lr = 1.0 / (self.L + 10) * 2  # 与Proximal GD保持一致的学习率

        x_prev = self.x_init.clone()
        x = self.x_init.clone()
        t = torch.tensor(1.0, device=self.device) # 将t初始化为Tensor
        objs = []

        for i in range(max_iter):
            # 计算加速点y
            if i > 0:
                y = x + ((t - 1) / t) * (x - x_prev)
            else:
                y = x.clone()

            # 计算梯度
            gradient = self.A.T @ (self.A @ y - self.b) # f(y)的梯度 = A^T(Ay - b)
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e3:
                gradient = gradient / grad_norm * 1e3

            # 更新x
            x_prev = x.clone()
            x_temp = y - lr * gradient
            x = self.soft_threshold(x_temp, lr * self.lambda_val)

            # 更新动量参数t
            t_new = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
            t = t_new

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

        return objs

    def fista_restart(self, max_iter=1000, lr=None):
        """FISTA+Restart: 带重启的加速proximal GD"""
        if lr is None:
            lr = 1.0 / (self.L + 10) * 2

        x_prev = self.x_init.clone()
        x = self.x_init.clone()
        t = torch.tensor(1.0, device=self.device)
        objs = []

        for i in range(max_iter):
            # 计算加速点y
            if i > 0:
                y = x + ((t - 1) / t) * (x - x_prev)
            else:
                y = x.clone()

            # 计算梯度
            gradient = self.A.T @ (self.A @ y - self.b)
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e3:
                gradient = gradient / grad_norm * 1e3

            # 更新x
            x_prev = x.clone()
            x_temp = y - lr * gradient
            x_new = self.soft_threshold(x_temp, lr * self.lambda_val)

            # 重启条件：f(y)的梯度和(x_new - x)的内积大于等于0
            restart_condition = torch.dot(gradient.flatten(), (x_new - x).flatten()) >= 0
            if restart_condition:
                t = torch.tensor(1.0, device=self.device)  # 重置动量参数
                y = x.clone()  # 重置加速点
            else:
                t_new = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
                t = t_new

            x = x_new

            obj = self.obj(x)
            objs.append(obj.item())

            if obj > 1e10 or np.isnan(obj.item()):
                objs.extend([1e10] * (max_iter - i - 1))
                break

        return objs

    # def smoothed_fista(self, max_iter=1000, lr=None, epsilon=1e-4):
    #     """Smoothed FISTA（Huber光滑化的加速GD）"""
    #     if lr is None:
    #         lr = 1.0 / (self.L + 10)
    #
    #     x_prev = self.x_init.clone()
    #     x = self.x_init.clone()
    #     # 将t初始化为Tensor，并放在正确的设备上
    #     t = torch.tensor(1.0, device=self.device)
    #     objs = []
    #
    #     for i in range(max_iter):
    #         # 计算加速点y
    #         if i > 0:
    #             y = x + ((t - 1) / t) * (x - x_prev)
    #         else:
    #             y = x.clone()
    #
    #         # 计算Huber光滑化梯度
    #         loss_gd = self.A.T @ (self.A @ y - self.b)
    #         l1_smooth_gd = self.lambda_val * y / torch.sqrt(y ** 2 + epsilon)
    #         gradient = loss_gd + l1_smooth_gd
    #
    #         # 梯度裁剪
    #         grad_norm = torch.norm(gradient)
    #         if grad_norm > 1e3:
    #             gradient = gradient / grad_norm * 1e3
    #
    #         # 更新x
    #         x_prev = x.clone()
    #         x = y - lr * gradient
    #         x = torch.clamp(x, -1e3, 1e3)
    #
    #         # 更新动量参数（使用Tensor操作）
    #         t_new = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
    #         t = t_new
    #
    #         obj = self.obj(x)
    #         objs.append(obj.item())
    #
    #         if obj > 1e10 or np.isnan(obj.item()):
    #             objs.extend([1e10] * (max_iter - i - 1))
    #             break
    #
    #     return objs


def create_lasso_problem(n_samples, n_features, sparsity, device):
    """创建Lasso问题实例"""
    A = torch.randn(n_samples, n_features, device=device) # 生成形状为(n_samples, n_features)的随机矩阵A，元素服从标准正态分布
    # 先计算列范数，避免除以0
    col_norms = torch.norm(A, dim=0, keepdim=True)
    col_norms[col_norms == 0] = 1.0  # 范数为0的列设为1，避免inf
    A = A / col_norms

    x_true = torch.zeros(n_features, device=device)
    nonzero_idx = torch.randperm(n_features)[:int(sparsity * n_features)] # 随机选择非0元素位置
    x_true[nonzero_idx] = torch.randn(len(nonzero_idx), device=device) * 0.1  # 真实解的范围

    b = A @ x_true + 0.01 * torch.randn(n_samples, device=device)  # Ax + \epsilon = b

    return A, b, x_true


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    n_trials = 5  # 试验次数
    n_samples = 100
    n_features = 200
    sparsity = 0.1 # 稀疏程度
    lambda_val = 0.01  # 正则化系数
    max_iter = 500  # 迭代次数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Starting {n_trials} random trials...")
    print(f"Problem size: {n_samples} × {n_features}")
    print(f"Device: {device}")

    # 算法配置
    algo_configs = {
        'Ordinary GD': {'color': 'blue', 'style': '-', 'width': 3},
        'Proximal GD': {'color': 'red', 'style': '-', 'width': 3},
        'Smoothed GD (neighbor=1e-4)': {'color': 'green', 'style': '-', 'width': 3},
        'Smoothed GD (neighbor=1e-6)': {'color': 'lightgreen', 'style': '--', 'width': 3},
        'Smoothed GD (neighbor=1e-8)': {'color': 'gray', 'style': '--', 'width': 3},
        'ADMM (punish=0.3)': {'color': 'brown', 'style': '-', 'width': 3},
        'ADMM (punish=0.5)': {'color': 'orange', 'style': '-', 'width': 3},
        'ADMM (punish=1.0)': {'color': 'yellow', 'style': '--', 'width': 3},
        'ADMM (punish=2.0)': {'color': 'pink', 'style': '--', 'width': 3},
        'Coordinate Descent': {'color': 'purple', 'style': '-', 'width': 3},
        # FISTA系列
        'FISTA': {'color': 'cyan', 'style': '-', 'width': 2},
        'FISTA+Restart': {'color': 'cyan', 'style': '--', 'width': 2},
        #'Smoothed FISTA (huber=1e-4)': {'color': 'magenta', 'style': '-', 'width': 2}
    }


    results = {name: [] for name in algo_configs.keys()}
    for i in range(n_trials):
        print(f"Processing trial {i + 1}/{n_trials}...")

        # 创建新的Lasso问题
        A, b, x_true = create_lasso_problem(n_samples, n_features, sparsity, device)
        lasso = LassoComparison(A, b, lambda_val, device)

        # 计算基准：lasso目标函数的近似的理论最小值
        cd_result = lasso.coordinate_descent(max_iter=500) # 以块坐标下降为基准
        f_star = min(cd_result)
        f_star = max(f_star, 0)

        # 运行所有算法
        results['Ordinary GD'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.ordinary_gd(max_iter=max_iter)]
        )

        results['Proximal GD'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.proximal_gd(max_iter=max_iter)]
        )

        results['Smoothed GD (neighbor=1e-4)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.smoothed_gd(max_iter=max_iter, epsilon=1e-4)]
        )

        results['Smoothed GD (neighbor=1e-6)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.smoothed_gd(max_iter=max_iter, epsilon=1e-6)]
        )

        results['Smoothed GD (neighbor=1e-8)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.smoothed_gd(max_iter=max_iter, epsilon=1e-8)]
        )

        results['ADMM (punish=0.3)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.admm(max_iter=max_iter, rho=0.3)]
        )

        results['ADMM (punish=0.5)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.admm(max_iter=max_iter, rho=0.5)]
        )

        results['ADMM (punish=1.0)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.admm(max_iter=max_iter, rho=1.0)]
        )

        results['ADMM (punish=2.0)'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.admm(max_iter=max_iter, rho=2.0)]
        )

        results['Coordinate Descent'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.coordinate_descent(max_iter=max_iter)]
        )

        # 新增FISTA系列
        results['FISTA'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.fista(max_iter=max_iter)]
        )

        results['FISTA+Restart'].append(
            [max(obj - f_star, 1e-10) for obj in lasso.fista_restart(max_iter=max_iter)]
        )

        # results['Smoothed FISTA (huber=1e-4)'].append(
        #     [max(obj - f_star, 1e-10) for obj in lasso.smoothed_fista(max_iter=max_iter, epsilon=1e-4)]
        # )

    # 绘图
    plt.figure(figsize=(12, 7))
    k_axis = np.arange(1, max_iter + 1)

    for name, histories in results.items():
        # 确保所有试验的曲线长度一致，取最短的长度
        min_len = min([len(h) for h in histories])
        data_matrix = np.array([h[:min_len] for h in histories])
        current_k_axis = k_axis[:min_len]  # 对应的x轴坐标

        cfg = algo_configs[name] # 当前算法的颜色、样式等

        # 绘制所有单次试验的曲线
        for single_trial in data_matrix:
            plt.plot(current_k_axis, single_trial,
                     color=cfg['color'],  # 与算法平均曲线同色
                     alpha=0.2,
                     linewidth=0.5,
                     linestyle=cfg['style'])  # 与平均曲线同线条样式

        # 绘制平均曲线
        mean_curve = np.mean(data_matrix, axis=0)  # 按列求平均，每个迭代步的均值
        plt.plot(current_k_axis, mean_curve,
                 color=cfg['color'],
                 linestyle=cfg['style'],
                 linewidth=cfg['width'],
                 label=name)

    # 图表设置
    plt.yscale('log')  # 对数坐标
    plt.xlabel('k', fontsize=12)
    plt.ylabel('$f(x_k) - f^*$', fontsize=12)
    plt.title(f'Lasso Comparison (n_trials={n_trials})', fontsize=14)
    plt.legend(fontsize=9, loc='upper right', framealpha=0.9)
    plt.grid(True,alpha=0.4)
    plt.ylim(bottom=1e-8, top=1)  # y轴范围
    plt.xlim(0, 150)  # x轴迭代次数范围

    plt.tight_layout()
    # plt.savefig('lasso_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

