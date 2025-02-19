import os
import torch
import numpy as np


class ModelBackUp:

    def __init__(self):
        self.backup = {}

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def backup_param(self, model):
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
    
    def calculate_gap(self, model):
        gap = 0
        for name, param in model.named_parameters():
            if name in self.backup:
                gap += (param.data - self.backup[name]).abs().sum()
        return gap

    def calculate_gradient(self, model):
        gap = 0
        for name, param in model.named_parameters():
            if name in self.backup and param.grad is not None:
                gap += param.grad.abs().sum()
        return gap
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        """
        Args:
            patience (int): Maximum number of epochs to wait when validation loss stops improving.
            delta (float): Minimum change in loss to qualify as an improvement. Used to define the threshold for loss improvement.
            path (str): Path to save the model.
            verbose (bool): If True, prints a message each time the model improves.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.verbose = verbose

    def __call__(self, metric_score):
        score = metric_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

import torch
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.embedding_vectors = []

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def attack_embedding(self, epsilon=1., alpha=0.3, embedding_vectors=[], is_first_attack=False):
        if is_first_attack:
            self.embedding_vectors = []
            for embedding_vector in embedding_vectors:
                self.embedding_vectors.append(embedding_vector.data.clone())
        for idx, param in enumerate(embedding_vectors):
            batch_size = param.shape[0]
            norm = torch.norm(param.grad.view(batch_size, -1), dim=1, keepdim=True)
            norm = norm + 1e-12
            r_at = alpha * param.grad / norm.view(batch_size, 1, 1)
            param.data.add_(r_at)
            param.data = self.project_embedding(idx, param.data, epsilon)
                

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def restore_vectors(self):
        for item in self.embedding_vectors:
            del item
        self.embedding_vectors = None
        for item in self.grad_backup:
            self.grad_backup[item] = None

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r # 多步累加扰动后可能超过扰动范围，因此设置 epsilon 
    
    # 多步累加扰动后可能超过扰动范围，因此设置 epsilon 
    def project_embedding(self, param_idx, param_data, epsilon):
        r = param_data - self.embedding_vectors[param_idx]
        batch_size = r.shape[0]
        norm = torch.norm(r.view(batch_size, -1), dim=1)
        mask = norm > epsilon # 判断是否超过扰动范围
        if mask.any():
            r[mask] = epsilon * r[mask] / norm[mask].view(mask.sum(), 1, 1) # 修改超过的扰动范围
        return self.embedding_vectors[param_idx] + r 

    def backup_grad(self): # 备份梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


def bhattacharyya_distance_pairwise(means1, stds1, means2, stds2):
    """
    计算一组高维独立正态分布之间的两两 Bhattacharyya 距离。

    参数：
    - means: 张量，形状为 (N, D)，N 个分布的均值向量
    - stds: 张量，形状为 (N, D)，N 个分布的标准差向量

    返回：
    - distance_matrix: 张量，形状为 (N, N)，Bhattacharyya 距离矩阵
    """
    # 计算方差
    vars1 = stds1 ** 2  # (N, D)
    vars2 = stds2 ** 2  # (N, D)
    # 扩展维度以进行广播
    means1 = means1.unsqueeze(1)  # (N, 1, D)
    means2 = means2.unsqueeze(0)  # (1, N, D)
    vars1 = vars1.unsqueeze(1)    # (N, 1, D)
    vars2 = vars2.unsqueeze(0)    # (1, N, D)
    # 计算均值差的平方
    delta_mean_squared = (means1 - means2) ** 2  # (N, N, D)
    # 计算平均方差
    sigma_squared = 0.5 * (vars1 + vars2)  # (N, N, D)
    # 添加数值稳定项
    eps = 1e-10
    sigma_squared = sigma_squared + eps
    # 计算第一项
    term1 = 0.125 * torch.sum(delta_mean_squared / sigma_squared, dim=2)  # (N, N)
    # 计算第二项
    term2 = 0.5 * torch.sum(torch.log(sigma_squared / torch.sqrt(vars1 * vars2 + eps)), dim=2)  # (N, N)
    # 计算 Bhattacharyya 距离矩阵
    distance_matrix = term1 + term2  # (N, N)
    coefficient = torch.exp(-distance_matrix) 
    return coefficient

def bhattacharyya_distance_independent_metric(mean0, std0, mean1, std1):
    """
    计算两个高维独立正态分布之间的 Bhattacharyya 距离。

    参数：
    - mean0: 张量，形状为 (batch_size, D)，第一个分布的均值向量
    - std0: 张量，形状为 (batch_size, D)，第一个分布的标准差向量
    - mean1: 张量，形状为 (batch_size, D)，第二个分布的均值向量
    - std1: 张量，形状为 (batch_size, D)，第二个分布的标准差向量
    返回：
    - distance: 标量，Bhattacharyya 距离
    """
    # 计算均值差向量
    delta_mean = mean1 - mean0
    # 计算平均方差向量
    sigma_squared = 0.5 * (std0 ** 2 + std1 ** 2)
    # 计算第一项
    eps = 1e-10
    term1 = 0.125 * torch.sum((delta_mean ** 2) / sigma_squared + eps, dim=1)
    # 计算第二项
    term2 = 0.5 * torch.sum(torch.log(sigma_squared / (std0 * std1 + eps)), dim=1)
    # 计算 Bhattacharyya 距离
    distance = term1 + term2
    return distance

def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    # 均值差异
    diff = mu1 - mu2
    # 协方差平均值
    sigma_m = 0.5 * (sigma1 + sigma2)
    
    # 计算均值项
    inv_sigma_m = torch.linalg.inv(sigma_m)  # 协方差矩阵的逆
    term1 = 0.125 * diff.T @ inv_sigma_m @ diff  # 均值差异项
    
    # 计算协方差项
    det_sigma1 = torch.det(sigma1)  # 协方差矩阵1的行列式
    det_sigma2 = torch.det(sigma2)  # 协方差矩阵2的行列式
    det_sigma_m = torch.det(sigma_m)  # 协方差平均矩阵的行列式
    term2 = 0.5 * torch.log(det_sigma_m / torch.sqrt(det_sigma1 * det_sigma2))  # 协方差项
    
    # Bhattacharyya 距离
    distance = term1 + term2
    return distance

def bhattacharyya_distance_independent(mean0, std0, mean1, std1):
    """
    计算两个高维独立正态分布之间的 Bhattacharyya 距离。

    参数：
    - mean0: 张量，形状为 (D,)，第一个分布的均值向量
    - std0: 张量，形状为 (D,)，第一个分布的标准差向量
    - mean1: 张量，形状为 (D,)，第二个分布的均值向量
    - std1: 张量，形状为 (D,)，第二个分布的标准差向量

    返回：
    - distance: 标量，Bhattacharyya 距离
    """
    # 计算均值差向量
    delta_mean = mean1 - mean0

    # 计算平均方差向量
    sigma_squared = 0.5 * (std0 ** 2 + std1 ** 2)

    # 计算第一项
    term1 = 0.125 * torch.sum((delta_mean ** 2) / sigma_squared)

    # 计算第二项
    term2 = 0.5 * torch.sum(torch.log(sigma_squared / (std0 * std1)))

    # 计算 Bhattacharyya 距离
    distance = term1 + term2

    return distance


def js_divergence_gaussians_full(mean1, variance1, mean2, variance2):
    """
    计算两组高斯分布之间所有可能组合的 JS 散度，支持 GPU 加速。

    参数：
    mean1 : torch.Tensor, shape (N1, D)
        第一组分布的均值矩阵。
    variance1 : torch.Tensor, shape (N1, D)
        第一组分布的方差矩阵。
    mean2 : torch.Tensor, shape (N2, D)
        第二组分布的均值矩阵。
    variance2 : torch.Tensor, shape (N2, D)
        第二组分布的方差矩阵。
    device : str 或 torch.device
        指定计算所使用的设备，例如 'cpu' 或 'cuda'。

    返回值：
    js_matrix : torch.Tensor, shape (N1, N2)
        所有分布组合的 JS 散度矩阵。
    """
    # 防止方差为零或负数
    variance1 = torch.clamp(variance1, min=1e-8)
    variance2 = torch.clamp(variance2, min=1e-8)

    # 扩展维度以便进行广播
    mean1_exp = mean1.unsqueeze(1)  # (N1, 1, D)
    variance1_exp = variance1.unsqueeze(1)  # (N1, 1, D)
    mean2_exp = mean2.unsqueeze(0)  # (1, N2, D)
    variance2_exp = variance2.unsqueeze(0)  # (1, N2, D)

    # 计算混合分布的均值和方差
    mean_m = 0.5 * (mean1_exp + mean2_exp)
    variance_m = 0.5 * (variance1_exp + variance2_exp) + 0.25 * (mean1_exp - mean2_exp) ** 2

    # 防止混合分布的方差为零或负数
    variance_m = torch.clamp(variance_m, min=1e-8)

    # 计算 D_KL(P || M)
    kl_pm = 0.5 * torch.sum(
        torch.log(variance_m / variance1_exp) +
        (variance1_exp + (mean1_exp - mean_m) ** 2) / variance_m - 1,
        dim=2
    )

    # 计算 D_KL(Q || M)
    kl_qm = 0.5 * torch.sum(
        torch.log(variance_m / variance2_exp) +
        (variance2_exp + (mean2_exp - mean_m) ** 2) / variance_m - 1,
        dim=2
    )

    # 计算 JS 散度矩阵
    js_matrix = 0.5 * (kl_pm + kl_qm)
    sim_matrix = torch.exp(-js_matrix)
    return sim_matrix
