import torch
import numpy as np

class DataGenerator:
    def __init__(self, config):
        self.config = config
        
        # --- 1. 定义物理域范围 (Heston Domain) ---
        # 资产价格 S: [0, 2K] 或更大，覆盖 Payoff 区域
        self.S_min = 0.0
        self.S_max = 80.0   # 设为 Strike Price (20) 的 4倍，保证覆盖充分
        
        # 随机波动率 v: Heston 的核心新增维度
        # 方差通常在 [0, 1.0] 之间波动
        self.v_min = 0.0
        self.v_max = 1.0
        
        # 时间 t (也就是 tau): [0, 1]
        self.T = 1.0
        
        # 批量大小
        self.batch_size = config['training']['batch_size']
        self.K = config['params']['kappa'] # 注意：这里如果没用到K可以删掉，主要是为了获取Strike Price
        self.Strike = 20.0 # 硬编码 Strike Price，或者从 config 读（如果你加了的话）

    def get_interior_points(self):
        """
        采样 PDE 内部点 (Collocation Points)
        返回: (S, v, t) 的 tensor，形状 [Batch_Size, 3]
        """
        # 随机采样 S, v, t
        S = (self.S_max - self.S_min) * torch.rand(self.batch_size, 1) + self.S_min
        v = (self.v_max - self.v_min) * torch.rand(self.batch_size, 1) + self.v_min
        t = self.T * torch.rand(self.batch_size, 1) # t is tau
        
        # 拼接成 [S, v, t]
        domain_points = torch.cat([S, v, t], dim=1)
        
        # 内部点不需要标签（无监督学习），只需要坐标去算 PDE Residual
        return domain_points

    def get_initial_condition_points(self):
        """
        采样初始条件 (Initial Condition / Payoff)
        对应 tau = 0 的时刻
        """
        N_ic = self.batch_size // 2 # 初始点稍微少一点没事，或者跟内部点一样多
        
        S = (self.S_max - self.S_min) * torch.rand(N_ic, 1) + self.S_min
        v = (self.v_max - self.v_min) * torch.rand(N_ic, 1) + self.v_min
        t = torch.zeros(N_ic, 1) # tau = 0
        
        # 计算 Payoff 真值: max(S - K, 0)
        # 注意 Payoff 和 v 无关，只和 S 有关
        payoff = torch.max(S - self.Strike, torch.zeros_like(S))
        
        ic_points = torch.cat([S, v, t], dim=1)
        return ic_points, payoff / 100

    def get_boundary_points(self):
        """
        采样边界条件 (Boundary Conditions)
        主要是 S_min (S=0) 和 S_max (S=80)
        v 方向通常不需要 Dirichlet 边界，让方程自然演化即可
        """
        N_bc = self.batch_size // 4
        
        # --- Lower Boundary (S=0) ---
        S_lb = torch.zeros(N_bc, 1) # S = 0
        v_lb = (self.v_max - self.v_min) * torch.rand(N_bc, 1) + self.v_min
        t_lb = self.T * torch.rand(N_bc, 1)
        val_lb = torch.zeros(N_bc, 1) # 资产为0时，期权价值为0
        
        # --- Upper Boundary (S=S_max) ---
        S_ub = torch.ones(N_bc, 1) * self.S_max
        v_ub = (self.v_max - self.v_min) * torch.rand(N_bc, 1) + self.v_min
        t_ub = self.T * torch.rand(N_bc, 1)
        # S很大时，Call Option ~ S - K*exp(-r*tau)
        r = self.config['params']['r']
        val_ub = (S_ub - self.Strike * torch.exp(-self.config['params']['r'] * t_ub)) / 100.0
        
        # 拼接
        lower = torch.cat([S_lb, v_lb, t_lb], dim=1)
        upper = torch.cat([S_ub, v_ub, t_ub], dim=1)
        
        return lower, val_lb, upper, val_ub