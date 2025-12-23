import torch
import torch.autograd as autograd

def pde_residual(model, S, t, r, sigma):
    """
    计算 Black-Scholes PDE 的残差 (标准 PINN 逻辑，无需改动)
    """
    # 确保输入需要梯度
    S.requires_grad_(True)
    t.requires_grad_(True)
    
    C = model(S, t)
    
    # 计算一阶导数 dC/dt, dC/dS
    grads = torch.ones_like(C)
    dC_dt = autograd.grad(C, t, grad_outputs=grads, create_graph=True, retain_graph=True)[0]
    dC_dS = autograd.grad(C, S, grad_outputs=grads, create_graph=True, retain_graph=True)[0]
    
    # 计算二阶导数 d^2C/dS^2
    d2C_dS2 = autograd.grad(dC_dS, S, grad_outputs=grads, create_graph=True, retain_graph=True)[0]
    
    # Black-Scholes PDE: dC/dt + 0.5 * sigma^2 * S^2 * d^2C/dS^2 + r * S * dC/dS - r * C = 0
    residual = dC_dt + 0.5 * (sigma**2) * (S**2) * d2C_dS2 + r * S * dC_dS - r * C
    return residual

def causal_loss(model, S_data, t_data, C_data, S_colloc, t_colloc, r, sigma, epsilon=1.0):
    """
    [Causal PINN 核心] 植入因果权重的 Loss 函数
    
    参数:
        epsilon (float): 控制因果惩罚力度的参数 (JAX 代码里的 tol)。
                         epsilon 越大，对“过去没学好”的惩罚越重。
    """
    # 1. Data Loss (边界/初始条件) - 保持不变
    # 这部分是地基，必须稳固，所以通常不需要加因果权重，或者单独算
    C_pred = model(S_data, t_data)
    loss_data = torch.mean((C_pred - C_data)**2)

    # 2. PDE Residual (物理方程残差) - 开始魔改！
    # -------------------------------------------------------------------------
    # [Step A] 排序: Causal PINN 必须严格按照时间 t 从小到大计算
    # -------------------------------------------------------------------------
    # 获取排序后的索引 (按时间 t 升序排列)
    sorted_indices = torch.argsort(t_colloc.flatten())
    #为什么要按时间排序：先处理好前一个时刻的误差，如果很大，就不能处理下一个时刻的误差。不排序的话，后面的torch.cumsum就会乱加

    # 对 S 和 t 进行重排
    S_sorted = S_colloc[sorted_indices]
    t_sorted = t_colloc[sorted_indices]
    
    # 在排序后的点上计算 PDE 残差
    # 注意：必须传入排序后的坐标，否则对应的物理意义就乱了
    pde_res = pde_residual(model, S_sorted, t_sorted, r, sigma)
    
    # -------------------------------------------------------------------------
    # [Step B] 计算因果权重 (The Causal Heart)
    # -------------------------------------------------------------------------
    # L_t: 每个时间点的平方残差 (Loss Vector)
    L_t = pde_res ** 2
    
    # 计算累积损失 (Cumulative Loss)
    # 对应论文公式里的积分或者矩阵乘法 M @ L_t
    # 意思：当前时刻 t 的惩罚，取决于 0 到 t 时刻所有残差的总和
    with torch.no_grad(): # 核心！计算权重时不需要反向传播梯度 (stop_gradient) 为什么截断梯度？不让他去调整参数，不然网络会偷懒，直接把loss的参数调成0，用来降低总的loss，但是实际上loss没变小。
        cumulative_loss = torch.cumsum(L_t, dim=0)
        # 计算权重 W = exp(-epsilon * cumulative_loss)
        weights = torch.exp(-epsilon * cumulative_loss)
    
    # -------------------------------------------------------------------------
    # [Step C] 加权 Loss
    # -------------------------------------------------------------------------
    # 用计算出的因果权重去压制后面的 Loss
    loss_pde = torch.mean(weights * L_t)
    
    # 返回总 Loss, 以及拆分项方便监控 (这里也返回 weights 均值，方便你观察权重是否衰减)
    return loss_data + loss_pde, loss_data, loss_pde, torch.mean(weights)