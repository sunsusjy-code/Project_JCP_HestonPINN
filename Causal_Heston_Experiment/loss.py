import torch
import torch.nn as nn

class CausalLoss(nn.Module):
    def __init__(self, config):
        super(CausalLoss, self).__init__()
        self.config = config
        # 从 config 中读取 Heston 参数
        self.kappa = config['params']['kappa']
        self.theta = config['params']['theta']
        self.sigma = config['params']['sigma'] # 这是 Vol of Vol
        self.rho = config['params']['rho']
        self.r = config['params']['r']
        
        # 读取因果参数 epsilon
        # 注意：在 Standard Baseline 中，我们把这个设为 0.0，让权重恒为 1
        self.epsilon = config['training']['epsilon']

    def get_pde_residual(self, u, S, v, t):
        """
        计算 Heston PDE 的残差 (Residual)
        方程 (在 tau 坐标系下):
        u_t - [ 0.5*v*S^2*u_ss + rho*sigma*v*S*u_sv + 0.5*sigma^2*v*u_vv 
                + r*S*u_s + kappa*(theta-v)*u_v - r*u ] = 0
        """
# === ✅ 修正开始：正确获取三个导数 ===
        # 1. 一阶导数
        # 注意：这里我们不加 [0]，而是直接解包 (unpack)
        grads = torch.autograd.grad(
            u, [S, v, t],
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )
        
        # grads 是一个元组 (du/dS, du/dv, du/dt)
        u_s = grads[0]  # du/dS
        u_v = grads[1]  # du/dv
        u_t = grads[2]  # du/dt
        # === 修正结束 ===

# 2. 二阶导数 (这里也要小心)
        # 对 u_s 求导，输入是 [S, v]，输出是 (d(u_s)/dS, d(u_s)/dv)
        grads_s = torch.autograd.grad(
            u_s, [S, v],
            grad_outputs=torch.ones_like(u_s),
            create_graph=True,
            retain_graph=True
        )
        u_ss = grads_s[0] # d2u/dS2
        u_sv = grads_s[1] # d2u/dSdv (混合导数)
        
        # 对 u_v 求导
        grads_v = torch.autograd.grad(
            u_v, [v],
            grad_outputs=torch.ones_like(u_v),
            create_graph=True,
            retain_graph=True
        )
        u_vv = grads_v[0] # d2u/dv2

        # 3. 组装 Heston PDE
        # Term 1: 扩散项 (Diffusion)
        term_S = 0.5 * v * (S ** 2) * u_ss
        term_v = 0.5 * (self.sigma ** 2) * v * u_vv
        term_Mix = self.rho * self.sigma * v * S * u_sv # 刚性来源
        
        # Term 2: 对流项 (Drift)
        drift_S = self.r * S * u_s
        drift_v = self.kappa * (self.theta - v) * u_v
        
        # Term 3: 贴现项 (Discount)
        discount = self.r * u

        # Residual = u_tau - (右边的项)
        f = u_t - (term_S + term_Mix + term_v + drift_S + drift_v - discount)
        
        return f

    def forward(self, model, domain_points, ic_points, ic_val, boundary_batch=None):
        # --- 1. PDE Loss (Physics) ---
        # 拆解输入: domain_points 是 [S, v, t]
        S = domain_points[:, 0:1].requires_grad_(True)
        v = domain_points[:, 1:2].requires_grad_(True)
        t = domain_points[:, 2:3].requires_grad_(True)
        
        # 预测
        u_pred = model(S, v, t)
        
        # 计算 PDE 残差
        L_t = self.get_pde_residual(u_pred, S, v, t) ** 2
        
        # === 核心逻辑：Standard vs Causal ===
        # 如果 config['training']['epsilon'] == 0.0 (Baseline任务)
        # 那么 weights = exp(0) = 1.0，就完全退化成了 Standard PINN
        
        # 按时间排序 (为了兼容 Causal 逻辑)
        idx = torch.argsort(t.flatten())
        L_sorted = L_t[idx]
        
        # 计算累积 Loss (Causal Integral)
        cumulative_loss = torch.cumsum(L_sorted, dim=0)
        
        # 计算权重 (停止梯度)
        with torch.no_grad():
            weights = torch.exp(-self.epsilon * cumulative_loss)
        
        # 加权 PDE Loss
        loss_pde = torch.mean(weights * L_sorted)
        
        # --- 2. Data Loss (Initial & Boundary) ---
        # 初始条件 (Payoff)
        S_ic = ic_points[:, 0:1]
        v_ic = ic_points[:, 1:2]
        t_ic = ic_points[:, 2:3]
        u_ic_pred = model(S_ic, v_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - ic_val) ** 2)
        
        # 边界条件 (如果有传入的话)
        loss_bc = 0.0
        if boundary_batch is not None:
            # boundary_batch = [lower, val_lb, upper, val_ub]
            lower, val_lb, upper, val_ub = boundary_batch
            
            # Lower Boundary (S=0)
            u_lb = model(lower[:,0:1], lower[:,1:2], lower[:,2:3])
            
            # Upper Boundary (S=S_max)
            u_ub = model(upper[:,0:1], upper[:,1:2], upper[:,2:3])
            
            loss_bc = torch.mean((u_lb - val_lb)**2) + torch.mean((u_ub - val_ub)**2)

        # 总 Loss
        total_loss = loss_pde + loss_ic + loss_bc
        
        # 返回各项 loss 以便记录日志 (Mean W 应该是 1.0)
        return total_loss, loss_pde, loss_ic, torch.mean(weights)