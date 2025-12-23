import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import seaborn as sns
import time

# ==========================================
# 1. 配置与参数 (Setup)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Black-Scholes 参数
r = 0.05      # 无风险利率
sigma = 0.25  # 波动率 (我们要学习的目标，这里先作为Ground Truth生成数据)
T = 1.0       # 到期时间
K = 50.0      # 行权价 (Strike Price)

# 训练参数
epochs = 2000
lr = 0.001

# ==========================================
# 2. 辅助函数：BS精确解 (用于对比验证)
# ==========================================
def black_scholes_call(S, t, K, r, sigma, T):
    # t is current time, so time to maturity is T - t
    # Avoid division by zero
    dt = T - t
    # small epsilon for stability
    dt = np.maximum(dt, 1e-10) 
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * dt) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

# ==========================================
# 3. 构建神经网络 (PINN Model)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 输入: S, t (2维) -> 输出: Option Price (1维)
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        
        # 初始化权重 (Xavier Initialization 往往收敛更快)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 训练逻辑 (Training Loop)
# ==========================================
def train():
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    # --- 训练数据采样 (Collocation Points) ---
    # S 范围 [0, 100], t 范围 [0, T]
    N_pde = 5000 # PDE 内部点
    S_pde = torch.rand(N_pde, 1) * 100.0
    t_pde = torch.rand(N_pde, 1) * T
    X_pde = torch.cat([S_pde, t_pde], dim=1).to(device)
    
    # 初始条件 t=T (注意: BS方程通常是反向求解，但为了简单，这里用 T 表示 Maturity)
    # Payoff: max(S-K, 0)
    N_init = 1000
    S_init = torch.rand(N_init, 1) * 100.0
    t_init = torch.ones(N_init, 1) * T # At maturity
    X_init = torch.cat([S_init, t_init], dim=1).to(device)
    u_init = torch.max(S_init - K, torch.zeros_like(S_init)).to(device) # Payoff
    
    # 边界条件 S=0 -> Price=0
    N_bound = 1000
    S_bound = torch.zeros(N_bound, 1)
    t_bound = torch.rand(N_bound, 1) * T
    X_bound = torch.cat([S_bound, t_bound], dim=1).to(device)
    u_bound = torch.zeros(N_bound, 1).to(device)
    
    print(">>> 开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. PDE Loss (Physics)
        X_pde.requires_grad = True
        u_pred = model(X_pde)
        
        # 计算导数
        grads = torch.autograd.grad(u_pred, X_pde, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_S = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_SS = torch.autograd.grad(u_S, X_pde, grad_outputs=torch.ones_like(u_S), create_graph=True)[0][:, 0:1]
        
        # Black-Scholes PDE: u_t + 0.5*sigma^2*S^2*u_SS + r*S*u_S - r*u = 0
        # 注意：这里我们求解的是backward equation的变体，或者视为转换后的forward。
        # 标准形式 Residual:
        f_res = u_t + 0.5 * (sigma**2) * (X_pde[:,0:1]**2) * u_SS + r * X_pde[:,0:1] * u_S - r * u_pred
        loss_pde = torch.mean(f_res ** 2)
        
        # 2. Boundary/Initial Loss (Data)
        u_init_pred = model(X_init)
        loss_init = torch.mean((u_init_pred - u_init) ** 2)
        
        u_bound_pred = model(X_bound)
        loss_bound = torch.mean((u_bound_pred - u_bound) ** 2)
        
        loss = loss_pde + loss_init + loss_bound
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")

    print(f"训练完成. 耗时: {time.time()-start_time:.2f}s")
    return model, loss_history

# ==========================================
# 5. 绘图与验证 (Execution & Plotting)
# ==========================================
if __name__ == "__main__":
    trained_model, loss_hist = train()
    
    # --- 图1: Loss 收敛曲线 ---
    plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label='Total Loss')
    plt.yscale('log')
    plt.title('Training Loss Convergence (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_convergence.png')
    print(">>> 已保存: loss_convergence.png")
    
    # --- 图2: 价格对比面 (Prediction vs Exact) ---
    # 生成网格
    s_plot = np.linspace(0, 100, 100)
    t_plot = np.linspace(0, T, 100)
    S_grid, T_grid = np.meshgrid(s_plot, t_plot)
    
    # 计算精确解
    Exact_grid = black_scholes_call(S_grid, T_grid, K, r, sigma, T)
    
    # 计算预测解
    X_test = torch.tensor(np.stack([S_grid.flatten(), T_grid.flatten()], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        Pred_grid = trained_model(X_test).cpu().numpy().reshape(100, 100)
        
    # 画图
    fig = plt.figure(figsize=(14, 6))
    
    # Exact
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(S_grid, T_grid, Exact_grid, cmap='viridis')
    ax1.set_title('Exact BS Solution')
    ax1.set_xlabel('Price S')
    ax1.set_ylabel('Time t')
    
    # PINN Prediction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(S_grid, T_grid, Pred_grid, cmap='plasma')
    ax2.set_title('PINN Prediction')
    ax2.set_xlabel('Price S')
    ax2.set_ylabel('Time t')
    
    plt.savefig('price_comparison.png')
    print(">>> 已保存: price_comparison.png")
    plt.show()