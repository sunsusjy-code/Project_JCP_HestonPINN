import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from collections import OrderedDict
import os


# ================= [最终修正版] =================
# 必须完全匹配你 model.pth 的结构 (2 -> 64 -> 64 -> 1)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),          # 输入 -> 64 (这里之前报错说 mismatch 50 vs 64)
            nn.Tanh(),
            nn.Linear(64, 64),         # 64 -> 64
            nn.Tanh(),
            nn.Linear(64, 1)           # 64 -> 输出
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        return self.net(inputs)

# ================= 2. Black-Scholes 真实解公式 =================
def black_scholes_call(S, t, K, r, sigma, T=1.0):
    tau = T - t
    tau = np.maximum(tau, 1e-10) # 避免除以0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - 
                  K * np.exp(-r * tau) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

# ================= 3. 主程序 =================
def run_comparison():
    # 参数设置 (根据你的 config.json)
    r = 0.05
    sigma = 0.25  # 注意：你的Standard和Causal最好参数一致
    K = 20.0
    T = 1.0
    
    # 路径设置
    path_causal = "Causal_BS_Experiment/BlackScholesPINN/model.pth"
    path_std = "Standard_BS_Baseline/BlackScholesPINN/model.pth"

    # 生成网格 (S: 1~40, t: 0~1)
    S = np.linspace(1, 40, 100)
    t = np.linspace(0, T, 100)
    S_grid, t_grid = np.meshgrid(S, t)
    
    S_tensor = torch.tensor(S_grid.flatten()[:, None], dtype=torch.float32)
    t_tensor = torch.tensor(t_grid.flatten()[:, None], dtype=torch.float32)

    # --- 1. 加载 Causal 模型 (蓝方) ---
    print(f"📦 Loading Causal Model from: {path_causal}")
    model_causal = PINN()
    if os.path.exists(path_causal):
        model_causal.load_state_dict(torch.load(path_causal))
        model_causal.eval()
        pred_causal = model_causal(S_tensor, t_tensor).detach().numpy().reshape(100, 100)
    else:
        print("❌ Error: Causal model not found!")
        return

    # --- 2. 加载 Standard 模型 (红方) ---
    print(f"📦 Loading Standard Model from: {path_std}")
    model_std = PINN()
    has_std = False
    if os.path.exists(path_std):
        try:
            model_std.load_state_dict(torch.load(path_std))
            model_std.eval()
            pred_std = model_std(S_tensor, t_tensor).detach().numpy().reshape(100, 100)
            has_std = True
        except Exception as e:
            print(f"⚠️ Standard model load failed: {e}")
            pred_std = np.zeros((100, 100))
    else:
        print("⚠️ Warning: Standard model file not found. Skipping Standard plot.")
        pred_std = np.zeros((100, 100))

    # --- 3. 计算真实解 & 误差 ---
    exact = black_scholes_call(S_grid, t_grid, K, r, sigma, T)
    error_causal = np.abs(exact - pred_causal)
    if has_std:
        error_std = np.abs(exact - pred_std)
    
    print(f"✅ Causal Max Error: {np.max(error_causal):.4f}")
    if has_std:
        print(f"✅ Standard Max Error: {np.max(error_std):.4f}")

    # ================= 4. 画图 (横向对比) =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: Causal Error (我们要展示的主角)
    im1 = axes[0].contourf(t_grid, S_grid, error_causal, levels=50, cmap='viridis')
    axes[0].set_title(f'Causal PINN Error\n(Max: {np.max(error_causal):.4f})')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Asset Price S')
    plt.colorbar(im1, ax=axes[0])

    # 图2: Standard Error (如果有的话)
    if has_std:
        im2 = axes[1].contourf(t_grid, S_grid, error_std, levels=50, cmap='viridis')
        axes[1].set_title(f'Standard PINN Error\n(Max: {np.max(error_std):.4f})')
    else:
        axes[1].text(0.5, 0.5, 'Standard Model Not Found', ha='center')
        axes[1].set_title('Standard PINN Error')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Asset Price S')
    
    # 图3: t=0 时刻的截面对比 (高光时刻)
    # t=0 在 grid 中对应 index 0
    axes[2].plot(S, exact[0, :], 'k-', label='Exact (Payoff)', linewidth=2)
    axes[2].plot(S, pred_causal[0, :], 'b--', label='Causal PINN', linewidth=2)
    if has_std:
        axes[2].plot(S, pred_std[0, :], 'r:', label='Standard PINN', linewidth=2, alpha=0.7)
    
    axes[2].set_title('Prediction at t=0 (Initial Condition)')
    axes[2].set_xlabel('Asset Price S')
    axes[2].set_ylabel('Call Price')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Error_Comparison.png', dpi=300)
    print("\n🎨 Plot saved as 'Error_Comparison.png'")

if __name__ == "__main__":
    run_comparison()