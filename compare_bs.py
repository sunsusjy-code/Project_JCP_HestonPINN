import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==========================================
# 1. 必须把新的动态 PINN 类复制到这里
# (为了避免跨文件夹 import 的路径麻烦，直接复制是最稳的)
# ==========================================
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, S, t):
        return self.net(torch.cat([S, t], dim=1))

# ==========================================
# 2. 定义加载函数 (学会读 Config)
# ==========================================
def load_model(folder_name):
    # A. 拼接路径
    base_path = os.path.join(folder_name, "BlackScholesPINN")
    config_path = os.path.join(base_path, "config.json")
    model_path = os.path.join(base_path, "model.pth")
    
    # B. 读取 Config (为了知道 layers 长什么样)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # C. 根据 Config 里的 layers 初始化模型
    model = PINN(config["layers"])
    
    # D. 加载训练好的权重
    if os.path.exists(model_path):
        print(f"📦 Loading {folder_name} Model...")
        # map_location='cpu' 保证即使没显卡也能跑
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval() # 开启预测模式
        return model, config
    else:
        print(f"⚠️ Warning: {folder_name} model not found.")
        return None, None

# ==========================================
# 3. 主程序
# ==========================================
def run_comparison():
    # 1. 加载两个模型
    causal_model, causal_config = load_model("Causal_BS_Experiment")
    standard_model, standard_config = load_model("Standard_BS_Baseline")
    
    # 2. 准备画布
    S = np.linspace(1, 40, 100)
    t = np.linspace(0, 1, 100)
    S_grid, t_grid = np.meshgrid(S, t)
    
    # 转成 Tensor
    S_tensor = torch.tensor(S_grid.flatten()[:, None], dtype=torch.float32)
    t_tensor = torch.tensor(t_grid.flatten()[:, None], dtype=torch.float32)
    
    # 3. 计算真实解 (Exact Solution) - 用于算 Error
    from scipy.stats import norm
    def black_scholes_exact(S, t, K, r, sigma, T):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        return S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)

    # 从 Causal Config 里读取参数 (假设两者参数一致)
    K = causal_config["K"]
    T = causal_config["T"]
    r = causal_config["r"]
    sigma = causal_config["sigma"]
    
    exact = black_scholes_exact(S_grid.flatten(), t_grid.flatten(), K, r, sigma, T)
    exact = exact.reshape(S_grid.shape)

    # 4. 预测并绘图
    plt.figure(figsize=(18, 5))
    
    # --- 画 Causal ---
    if causal_model:
        with torch.no_grad():
            pred_c = causal_model(S_tensor, t_tensor).numpy().reshape(S_grid.shape)
        error_c = np.abs(pred_c - exact)
        max_err_c = np.max(error_c)
        print(f"✅ Causal Max Error: {max_err_c:.4f}")
        
        plt.subplot(1, 3, 1)
        plt.contourf(t_grid, S_grid, error_c, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title(f'Causal PINN Error\n(Max: {max_err_c:.4f})')
        plt.xlabel('Time t'); plt.ylabel('Asset Price S')

    # --- 画 Standard ---
    if standard_model:
        with torch.no_grad():
            pred_s = standard_model(S_tensor, t_tensor).numpy().reshape(S_grid.shape)
        error_s = np.abs(pred_s - exact)
        max_err_s = np.max(error_s)
        print(f"✅ Standard Max Error: {max_err_s:.4f}")

        plt.subplot(1, 3, 2)
        plt.contourf(t_grid, S_grid, error_s, levels=50, cmap='viridis') # 保持和左边一样的色阶
        plt.title(f'Standard PINN Error\n(Max: {max_err_s:.4f})')
        plt.xlabel('Time t'); plt.ylabel('Asset Price S')

    # --- 画初始条件对比 (t=0) ---
    plt.subplot(1, 3, 3)
    # 取 t=0 的切片 (对应 t_grid 的第一行)
    plt.plot(S, exact[0, :], 'k-', linewidth=2, label='Exact (Payoff)')
    if causal_model:
        plt.plot(S, pred_c[0, :], 'b--', linewidth=2, label='Causal PINN')
    if standard_model:
        plt.plot(S, pred_s[0, :], 'r:', linewidth=2, label='Standard PINN')
    
    plt.title('Prediction at t=0 (Initial Condition)')
    plt.xlabel('Asset Price S')
    plt.ylabel('Call Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('Error_Comparison.png', dpi=300)
    print("🎨 Plot saved as 'Error_Comparison.png'")

if __name__ == "__main__":
    run_comparison()