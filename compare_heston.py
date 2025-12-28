import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ==============================================================================
# 1. 定义 Heston PINN 结构 (必须和 training 时一致)
# ==============================================================================
class PINN(nn.Module):
    def __init__(self, config):
        super(PINN, self).__init__()
        layers = config['layers']
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, S, v, t):
        # ⚠️ 注意：这里必须包含我们在 model.py 里加的归一化
        S_norm = S / 100.0
        inputs = torch.cat([S_norm, v, t], dim=1)
        return self.net(inputs)

# ==============================================================================
# 2. 加载两个模型
# ==============================================================================
def load_model(folder_name):
    config_path = os.path.join(folder_name, "config.json")
    model_path = os.path.join(folder_name, "model.pth")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    
    model = PINN(config)
    # 加载权重 (map_location='cpu' 确保在没 GPU 的时候也能跑)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, config

print("⚖️ Loading Models...")
# 确保这里的文件夹名字和你左侧目录完全一致！
try:
    model_std, config_std = load_model("Standard_Heston_Baseline")
    model_causal, config_causal = load_model("Causal_Heston_Experiment")
    print("✅ Models Loaded Successfully.")
except Exception as e:
    print(e)
    exit()

# ==============================================================================
# 3. 准备测试数据 (Grid) - Heston 专属 3D 网格
# ==============================================================================
S_min, S_max = 0.0, 80.0
T_max = 1.0
N = 200

S_test = np.linspace(S_min, S_max, N)
t_test = np.linspace(0, T_max, N)
S_grid, t_grid = np.meshgrid(S_test, t_test)

S_flat = torch.tensor(S_grid.flatten()[:, None], dtype=torch.float32)
t_flat = torch.tensor(t_grid.flatten()[:, None], dtype=torch.float32)

# 🎯 关键点：固定 v 进行切片对比
# 我们选择 High Volatility (v=0.1) 因为这里 Standard 最容易出错
v_val = 0.1
v_flat = torch.full_like(S_flat, v_val)

# ==============================================================================
# 4. 预测与对比
# ==============================================================================
print(f"🔮 Predicting at v={v_val}...")
with torch.no_grad():
    # 预测并还原真实价格 (* 100)
    pred_std = model_std(S_flat, v_flat, t_flat).numpy() * 100.0
    pred_causal = model_causal(S_flat, v_flat, t_flat).numpy() * 100.0

# Reshape
Z_std = pred_std.reshape(N, N)
Z_causal = pred_causal.reshape(N, N)

# 计算差异 (Difference)
Z_diff = np.abs(Z_std - Z_causal)
max_diff = np.max(Z_diff)
print(f"📊 Max Difference between models: {max_diff:.4f}")

# ==============================================================================
# 5. 画图：三张图对比 (Heatmaps)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Standard
im1 = axes[0].contourf(t_grid, S_grid, Z_std, levels=50, cmap='viridis')
axes[0].set_title("Standard PINN Prediction")
axes[0].set_xlabel("Time (t)")
axes[0].set_ylabel("Price (S)")
plt.colorbar(im1, ax=axes[0])

# Plot 2: Causal
im2 = axes[1].contourf(t_grid, S_grid, Z_causal, levels=50, cmap='viridis')
axes[1].set_title("Causal PINN Prediction")
axes[1].set_xlabel("Time (t)")
axes[1].set_ylabel("Price (S)")
plt.colorbar(im2, ax=axes[1])

# Plot 3: Difference (显微镜模式)
im3 = axes[2].contourf(t_grid, S_grid, Z_diff, levels=50, cmap='inferno')
axes[2].set_title(f"Difference |Std - Causal| (Max={max_diff:.2f})")
axes[2].set_xlabel("Time (t)")
axes[2].set_ylabel("Price (S)")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("Heston_Comparison_Heatmap.png")
print("✅ Heston_Comparison_Heatmap.png saved.")

# ==============================================================================
# 6. 画图：2D 折线图 (Line Plot) - 细节对比
# ==============================================================================
plt.figure(figsize=(10, 6))

# 切片 1: t = 0.5
idx_t1 = int(N * 0.5) 
plt.plot(S_test, Z_std[idx_t1, :], 'r--', label='Standard (t=0.5)', linewidth=2)
plt.plot(S_test, Z_causal[idx_t1, :], 'b-', label='Causal (t=0.5)', linewidth=2, alpha=0.7)

# 切片 2: t = 0.9 (接近到期，最难)
idx_t2 = int(N * 0.9)
plt.plot(S_test, Z_std[idx_t2, :], 'm--', label='Standard (t=0.9)', linewidth=2)
plt.plot(S_test, Z_causal[idx_t2, :], 'c-', label='Causal (t=0.9)', linewidth=2, alpha=0.7)

plt.title(f"Price vs Asset S (Slice at v={v_val})")
plt.xlabel("Asset Price S")
plt.ylabel("Option Price C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Heston_Comparison_LinePlot.png")
print("✅ Heston_Comparison_LinePlot.png saved.")