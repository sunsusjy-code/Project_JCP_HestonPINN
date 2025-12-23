import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
from data import generate_synthetic_data
from train import train
from black_scholes import BlackScholesPINN

def main(config_path):
    # 1. Load config from JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Generate synthetic training data
    bs = BlackScholesPINN(config)
    bs.train()
    bs.export()
    print(f"\n✅ Model saved to {config.get('model_path', 'model.pth')}")

# ================= [修正] 画图范围要匹配 Config =================
    print("🎨 Generating plots...")
    
    # [修正点 1] 读取 config 里的范围
    min_s = config.get("min_S", 0)
    max_s = config.get("max_S", 100) # 如果没读到就默认 100
    
    # [修正点 2] 使用正确的范围生成测试点
    S_test = np.linspace(min_s, max_s, 100)  # 从 1 画到 40
    t_test = np.linspace(0, 1, 100)
    S_grid, t_grid = np.meshgrid(S_test, t_test)
    
    # ... (后面的代码不变) ...
    
    # 转换为 Tensor
    S_tensor = torch.tensor(S_grid.flatten()[:, None], dtype=torch.float32)
    t_tensor = torch.tensor(t_grid.flatten()[:, None], dtype=torch.float32)
    
    # 预测
    bs.model.eval()
    with torch.no_grad():
        C_pred = bs.model(S_tensor, t_tensor).numpy().reshape(100, 100)

    # 画图：预测结果热力图
    plt.figure(figsize=(6, 5))
    plt.contourf(t_grid, S_grid, C_pred, levels=50, cmap='viridis')
    plt.colorbar(label='Call Price C(S,t)')
    plt.xlabel('Time t')
    plt.ylabel('Asset Price S')
    plt.title('Causal PINN Prediction')
    plt.savefig('prediction_result.png') # 保存图片
    print("✅ Plot saved to prediction_result.png")
    # =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN on the Black-Scholes equation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (default: config.json)"
    )
    args = parser.parse_args()
    main(args.config)
