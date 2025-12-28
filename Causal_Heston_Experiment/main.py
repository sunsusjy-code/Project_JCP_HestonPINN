import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 引用我们自己的模块
from model import PINN
from data import DataGenerator
from train import train

# ==============================================================================
# 📝 Logger 类：同时将控制台输出保存到文件
# ==============================================================================
class Logger(object):
    def __init__(self, filename='training.log'):
        self.terminal = sys.stdout
        self.log = open(filename, "a") # "a" 表示追加模式 (Append)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 立即写入硬盘，防止程序崩溃导致日志丢失

    def flush(self):
        # needed for python 3 compatibility
        pass

# ==============================================================================
# 🚀 Main 函数
# ==============================================================================
def main(config_path):
    # --- 1. 加载配置 (Load Config) ---
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"📖 Configuration loaded from {config_path}")

    # --- 2. 初始化组件 (Init Components) ---
    # 这里的 inputs 是 3 (对应 S, v, t)
    print(f"🤖 Initializing Model with layers: {config['layers']}")
    model = PINN(config) 
    
    # 初始化数据生成器 (负责采样和归一化)
    data_gen = DataGenerator(config)

    # --- 3. 开始训练 (Start Training) ---
    # 调用 train.py 里的训练循环
    train(model, config, data_gen)
    
    # --- 4. 保存模型 (Save Model) ---
    torch.save(model.state_dict(), "model.pth")
    print(f"\n✅ Model saved to model.pth")

    # ==========================================================================
    # 🎨 3D 可视化适配：多切片分析 (Multi-Slice Visualization)
    # 目的：为了全方位展示模型在不同波动率环境下的稳定性 (给教授看要做得全面)
    # ==========================================================================
    print("\n🎨 Generating Heston plots (Multi-Slice Analysis)...")
    
    model.eval() # 切换到评估模式
    
    # A. 准备基础网格 (S, t)
    S_min, S_max = 0.0, 80.0
    T_max = 1.0
    
    S_test = np.linspace(S_min, S_max, 100)
    t_test = np.linspace(0, T_max, 100)
    S_grid, t_grid = np.meshgrid(S_test, t_test)
    
    # 拉平网格以便输入网络
    S_flat = S_grid.flatten()[:, None]
    t_flat = t_grid.flatten()[:, None]

    # B. 定义三个波动率切片 (Low, Mean, High)
    # theta = 0.04 (长期均值)
    theta = config['params']['theta']
    
    # 我们画三张图：
    # 1. Low Vol (v=0.01): 市场平静
    # 2. Mean Vol (v=theta): 市场正常
    # 3. High Vol (v=0.1): 市场动荡 (最考验模型稳定性)
    slices = [
        {"val": 0.01,  "name": "Low_Vol"},
        {"val": theta, "name": "Mean_Vol_Theta"},
        {"val": 0.1,   "name": "High_Vol"}
    ]

    for item in slices:
        v_val = item["val"]
        name = item["name"]
        print(f"   ... Plotting Slice: v = {v_val} ({name})")

        # 1. 构造 v 维度输入 (全部填充为当前切片值)
        v_flat = np.full_like(S_flat, v_val)
        
        # 2. 拼接成 [N, 3] 的 Tensor (S, v, t)
        # 注意：这里 S 不需要手动除以 100，因为 model.forward 内部已经写了 S/100
        input_tensor = torch.tensor(
            np.concatenate([S_flat, v_flat, t_flat], axis=1), 
            dtype=torch.float32
        )
        
        # 3. 预测
        with torch.no_grad():
            # 预测输出的是归一化后的价格 (0 ~ 0.6)
            C_pred_norm = model(input_tensor[:,0:1], input_tensor[:,1:2], input_tensor[:,2:3])
            
            # [关键步骤] 还原真实价格！
            # 因为我们在 data.py 里把目标除以了 100，所以这里要乘回 100
            # 这样画出来的图 Colorbar 才是 0~60，符合物理直觉
            C_pred_real = C_pred_norm.numpy() * 100.0
            
            # Reshape 成网格形状
            C_pred_grid = C_pred_real.reshape(100, 100)

        # 4. 画图并保存
        plt.figure(figsize=(6, 5))
        plt.contourf(t_grid, S_grid, C_pred_grid, levels=50, cmap='viridis')
        plt.colorbar(label=f'Call Price (v={v_val})')
        plt.xlabel('Time t (tau)')
        plt.ylabel('Asset Price S')
        plt.title(f'Heston PINN Prediction (v={v_val})')
        
        # 保存图片
        filename = f'prediction_{name}.png'
        plt.savefig(filename)
        plt.close() # 关闭画布，防止内存泄漏
        print(f"      -> Saved to {filename}")

    print("✅ All plots generated successfully.")
    # ==========================================================================

if __name__ == "__main__":
    # --- 启动日志记录 ---
    # 这行代码会将 print 的内容同时写入 training_log.txt
    sys.stdout = Logger("training_log.txt")
    
    parser = argparse.ArgumentParser(description="Train Causal PINN for Heston Model")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    args = parser.parse_args()
    
    main(args.config)