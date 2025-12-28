import torch
import torch.optim as optim
from loss import CausalLoss  # 导入我们在 Step 3 写好的类

def train(model, config, data_generator):
    # 1. 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # 2. 初始化 Loss 计算器 (Heston PDE 就在这里面)
    criterion = CausalLoss(config)
    
    # 3. 读取 Epochs
    epochs = config['training']['epochs']
    
    print(f"🚀 Start Training Heston Model...")
    print(f"⚙️  Config: Epsilon={config['training']['epsilon']} (Should be 0.0 for Baseline)")

    # --- Training Loop ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # A. 从 DataGenerator 获取 3D 数据
        # 内部点 (S, v, t)
        domain_points = data_generator.get_interior_points()
        
        # 初始条件 (S, v, 0) 和 Payoff
        ic_points, ic_val = data_generator.get_initial_condition_points()
        
        # 边界条件 (S=0, S=max)
        boundary_batch = data_generator.get_boundary_points()
        
        # B. 计算 Loss (调用 CausalLoss 的 forward)
        # 注意：这里会自动根据 epsilon=0 退化为 Standard Loss
        total_loss, loss_pde, loss_ic, mean_w = criterion(
            model, domain_points, ic_points, ic_val, boundary_batch
        )
        
        # C. 反向传播
        total_loss.backward()
        optimizer.step()

        # D. 打印日志
        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | Total: {total_loss.item():.6f} | "
                  f"PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f} | "
                  f"Mean W: {mean_w.item():.4f}")

    print("✅ Training Finished.")