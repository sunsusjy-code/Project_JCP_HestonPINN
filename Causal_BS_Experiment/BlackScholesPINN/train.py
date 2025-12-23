from data import generate_collocation_points
# [Change 1] 导入新的 Loss 函数
from loss import causal_loss

def train(model, optimizer, config, S_data, t_data, C_data):
    # [Tip] 从配置中读取 epsilon，如果没有就默认为 0 (退化回标准 PINN) 或 10 (开启因果)
    # 建议你在 config.json 里加上 "epsilon": 10
    epsilon_val = config["epsilon"] # 如果 json 里忘了写，直接报错，不许瞎跑！
    
    print(f"Start Training with Epsilon = {epsilon_val}")

    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        S_colloc, t_colloc = generate_collocation_points(config)
        
        # [Change 2] 调用 causal_loss，并传入 epsilon
        # 注意：现在它返回 4 个值，多了一个 mean_weights
        loss, loss_data, loss_pde, mean_weights = causal_loss(
            model, S_data, t_data, C_data, S_colloc, t_colloc, 
            config["r"], config["sigma"], epsilon=epsilon_val
        )
        
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            # [Change 3] 打印 W (权重均值)
            # 如果 W 一直是 1.0，说明是标准 PINN；如果 W < 1.0，说明因果机制生效了
            print(f"Epoch {epoch} | Total: {loss.item():.6f} | Data: {loss_data.item():.6f} | PDE: {loss_pde.item():.6f} | Mean W: {mean_weights.item():.4f}")