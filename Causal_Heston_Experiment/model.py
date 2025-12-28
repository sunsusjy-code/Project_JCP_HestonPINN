import torch
import torch.nn as nn

class PINN(nn.Module):
    # 修改点：现在接收 config 字典，而不是直接接收 layers 列表
    def __init__(self, config):
        super(PINN, self).__init__()
        self.config = config
        
        # [关键修正] 从 config 字典里提取 layers 列表
        # 如果 config 里没有 'layers'，这行会报错，提醒你检查 json
        layers = config['layers'] 

        # 1. 准备一个空列表
        modules = []

        # 2. 循环构建每一层
        for i in range(len(layers) - 1):
            
            # A. 添加全连接层
            modules.append(nn.Linear(layers[i], layers[i+1]))

            # B. 添加激活函数 (Tanh)
            # 注意：最后一层后面不加激活函数
            if i < len(layers) - 2:
                modules.append(nn.Tanh())

        # 3. 塞进 Sequential
        self.net = nn.Sequential(*modules)

# ... (前面的 __init__ 不用变) ...

    def forward(self, S, v, t):
        # === 🚑 紧急修复：输入归一化 ===
        # 神经网络喜欢 [0, 1] 左右的小数字
        # S 的物理范围是 [0, 80]，我们除以 100.0 把它缩放到 [0, 0.8]
        S_norm = S / 100.0
        
        # v (0~1) 和 t (0~1) 本身就很小，不需要动
        
        # 拼接归一化后的输入
        inputs = torch.cat([S_norm, v, t], dim=1)
        
        return self.net(inputs)