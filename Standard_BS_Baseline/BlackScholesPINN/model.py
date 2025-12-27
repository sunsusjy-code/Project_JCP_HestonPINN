import torch
import torch.nn as nn

class PINN(nn.Module):
    # 修改点 1: __init__ 现在必须接收一个 layers 列表参数
    # 例如: layers = [2, 50, 50, 50, 1]
    def __init__(self, layers):
        super(PINN, self).__init__()
        
        # 1. 准备一个空列表，像搭积木一样把层放进去
        modules = []
        
        # 2. 循环构建每一层
        # len(layers) - 1 代表有几个“缝隙”（即有几层网络）
        for i in range(len(layers) - 1):
            
            # A. 添加全连接层 (Linear)
            # 比如从 2 -> 50, 或者 50 -> 50
            modules.append(nn.Linear(layers[i], layers[i+1]))
            
            # B. 添加激活函数 (Tanh)
            # 注意：最后一层 (输出层) 后面通常不加激活函数，或者根据需求加
            # 这里 logic 是：只要不是最后一层，就加 Tanh
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        
        # 3. 把积木列表用 * 号拆包，塞进 Sequential 里
        self.net = nn.Sequential(*modules)
        

    def forward(self, S, t):
        # 这里的输入拼接保持不变
        return self.net(torch.cat([S, t], dim=1))