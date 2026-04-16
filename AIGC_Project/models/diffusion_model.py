import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.hidden_dims = hidden_dims
        
        # 编码器
        self.encoders = nn.ModuleList()
        prev_dim = in_channels
        for dim in hidden_dims:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            prev_dim = dim
        
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_dim, prev_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(prev_dim * 2, prev_dim * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 解码器
        self.decoders = nn.ModuleList()
        prev_dim = prev_dim * 2  # 瓶颈输出是prev_dim * 2
        for dim in reversed(hidden_dims):
            self.decoders.append(nn.Sequential(
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ))
            prev_dim = dim
        
        # 输出层
        self.output = nn.Conv2d(prev_dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # 编码器前向传播
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码器前向传播
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
        
        # 输出
        x = self.output(x)
        return x

class DiffusionModel:
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_steps=1000):
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        
        # 预计算beta、alpha等参数
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffusion(self, x_0, t):
        """前向扩散过程"""
        noise = torch.randn_like(x_0)
        alpha_cum_prod_t = self.alpha_cum_prod[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_t = torch.sqrt(alpha_cum_prod_t) * x_0 + torch.sqrt(1 - alpha_cum_prod_t) * noise
        return x_t, noise
    
    def sample(self, batch_size, image_size):
        """逆向采样过程"""
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.tensor([t], device=device).repeat(batch_size)
            
            # 预测噪声
            noise_pred = self.model(x)
            
            # 计算参数
            alpha_t = self.alphas[t]
            alpha_cum_prod_t = self.alpha_cum_prod[t]
            beta_t = self.betas[t]
            
            # 更新x
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cum_prod_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        
        return x
    
    def train_step(self, x_0, optimizer):
        """训练步骤"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=x_0.device)
        
        # 前向扩散
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 预测噪声
        noise_pred = self.model(x_t)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()