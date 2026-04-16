import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的VGG19模型
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:36])  # 取前36层
        
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 提取特征
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [3, 8, 17, 26, 35]:  # 选择特定层的特征
                features.append(x)
        return features

class StyleTransfer:
    def __init__(self):
        self.vgg = VGG19().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def content_loss(self, content_features, generated_features):
        """计算内容损失"""
        loss = 0
        for c, g in zip(content_features, generated_features):
            loss += F.mse_loss(c, g)
        return loss
    
    def gram_matrix(self, features):
        """计算Gram矩阵"""
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram = torch.matmul(features, features.transpose(1, 2))
        return gram / (channels * height * width)
    
    def style_loss(self, style_features, generated_features):
        """计算风格损失"""
        loss = 0
        for s, g in zip(style_features, generated_features):
            s_gram = self.gram_matrix(s)
            g_gram = self.gram_matrix(g)
            loss += F.mse_loss(s_gram, g_gram)
        return loss
    
    def transfer(self, content_image, style_image, iterations=100, content_weight=1, style_weight=1000):
        """执行风格迁移"""
        # 转换为张量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        content_tensor = content_image.to(device)
        style_tensor = style_image.to(device)
        
        # 生成图像初始化为内容图像
        generated = content_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([generated])
        
        # 提取特征
        content_features = self.vgg(content_tensor)
        style_features = self.vgg(style_tensor)
        
        # 优化过程
        iteration = [0]
        while iteration[0] < iterations:
            def closure():
                optimizer.zero_grad()
                generated_features = self.vgg(generated)
                
                # 计算损失
                c_loss = self.content_loss(content_features, generated_features)
                s_loss = self.style_loss(style_features, generated_features)
                total_loss = content_weight * c_loss + style_weight * s_loss
                
                total_loss.backward()
                iteration[0] += 1
                return total_loss
            
            optimizer.step(closure)
        
        return generated